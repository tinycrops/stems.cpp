// stft.h — torch.stft / torch.istft as HTDemucs calls them, on the host.
//
// Only what HTDemucs needs: n_fft a power of two, Hann window (torch's periodic default),
// center=True with reflect padding, normalized=True, onesided. Double precision inside; the
// spectrogram is tiny next to the network (336 frames per 7.8 s segment), so there is no
// reason to put it on the GPU or to trade accuracy for speed here.
#pragma once

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace st {

using cplx = std::complex<double>;

class FFT {
public:
    explicit FFT(int n) : n_(n), rev_(n), tw_(n / 2) {
        if (n < 2 || (n & (n - 1))) throw std::invalid_argument("FFT size must be a power of two");
        int bits = 0;
        while ((1 << bits) < n) bits++;
        for (int i = 0; i < n; i++) {
            int r = 0;
            for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
            rev_[i] = r;
        }
        for (int i = 0; i < n / 2; i++) tw_[i] = std::polar(1.0, -2.0 * M_PI * i / n);
    }
    int size() const { return n_; }

    // In-place, unnormalised. inverse=true uses conj twiddles (no 1/n).
    void run(std::vector<cplx>& a, bool inverse) const {
        for (int i = 0; i < n_; i++) if (i < rev_[i]) std::swap(a[i], a[rev_[i]]);
        for (int len = 2; len <= n_; len <<= 1) {
            const int step = n_ / len;
            for (int i = 0; i < n_; i += len) {
                for (int j = 0; j < len / 2; j++) {
                    cplx w = tw_[j * step];
                    if (inverse) w = std::conj(w);
                    const cplx u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                }
            }
        }
    }

private:
    int n_;
    std::vector<int> rev_;
    std::vector<cplx> tw_;
};

// torch's reflect padding: the edge sample is not repeated.
inline std::vector<double> reflect_pad(const float* x, int n, int left, int right) {
    if (n <= left || n <= right) throw std::runtime_error("reflect pad larger than the signal");
    std::vector<double> y((size_t)n + left + right);
    for (int i = 0; i < (int)y.size(); i++) {
        int j = i - left;
        if (j < 0) j = -j;
        if (j >= n) j = 2 * (n - 1) - j;
        y[i] = x[j];
    }
    return y;
}

inline std::vector<double> hann_periodic(int n) {
    std::vector<double> w(n);
    for (int i = 0; i < n; i++) w[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / n);
    return w;
}

// HTDemucs._spec for one channel: returns [F = nfft/2][T = ceil(len/hop)] complex, laid out
// f-major (spec[f * T + t]). Drops the Nyquist bin and the two edge frames on each side,
// exactly as the model does, so the output length is len / hop when hop divides len.
inline std::vector<cplx> htdemucs_spec(const FFT& fft, const float* x, int len, int& n_frames) {
    const int nfft = fft.size(), hl = nfft / 4;
    const int le = (len + hl - 1) / hl;
    const int pad = hl / 2 * 3;
    // pad1d(x, (pad, pad + le*hl - len), reflect) then torch.stft's own center pad of nfft/2.
    std::vector<float> xf;
    {
        std::vector<double> p = reflect_pad(x, len, pad, pad + le * hl - len);
        xf.assign(p.begin(), p.end());
    }
    std::vector<double> xp = reflect_pad(xf.data(), (int)xf.size(), nfft / 2, nfft / 2);
    const std::vector<double> win = hann_periodic(nfft);
    const double norm = 1.0 / std::sqrt((double)nfft);
    const int F = nfft / 2;
    n_frames = le;
    std::vector<cplx> out((size_t)F * le);
    std::vector<cplx> buf(nfft);
    for (int t = 0; t < le; t++) {
        const size_t start = (size_t)(t + 2) * hl;       // skip the first two frames
        for (int i = 0; i < nfft; i++) buf[i] = cplx(xp[start + i] * win[i], 0.0);
        fft.run(buf, false);
        for (int f = 0; f < F; f++) out[(size_t)f * le + t] = buf[f] * norm;
    }
    return out;
}

// HTDemucs._ispec for one channel: spec [F][T] (f-major, Nyquist dropped) -> `length` samples.
inline std::vector<float> htdemucs_ispec(const FFT& fft, const std::vector<cplx>& spec,
                                         int n_frames, int length) {
    const int nfft = fft.size(), hl = nfft / 4, F = nfft / 2;
    const int pad = hl / 2 * 3;
    const int le_out = hl * ((length + hl - 1) / hl) + 2 * pad;
    const int frames = n_frames + 4;                     // two zero frames each side
    const std::vector<double> win = hann_periodic(nfft);
    const size_t full = (size_t)nfft + (size_t)hl * (frames - 1);
    std::vector<double> y(full, 0.0), env(full, 0.0);
    std::vector<cplx> buf(nfft);
    const double norm = std::sqrt((double)nfft) / nfft;  // undo normalized=True, then irfft's 1/n
    for (int t = 0; t < frames; t++) {
        const int st = t - 2;
        const size_t off = (size_t)t * hl;
        if (st >= 0 && st < n_frames) {
            buf[0] = cplx(spec[(size_t)0 * n_frames + st].real(), 0.0);   // c2r ignores DC imag
            for (int f = 1; f < F; f++) {
                buf[f] = spec[(size_t)f * n_frames + st];
                buf[nfft - f] = std::conj(buf[f]);
            }
            buf[F] = cplx(0.0, 0.0);                     // the Nyquist bin HTDemucs zero-pads
            fft.run(buf, true);
            for (int i = 0; i < nfft; i++) y[off + i] += buf[i].real() * norm * win[i];
        }
        for (int i = 0; i < nfft; i++) env[off + i] += win[i] * win[i];
    }
    // center=True drops nfft/2 at the start; then HTDemucs crops `pad` more.
    std::vector<float> out(length);
    for (int i = 0; i < length; i++) {
        const size_t j = (size_t)nfft / 2 + pad + i;
        if ((int)(j - nfft / 2) >= le_out || j >= full) { out[i] = 0.0f; continue; }
        out[i] = env[j] > 1e-11 ? (float)(y[j] / env[j]) : 0.0f;
    }
    return out;
}

} // namespace st
