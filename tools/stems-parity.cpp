// stems-parity — compare the ggml port against PyTorch reference arrays.
//
//   stems-parity --model models/htdemucs-f32.gguf --refs tests/refs/htdemucs \
//                [--wav tests/data/test.wav] [--dump /tmp/taps]
//
// Checks one raw segment forward (seg_in.npy -> seg_out.npy) and, with --wav, the whole
// separate path against full.npy. Reports cosine similarity and SNR (dB) per source.
// Exit status is non-zero if any cosine falls below --min-cos (default 0.9999).
#include "htdemucs.h"
#include "npy.h"
#include "wav.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

struct Stat { double cos, snr, maxabs; };

Stat compare(const float* a, const float* ref, size_t n) {
    double dot = 0, na = 0, nr = 0, err = 0, mx = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = (double)a[i] - ref[i];
        dot += (double)a[i] * ref[i];
        na += (double)a[i] * a[i];
        nr += (double)ref[i] * ref[i];
        err += d * d;
        mx = std::max(mx, std::fabs(d));
    }
    return {dot / std::sqrt(na * nr + 1e-30), 10.0 * std::log10(nr / (err + 1e-30)), mx};
}

bool report(const st::HTDemucs& m, const char* what, const float* got, const float* ref,
            size_t per_source, double min_cos) {
    bool ok = true;
    for (size_t s = 0; s < m.sources().size(); s++) {
        const Stat r = compare(got + s * per_source, ref + s * per_source, per_source);
        ok = ok && r.cos >= min_cos;
        printf("  %-5s %-8s cos %.7f  snr %6.1f dB  max|d| %.2e%s\n", what, m.sources()[s].c_str(),
               r.cos, r.snr, r.maxabs, r.cos >= min_cos ? "" : "  FAIL");
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    std::string model, refs, wav, dump, device;
    double min_cos = 0.9999;
    for (int i = 1; i < argc; i++) {
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); } return argv[++i]; };
        if (!strcmp(argv[i], "--model")) model = next();
        else if (!strcmp(argv[i], "--refs")) refs = next();
        else if (!strcmp(argv[i], "--wav")) wav = next();
        else if (!strcmp(argv[i], "--dump")) dump = next();
        else if (!strcmp(argv[i], "--device")) device = next();
        else if (!strcmp(argv[i], "--min-cos")) min_cos = std::stod(next());
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (model.empty() || refs.empty()) {
        fprintf(stderr, "usage: stems-parity --model M.gguf --refs DIR [--wav W.wav] [--dump DIR] [--device cpu]\n");
        return 2;
    }
    try {
        st::HTDemucs m(model, device.empty() ? nullptr : device.c_str());
        printf("%s on %s\n", m.name().c_str(), m.backend_name());
        bool ok = true;

        std::vector<size_t> sh;
        const std::vector<float> seg = st::read_npy_f32(refs + "/seg_in.npy", sh);
        const std::vector<float> seg_ref = st::read_npy_f32(refs + "/seg_out.npy", sh);
        auto t0 = std::chrono::steady_clock::now();
        const std::vector<float> out = m.forward(0, seg.data(), dump);
        auto t1 = std::chrono::steady_clock::now();
        printf("segment forward: %.2f s\n", std::chrono::duration<double>(t1 - t0).count());
        ok &= report(m, "seg", out.data(), seg_ref.data(), out.size() / m.sources().size(), min_cos);

        if (!wav.empty()) {
            int n = 0, ch = 0, sr = 0;
            std::vector<float> audio = st::read_wav_planar(wav, n, ch, sr);
            if (sr != m.samplerate() || ch != m.audio_channels()) {
                fprintf(stderr, "parity wav must be %d Hz, %d channels\n", m.samplerate(), m.audio_channels());
                return 2;
            }
            const std::vector<float> full_ref = st::read_npy_f32(refs + "/full.npy", sh);
            t0 = std::chrono::steady_clock::now();
            const std::vector<float> full = m.separate(audio.data(), n);
            t1 = std::chrono::steady_clock::now();
            printf("separate %.1f s of audio: %.2f s\n", (double)n / sr,
                   std::chrono::duration<double>(t1 - t0).count());
            ok &= report(m, "full", full.data(), full_ref.data(), full.size() / m.sources().size(), min_cos);
        }
        printf(ok ? "PASS\n" : "FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
