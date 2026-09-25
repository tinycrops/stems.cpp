// audio.h — getting arbitrary WAV input into the model's format, and stems back out.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace st {

// Band-limited resampling: a Kaiser-windowed sinc, 32 zero crossings, cutoff at 0.95 of the
// lower Nyquist. Only used when the input is not already at the model rate (44.1 kHz);
// separation quality depends on not aliasing the top octave into the stems.
inline std::vector<float> resample_planar(const std::vector<float>& in, int n, int ch,
                                          int src_rate, int dst_rate, int& out_n) {
    if (src_rate == dst_rate) { out_n = n; return in; }
    const double ratio = (double)dst_rate / src_rate;
    out_n = (int)std::llround(n * ratio);
    const double cutoff = 0.95 * std::min(1.0, ratio);
    const int zc = 32;
    const double half = zc / cutoff;            // half-width in input samples
    const double beta = 8.6;
    auto bessel_i0 = [](double x) {
        double s = 1, t = 1;
        for (int k = 1; k < 30; k++) { t *= (x / (2 * k)) * (x / (2 * k)); s += t; }
        return s;
    };
    const double i0b = bessel_i0(beta);
    // Tabulate the kernel over |x| in [0, half] once; interpolate linearly between entries.
    const int table_n = 1 << 15;
    std::vector<double> table(table_n + 2);
    for (int t = 0; t <= table_n + 1; t++) {
        const double x = std::min((double)t / table_n, 1.0) * half;
        const double r = x / half;
        const double arg = M_PI * x * cutoff;
        const double sinc = arg < 1e-12 ? 1.0 : std::sin(arg) / arg;
        table[t] = r >= 1 ? 0.0 : sinc * bessel_i0(beta * std::sqrt(1 - r * r)) / i0b * cutoff;
    }
    std::vector<float> out((size_t)out_n * ch);
    std::vector<double> acc(ch);
    for (int i = 0; i < out_n; i++) {
        const double c = i / ratio;
        const int lo = std::max(0, (int)std::ceil(c - half)), hi = std::min(n - 1, (int)std::floor(c + half));
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int j = lo; j <= hi; j++) {
            const double pos = std::fabs(j - c) / half * table_n;
            const int t = (int)pos;
            if (t >= table_n) continue;
            const double fr = pos - t;
            const double w = table[t] + (table[t + 1] - table[t]) * fr;
            for (int k = 0; k < ch; k++) acc[k] += w * in[(size_t)k * n + j];
        }
        for (int k = 0; k < ch; k++) out[(size_t)k * out_n + i] = (float)acc[k];
    }
    return out;
}

// demucs.audio.convert_audio_channels: mono is duplicated, extra channels are dropped.
inline std::vector<float> to_channels(const std::vector<float>& in, int n, int ch, int want) {
    if (ch == want) return in;
    std::vector<float> out((size_t)want * n);
    for (int c = 0; c < want; c++) {
        const int src = ch == 1 ? 0 : std::min(c, ch - 1);
        std::copy(in.begin() + (size_t)src * n, in.begin() + (size_t)(src + 1) * n,
                  out.begin() + (size_t)c * n);
    }
    return out;
}

// Interleaved WAV bytes: 16-bit PCM or 32-bit float.
inline std::string wav_bytes(const float* planar, int n, int ch, int sr, bool float32) {
    const int bps = float32 ? 4 : 2;
    const uint32_t data = (uint32_t)n * ch * bps;
    std::string b;
    b.reserve(44 + data);
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) b.push_back((char)((v >> (8 * i)) & 0xff)); };
    auto u16 = [&](uint16_t v) { b.push_back((char)(v & 0xff)); b.push_back((char)(v >> 8)); };
    b += "RIFF"; u32(36 + data); b += "WAVEfmt "; u32(16);
    u16(float32 ? 3 : 1); u16((uint16_t)ch); u32((uint32_t)sr); u32((uint32_t)sr * ch * bps);
    u16((uint16_t)(ch * bps)); u16((uint16_t)(bps * 8));
    b += "data"; u32(data);
    for (int i = 0; i < n; i++)
        for (int c = 0; c < ch; c++) {
            const float v = planar[(size_t)c * n + i];
            if (float32) {
                const char* p = reinterpret_cast<const char*>(&v);
                b.append(p, 4);
            } else {
                const float cl = std::max(-1.0f, std::min(1.0f, v));
                u16((uint16_t)(int16_t)std::lrint(cl * 32767.0f));
            }
        }
    return b;
}

// demucs --clip-mode rescale: scale a stem down only if it would clip.
inline void rescale_if_clipping(float* planar, size_t n) {
    float peak = 0;
    for (size_t i = 0; i < n; i++) peak = std::max(peak, std::fabs(planar[i]));
    const float d = std::max(1.01f * peak, 1.0f);
    if (d > 1.0f) for (size_t i = 0; i < n; i++) planar[i] /= d;
}

inline void write_file(const std::string& path, const std::string& bytes) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    const size_t w = fwrite(bytes.data(), 1, bytes.size(), f);
    fclose(f);
    if (w != bytes.size()) throw std::runtime_error("short write to " + path);
}

} // namespace st
