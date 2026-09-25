// Ported from betweentwomidnights/audiocraft.cpp (MIT, (c) 2026 betweentwomidnights), itself from sa3.cpp.
// Ported from sa3.cpp src/wav.h (namespace sa3 -> ac, SA3_* env vars -> STEMS_*).
// wav.h — minimal 16-bit PCM WAV writer.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace st {

namespace wav_detail {

struct FileHandle {
    FILE* f = nullptr;
    ~FileHandle() { if (f) fclose(f); }
};

inline uint16_t u16le(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

inline uint32_t u32le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline int32_t s24le(const uint8_t* p) {
    uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
    if (v & 0x800000u) v |= 0xff000000u;
    return (int32_t)v;
}

inline int32_t s32le(const uint8_t* p) {
    uint32_t u = u32le(p);
    int32_t v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

inline void read_exact(FILE* f, void* dst, size_t bytes, const std::string& what) {
    if (bytes && fread(dst, 1, bytes, f) != bytes) {
        throw std::runtime_error("unexpected end of file while reading " + what);
    }
}

inline std::vector<int16_t> interleave_planar_i16(const float* data, int n_samples,
                                                  int n_ch, int channel_stride) {
    if (!data || n_samples < 0 || n_ch < 1 || channel_stride < n_samples)
        throw std::invalid_argument("invalid planar audio geometry");
    auto clip = [](float v) {
        v = v < -1.f ? -1.f : (v > 1.f ? 1.f : v);
        return (int16_t)(v * 32767.0f);
    };
    std::vector<int16_t> inter((size_t)n_samples * n_ch);
    for (int s = 0; s < n_samples; ++s)
        for (int c = 0; c < n_ch; ++c)
            inter[(size_t)s*n_ch + c] = clip(data[(size_t)c*channel_stride + s]);
    return inter;
}

} // namespace wav_detail

// Write interleaved 16-bit WAV from planar f32 channels whose source planes may be
// longer than the requested output. channel_stride is the distance between channel
// starts; this matters when cropping a decoded buffer without first repacking it.
inline void write_wav_planar_strided(const std::string& path, const float* data, int n_samples,
                                     int n_ch, int sample_rate, int channel_stride) {
    std::vector<int16_t> inter = wav_detail::interleave_planar_i16(
        data, n_samples, n_ch, channel_stride);

    const uint32_t data_bytes = (uint32_t)inter.size() * sizeof(int16_t);
    const uint32_t byte_rate  = sample_rate * n_ch * 2;
    FILE* f = fopen(path.c_str(), "wb");
    auto u32 = [&](uint32_t v){ fwrite(&v, 4, 1, f); };
    auto u16 = [&](uint16_t v){ fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); u32(36 + data_bytes); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); u32(16); u16(1); u16((uint16_t)n_ch);
    u32(sample_rate); u32(byte_rate); u16((uint16_t)(n_ch*2)); u16(16);
    fwrite("data", 1, 4, f); u32(data_bytes);
    fwrite(inter.data(), 1, data_bytes, f);
    fclose(f);
}

// Write a complete tightly-packed planar buffer.
inline void write_wav_planar(const std::string& path, const float* data, int n_samples,
                             int n_ch, int sample_rate) {
    write_wav_planar_strided(path, data, n_samples, n_ch, sample_rate, n_samples);
}

// Same WAV bytes in memory (for an HTTP response body, etc.) instead of a file.
inline std::string wav_planar_bytes(const float* data, int n_samples, int n_ch, int sample_rate) {
    auto clip = [](float v){ v = v < -1.f ? -1.f : (v > 1.f ? 1.f : v); return (int16_t)(v * 32767.0f); };
    std::vector<int16_t> inter((size_t)n_samples * n_ch);
    for (int s = 0; s < n_samples; s++)
        for (int c = 0; c < n_ch; c++)
            inter[(size_t)s*n_ch + c] = clip(data[(size_t)c*n_samples + s]);
    const uint32_t data_bytes = (uint32_t)inter.size() * sizeof(int16_t);
    const uint32_t byte_rate  = sample_rate * n_ch * 2;
    std::string out;
    auto bytes = [&](const void* p, size_t n){ out.append((const char*)p, n); };
    auto u32 = [&](uint32_t v){ bytes(&v, 4); };
    auto u16 = [&](uint16_t v){ bytes(&v, 2); };
    bytes("RIFF", 4); u32(36 + data_bytes); bytes("WAVE", 4);
    bytes("fmt ", 4); u32(16); u16(1); u16((uint16_t)n_ch);
    u32(sample_rate); u32(byte_rate); u16((uint16_t)(n_ch*2)); u16(16);
    bytes("data", 4); u32(data_bytes);
    bytes(inter.data(), data_bytes);
    return out;
}

// Parse a PCM/float WAV already in memory into planar f32 channels
// (ch0[0..n-1], ch1[0..n-1], ...), the same layout write_wav_planar consumes.
//
// Memory rather than a path because both services move audio as base64 in JSON: the bytes
// arrive over a socket and never touch the filesystem. `read_wav_planar` is this with a
// file read in front of it.
//
// `what` only names the source in error messages.
inline std::vector<float> parse_wav_planar(const std::string& bytes, int& n_samples,
                                           int& n_ch, int& sample_rate,
                                           const std::string& what = "wav") {
    using namespace wav_detail;

    const uint8_t* base = (const uint8_t*)bytes.data();
    const size_t size = bytes.size();
    size_t at = 0;
    auto need = [&](size_t n) {
        if (at + n > size) throw std::runtime_error(what + ": truncated WAV");
    };

    need(12);
    if (memcmp(base, "RIFF", 4) || memcmp(base + 8, "WAVE", 4))
        throw std::runtime_error(what + " is not a RIFF/WAVE file");
    at = 12;

    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    const uint8_t* data = nullptr;
    size_t data_size = 0;

    while (at + 8 <= size) {
        const uint8_t* id = base + at;
        const uint32_t sz = u32le(base + at + 4);
        at += 8;
        // A declared size past the end is a truncated file; take what is actually there
        // rather than reading off the end, which is how a half-uploaded chunk arrives.
        const size_t avail = std::min((size_t)sz, size - at);
        if (!memcmp(id, "fmt ", 4)) {
            if (avail < 16) throw std::runtime_error(what + ": invalid fmt chunk");
            fmt = u16le(base + at);
            ch = u16le(base + at + 2);
            rate = u32le(base + at + 4);
            bits = u16le(base + at + 14);
            // WAVE_FORMAT_EXTENSIBLE stores the real codec in the SubFormat GUID.
            if (fmt == 0xfffe && avail >= 40) {
                const uint16_t sub_format = u16le(base + at + 24);
                if (sub_format == 1 || sub_format == 3) fmt = sub_format;
            }
        } else if (!memcmp(id, "data", 4)) {
            data = base + at;
            data_size = avail;
        }
        at += (size_t)sz + (sz & 1);   // chunks are word-aligned
    }

    if ((fmt != 1 && fmt != 3) || ch == 0 || rate == 0 || !data || data_size == 0) {
        throw std::runtime_error(what + ": unsupported or empty WAV (fmt=" + std::to_string(fmt) +
                                 " bits=" + std::to_string(bits) + " ch=" + std::to_string(ch) + ")");
    }
    if (!(bits == 16 || bits == 24 || bits == 32 || (fmt == 3 && bits == 64))) {
        throw std::runtime_error(what + ": unsupported WAV bit depth " + std::to_string(bits));
    }
    if (fmt == 3 && !(bits == 32 || bits == 64)) {
        throw std::runtime_error(what + ": unsupported float WAV bit depth " + std::to_string(bits));
    }

    const size_t bytes_per_sample = bits / 8;
    const size_t frame_bytes = bytes_per_sample * ch;
    if (frame_bytes == 0 || data_size < frame_bytes) {
        throw std::runtime_error(what + ": WAV data chunk is too small");
    }

    n_ch = ch;
    sample_rate = (int)rate;
    n_samples = (int)(data_size / frame_bytes);
    std::vector<float> planar((size_t)n_samples * n_ch);
    for (int s = 0; s < n_samples; s++) {
        for (int c = 0; c < n_ch; c++) {
            const uint8_t* p = data + ((size_t)s * n_ch + c) * bytes_per_sample;
            float v = 0.0f;
            if (fmt == 3) {
                if (bits == 32) {
                    std::memcpy(&v, p, sizeof(float));
                } else {
                    double d = 0.0;
                    std::memcpy(&d, p, sizeof(double));
                    v = (float)d;
                }
            } else if (bits == 16) {
                int16_t sample = (int16_t)u16le(p);
                v = sample / 32768.0f;
            } else if (bits == 24) {
                v = s24le(p) / 8388608.0f;
            } else {
                v = s32le(p) / 2147483648.0f;
            }
            planar[(size_t)c*n_samples + s] = std::max(-1.0f, std::min(1.0f, v));
        }
    }
    return planar;
}

inline std::vector<float> read_wav_planar(const std::string& path, int& n_samples,
                                          int& n_ch, int& sample_rate) {
    wav_detail::FileHandle fh{fopen(path.c_str(), "rb")};
    if (!fh.f) throw std::runtime_error("cannot open " + path);
    std::string bytes;
    char buf[1 << 16];
    for (;;) {
        const size_t got = fread(buf, 1, sizeof(buf), fh.f);
        if (!got) break;
        bytes.append(buf, got);
    }
    return parse_wav_planar(bytes, n_samples, n_ch, sample_rate, path);
}

} // namespace st
