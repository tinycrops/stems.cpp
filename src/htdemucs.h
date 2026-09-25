// htdemucs.h — Hybrid Transformer Demucs (v4) source separation on ggml.
//
// One class covers every published HTDemucs checkpoint: `htdemucs` (4 stems), `htdemucs_6s`
// (6 stems) and `htdemucs_ft` (a bag of four per-source fine-tunes). The network runs as one
// ggml graph per 7.8 s segment; the spectrogram, normalisation and the segment/overlap/shift
// bookkeeping of demucs.apply run on the host, reproduced exactly so outputs match
// `demucs --shifts 0` sample for sample (see docs/PARITY.md).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace st {

struct SeparateOptions {
    // Random time shifts averaged for time equivariance (demucs --shifts). Each shift is a
    // full extra pass. 0 = one deterministic pass (the default here; demucs' CLI uses 1).
    int shifts = 0;
    // Overlap between consecutive segments (demucs --overlap).
    float overlap = 0.25f;
    // Seed for the shift offsets.
    uint32_t seed = 0;
    // Called with (done, total) segment passes.
    std::function<void(int, int)> progress;
    // Polled before every segment pass; returning true abandons the job with Cancelled.
    std::function<bool()> should_cancel;
};

struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("cancelled") {}
};

class HTDemucs {
public:
    // device: nullptr/"" = best GPU (STEMS_DEVICE / STEMS_GPU env vars apply), "cpu" = CPU.
    explicit HTDemucs(const std::string& gguf_path, const char* device = nullptr, int cpu_threads = 0);
    ~HTDemucs();
    HTDemucs(const HTDemucs&) = delete;
    HTDemucs& operator=(const HTDemucs&) = delete;

    const std::string& name() const;
    const std::vector<std::string>& sources() const;
    int samplerate() const;
    int audio_channels() const;
    int segment_samples() const;
    int n_models() const;
    const char* backend_name() const;

    // Full separation. mix is planar [audio_channels][len] at samplerate(); returns planar
    // [S][audio_channels][len]. Normalises by the mono mix exactly as `demucs` does.
    std::vector<float> separate(const float* mix, int len, const SeparateOptions& opt = {}) const;

    // One raw model forward on exactly segment_samples() of already-normalised audio:
    // planar [C][N] -> [S][C][N]. For the parity tests. If dump_dir is non-empty, the
    // intermediate activations are written there as raw float32 (same memory layout as
    // the PyTorch arrays in tests/refs).
    std::vector<float> forward(int model_index, const float* seg, const std::string& dump_dir = "") const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace st
