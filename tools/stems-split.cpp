// stems-split — separate a WAV into stems.
//
//   stems-split --model models/htdemucs-f32.gguf --input song.wav --out stems/
//   stems-split --model models/htdemucs-f32.gguf --input loop.wav --out stems/ --two-stems drums
//
// Writes <out>/<source>.wav for each source (16-bit, rescaled only if it would clip, as
// demucs does; --float32 for float WAVs). --two-stems X writes X.wav and no_X.wav, where
// no_X is the sum of the other sources -- the demucs definition, so it is the model's own
// estimate of the rest, not the mix minus X.
#include "audio.h"
#include "htdemucs.h"
#include "wav.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

namespace {

void usage() {
    fprintf(stderr,
        "usage: stems-split --model M.gguf --input IN.wav [--out DIR]\n"
        "                   [--two-stems SOURCE] [--stems a,b,...] [--shifts N] [--overlap F]\n"
        "                   [--seed N] [--float32] [--device cpu|gpu] [--threads N]\n");
}

void make_dir(const std::string& d) {
#ifdef _WIN32
    _mkdir(d.c_str());
#else
    mkdir(d.c_str(), 0755);
#endif
}

} // namespace

int main(int argc, char** argv) {
    std::string model, input, out = "stems", two, only, device;
    st::SeparateOptions opt;
    bool f32 = false;
    int threads = 0;
    for (int i = 1; i < argc; i++) {
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(2); }
            return argv[++i];
        };
        const char* a = argv[i];
        if (!strcmp(a, "--model") || !strcmp(a, "-m")) model = next();
        else if (!strcmp(a, "--input") || !strcmp(a, "-i")) input = next();
        else if (!strcmp(a, "--out") || !strcmp(a, "-o")) out = next();
        else if (!strcmp(a, "--two-stems")) two = next();
        else if (!strcmp(a, "--stems")) only = next();
        else if (!strcmp(a, "--shifts")) opt.shifts = std::stoi(next());
        else if (!strcmp(a, "--overlap")) opt.overlap = std::stof(next());
        else if (!strcmp(a, "--seed")) opt.seed = (uint32_t)std::stoul(next());
        else if (!strcmp(a, "--float32")) f32 = true;
        else if (!strcmp(a, "--device")) device = next();
        else if (!strcmp(a, "--threads")) threads = std::stoi(next());
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
        else { fprintf(stderr, "unknown argument %s\n", a); usage(); return 2; }
    }
    if (model.empty() || input.empty()) { usage(); return 2; }

    try {
        st::HTDemucs m(model, device.empty() ? nullptr : device.c_str(), threads);
        const auto& src = m.sources();
        int n = 0, ch = 0, sr = 0;
        std::vector<float> audio = st::read_wav_planar(input, n, ch, sr);
        audio = st::to_channels(audio, n, ch, m.audio_channels());
        int len = n;
        if (sr != m.samplerate()) audio = st::resample_planar(audio, n, m.audio_channels(), sr, m.samplerate(), len);
        const int C = m.audio_channels(), R = m.samplerate();

        int two_idx = -1;
        if (!two.empty()) {
            for (size_t s = 0; s < src.size(); s++) if (src[s] == two) two_idx = (int)s;
            if (two_idx < 0) { fprintf(stderr, "--two-stems: %s is not one of this model's sources\n", two.c_str()); return 2; }
        }

        fprintf(stderr, "[stems] %s (%zu stems, %d model%s) on %s: %.1f s of audio\n", m.name().c_str(),
                src.size(), m.n_models(), m.n_models() > 1 ? "s" : "", m.backend_name(), (double)len / R);
        opt.progress = [](int done, int total) {
            fprintf(stderr, "\r[stems] segment %d/%d", done, total);
            if (done == total) fprintf(stderr, "\n");
        };
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> stems = m.separate(audio.data(), len, opt);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[stems] separated in %.2f s (%.1fx realtime)\n", secs, (double)len / R / secs);

        make_dir(out);
        const size_t per = (size_t)C * len;
        auto emit = [&](const std::string& name, std::vector<float> data) {
            if (!f32) st::rescale_if_clipping(data.data(), data.size());
            const std::string path = out + "/" + name + ".wav";
            st::write_file(path, st::wav_bytes(data.data(), len, C, R, f32));
            printf("%s\n", path.c_str());
        };
        if (two_idx >= 0) {
            std::vector<float> keep(stems.begin() + two_idx * per, stems.begin() + (two_idx + 1) * per);
            std::vector<float> rest(per, 0.0f);
            for (size_t s = 0; s < src.size(); s++)
                if ((int)s != two_idx)
                    for (size_t i = 0; i < per; i++) rest[i] += stems[s * per + i];
            emit(two, std::move(keep));
            emit("no_" + two, std::move(rest));
        } else {
            for (size_t s = 0; s < src.size(); s++) {
                if (!only.empty() && ("," + only + ",").find("," + src[s] + ",") == std::string::npos) continue;
                emit(src[s], std::vector<float>(stems.begin() + s * per, stems.begin() + (s + 1) * per));
            }
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
