// stems-server -- HTDemucs source separation over HTTP, on :8010.
//
// Same shape as audiocraft.cpp's services (one job at a time, a session to poll, base64 WAV
// in and out) so gary4juce can drive it with the client code it already has for terry:
//
//   GET  /health
//   GET  /api/models                      the GGUFs in --models-dir and their sources
//   POST /separate                        JSON in, JSON out with every stem, synchronous
//   POST /api/juce/separate_audio         -> {success, session_id}, runs in background
//   GET  /api/juce/poll_status/<id>       -> progress, then {stems: {name: b64 wav}}
//
// Request fields: audio_data (base64 WAV, any rate/channel count), and optionally
// model ("htdemucs", "htdemucs_6s", "htdemucs_ft"; default htdemucs), two_stems (a source
// name: returns it and no_<name>), stems (array of source names to return), shifts,
// overlap, seed, float32 (return float WAVs instead of 16-bit).
//
// The model stays resident between requests: HTDemucs is 168 MB of weights, and loading it
// is most of the latency on a short loop. Asking for a different model swaps it.
#include "audio.h"
#include "htdemucs.h"
#include "serve/http.h"

#include "httplib.h"
#include "yyjson.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

namespace {

using namespace st::serve;

std::string g_models_dir;
std::string g_device;
Sessions g_sessions;
std::unique_ptr<st::HTDemucs> g_model;     // guarded by g_sessions.gpu()
std::string g_model_path;

std::string env_or(const char* key, const std::string& fallback) {
    const char* v = getenv(key);
    return (v && *v) ? std::string(v) : fallback;
}

bool file_exists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

std::vector<std::string> list_ggufs() {
    std::vector<std::string> out;
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((g_models_dir + "\\*.gguf").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do out.push_back(fd.cFileName); while (FindNextFileA(h, &fd));
        FindClose(h);
    }
#else
    if (DIR* d = opendir(g_models_dir.c_str())) {
        while (dirent* e = readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0) out.push_back(n);
        }
        closedir(d);
    }
#endif
    std::sort(out.begin(), out.end());
    return out;
}

// "htdemucs" -> models/htdemucs-f16.gguf or -f32.gguf, whichever exists (f16 first).
std::string find_model(const std::string& name) {
    for (const char* suffix : {"-f16.gguf", "-f32.gguf", ".gguf"}) {
        const std::string path = g_models_dir + "/" + name + suffix;
        if (file_exists(path)) return path;
    }
    return {};
}

struct Json {
    yyjson_doc* doc = nullptr;
    yyjson_val* root = nullptr;
    explicit Json(const std::string& body) {
        doc = yyjson_read(body.data(), body.size(), 0);
        root = doc ? yyjson_doc_get_root(doc) : nullptr;
        if (!root || !yyjson_is_obj(root)) throw std::runtime_error("body is not a JSON object");
    }
    ~Json() { if (doc) yyjson_doc_free(doc); }
    Json(const Json&) = delete;
    Json& operator=(const Json&) = delete;

    std::string str(const char* key, const std::string& fallback = {}) const {
        yyjson_val* v = yyjson_obj_get(root, key);
        return (v && yyjson_is_str(v)) ? std::string(yyjson_get_str(v)) : fallback;
    }
    double num(const char* key, double fallback) const {
        yyjson_val* v = yyjson_obj_get(root, key);
        if (!v) return fallback;
        if (yyjson_is_num(v)) return yyjson_get_num(v);
        if (yyjson_is_str(v)) { try { return std::stod(yyjson_get_str(v)); } catch (...) {} }
        return fallback;
    }
    bool flag(const char* key) const {
        yyjson_val* v = yyjson_obj_get(root, key);
        return v && yyjson_is_bool(v) && yyjson_get_bool(v);
    }
    std::vector<std::string> strs(const char* key) const {
        std::vector<std::string> out;
        yyjson_val* v = yyjson_obj_get(root, key);
        if (v && yyjson_is_arr(v)) {
            size_t i, n;
            yyjson_val* e;
            yyjson_arr_foreach(v, i, n, e) if (yyjson_is_str(e)) out.push_back(yyjson_get_str(e));
        } else if (v && yyjson_is_str(v)) {
            out.push_back(yyjson_get_str(v));
        }
        return out;
    }
};

struct SeparateRequest {
    std::string audio_b64;
    std::string model = "htdemucs";
    std::string two_stems;
    std::vector<std::string> stems;
    st::SeparateOptions opt;
    bool float32 = false;
};

SeparateRequest parse(const Json& body) {
    SeparateRequest r;
    r.audio_b64 = body.str("audio_data");
    if (r.audio_b64.empty()) throw std::invalid_argument("audio_data is required");
    r.model = body.str("model", body.str("model_name", "htdemucs"));
    r.two_stems = body.str("two_stems");
    r.stems = body.strs("stems");
    r.opt.shifts = (int)body.num("shifts", 0);
    r.opt.overlap = (float)body.num("overlap", 0.25);
    r.opt.seed = (uint32_t)resolve_seed((long long)body.num("seed", -1));
    r.float32 = body.flag("float32");
    if (r.opt.shifts < 0 || r.opt.shifts > 10) throw std::invalid_argument("shifts must be 0..10");
    if (!(r.opt.overlap >= 0.0f && r.opt.overlap < 1.0f)) throw std::invalid_argument("overlap must be in [0, 1)");
    return r;
}

// Runs under the gpu lock. Returns the "stems" JSON object: {"drums": "<b64>", ...}.
std::string run_separate(SeparateRequest req, const std::string& session) {
    const std::string path = find_model(req.model);
    if (path.empty()) throw std::runtime_error("no GGUF for model '" + req.model + "' in " + g_models_dir);
    if (!g_model || g_model_path != path) {
        g_model.reset();
        g_model = std::make_unique<st::HTDemucs>(path, g_device.empty() ? nullptr : g_device.c_str());
        g_model_path = path;
    }
    st::HTDemucs& m = *g_model;
    const auto& src = m.sources();
    int two = -1;
    if (!req.two_stems.empty()) {
        for (size_t s = 0; s < src.size(); s++) if (src[s] == req.two_stems) two = (int)s;
        if (two < 0) throw std::invalid_argument("two_stems: '" + req.two_stems + "' is not a source of " + req.model);
    }
    for (const auto& n : req.stems)
        if (std::find(src.begin(), src.end(), n) == src.end())
            throw std::invalid_argument("stems: '" + n + "' is not a source of " + req.model);

    int n = 0, ch = 0, sr = 0;
    std::vector<float> audio = decode_audio(req.audio_b64, n, ch, sr);
    const int C = m.audio_channels(), R = m.samplerate();
    audio = st::to_channels(audio, n, ch, C);
    int len = n;
    if (sr != R) audio = st::resample_planar(audio, n, C, sr, R, len);

    req.opt.progress = [&session](int done, int total) { g_sessions.set_progress(session, done, total); };
    const std::vector<float> stems = m.separate(audio.data(), len, req.opt);
    const size_t per = (size_t)C * len;

    std::string out = "{";
    auto emit = [&](const std::string& name, std::vector<float> data) {
        if (!req.float32) st::rescale_if_clipping(data.data(), data.size());
        if (out.size() > 1) out += ',';
        out += json_string(name) + ":" + json_string(b64_encode(st::wav_bytes(data.data(), len, C, R, req.float32)));
    };
    if (two >= 0) {
        std::vector<float> rest(per, 0.0f);
        for (size_t s = 0; s < src.size(); s++)
            if ((int)s != two) for (size_t i = 0; i < per; i++) rest[i] += stems[s * per + i];
        emit(src[two], std::vector<float>(stems.begin() + two * per, stems.begin() + (two + 1) * per));
        emit("no_" + src[two], std::move(rest));
    } else {
        for (size_t s = 0; s < src.size(); s++) {
            if (!req.stems.empty() && std::find(req.stems.begin(), req.stems.end(), src[s]) == req.stems.end()) continue;
            emit(src[s], std::vector<float>(stems.begin() + s * per, stems.begin() + (s + 1) * per));
        }
    }
    return out + "}";
}

void usage() {
    fprintf(stderr,
        "usage: stems-server [--port 8010] [--host 127.0.0.1] [--models-dir DIR] [--device NAME]\n"
        "\n"
        "HTDemucs stem separation over HTTP. Models are found in --models-dir (or\n"
        "STEMS_MODELS_DIR) as <name>-f16.gguf / <name>-f32.gguf.\n");
}

} // namespace

int main(int argc, char** argv) {
    int port = 8010;
    std::string host = "127.0.0.1";
    g_models_dir = env_or("STEMS_MODELS_DIR", "models");
    g_device = env_or("STEMS_DEVICE", "");
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", what); exit(2); }
            return argv[++i];
        };
        if (a == "--port") port = atoi(next("--port").c_str());
        else if (a == "--host") host = next("--host");
        else if (a == "--models-dir") g_models_dir = next("--models-dir");
        else if (a == "--device") g_device = next("--device");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "error: unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    if (list_ggufs().empty())
        fprintf(stderr, "[stems] warning: no .gguf in %s (run models.sh)\n", g_models_dir.c_str());

    httplib::Server server;
    server.set_payload_max_length(1024ull * 1024 * 1024);   // a whole song as base64 WAV

    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        const bool ready = !find_model("htdemucs").empty();
        res.status = ready ? 200 : 503;
        res.set_content(JsonObject()
                            .str("status", ready ? "healthy" : "degraded")
                            .str("service", "stems-localhost")
                            .boolean("model_loaded", (bool)g_model)
                            .str("models_dir", g_models_dir)
                            .str("device", g_device.empty() ? "auto" : g_device)
                            .str("backend", "stems.cpp")
                            .str(),
                        "application/json");
    });

    server.Get("/api/models", [](const httplib::Request&, httplib::Response& res) {
        std::string body = "{\"models\":[";
        bool first = true;
        for (const std::string& f : list_ggufs()) {
            if (!first) body += ',';
            first = false;
            body += json_string(f);
        }
        body += "],\"known\":{\"htdemucs\":[\"drums\",\"bass\",\"other\",\"vocals\"],"
                "\"htdemucs_ft\":[\"drums\",\"bass\",\"other\",\"vocals\"],"
                "\"htdemucs_6s\":[\"drums\",\"bass\",\"other\",\"vocals\",\"guitar\",\"piano\"]}}";
        res.set_content(body, "application/json");
    });

    server.Post("/separate", [](const httplib::Request& req, httplib::Response& res) {
        try {
            Json body(req.body);
            SeparateRequest r = parse(body);
            const std::string session = g_sessions.create({});
            std::lock_guard<std::mutex> lock(g_sessions.gpu());
            const std::string stems = run_separate(std::move(r), session);
            res.set_content(JsonObject().boolean("success", true).str("model", g_model->name())
                                .num("sample_rate", g_model->samplerate()).raw("stems", stems).str(),
                            "application/json");
        } catch (const std::invalid_argument& e) {
            res.status = 400;
            res.set_content(json_error(e.what()), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json_error(e.what()), "application/json");
        }
    });

    server.Post("/api/juce/separate_audio", [](const httplib::Request& req, httplib::Response& res) {
        try {
            Json body(req.body);
            SeparateRequest r = parse(body);
            Session initial;
            initial.status = "queued";
            initial.seed = r.opt.seed;
            initial.model = r.model;
            const std::string id = g_sessions.create(initial);
            run_job(g_sessions, id, [r](const std::string& session) {
                const std::string stems = run_separate(r, session);
                g_sessions.update(session, [&](Session& s) {
                    s.status = "completed";
                    s.progress = 100;
                    s.result_json = stems;
                    s.finished = now_s();
                });
            });
            res.set_content(JsonObject().boolean("success", true).str("session_id", id)
                                .str("message", "Separation started")
                                .str("note", "Poll /api/juce/poll_status/{session_id} for progress and results")
                                .str(),
                            "application/json");
        } catch (const std::invalid_argument& e) {
            res.status = 400;
            res.set_content(json_error(e.what()), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json_error(e.what()), "application/json");
        }
    });

    server.Get(R"(/api/juce/poll_status/(\w+))", [](const httplib::Request& req, httplib::Response& res) {
        Session s;
        if (!g_sessions.get(req.matches[1], s)) {
            res.status = 404;
            res.set_content(json_error("Session not found"), "application/json");
            return;
        }
        const bool running = s.status == "warming" || s.status == "processing" || s.status == "queued";
        JsonObject o;
        o.boolean("success", true)
         .str("status", s.status)
         .num("progress", s.progress)
         .boolean("separation_in_progress", running)
         .str("model", s.model);
        if (s.status == "completed" && !s.result_json.empty()) o.raw("stems", s.result_json);
        if (s.status == "failed") o.str("error", s.error.empty() ? "Unknown error" : s.error);
        res.set_content(o.str(), "application/json");
    });

    fprintf(stderr, "[stems] stems-server listening on http://%s:%d (models: %s)\n",
            host.c_str(), port, g_models_dir.c_str());
    if (!server.listen(host.c_str(), port)) {
        fprintf(stderr, "error: could not bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
