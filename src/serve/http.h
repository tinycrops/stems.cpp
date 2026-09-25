// Ported from betweentwomidnights/audiocraft.cpp src/serve/http.h (MIT).
// serve/http.h -- the parts of an HTTP service both gary and terry need.
//
// Neither service is complicated: a JSON request comes in carrying base64 audio, a job runs
// on a worker thread, and the client polls for progress and the base64 result. What is
// fiddly is the *shape* of the replies, because a JUCE plugin is already reading them --
// gary4juce polls `/api/juce/poll_status/<id>` and expects particular field names, so the
// C++ services must answer identically or the plugin silently stops updating.
//
// The two servers keep their own routes and payloads; this is the machinery under them.
#pragma once

#include "wav.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace st::serve {

// --- base64 -------------------------------------------------------------------------------
//
// Audio crosses these APIs as a base64 WAV in a JSON string, which is what gary4juce sends
// and what the Python services return.

inline std::string b64_encode(const std::string& in) {
    static const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t v = ((uint8_t)in[i] << 16) | ((uint8_t)in[i + 1] << 8) | (uint8_t)in[i + 2];
        out += T[(v >> 18) & 63];
        out += T[(v >> 12) & 63];
        out += T[(v >> 6) & 63];
        out += T[v & 63];
    }
    if (i + 1 == in.size()) {
        const uint32_t v = (uint8_t)in[i] << 16;
        out += T[(v >> 18) & 63];
        out += T[(v >> 12) & 63];
        out += "==";
    } else if (i + 2 == in.size()) {
        const uint32_t v = ((uint8_t)in[i] << 16) | ((uint8_t)in[i + 1] << 8);
        out += T[(v >> 18) & 63];
        out += T[(v >> 12) & 63];
        out += T[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

// Decodes standard and URL-safe alphabets, ignoring whitespace and any `data:` prefix a
// browser may have left on the front. Throws on a character that is neither.
inline std::string b64_decode(const std::string& in) {
    static int8_t table[256];
    static const bool ready = [] {
        for (int i = 0; i < 256; ++i) table[i] = -1;
        const char* T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) table[(uint8_t)T[i]] = (int8_t)i;
        table[(uint8_t)'-'] = 62;   // url-safe
        table[(uint8_t)'_'] = 63;
        return true;
    }();
    (void)ready;

    size_t begin = 0;
    const size_t comma = in.find(',');
    if (comma != std::string::npos && in.compare(0, 5, "data:") == 0) begin = comma + 1;

    std::string out;
    out.reserve(in.size() / 4 * 3);
    uint32_t acc = 0;
    int bits = 0;
    for (size_t i = begin; i < in.size(); ++i) {
        const unsigned char c = (unsigned char)in[i];
        if (c == '=' ) break;
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const int8_t d = table[c];
        if (d < 0) throw std::runtime_error("audio_data is not valid base64");
        acc = (acc << 6) | (uint32_t)d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += (char)((acc >> bits) & 0xff);
        }
    }
    return out;
}

// --- JSON output ---------------------------------------------------------------------------
//
// Replies are small and their shape is fixed by the clients, so they are built directly
// rather than through a document model. yyjson parses the requests.

inline std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

inline std::string json_string(const std::string& s) { return "\"" + json_escape(s) + "\""; }

inline std::string json_number(double v) {
    if (!std::isfinite(v)) return "null";
    char b[64];
    snprintf(b, sizeof b, "%.9g", v);
    return b;
}

// Builds a flat object, one field at a time. Values arrive already encoded.
class JsonObject {
public:
    JsonObject& raw(const std::string& key, const std::string& value) {
        if (!body_.empty()) body_ += ',';
        body_ += json_string(key);
        body_ += ':';
        body_ += value;
        return *this;
    }
    JsonObject& str(const std::string& key, const std::string& value) {
        return raw(key, json_string(value));
    }
    JsonObject& num(const std::string& key, double value) {
        return raw(key, json_number(value));
    }
    JsonObject& boolean(const std::string& key, bool value) {
        return raw(key, value ? "true" : "false");
    }
    std::string str() const { return "{" + body_ + "}"; }

private:
    std::string body_;
};

inline std::string json_error(const std::string& message) {
    return JsonObject().boolean("success", false).str("error", message).str();
}

// --- sessions -------------------------------------------------------------------------------

inline double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// The 32-hex-character id both services generate. gary4juce treats it as opaque.
inline std::string new_session_id() {
    static std::mt19937_64 rng(std::random_device{}());
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    char buf[33];
    snprintf(buf, sizeof buf, "%016llx%016llx",
             (unsigned long long)rng(), (unsigned long long)rng());
    return std::string(buf, 32);
}

// -1, absent or blank means "pick one". The range matches the Python services' so the number
// stays short enough to read out over chat, which is how these get compared between machines.
inline uint64_t resolve_seed(long long requested) {
    if (requested >= 0) return (uint64_t)requested;
    static std::mt19937_64 rng(std::random_device{}());
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    return rng() % 100000;
}

struct Session {
    std::string status = "queued";      // queued | warming | processing | completed | failed
    int progress = 0;                   // 0..100
    uint64_t seed = 0;
    std::string audio_b64;              // the result, filled on completion
    std::string original_b64;           // what came in, so terry can undo
    std::string result_json;            // stems: {"drums": "<b64 wav>", ...}
    std::string error;
    std::string model;                  // gary: which checkpoint this session used
    int prompt_duration = 0;            // gary: so a retry can reuse it
    std::string description;
    double created = 0.0;
    double finished = 0.0;
};

// A registry of sessions, and the one lock that serializes generation.
//
// Both services run **one job at a time** deliberately. The Python ones do too, and the
// reason is the same: a request holds most of a GPU, so a second concurrent generation does
// not go faster, it goes out of memory.
class Sessions {
public:
    explicit Sessions(size_t keep = 64) : keep_(keep) {}

    std::string create(const Session& initial) {
        const std::string id = new_session_id();
        std::lock_guard<std::mutex> lock(mtx_);
        Session s = initial;
        s.created = now_s();
        map_[id] = s;
        order_.push_back(id);
        sweep();
        return id;
    }

    bool get(const std::string& id, Session& out) const {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(id);
        if (it == map_.end()) return false;
        out = it->second;
        return true;
    }

    template <typename Fn>
    bool update(const std::string& id, Fn&& fn) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = map_.find(id);
        if (it == map_.end()) return false;
        fn(it->second);
        return true;
    }

    void set_progress(const std::string& id, int done, int total) {
        const int pct = total > 0 ? (int)((long long)done * 100 / total) : 0;
        update(id, [&](Session& s) {
            s.status = "processing";
            s.progress = pct < 0 ? 0 : (pct > 100 ? 100 : pct);
        });
    }

    void fail(const std::string& id, const std::string& message) {
        update(id, [&](Session& s) {
            s.status = "failed";
            s.error = message;
            s.finished = now_s();
        });
    }

    // The generation lock. Held for the whole of one job.
    std::mutex& gpu() { return gpu_; }

private:
    // Finished sessions hold a base64 WAV each -- a 30 s stereo result is about 8 MB of
    // string -- so the oldest are dropped once there are enough of them. The client has
    // already polled its result by then; `undo_transform` is the only thing that reaches
    // back, and it does so within a session's lifetime.
    void sweep() {
        while (order_.size() > keep_) {
            map_.erase(order_.front());
            order_.pop_front();
        }
    }

    mutable std::mutex mtx_;
    std::mutex gpu_;
    std::unordered_map<std::string, Session> map_;
    std::deque<std::string> order_;
    size_t keep_;
};

// Run `work` on a detached thread under the generation lock, recording failures on the
// session rather than letting them escape into the HTTP thread.
template <typename Fn>
void run_job(Sessions& sessions, const std::string& id, Fn work) {
    std::thread([&sessions, id, work = std::move(work)]() mutable {
        std::lock_guard<std::mutex> lock(sessions.gpu());
        sessions.update(id, [](Session& s) { s.status = "warming"; });
        try {
            work(id);
        } catch (const std::exception& e) {
            sessions.fail(id, e.what());
        } catch (...) {
            sessions.fail(id, "unknown error");
        }
    }).detach();
}

// --- audio helpers --------------------------------------------------------------------------

// base64 WAV -> planar float channels.
inline std::vector<float> decode_audio(const std::string& b64, int& n_samples, int& n_ch,
                                       int& sample_rate) {
    const std::string bytes = b64_decode(b64);
    if (bytes.size() < 44) throw std::runtime_error("audio_data is too short to be a WAV");
    return parse_wav_planar(bytes, n_samples, n_ch, sample_rate, "audio_data");
}

inline std::string encode_audio(const std::vector<float>& planar, int n_samples, int n_ch,
                                int sample_rate) {
    return b64_encode(wav_planar_bytes(planar.data(), n_samples, n_ch, sample_rate));
}

} // namespace st::serve
