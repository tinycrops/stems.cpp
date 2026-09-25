// libstems.cpp — the V1 C ABI over st::HTDemucs. See libstems_v1.h for the contract.
#define STEMS_BUILD_DLL
#include "libstems_v1.h"

#include "audio.h"
#include "htdemucs.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#define STEMS_RUNTIME_VERSION "stems.cpp 0.1.0 (libstems abi 1)"

struct stems_context {
    std::unique_ptr<st::HTDemucs> model;
};

namespace {

void set_error(stems_error_v1* e, stems_status_v1 code, const char* msg) {
    if (!e || e->size < STEMS_ERROR_V1_MIN_SIZE) return;
    e->code = code;
    snprintf(e->message, sizeof(e->message), "%s", msg ? msg : "");
}

stems_status_v1 fail(stems_error_v1* e, stems_status_v1 code, const std::string& msg) {
    set_error(e, code, msg.c_str());
    return code;
}

// Zero the fields past `size` is the caller's job; these write defaults into what they own.
template <typename T>
bool sized(const T* p, uint32_t min) { return p && p->size >= min; }

const char* STEMS_CALL runtime_version() { return STEMS_RUNTIME_VERSION; }

void STEMS_CALL error_init(stems_error_v1* e) {
    if (!sized(e, STEMS_ERROR_V1_MIN_SIZE)) return;
    e->code = STEMS_STATUS_OK_V1;
    e->message[0] = '\0';
}

void STEMS_CALL context_config_init(stems_context_config_v1* c) {
    if (!sized(c, STEMS_CONTEXT_CONFIG_V1_MIN_SIZE)) return;
    c->model_path = nullptr;
    c->device = nullptr;
    c->cpu_threads = 0;
}

void STEMS_CALL audio_view_init(stems_audio_view_v1* a) {
    if (!sized(a, STEMS_AUDIO_VIEW_V1_MIN_SIZE)) return;
    a->samples = nullptr;
    a->n_samples = 0;
    a->n_channels = 0;
    a->sample_rate = 0;
    a->layout = STEMS_AUDIO_PLANAR_V1;
    a->reserved = 0;
}

void STEMS_CALL request_init(stems_request_v1* r) {
    if (!sized(r, STEMS_REQUEST_V1_MIN_SIZE)) return;
    r->input.size = sizeof(stems_audio_view_v1);
    audio_view_init(&r->input);
    r->shifts = 0;
    r->overlap = 0.25f;
    r->seed = 0;
    r->on_progress = nullptr;
    r->should_cancel = nullptr;
    r->callback_user = nullptr;
}

void STEMS_CALL result_init(stems_result_v1* r) {
    if (!sized(r, STEMS_RESULT_V1_MIN_SIZE)) return;
    r->samples = nullptr;
    r->n_sources = r->n_channels = 0;
    r->n_samples = 0;
    r->sample_rate = 0;
    r->reserved = 0;
}

void STEMS_CALL model_info_init(stems_model_info_v1* i) {
    if (!sized(i, STEMS_MODEL_INFO_V1_MIN_SIZE)) return;
    const uint32_t size = i->size;
    memset(i, 0, STEMS_MODEL_INFO_V1_MIN_SIZE);
    i->size = size;
}

stems_status_v1 STEMS_CALL context_create(const stems_context_config_v1* cfg, stems_context** out,
                                          stems_error_v1* err) {
    if (!out) return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "out_context is NULL");
    *out = nullptr;
    if (!sized(cfg, STEMS_CONTEXT_CONFIG_V1_MIN_SIZE))
        return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "config is NULL or config.size is too small");
    if (!cfg->model_path || !*cfg->model_path)
        return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "config.model_path is required");
    try {
        auto ctx = std::make_unique<stems_context>();
        ctx->model = std::make_unique<st::HTDemucs>(cfg->model_path,
            (cfg->device && *cfg->device) ? cfg->device : nullptr, cfg->cpu_threads);
        if (ctx->model->sources().size() > STEMS_MAX_SOURCES_V1)
            return fail(err, STEMS_STATUS_MODEL_ERROR_V1, "model has more sources than ABI V1 can describe");
        *out = ctx.release();
        return STEMS_STATUS_OK_V1;
    } catch (const std::bad_alloc&) {
        return fail(err, STEMS_STATUS_OUT_OF_MEMORY_V1, "out of memory loading the model");
    } catch (const std::exception& e) {
        return fail(err, STEMS_STATUS_MODEL_ERROR_V1, e.what());
    }
}

void STEMS_CALL context_destroy(stems_context* ctx) { delete ctx; }

stems_status_v1 STEMS_CALL model_info(stems_context* ctx, stems_model_info_v1* info, stems_error_v1* err) {
    if (!ctx || !ctx->model) return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "context is NULL");
    if (!sized(info, STEMS_MODEL_INFO_V1_MIN_SIZE))
        return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "info is NULL or info.size is too small");
    const st::HTDemucs& m = *ctx->model;
    info->name = m.name().c_str();
    info->backend = m.backend_name();
    info->n_sources = (uint32_t)m.sources().size();
    for (uint32_t s = 0; s < STEMS_MAX_SOURCES_V1; s++)
        info->source_names[s] = s < info->n_sources ? m.sources()[s].c_str() : nullptr;
    info->sample_rate = (uint32_t)m.samplerate();
    info->audio_channels = (uint32_t)m.audio_channels();
    info->n_models = (uint32_t)m.n_models();
    info->segment_samples = (uint32_t)m.segment_samples();
    return STEMS_STATUS_OK_V1;
}

stems_status_v1 STEMS_CALL separate(stems_context* ctx, const stems_request_v1* req,
                                    stems_result_v1* res, stems_error_v1* err) {
    if (!ctx || !ctx->model) return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "context is NULL");
    if (!sized(req, STEMS_REQUEST_V1_MIN_SIZE))
        return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "request is NULL or request.size is too small");
    if (!sized(res, STEMS_RESULT_V1_MIN_SIZE))
        return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "result is NULL or result.size is too small");
    const stems_audio_view_v1& in = req->input;
    if (in.size < STEMS_AUDIO_VIEW_V1_MIN_SIZE || !in.samples || in.n_samples == 0 ||
        in.n_channels == 0 || in.sample_rate == 0)
        return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "request.input needs samples, n_samples, n_channels and sample_rate");
    if (in.layout != STEMS_AUDIO_PLANAR_V1 && in.layout != STEMS_AUDIO_INTERLEAVED_V1)
        return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "request.input.layout is not a known layout");
    if (in.n_samples > (uint64_t)INT32_MAX / 4)
        return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "input is too long");
    if (req->shifts < 0 || req->shifts > 10)
        return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "shifts must be 0..10");
    if (!(req->overlap >= 0.0f && req->overlap < 1.0f))
        return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, "overlap must be in [0, 1)");

    try {
        st::HTDemucs& m = *ctx->model;
        const int n = (int)in.n_samples, ch = (int)in.n_channels, sr = (int)in.sample_rate;
        const int C = m.audio_channels(), R = m.samplerate(), S = (int)m.sources().size();

        std::vector<float> audio((size_t)n * ch);
        if (in.layout == STEMS_AUDIO_PLANAR_V1) {
            std::memcpy(audio.data(), in.samples, audio.size() * sizeof(float));
        } else {
            for (int i = 0; i < n; i++)
                for (int c = 0; c < ch; c++) audio[(size_t)c * n + i] = in.samples[(size_t)i * ch + c];
        }
        audio = st::to_channels(audio, n, ch, C);
        int len = n;
        if (sr != R) audio = st::resample_planar(audio, n, C, sr, R, len);

        st::SeparateOptions opt;
        opt.shifts = req->shifts;
        opt.overlap = req->overlap;
        opt.seed = req->seed;
        if (req->on_progress) {
            opt.progress = [req](int done, int total) {
                stems_progress_v1 p;
                p.size = sizeof(p);
                p.pass = done;
                p.total = total;
                p.fraction = total > 0 ? (float)done / (float)total : 0.0f;
                req->on_progress(req->callback_user, &p);
            };
        }
        if (req->should_cancel)
            opt.should_cancel = [req]() { return req->should_cancel(req->callback_user) != 0; };

        std::vector<float> stems = m.separate(audio.data(), len, opt);
        if (sr != R) {
            int back = 0;
            stems = st::resample_planar(stems, len, S * C, R, sr, back);
            if (back != n) {   // rounding: pin the length to the input's exactly
                std::vector<float> fixed((size_t)S * C * n, 0.0f);
                for (int k = 0; k < S * C; k++)
                    std::memcpy(&fixed[(size_t)k * n], &stems[(size_t)k * back],
                                (size_t)std::min(n, back) * sizeof(float));
                stems.swap(fixed);
            }
        }

        float* out = (float*)std::malloc(stems.size() * sizeof(float));
        if (!out) return fail(err, STEMS_STATUS_OUT_OF_MEMORY_V1, "cannot allocate the result");
        std::memcpy(out, stems.data(), stems.size() * sizeof(float));
        res->samples = out;
        res->n_sources = (uint32_t)S;
        res->n_channels = (uint32_t)C;
        res->n_samples = (uint64_t)n;
        res->sample_rate = (uint32_t)sr;
        return STEMS_STATUS_OK_V1;
    } catch (const st::Cancelled&) {
        return fail(err, STEMS_STATUS_CANCELLED_V1, "cancelled");
    } catch (const std::bad_alloc&) {
        return fail(err, STEMS_STATUS_OUT_OF_MEMORY_V1, "out of memory");
    } catch (const std::invalid_argument& e) {
        return fail(err, STEMS_STATUS_INVALID_ARGUMENT_V1, e.what());
    } catch (const std::exception& e) {
        return fail(err, STEMS_STATUS_INTERNAL_ERROR_V1, e.what());
    }
}

void STEMS_CALL result_free(stems_result_v1* r) {
    if (!r || r->size < STEMS_RESULT_V1_MIN_SIZE) return;
    std::free(r->samples);
    r->samples = nullptr;
    r->n_sources = r->n_channels = 0;
    r->n_samples = 0;
}

const stems_api_v1 kApiV1 = [] {
    stems_api_v1 a;
    std::memset(&a, 0, sizeof(a));
    a.size = sizeof(a);
    a.abi_version = STEMS_ABI_VERSION_1;
    a.runtime_version = runtime_version;
    a.error_init = error_init;
    a.context_config_init = context_config_init;
    a.audio_view_init = audio_view_init;
    a.request_init = request_init;
    a.result_init = result_init;
    a.model_info_init = model_info_init;
    a.context_create = context_create;
    a.context_destroy = context_destroy;
    a.model_info = model_info;
    a.separate = separate;
    a.result_free = result_free;
    return a;
}();

} // namespace

extern "C" STEMS_API const stems_api_v1* STEMS_CALL stems_get_api(uint32_t abi_version) {
    return abi_version == STEMS_ABI_VERSION_1 ? &kApiV1 : nullptr;
}
