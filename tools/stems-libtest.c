/* stems-libtest — pure-C smoke test and minimal usage example for the libstems V1 ABI.
 *
 * The whole call sequence an embedded host (JUCE / iPlug2 plugin, iOS app, Tauri via FFI) uses:
 *
 *   stems_get_api -> context_create -> model_info -> request_init -> separate
 *                 -> use result.samples -> result_free -> context_destroy
 *
 * Every V1 struct is used the same way: zero it, set `size`, let the library's initializer
 * write the defaults, then set what you need.
 *
 * It reads the WAV as INTERLEAVED on purpose (hosts usually hold interleaved buffers), so the
 * layout conversion is exercised too. Stems are written as float32 WAVs.
 *
 *   usage: stems-libtest MODEL.gguf IN.wav OUT_DIR [cancel_after_passes]
 */
#include "libstems_v1.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int cancel_after; int passes; } host_state;

static void STEMS_CALL on_progress(void* user, const stems_progress_v1* p) {
    host_state* h = (host_state*)user;
    h->passes = p->pass;
    printf("  [%3.0f%%] pass %d/%d\n", p->fraction * 100.0f, p->pass, p->total);
    fflush(stdout);
}

static int32_t STEMS_CALL should_cancel(void* user) {
    host_state* h = (host_state*)user;
    return h->cancel_after > 0 && h->passes >= h->cancel_after;
}

static uint32_t rd32(const unsigned char* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t rd16(const unsigned char* p) { return (uint16_t)(p[0] | p[1] << 8); }

/* 16-bit PCM or 32-bit float WAV -> interleaved float. */
static float* read_wav(const char* path, uint64_t* n, uint32_t* ch, uint32_t* sr) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* b = (unsigned char*)malloc((size_t)size);
    if (!b || fread(b, 1, (size_t)size, f) != (size_t)size) { fclose(f); free(b); return NULL; }
    fclose(f);
    uint16_t fmt = 0, bits = 0;
    const unsigned char* data = NULL;
    uint32_t data_len = 0;
    for (long at = 12; at + 8 <= size;) {
        uint32_t len = rd32(b + at + 4);
        if (!memcmp(b + at, "fmt ", 4)) {
            fmt = rd16(b + at + 8); *ch = rd16(b + at + 10); *sr = rd32(b + at + 12); bits = rd16(b + at + 22);
            if (fmt == 0xfffe) fmt = rd16(b + at + 32);
        } else if (!memcmp(b + at, "data", 4)) {
            data = b + at + 8;
            data_len = len;
            if ((long)(at + 8 + len) > size) data_len = (uint32_t)(size - at - 8);
        }
        at += 8 + len + (len & 1);
    }
    if (!data || !(fmt == 1 && bits == 16) && !(fmt == 3 && bits == 32)) { free(b); return NULL; }
    const uint32_t bps = bits / 8;
    *n = data_len / (bps * *ch);
    float* out = (float*)malloc((size_t)*n * *ch * sizeof(float));
    for (uint64_t i = 0; i < *n * *ch; i++) {
        if (fmt == 3) memcpy(&out[i], data + i * 4, 4);
        else out[i] = (int16_t)rd16(data + i * 2) / 32768.0f;
    }
    free(b);
    return out;
}

static void write_wav_f32(const char* path, const float* planar, uint64_t n, uint32_t ch, uint32_t sr) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    const uint32_t data = (uint32_t)(n * ch * 4), chunk = 36 + data, fmtlen = 16, rate = sr * ch * 4;
    const uint16_t fmt = 3, c16 = (uint16_t)ch, align = (uint16_t)(ch * 4), bits = 32;
    fwrite("RIFF", 1, 4, f); fwrite(&chunk, 4, 1, f); fwrite("WAVEfmt ", 1, 8, f); fwrite(&fmtlen, 4, 1, f);
    fwrite(&fmt, 2, 1, f); fwrite(&c16, 2, 1, f); fwrite(&sr, 4, 1, f); fwrite(&rate, 4, 1, f);
    fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f); fwrite("data", 1, 4, f); fwrite(&data, 4, 1, f);
    for (uint64_t i = 0; i < n; i++)
        for (uint32_t c = 0; c < ch; c++) fwrite(&planar[c * n + i], 4, 1, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: stems-libtest MODEL.gguf IN.wav OUT_DIR [cancel_after_passes]\n"); return 2; }
    host_state host = {argc > 4 ? atoi(argv[4]) : 0, 0};

    const stems_api_v1* api = stems_get_api(STEMS_ABI_VERSION_1);
    if (!api) { fprintf(stderr, "library does not provide ABI v1\n"); return 1; }
    if (stems_get_api(99) != NULL) { fprintf(stderr, "unknown ABI version was accepted\n"); return 1; }
    printf("%s\n", api->runtime_version());

    stems_error_v1 err;
    memset(&err, 0, sizeof err); err.size = sizeof err; api->error_init(&err);

    stems_context_config_v1 cfg;
    memset(&cfg, 0, sizeof cfg); cfg.size = sizeof cfg; api->context_config_init(&cfg);
    cfg.model_path = argv[1];
    cfg.device = getenv("STEMS_DEVICE");

    stems_context* ctx = NULL;
    if (api->context_create(&cfg, &ctx, &err) != STEMS_STATUS_OK_V1) {
        fprintf(stderr, "context_create failed (%d): %s\n", err.code, err.message);
        return 1;
    }

    stems_model_info_v1 info;
    memset(&info, 0, sizeof info); info.size = sizeof info; api->model_info_init(&info);
    if (api->model_info(ctx, &info, &err) != STEMS_STATUS_OK_V1) { fprintf(stderr, "%s\n", err.message); return 1; }
    printf("%s on %s: %u stems at %u Hz\n", info.name, info.backend, info.n_sources, info.sample_rate);

    uint64_t n = 0; uint32_t ch = 0, sr = 0;
    float* audio = read_wav(argv[2], &n, &ch, &sr);
    if (!audio) { fprintf(stderr, "cannot read %s (16-bit PCM or float32 WAV)\n", argv[2]); return 1; }

    stems_request_v1 req;
    memset(&req, 0, sizeof req); req.size = sizeof req; api->request_init(&req);
    req.input.samples = audio;
    req.input.n_samples = n;
    req.input.n_channels = ch;
    req.input.sample_rate = sr;
    req.input.layout = STEMS_AUDIO_INTERLEAVED_V1;
    req.on_progress = on_progress;
    req.should_cancel = should_cancel;
    req.callback_user = &host;

    stems_result_v1 res;
    memset(&res, 0, sizeof res); res.size = sizeof res; api->result_init(&res);
    const stems_status_v1 st = api->separate(ctx, &req, &res, &err);
    free(audio);
    if (st == STEMS_STATUS_CANCELLED_V1) {
        printf("cancelled after %d passes, as asked\n", host.passes);
        api->result_free(&res);
        api->context_destroy(ctx);
        return host.cancel_after > 0 ? 0 : 1;
    }
    if (st != STEMS_STATUS_OK_V1) { fprintf(stderr, "separate failed (%d): %s\n", err.code, err.message); return 1; }

    for (uint32_t s = 0; s < res.n_sources; s++) {
        char path[4096];
        snprintf(path, sizeof path, "%s/%s.wav", argv[3], info.source_names[s]);
        write_wav_f32(path, res.samples + (size_t)s * res.n_channels * res.n_samples, res.n_samples, res.n_channels, res.sample_rate);
        printf("%s\n", path);
    }
    api->result_free(&res);
    api->result_free(&res);   /* safe to call twice */
    api->context_destroy(ctx);
    return 0;
}
