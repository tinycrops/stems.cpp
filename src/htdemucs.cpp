// htdemucs.cpp — see htdemucs.h. Layout conventions used throughout:
//
//   frequency branch  [T, F, C]  (ggml ne; memory-identical to torch's (C, F, T))
//   time branch       [L, 1, C]  (memory-identical to torch's (C, L))
//   transformer       [D, N]     tokens in torch's (t1 fr) order, D fastest
//
// Every convolution is a 2D im2col + matmul in F32 (the converter writes Conv1d kernels with a
// unit height), so the frequency branch's (8, 1) kernels, the time branch's 1D kernels and the
// DConv residuals share one code path. Transposed convolutions are a matmul followed by an
// explicit overlap-add of the two stride-sized halves of each kernel (K == 2 * stride holds for
// every HTDemucs layer), which keeps the graph on ops that every ggml backend implements.
#include "htdemucs.h"

#include "gguf_model.h"
#include "stft.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

namespace st {

namespace {

constexpr int kDepth = 4;
constexpr int kNfft = 4096;
constexpr int kHop = kNfft / 4;
constexpr int kKernel = 8;
constexpr int kStride = 4;
constexpr int kHeads = 8;
constexpr int kLayers = 5;
constexpr float kEps = 1e-5f;

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor* in_x = nullptr;     // [T, F, 2*C] normalised CaC spectrogram
    ggml_tensor* in_xt = nullptr;    // [L, 1, C]   normalised waveform
    ggml_tensor* pos2d = nullptr;    // [D, F'*T]
    ggml_tensor* pos1d = nullptr;    // [D, L']
    ggml_tensor* out_x = nullptr;    // [T, F, S*2*C]
    ggml_tensor* out_xt = nullptr;   // [L, 1, S*C]
    std::vector<ggml_tensor*> taps;
    // Re-uploaded before every compute: the allocator reuses an input's memory once its last
    // consumer has run, so a constant set once at build time is garbage on the second segment.
    std::vector<float> pos2d_data, pos1d_data;
    ~Graph() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx) ggml_free(ctx);
    }
};

// ---------------------------------------------------------------------------------------------
// Graph building blocks

struct Builder {
    ggml_context* ctx;
    const GgufModel& m;
    std::string pfx;
    std::vector<ggml_tensor*>* taps;

    ggml_tensor* W(const std::string& n) const { return m.get(pfx + n); }

    ggml_tensor* tap(ggml_tensor* t, const std::string& name) {
        if (taps) {
            t = ggml_cont(ctx, t);
            ggml_set_name(t, name.c_str());
            ggml_set_output(t);
            taps->push_back(t);
        }
        return t;
    }

    // x: [W, H, IC] (or [W, H, IC, N]); k: [KW, KH, IC, OC] -> [OW, OH, OC(, N)]
    ggml_tensor* conv2d(ggml_tensor* k, ggml_tensor* x, int s0, int s1, int p0, int p1,
                        int d0 = 1, int d1 = 1, ggml_tensor* bias = nullptr) {
        ggml_tensor* im = ggml_im2col(ctx, k, x, s0, s1, p0, p1, d0, d1, true, GGML_TYPE_F32);
        // im: [IC*KH*KW, OW, OH, N]
        ggml_tensor* y = ggml_mul_mat(ctx,
            ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2] * im->ne[3]),
            ggml_reshape_2d(ctx, k, k->ne[0] * k->ne[1] * k->ne[2], k->ne[3]));
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        // y: [OW*OH*N, OC]
        if (im->ne[3] == 1) {
            y = ggml_reshape_3d(ctx, y, im->ne[1], im->ne[2], k->ne[3]);
        } else {
            y = ggml_reshape_4d(ctx, y, im->ne[1], im->ne[2], im->ne[3], k->ne[3]);
            y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 1, 3, 2));
        }
        if (bias) y = ggml_add(ctx, y, bias);
        return y;
    }

    // GLU over the channel axis (ne2) of [W, H, 2C].
    ggml_tensor* glu(ggml_tensor* x) {
        const int64_t c = x->ne[2] / 2;
        ggml_tensor* a = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], c, x->nb[1], x->nb[2], 0);
        ggml_tensor* b = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], c, x->nb[1], x->nb[2], c * x->nb[2]);
        return ggml_mul(ctx, a, ggml_sigmoid(ctx, b));
    }

    // GroupNorm(1, C) applied to each row h of [W, H, C] independently over (W, C) — torch
    // folds H into the batch before these norms (the DConv reshape), so this is that.
    ggml_tensor* group_norm_rows(ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
        const int64_t W_ = x->ne[0], H = x->ne[1], C = x->ne[2];
        ggml_tensor* y;
        if (H == 1) {
            y = ggml_group_norm(ctx, ggml_reshape_4d(ctx, x, W_, 1, C, 1), 1, kEps);
            y = ggml_reshape_3d(ctx, y, W_, 1, C);
        } else {
            ggml_tensor* p = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));   // [W, C, H]
            y = ggml_group_norm(ctx, ggml_reshape_4d(ctx, p, W_, C, 1, H), 1, kEps);
            y = ggml_reshape_3d(ctx, y, W_, C, H);
            y = ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3));                  // [W, H, C]
        }
        return ggml_add(ctx, ggml_mul(ctx, y, w), b);
    }

    // DConv: two dilated residual branches along W, rows of H independent.
    ggml_tensor* dconv(const std::string& p, ggml_tensor* x) {
        for (int d = 0; d < 2; d++) {
            const std::string l = p + ".dconv.layers." + std::to_string(d);
            const int dil = 1 << d;
            ggml_tensor* h = conv2d(W(l + ".0.weight"), x, 1, 1, dil, 0, dil, 1, W(l + ".0.bias"));
            h = ggml_gelu_erf(ctx, group_norm_rows(h, W(l + ".1.weight"), W(l + ".1.bias")));
            h = conv2d(W(l + ".3.weight"), h, 1, 1, 0, 0, 1, 1, W(l + ".3.bias"));
            h = glu(group_norm_rows(h, W(l + ".4.weight"), W(l + ".4.bias")));
            x = ggml_add(ctx, x, ggml_mul(ctx, h, W(l + ".6.scale")));
        }
        return x;
    }

    // HEncLayer. freq: [T, F, C] with an (8, 1) kernel over F; time: [L, 1, C].
    ggml_tensor* encoder(const std::string& p, ggml_tensor* x, bool freq) {
        ggml_tensor* y;
        if (freq) {
            y = conv2d(W(p + ".conv.weight"), x, 1, kStride, 0, kKernel / 4, 1, 1, W(p + ".conv.bias"));
        } else {
            const int64_t le = x->ne[0];
            if (le % kStride) x = ggml_pad(ctx, x, (int)(kStride - le % kStride), 0, 0, 0);
            y = conv2d(W(p + ".conv.weight"), x, kStride, 1, kKernel / 4, 0, 1, 1, W(p + ".conv.bias"));
        }
        y = ggml_gelu_erf(ctx, y);
        y = dconv(p, y);
        y = conv2d(W(p + ".rewrite.weight"), y, 1, 1, 0, 0, 1, 1, W(p + ".rewrite.bias"));
        return glu(y);
    }

    // ConvTranspose with K == 2*stride along axis `ax` (0 = W / time, 1 = H / freq) of
    // [W, H, IC]; weight is the converter's matmul layout ne [IC, OC*K].
    ggml_tensor* conv_tr(ggml_tensor* x, ggml_tensor* wmm, ggml_tensor* bias, int ax) {
        const int64_t W_ = x->ne[0], H = x->ne[1], IC = x->ne[2];
        const int64_t OC = wmm->ne[1] / kKernel;
        ggml_tensor* xp = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));         // [IC, W, H]
        ggml_tensor* y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, xp, IC, W_ * H), wmm);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        y = ggml_reshape_4d(ctx, y, W_, H, kKernel, OC);                             // [W, H, K, OC]
        auto half = [&](int k0) {
            ggml_tensor* v = ggml_view_4d(ctx, y, W_, H, kStride, OC, y->nb[1], y->nb[2], y->nb[3],
                                          (size_t)k0 * y->nb[2]);
            if (ax == 1) {   // out index along H = h*stride + k
                v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));               // [W, k, H, OC]
                return ggml_reshape_3d(ctx, v, W_, kStride * H, OC);
            }
            // along W (H == 1): out index = w*stride + k
            v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));                   // [k, W, H, OC]
            return ggml_reshape_3d(ctx, v, kStride * W_, H, OC);
        };
        ggml_tensor* lo = half(0);
        ggml_tensor* hi = half(kStride);
        if (ax == 1) {
            lo = ggml_pad_ext(ctx, lo, 0, 0, 0, kStride, 0, 0, 0, 0);
            hi = ggml_pad_ext(ctx, hi, 0, 0, kStride, 0, 0, 0, 0, 0);
        } else {
            lo = ggml_pad_ext(ctx, lo, 0, kStride, 0, 0, 0, 0, 0, 0);
            hi = ggml_pad_ext(ctx, hi, kStride, 0, 0, 0, 0, 0, 0, 0);
        }
        return ggml_add(ctx, ggml_add(ctx, lo, hi), bias);
    }

    // HDecLayer. length is the time branch's target length (ignored for freq).
    ggml_tensor* decoder(const std::string& p, ggml_tensor* x, ggml_tensor* skip, bool freq,
                         bool last, int64_t length) {
        x = ggml_add(ctx, x, skip);
        ggml_tensor* y = freq
            ? conv2d(W(p + ".rewrite.weight"), x, 1, 1, 1, 1, 1, 1, W(p + ".rewrite.bias"))
            : conv2d(W(p + ".rewrite.weight"), x, 1, 1, 1, 0, 1, 1, W(p + ".rewrite.bias"));
        y = dconv(p, glu(y));
        ggml_tensor* z = conv_tr(y, W(p + ".conv_tr.weight_mm"), W(p + ".conv_tr.bias"), freq ? 1 : 0);
        const int pad = kKernel / 4;
        if (freq) {
            const int64_t F = z->ne[1] - 2 * pad;
            z = ggml_view_3d(ctx, z, z->ne[0], F, z->ne[2], z->nb[1], z->nb[2], (size_t)pad * z->nb[1]);
        } else {
            z = ggml_view_3d(ctx, z, length, 1, z->ne[2], z->nb[1], z->nb[2], (size_t)pad * z->nb[0]);
        }
        z = ggml_cont(ctx, z);
        return last ? z : ggml_gelu_erf(ctx, z);
    }

    // ------------------------------------------------------------------ transformer
    ggml_tensor* layer_norm(ggml_tensor* x, const std::string& n) {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, kEps), W(n + ".weight")), W(n + ".bias"));
    }
    ggml_tensor* linear(ggml_tensor* x, const std::string& n) {
        ggml_tensor* y = ggml_mul_mat(ctx, W(n + ".weight"), x);
        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
        return ggml_add(ctx, y, W(n + ".bias"));
    }
    // MyGroupNorm(1, D) on (B, N, D): one mean/var over the whole sequence.
    ggml_tensor* norm_out(ggml_tensor* x, const std::string& n) {
        ggml_tensor* y = ggml_group_norm(ctx, ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1], 1, 1), 1, kEps);
        y = ggml_reshape_2d(ctx, y, x->ne[0], x->ne[1]);
        return ggml_add(ctx, ggml_mul(ctx, y, W(n + ".weight")), W(n + ".bias"));
    }
    ggml_tensor* attention(ggml_tensor* q_in, ggml_tensor* kv_in, const std::string& n) {
        const int64_t D = q_in->ne[0], Nq = q_in->ne[1], Nk = kv_in->ne[1], hd = D / kHeads;
        auto heads = [&](ggml_tensor* t, int64_t N) {
            return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, t, hd, kHeads, N), 0, 2, 1, 3));
        };
        ggml_tensor* q = heads(linear(q_in, n + ".q_proj"), Nq);            // [hd, Nq, H]
        ggml_tensor* k = heads(linear(kv_in, n + ".k_proj"), Nk);
        ggml_tensor* v = heads(linear(kv_in, n + ".v_proj"), Nk);
        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);                            // [Nk, Nq, H]
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / std::sqrt((float)hd), 0.0f);
        ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));  // [Nk, hd, H]
        ggml_tensor* o = ggml_mul_mat(ctx, vt, kq);                           // [hd, Nq, H]
        ggml_mul_mat_set_prec(o, GGML_PREC_F32);
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));                 // [hd, H, Nq]
        return linear(ggml_reshape_2d(ctx, o, D, Nq), n + ".out_proj");
    }
    ggml_tensor* ff(ggml_tensor* x, const std::string& n) {
        return linear(ggml_gelu_erf(ctx, linear(x, n + ".linear1")), n + ".linear2");
    }
    ggml_tensor* self_layer(ggml_tensor* x, const std::string& n) {
        ggml_tensor* h = layer_norm(x, n + ".norm1");
        x = ggml_add(ctx, x, ggml_mul(ctx, attention(h, h, n + ".self_attn"), W(n + ".gamma_1.scale")));
        x = ggml_add(ctx, x, ggml_mul(ctx, ff(layer_norm(x, n + ".norm2"), n), W(n + ".gamma_2.scale")));
        return norm_out(x, n + ".norm_out");
    }
    ggml_tensor* cross_layer(ggml_tensor* q, ggml_tensor* k, const std::string& n) {
        ggml_tensor* x = ggml_add(ctx, q, ggml_mul(ctx, attention(layer_norm(q, n + ".norm1"),
                                  layer_norm(k, n + ".norm2"), n + ".cross_attn"), W(n + ".gamma_1.scale")));
        x = ggml_add(ctx, x, ggml_mul(ctx, ff(layer_norm(x, n + ".norm3"), n), W(n + ".gamma_2.scale")));
        return norm_out(x, n + ".norm_out");
    }
};

// Positional embeddings, computed on the host exactly as demucs.transformer does.
std::vector<float> sin_embedding_1d(int T, int D, double max_period) {
    std::vector<float> e((size_t)T * D);
    const int half = D / 2;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < half; i++) {
            const float phase = (float)t / std::pow((float)max_period, (float)i / (float)(half - 1));
            e[(size_t)t * D + i] = std::cos(phase);
            e[(size_t)t * D + half + i] = std::sin(phase);
        }
    return e;   // [D, T] in ggml ne
}

std::vector<float> sin_embedding_2d(int D, int Fr, int T1, double max_period) {
    // torch: pe [D, Fr, T1]; then "b c fr t1 -> b (t1 fr) c" -> ggml ne [D, Fr*T1].
    std::vector<float> e((size_t)D * Fr * T1);
    const int dm = D / 2;
    for (int i = 0; i < dm / 2; i++) {
        const float div = std::exp((float)(2 * i) * -(std::log((float)max_period) / (float)dm));
        for (int t = 0; t < T1; t++)
            for (int f = 0; f < Fr; f++) {
                float* v = &e[((size_t)t * Fr + f) * D];
                v[2 * i] = std::sin((float)t * div);
                v[2 * i + 1] = std::cos((float)t * div);
                v[dm + 2 * i] = std::sin((float)f * div);
                v[dm + 2 * i + 1] = std::cos((float)f * div);
            }
    }
    return e;
}

double mean_of(const float* x, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; i++) s += x[i];
    return s / n;
}
double std_of(const float* x, size_t n, double mean) {   // unbiased, like torch.std
    double s = 0;
    for (size_t i = 0; i < n; i++) { const double d = x[i] - mean; s += d * d; }
    return std::sqrt(s / (n - 1));
}

} // namespace

// ---------------------------------------------------------------------------------------------

struct HTDemucs::Impl {
    ggml_backend_t backend = nullptr;
    GgufModel model;
    std::string name;
    std::vector<std::string> sources;
    int sr = 44100, channels = 2, seg_num = 39, seg_den = 5, n_models = 1;
    std::vector<float> bag;                 // [n_models][S]
    FFT fft{kNfft};

    int segment_samples() const { return (int)((int64_t)seg_num * sr / seg_den); }

    // Frame/length bookkeeping of one segment, identical to the torch forward.
    int n_frames() const { return (segment_samples() + kHop - 1) / kHop; }

    std::unique_ptr<Graph> build(int mi, bool with_taps) const {
        auto g = std::make_unique<Graph>();
        const size_t n_nodes = 8192;
        ggml_init_params ip = {ggml_tensor_overhead() * n_nodes * 2 + ggml_graph_overhead_custom(n_nodes, false),
                               nullptr, true};
        g->ctx = ggml_init(ip);
        ggml_context* ctx = g->ctx;
        Builder b{ctx, model, "m" + std::to_string(mi) + ".", with_taps ? &g->taps : nullptr};
        const int S = (int)sources.size(), C = channels;
        const int T = n_frames(), F = kNfft / 2, L = segment_samples();

        g->in_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, F, 2 * C);
        g->in_xt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, L, 1, C);
        ggml_set_input(g->in_x);
        ggml_set_input(g->in_xt);

        ggml_tensor* x = g->in_x;
        ggml_tensor* xt = g->in_xt;
        std::vector<ggml_tensor*> saved, saved_t;
        std::vector<int64_t> lengths_t;
        for (int i = 0; i < kDepth; i++) {
            const std::string e = std::to_string(i);
            lengths_t.push_back(xt->ne[0]);
            xt = b.tap(b.encoder("tencoder." + e, xt, false), "tenc" + e);
            saved_t.push_back(xt);
            x = b.encoder("encoder." + e, x, true);
            if (i == 0) x = ggml_add(ctx, x, b.W("freq_emb"));
            x = b.tap(x, "enc" + e);
            saved.push_back(x);
        }

        // Cross-transformer at the bottom.
        const int64_t Fb = x->ne[1], Tb = x->ne[0], Cb = x->ne[2], Lb = xt->ne[0];
        ggml_tensor* tok = ggml_cont(ctx, ggml_permute(ctx, x, 2, 1, 0, 3));          // [C, F, T]
        // htdemucs/_ft widen 384 -> 512 around the transformer (bottom_channels); _6s does not.
        const bool bottom = model.has(b.pfx + "channel_upsampler.weight");
        tok = ggml_reshape_2d(ctx, tok, Cb, Fb * Tb);
        ggml_tensor* tokt = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, xt, Lb, Cb)));
        if (bottom) {
            tok = b.linear(tok, "channel_upsampler");
            tokt = b.linear(tokt, "channel_upsampler_t");
        }
        const int64_t D = tok->ne[0];
        g->pos2d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Fb * Tb);
        g->pos1d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, Lb);
        ggml_set_input(g->pos2d);
        ggml_set_input(g->pos1d);
        tok = ggml_add(ctx, b.layer_norm(tok, "crosstransformer.norm_in"), g->pos2d);
        tokt = ggml_add(ctx, b.layer_norm(tokt, "crosstransformer.norm_in_t"), g->pos1d);
        for (int l = 0; l < kLayers; l++) {
            const std::string n = "crosstransformer.layers." + std::to_string(l);
            const std::string nt = "crosstransformer.layers_t." + std::to_string(l);
            if (l % 2 == 0) {
                tok = b.self_layer(tok, n);
                tokt = b.self_layer(tokt, nt);
            } else {
                ggml_tensor* old = tok;
                tok = b.cross_layer(tok, tokt, n);
                tokt = b.cross_layer(tokt, old, nt);
            }
        }
        b.tap(ggml_reshape_3d(ctx, tok, D, Fb, Tb), "xformer_tok");   // [D, F, T]; see compare.py
        b.tap(tokt, "xformer_t_tok");                                  // [D, L]
        if (bottom) tok = b.linear(tok, "channel_downsampler");
        x = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, tok, Cb, Fb, Tb), 2, 1, 0, 3));  // [T, F, C]
        if (bottom) tokt = b.linear(tokt, "channel_downsampler_t");
        xt = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, tokt)), Lb, 1, Cb);

        for (int i = 0; i < kDepth; i++) {
            const std::string d = std::to_string(i);
            const bool last = i == kDepth - 1;
            x = b.tap(b.decoder("decoder." + d, x, saved[kDepth - 1 - i], true, last, 0), "dec" + d);
            xt = b.tap(b.decoder("tdecoder." + d, xt, saved_t[kDepth - 1 - i], false, last,
                                 lengths_t[kDepth - 1 - i]), "tdec" + d);
        }
        if (x->ne[2] != S * 2 * C || xt->ne[2] != S * C || x->ne[1] != F || xt->ne[0] != L)
            throw std::runtime_error("htdemucs: unexpected output shape");
        g->out_x = x;
        g->out_xt = xt;
        ggml_set_output(x);
        ggml_set_output(xt);

        g->gf = ggml_new_graph_custom(ctx, n_nodes, false);
        ggml_build_forward_expand(g->gf, x);
        ggml_build_forward_expand(g->gf, xt);
        for (ggml_tensor* t : g->taps) ggml_build_forward_expand(g->gf, t);

        g->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(g->alloc, g->gf))
            throw std::runtime_error("htdemucs: failed to allocate the compute graph");

        g->pos2d_data = sin_embedding_2d((int)D, (int)Fb, (int)Tb, 10000.0);
        g->pos1d_data = sin_embedding_1d((int)Lb, (int)D, 10000.0);
        return g;
    }

    // HTDemucs.forward on one segment of normalised audio: [C][L] -> [S][C][L].
    void run(Graph& g, const float* seg, float* out, const std::string& dump_dir = "") const {
        const int S = (int)sources.size(), C = channels, L = segment_samples();
        const int T = n_frames(), F = kNfft / 2;

        // Spectrogram, complex-as-channels: channel index c*2 + {re, im}; layout [T, F, 2C].
        std::vector<std::vector<cplx>> z(C);
        std::vector<float> mag((size_t)2 * C * F * T);
        for (int c = 0; c < C; c++) {
            int frames = 0;
            z[c] = htdemucs_spec(fft, seg + (size_t)c * L, L, frames);
            for (size_t i = 0; i < (size_t)F * T; i++) {
                mag[(size_t)(2 * c) * F * T + i] = (float)z[c][i].real();
                mag[(size_t)(2 * c + 1) * F * T + i] = (float)z[c][i].imag();
            }
        }
        const double mean = mean_of(mag.data(), mag.size());
        const double sd = std_of(mag.data(), mag.size(), mean);
        for (float& v : mag) v = (float)((v - mean) / (1e-5 + sd));

        std::vector<float> wav(seg, seg + (size_t)C * L);
        const double meant = mean_of(wav.data(), wav.size());
        const double sdt = std_of(wav.data(), wav.size(), meant);
        for (float& v : wav) v = (float)((v - meant) / (1e-5 + sdt));

        ggml_backend_tensor_set(g.in_x, mag.data(), 0, mag.size() * sizeof(float));
        ggml_backend_tensor_set(g.in_xt, wav.data(), 0, wav.size() * sizeof(float));
        ggml_backend_tensor_set(g.pos2d, g.pos2d_data.data(), 0, g.pos2d_data.size() * sizeof(float));
        ggml_backend_tensor_set(g.pos1d, g.pos1d_data.data(), 0, g.pos1d_data.size() * sizeof(float));
        std::string err;
        if (!graph_compute_checked(backend, g.gf, "htdemucs", err)) throw std::runtime_error(err);

        std::vector<float> ox((size_t)S * 2 * C * F * T), oxt((size_t)S * C * L);
        ggml_backend_tensor_get(g.out_x, ox.data(), 0, ox.size() * sizeof(float));
        ggml_backend_tensor_get(g.out_xt, oxt.data(), 0, oxt.size() * sizeof(float));

        if (!dump_dir.empty()) {
            for (ggml_tensor* t : g.taps) {
                std::vector<float> buf(ggml_nelements(t));
                ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
                FILE* f = fopen((dump_dir + "/" + ggml_get_name(t) + ".f32").c_str(), "wb");
                if (f) { fwrite(buf.data(), sizeof(float), buf.size(), f); fclose(f); }
            }
        }

        std::vector<cplx> spec((size_t)F * T);
        for (int s = 0; s < S; s++)
            for (int c = 0; c < C; c++) {
                const float* re = &ox[(size_t)(s * 2 * C + 2 * c) * F * T];
                const float* im = &ox[(size_t)(s * 2 * C + 2 * c + 1) * F * T];
                for (size_t i = 0; i < (size_t)F * T; i++)
                    spec[i] = cplx(re[i] * sd + mean, im[i] * sd + mean);
                const std::vector<float> y = htdemucs_ispec(fft, spec, T, L);
                float* o = out + ((size_t)s * C + c) * L;
                const float* t = &oxt[((size_t)s * C + c) * L];
                for (int i = 0; i < L; i++) o[i] = (float)(t[i] * sdt + meant) + y[i];
            }
    }
};

HTDemucs::HTDemucs(const std::string& path, const char* device, int cpu_threads) : p_(new Impl) {
    p_->backend = make_backend(cpu_threads, device);
    p_->model = load_gguf(path.c_str(), p_->backend);
    const GgufModel& m = p_->model;
    gguf_context* gg = m.gguf;
    if (m.string("general.architecture") != "htdemucs")
        throw std::runtime_error(path + ": not an htdemucs GGUF");
    p_->name = m.string("stems.model");
    const int ks = gguf_find_key(gg, "stems.sources");
    if (ks < 0) throw std::runtime_error(path + ": missing stems.sources");
    for (size_t i = 0; i < gguf_get_arr_n(gg, ks); i++) p_->sources.push_back(gguf_get_arr_str(gg, ks, i));
    p_->sr = (int)m.u32("stems.samplerate");
    p_->channels = (int)m.u32("stems.audio_channels");
    p_->seg_num = (int)m.u32("stems.segment_num");
    p_->seg_den = (int)m.u32("stems.segment_den");
    p_->n_models = (int)m.u32("stems.n_models");
    const int kw = gguf_find_key(gg, "stems.bag_weights");
    if (kw < 0 || gguf_get_arr_type(gg, kw) != GGUF_TYPE_FLOAT32) throw std::runtime_error(path + ": bad stems.bag_weights");
    const float* w = static_cast<const float*>(gguf_get_arr_data(gg, kw));
    p_->bag.assign(w, w + gguf_get_arr_n(gg, kw));
    if (p_->bag.size() != (size_t)p_->n_models * p_->sources.size())
        throw std::runtime_error(path + ": stems.bag_weights has the wrong size");
}

HTDemucs::~HTDemucs() {
    if (p_) {
        p_->model.free();
        if (p_->backend) ggml_backend_free(p_->backend);
    }
}

const std::string& HTDemucs::name() const { return p_->name; }
const std::vector<std::string>& HTDemucs::sources() const { return p_->sources; }
int HTDemucs::samplerate() const { return p_->sr; }
int HTDemucs::audio_channels() const { return p_->channels; }
int HTDemucs::segment_samples() const { return p_->segment_samples(); }
int HTDemucs::n_models() const { return p_->n_models; }
const char* HTDemucs::backend_name() const { return ggml_backend_name(p_->backend); }

std::vector<float> HTDemucs::forward(int mi, const float* seg, const std::string& dump_dir) const {
    auto g = p_->build(mi, !dump_dir.empty());
    std::vector<float> out((size_t)p_->sources.size() * p_->channels * p_->segment_samples());
    p_->run(*g, seg, out.data(), dump_dir);
    return out;
}

// demucs.apply.apply_model, reproduced: bag -> shifts -> split -> one padded segment.
//
// A "view" is TensorChunk: a window [offset, offset+length) onto an underlying buffer, whose
// padded() reads real samples from outside the window when they exist and zeros beyond the
// buffer. That detail matters: the last segment of a split is centre-padded with audio from
// *before* it, not with silence, and matching that is what makes the outputs line up.
std::vector<float> HTDemucs::separate(const float* mix_in, int len, const SeparateOptions& opt) const {
    const Impl& P = *p_;
    const int S = (int)P.sources.size(), C = P.channels, seg = P.segment_samples();
    if (len <= 0) throw std::invalid_argument("empty input");

    // Normalise by the mono mix (demucs.separate).
    std::vector<float> mono(len);
    for (int i = 0; i < len; i++) {
        double s = 0;
        for (int c = 0; c < C; c++) s += mix_in[(size_t)c * len + i];
        mono[i] = (float)(s / C);
    }
    const double ref_mean = mean_of(mono.data(), len);
    const double ref_std = std::max(std_of(mono.data(), len, ref_mean), 1e-8);
    std::vector<float> mix((size_t)C * len);
    for (size_t i = 0; i < mix.size(); i++) mix[i] = (float)((mix_in[i] - ref_mean) / ref_std);

    struct View { const float* buf; int total; int offset; int length; };
    auto padded = [&](const View& v, int target, std::vector<float>& out) {
        const int delta = target - v.length;
        const int start = v.offset - delta / 2;
        out.assign((size_t)C * target, 0.0f);
        for (int c = 0; c < C; c++)
            for (int i = 0; i < target; i++) {
                const int j = start + i;
                if (j >= 0 && j < v.total) out[(size_t)c * target + i] = v.buf[(size_t)c * v.total + j];
            }
    };

    const int stride = (int)((1.0 - (double)opt.overlap) * seg);
    if (stride <= 0) throw std::invalid_argument("overlap must be < 1");
    std::vector<float> weight(seg);
    {
        const int half = seg / 2;
        for (int i = 0; i < half; i++) weight[i] = (float)(i + 1);
        for (int i = half; i < seg; i++) weight[i] = (float)(seg - i);
        const float mx = *std::max_element(weight.begin(), weight.end());
        for (float& w : weight) w /= mx;
    }

    const int shifts = std::max(0, opt.shifts);
    const int passes_per_split = (len + stride - 1) / stride;
    const int total_passes = P.n_models * std::max(1, shifts) * passes_per_split;
    int done = 0;
    std::mt19937 rng(opt.seed);

    std::vector<float> estimates((size_t)S * C * len, 0.0f);
    std::vector<float> totals(S, 0.0f);
    std::vector<float> seg_in, seg_out((size_t)S * C * seg);

    for (int mi = 0; mi < P.n_models; mi++) {
        auto g = P.build(mi, false);

        // split=True over one view -> [S][C][view.length]
        auto split = [&](const View& v) {
            std::vector<float> out((size_t)S * C * v.length, 0.0f);
            std::vector<float> sum_w(v.length, 0.0f);
            for (int off = 0; off < v.length; off += stride) {
                View chunk{v.buf, v.total, v.offset + off, std::min(seg, v.length - off)};
                padded(chunk, seg, seg_in);
                P.run(*g, seg_in.data(), seg_out.data());
                const int trim = (seg - chunk.length) / 2;          // center_trim
                for (int sc = 0; sc < S * C; sc++)
                    for (int i = 0; i < chunk.length; i++)
                        out[(size_t)sc * v.length + off + i] += weight[i] * seg_out[(size_t)sc * seg + trim + i];
                for (int i = 0; i < chunk.length; i++) sum_w[off + i] += weight[i];
                if (opt.progress) opt.progress(++done, total_passes);
            }
            for (int sc = 0; sc < S * C; sc++)
                for (int i = 0; i < v.length; i++) out[(size_t)sc * v.length + i] /= sum_w[i];
            return out;
        };

        std::vector<float> res;
        if (shifts == 0) {
            res = split(View{mix.data(), len, 0, len});
        } else {
            const int max_shift = P.sr / 2;
            std::vector<float> pm;
            padded(View{mix.data(), len, 0, len}, len + 2 * max_shift, pm);
            res.assign((size_t)S * C * len, 0.0f);
            std::uniform_int_distribution<int> dist(0, max_shift);
            for (int k = 0; k < shifts; k++) {
                const int offset = dist(rng);
                const View shifted{pm.data(), len + 2 * max_shift, offset, len + max_shift - offset};
                const std::vector<float> o = split(shifted);
                for (int sc = 0; sc < S * C; sc++)
                    for (int i = 0; i < len; i++)
                        res[(size_t)sc * len + i] += o[(size_t)sc * shifted.length + max_shift - offset + i];
            }
            for (float& v : res) v /= shifts;
        }

        for (int s = 0; s < S; s++) {
            const float w = P.bag[(size_t)mi * S + s];
            totals[s] += w;
            if (w == 0.0f) continue;
            for (size_t i = 0; i < (size_t)C * len; i++)
                estimates[(size_t)s * C * len + i] += w * res[(size_t)s * C * len + i];
        }
    }

    for (int s = 0; s < S; s++)
        for (size_t i = 0; i < (size_t)C * len; i++) {
            float& v = estimates[(size_t)s * C * len + i];
            v = (float)(v / totals[s] * ref_std + ref_mean);
        }
    return estimates;
}

} // namespace st
