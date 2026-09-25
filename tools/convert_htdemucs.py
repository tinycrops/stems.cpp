#!/usr/bin/env python3
"""Convert a pretrained HTDemucs (v4) model to GGUF.

    python tools/convert_htdemucs.py htdemucs      models/htdemucs-f32.gguf
    python tools/convert_htdemucs.py htdemucs_6s   models/htdemucs_6s-f32.gguf
    python tools/convert_htdemucs.py htdemucs_ft   models/htdemucs_ft-f32.gguf

Needs `demucs` (and so torch) at conversion time only. The C++ side never touches Python.

`htdemucs_ft` is a *bag* of four HTDemucs models, one fine-tuned per source, combined with
one-hot weights. All members are written into one file under `m0.`, `m1.`, ... and the bag
weights go in the metadata, so the runtime reproduces BagOfModels exactly.

Everything constant is folded here so the ggml graph is a plain composition of stock ops:
  * every Conv1d is written as a Conv2d with a unit height, so one im2col path serves the
    frequency branch, the time branch and the DConv residuals alike;
  * ConvTranspose weights are written as a matmul layout [OC*K, IC] (ggml ne [IC, OC*K]);
    the graph does the overlap-add itself, which keeps it on ops every backend has;
  * per-channel vectors that act on a [W, H, C] activation are written as (C, 1, 1);
  * the frequency embedding is pre-multiplied by its scale (10) and freq_emb (0.2);
  * MultiheadAttention's packed in_proj is split into q/k/v.
"""

import argparse
import sys
from fractions import Fraction

import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType

ARCH = "htdemucs"

# The hyperparameters the C++ graph hard-codes. The converter refuses anything else rather
# than writing a file that would load and then compute nonsense.
EXPECTED = {
    "audio_channels": 2, "channels": 48, "channels_time": None, "growth": 2, "nfft": 4096,
    "wiener_iters": 0, "cac": True, "depth": 4, "rewrite": True, "multi_freqs": [],
    "freq_emb": 0.2, "emb_scale": 10, "kernel_size": 8, "stride": 4, "context": 1,
    "context_enc": 0, "norm_starts": 4, "dconv_mode": 3, "dconv_depth": 2, "dconv_comp": 8,
    "t_layers": 5, "t_hidden_scale": 4.0, "t_heads": 8,
    "t_emb": "sin", "t_max_period": 10000.0, "t_weight_pos_embed": 1.0, "t_norm_in": True,
    "t_norm_in_group": False, "t_group_norm": False, "t_norm_first": True, "t_norm_out": True,
    "t_layer_scale": True, "t_gelu": True, "t_sparse_self_attn": False,
    "t_sparse_cross_attn": False, "t_cross_first": False, "use_train_segment": True,
    "t_sin_random_shift": 0,
}


def check_config(kw):
    bad = {k: (kw.get(k, "<missing>"), v) for k, v in EXPECTED.items()
           if kw.get(k, v) != v}
    # htdemucs/_ft project 384 -> 512 channels around the transformer; htdemucs_6s does not.
    if kw.get("bottom_channels", 0) not in (0, 512):
        bad["bottom_channels"] = (kw.get("bottom_channels"), "0 or 512")
    if bad:
        raise SystemExit("unsupported HTDemucs config (got, expected): %r" % bad)


def f32(t):
    return t.detach().cpu().float().numpy()


def conv_as_2d(w):
    """Conv1d (OC, IC, K) -> (OC, IC, 1, K): ggml kernel ne [K, 1, IC, OC]."""
    return w.reshape(w.shape[0], w.shape[1], 1, w.shape[2]) if w.ndim == 3 else w


def conv_tr_mm(w):
    """ConvTranspose (IC, OC, K[, 1]) -> (OC*K, IC) with row = oc*K + k."""
    if w.ndim == 4:
        assert w.shape[3] == 1, w.shape
        w = w[..., 0]
    ic, oc, k = w.shape
    return np.ascontiguousarray(w.transpose(1, 2, 0).reshape(oc * k, ic))


def chan(v):
    """Per-channel vector applied to a [W, H, C] activation: (C,) -> (C, 1, 1)."""
    return v.reshape(-1, 1, 1)


def convert_model(sd, prefix, add):
    for name, t in sd.items():
        a = f32(t)
        parts = name.split(".")
        if name.startswith(("encoder.", "decoder.", "tencoder.", "tdecoder.")):
            if "conv_tr" in parts:
                add(prefix + name.replace(".weight", ".weight_mm"), conv_tr_mm(a)
                    if parts[-1] == "weight" else chan(a))
            elif a.ndim >= 3:                       # conv / rewrite / dconv convs
                add(prefix + name, conv_as_2d(a))
            else:                                   # conv biases, group norms, layer scales
                add(prefix + name, chan(a))
        elif name == "freq_emb.embedding.weight":
            # ScaledEmbedding.forward multiplies by its scale; HTDemucs then by freq_emb.
            emb = a * EXPECTED["emb_scale"] * EXPECTED["freq_emb"]   # (F, C)
            add(prefix + "freq_emb", np.ascontiguousarray(emb.T[:, :, None]))  # ne [1, F, C]
        elif name.startswith("channel_"):
            add(prefix + name, a[..., 0] if a.ndim == 3 else a)
        elif name.startswith("crosstransformer."):
            if name.endswith("in_proj_weight") or name.endswith("in_proj_bias"):
                base = name.rsplit(".", 1)[0]
                kind = "weight" if name.endswith("weight") else "bias"
                for i, p in enumerate("qkv"):
                    add(prefix + f"{base}.{p}_proj.{kind}", np.ascontiguousarray(
                        np.split(a, 3, axis=0)[i]))
            else:
                add(prefix + name, a)
        else:
            raise SystemExit(f"unexpected tensor {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name", help="demucs pretrained name: htdemucs, htdemucs_6s, htdemucs_ft")
    ap.add_argument("out")
    ap.add_argument("--f16", action="store_true",
                    help="store transformer matmul weights as F16 (convs stay F32)")
    args = ap.parse_args()

    from demucs.pretrained import get_model
    from demucs.apply import BagOfModels

    bag = get_model(args.name)
    models = bag.models if isinstance(bag, BagOfModels) else [bag]
    weights = bag.weights if isinstance(bag, BagOfModels) else [[1.0] * len(models[0].sources)]
    ref = models[0]
    for m in models:
        kw = m._init_args_kwargs[1]
        check_config(kw)
        if m.sources != ref.sources or m.samplerate != ref.samplerate or m.segment != ref.segment:
            raise SystemExit("bag members disagree on sources/samplerate/segment")

    seg = Fraction(ref.segment)
    w = GGUFWriter(args.out, ARCH)
    w.add_name(args.name)
    w.add_string("stems.model", args.name)
    w.add_array("stems.sources", list(ref.sources))
    w.add_uint32("stems.samplerate", int(ref.samplerate))
    w.add_uint32("stems.audio_channels", int(ref.audio_channels))
    w.add_uint32("stems.segment_num", seg.numerator)
    w.add_uint32("stems.segment_den", seg.denominator)
    w.add_uint32("stems.n_models", len(models))
    w.add_array("stems.bag_weights", [float(x) for row in weights for x in row])
    w.add_string("stems.license", "MIT (facebookresearch/demucs code and weights)")

    n = 0

    def add(name, arr):
        nonlocal n
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        is_mm = name.split(".")[-1] == "weight" and arr.ndim == 2 and ".crosstransformer." in "." + name
        if args.f16 and is_mm:
            w.add_tensor(name, arr.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
        else:
            w.add_tensor(name, arr)
        n += arr.size

    for i, m in enumerate(models):
        convert_model(m.state_dict(), f"m{i}.", add)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{args.name}: {len(models)} model(s), sources={ref.sources}, "
          f"segment={seg} s, {n / 1e6:.1f}M params -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
