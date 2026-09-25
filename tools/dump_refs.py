#!/usr/bin/env python3
"""Dump PyTorch reference outputs for the parity tests.

    python tools/dump_refs.py htdemucs tests/data/test.wav tests/refs/htdemucs

Writes, as .npy:
  full.npy        [S, C, L]  the whole `demucs` separate path (normalise by the mono mix,
                             apply_model split=True overlap=0.25 shifts=0, denormalise)
  seg_in.npy      [C, N]     the first segment exactly as apply_model feeds it to the model
  seg_out.npy     [S, C, N]  the model's raw output on that segment
  enc{i}.npy / tenc{i}.npy   encoder outputs (freq [C, F, T] / time [C, L])
  xformer.npy / xformer_t.npy  cross-transformer outputs
  dec{i}.npy / tdec{i}.npy   decoder outputs

shifts=0 because demucs' default (shifts=1) is a *random* time shift; it cannot be compared
sample-for-sample. The C++ CLI implements shifts with its own seeded RNG.
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("wav")
    ap.add_argument("out")
    ap.add_argument("--overlap", type=float, default=0.25)
    args = ap.parse_args()

    from demucs.pretrained import get_model
    from demucs.apply import apply_model, BagOfModels, TensorChunk

    torch.set_num_threads(os.cpu_count())
    os.makedirs(args.out, exist_ok=True)
    save = lambda n, t: np.save(os.path.join(args.out, n + ".npy"),
                                (t.detach().numpy() if torch.is_tensor(t) else t).astype(np.float32))

    bag = get_model(args.name)
    bag.eval()
    data, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    assert sr == 44100, sr
    wav = torch.from_numpy(data.T.copy())

    ref = wav.mean(0)
    mix = (wav - ref.mean()) / ref.std()
    with torch.no_grad():
        out = apply_model(bag, mix[None], shifts=0, split=True, overlap=args.overlap)[0]
    out = out * ref.std() + ref.mean()
    save("full", out)

    # One segment through the first model, with the intermediate taps.
    model = bag.models[0] if isinstance(bag, BagOfModels) else bag
    seglen = int(model.segment * model.samplerate)
    seg = TensorChunk(mix[None], 0, seglen).padded(seglen)
    save("seg_in", seg[0])
    taps = {}

    def hook(name):
        def f(_m, _i, o):
            taps[name] = o
        return f

    for i, m in enumerate(model.encoder):
        m.register_forward_hook(hook(f"enc{i}"))
    for i, m in enumerate(model.tencoder):
        m.register_forward_hook(hook(f"tenc{i}"))
    for i, m in enumerate(model.decoder):
        m.register_forward_hook(hook(f"dec{i}"))
    for i, m in enumerate(model.tdecoder):
        m.register_forward_hook(hook(f"tdec{i}"))
    model.crosstransformer.register_forward_hook(hook("xformer"))
    with torch.no_grad():
        y = model(seg)
    save("seg_out", y[0])
    for k, v in taps.items():
        if k == "xformer":
            save("xformer", v[0][0])
            save("xformer_t", v[1][0])
        elif k.startswith(("dec", "tdec")):
            save(k, v[0][0])
        else:
            save(k, v[0])
    print(f"wrote {len(taps) + 3} arrays to {args.out}; full={tuple(out.shape)}")


if __name__ == "__main__":
    main()
