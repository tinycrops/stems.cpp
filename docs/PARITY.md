# Parity with PyTorch

Measured 2026-09-24 on CPU (Xeon E3-1535M v6) against `demucs 4.0.1` / torch 2.14 CPU.
Test audio: demucs' own public `test.mp3` (20.0 s), decoded to 44.1 kHz stereo float.

Two checks per model:

- **seg** is one raw `model(x)` on the first 7.8 s segment, exactly as `apply_model` feeds it.
- **full** is the whole `demucs` separate path: normalise by the mono mix, then
  `apply_model(split=True, overlap=0.25, shifts=0)`, then denormalise. That exercises the three
  overlapping segments, the triangular cross-fade, the centre-padding of the last segment
  and the trim.

`shifts=0` because demucs' default of one shift is a *random* offset and can't be compared
sample for sample. `--shifts N` is implemented the same way, with a seeded RNG.

SNR (dB) of the C++ output against torch, per stem:

| model | check | drums | bass | other | vocals | guitar | piano |
|---|---|---|---|---|---|---|---|
| htdemucs | seg | 111.4 | 85.9 | 124.4 | 79.0 | | |
| htdemucs | full | 118.2 | 85.0 | 124.5 | 78.7 | | |
| htdemucs_6s | seg | 118.0 | 78.9 | 125.4 | 76.1 | 78.9 | 81.3 |
| htdemucs_6s | full | 124.4 | 83.6 | 126.1 | 75.8 | 77.8 | 81.5 |
| htdemucs_ft | seg | 111.8 | 121.9 | 122.7 | 112.0 | | |
| htdemucs_ft | full | 117.3 | 81.0 | 124.3 | 71.1 | | |

Cosine similarity is 1.0000000 to seven places everywhere. The largest absolute sample error
is 1.2e-5. The spread between stems is set by how quiet the stem is in this clip, not by
anything in the port.

## Reproduce

```bash
python -m venv .venv-ref && .venv-ref/bin/pip install torch demucs==4.0.1 gguf soundfile numpy
curl -L -o tests/data/test.mp3 https://raw.githubusercontent.com/adefossez/demucs/main/test.mp3
ffmpeg -i tests/data/test.mp3 -ar 44100 -ac 2 -c:a pcm_f32le tests/data/test.wav
for m in htdemucs htdemucs_6s htdemucs_ft; do
  .venv-ref/bin/python tools/convert_htdemucs.py $m models/$m-f32.gguf
  .venv-ref/bin/python tools/dump_refs.py $m tests/data/test.wav tests/refs/$m
  build/bin/stems-parity --model models/$m-f32.gguf --refs tests/refs/$m --wav tests/data/test.wav
done
```

`stems-parity --dump DIR` also writes every encoder, transformer and decoder activation as raw
float32. The frequency-branch tensors are memory-identical to the torch arrays in `tests/refs`.
The two transformer taps are token-major, so transpose them before comparing.

## One bug worth knowing about

The first full-path run failed completely (cosine 0.3) while the single segment matched
exactly. The cause: the transformer's positional embeddings were uploaded once, when the graph
was built. ggml's graph allocator reuses an input tensor's memory once its last consumer has run,
so from the second segment on the embeddings were overwritten scratch. Every input is now
re-uploaded before each compute. Any ggml port that reuses a graph across calls has the same trap.

## Speed

On this CPU, `htdemucs` separates at about 0.45x realtime. The CUDA build compiles for Pascal
(`-DCMAKE_CUDA_ARCHITECTURES=61`), but the GPU numbers weren't measured: the P4000 test box
rebooted into a kernel without its NVIDIA module partway through. TODO.
