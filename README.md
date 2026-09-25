# stems.cpp

Stem separation for the gary ecosystem: Meta's **HTDemucs v4** in C++ on
[ggml](https://github.com/betweentwomidnights/ggml). No PyTorch, no Python at inference time.
A sibling of [sa3.cpp](https://github.com/betweentwomidnights/sa3.cpp) and
[audiocraft.cpp](https://github.com/betweentwomidnights/audiocraft.cpp): same ggml fork, same pin,
same build scripts, same service shape. It exists because gary4juce's Carey extract tab says
"if you have a stem separator it may work better". This is that separator.

All three published checkpoints run and match PyTorch **sample for sample**. That means
`demucs --shifts 0` on a 20 s song, through the whole split/overlap/trim path, at 71–126 dB SNR
per stem (float32 rounding):

| model | stems | params | GGUF |
|---|---|---|---|
| `htdemucs` | drums, bass, other, vocals | 42 M | 168 MB |
| `htdemucs_6s` | + guitar, piano | 27 M | 110 MB |
| `htdemucs_ft` | drums, bass, other, vocals (bag of 4 per-source fine-tunes, best quality, 4x the time) | 168 M | 672 MB |

See [docs/PARITY.md](docs/PARITY.md) for the numbers and how to reproduce them.

## Build

```bash
git clone --recurse-submodules https://github.com/tinycrops/stems.cpp.git
cd stems.cpp
./build.sh cpu          # or: cuda | vulkan | metal | all   (windows: build.cmd cuda)
./models.sh             # htdemucs; ./models.sh all for the other two
```

`build.sh` runs 4 compile jobs by default (`JOBS=8 ./build.sh cuda` to change it). An unbounded
`-j` on ggml's CUDA kernels used up 32 GB and took the build machine down.
Older nvcc with a newer gcc: `./build.sh cuda -DCMAKE_CUDA_HOST_COMPILER=g++-12`.

## Run

```bash
stems-split -m models/htdemucs-f32.gguf -i song.wav -o stems/                     # 4 stems
stems-split -m models/htdemucs-f32.gguf -i loop.wav -o stems/ --two-stems drums   # drums + no_drums
stems-split -m models/htdemucs_6s-f32.gguf -i song.wav -o stems/ --stems guitar,piano
```

Any WAV works: 16/24/32-bit or float, any sample rate (band-limited resample to 44.1 kHz),
mono or stereo. Stems come out 16-bit, scaled down only if they would clip (demucs'
`--clip-mode rescale`), or `--float32`. `--shifts N` averages N random time shifts, like demucs,
at N times the cost. `--overlap` defaults to 0.25.

`no_X` from `--two-stems X` is the **sum of the other stems**. That is demucs' definition, and it
is not the same as mix minus X.

## Service

```bash
stems-server --port 8010 --models-dir models
```

```
GET  /health
GET  /api/models
POST /separate                   JSON in, {success, model, sample_rate, stems: {name: b64 wav}}
POST /api/juce/separate_audio    -> {success, session_id}
GET  /api/juce/poll_status/<id>  -> {status, progress, separation_in_progress, stems on completion}
```

Request: `audio_data` (base64 WAV) and optionally `model` (`htdemucs` | `htdemucs_6s` |
`htdemucs_ft`), `two_stems`, `stems` (list), `shifts`, `overlap`, `seed`, `float32`. The session
and poll shape is audiocraft.cpp's, so gary4juce can reuse the client code it already has for
terry. One job at a time. The model stays resident between requests and swaps when a different
one is asked for.

## C ABI (libstems)

For embedding in a host (a JUCE/iPlug2 plugin, the iOS app, Tauri over FFI), the build also
produces `libstems.so` / `stems.dll` / `libstems.dylib`, or `libstems.a` with `-DSTEMS_STATIC=ON`
(forced on for iOS). The contract is `src/libstems_v1.h`. It is built the same way as sa3.cpp's
`libsa3_v1.h`: one exported symbol, `stems_get_api(STEMS_ABI_VERSION_1)`, returns a function
table. Structs are size-tagged (zero, set `size`, call the `*_init`, then fill in). The library
owns results and you free them with `result_free`. Progress and cancel are callbacks.

```c
const stems_api_v1* api = stems_get_api(STEMS_ABI_VERSION_1);
stems_context_config_v1 cfg = {sizeof cfg}; api->context_config_init(&cfg);
cfg.model_path = "models/htdemucs-f32.gguf";
stems_context* ctx; stems_error_v1 err = {sizeof err};
api->context_create(&cfg, &ctx, &err);

stems_request_v1 req = {sizeof req}; api->request_init(&req);
req.input.samples = buf; req.input.n_samples = n; req.input.n_channels = 2;
req.input.sample_rate = 48000; req.input.layout = STEMS_AUDIO_INTERLEAVED_V1;
stems_result_v1 res = {sizeof res}; api->result_init(&res);
api->separate(ctx, &req, &res, &err);   /* res.samples: [source][channel][sample] at 48 kHz */
api->result_free(&res);
api->context_destroy(ctx);
```

Input can be any rate, any channel count, planar or interleaved. Stems come back at the input's
sample rate and exact length, so a host never resamples. ggml is linked in statically and kept
private: `stems_get_api` is the only exported symbol, so a plugin can load libstems next to
libsa3 without their ggml copies colliding. `tools/stems-libtest.c` is a complete example in
plain C. Its output is byte-identical to `stems-split --float32`.

## How it's put together

- `src/htdemucs.cpp` holds the network as one ggml graph per 7.8 s segment: freq branch, time branch,
  and the 5-layer cross-transformer between them. Every conv is an F32 im2col plus matmul. Transposed convs
  are a matmul plus an explicit overlap-add, so only ops every backend has are used.
- `src/stft.h`: torch's `stft`/`istft` as HTDemucs calls them, on the host in double precision.
- `HTDemucs::separate` reproduces `demucs.apply.apply_model` (bag → shifts → split → segment),
  including the detail that the last segment is centre-padded with real audio from before it,
  not with silence.
- `tools/convert_htdemucs.py` converts the official checkpoints to GGUF. It needs `demucs` at
  conversion time only. `tools/dump_refs.py` and `stems-parity` are the parity harness.

## Credits

[demucs](https://github.com/facebookresearch/demucs) (Défossez et al., MIT, code and weights),
[ggml](https://github.com/ggml-org/ggml). The GGUF loader, WAV reader, HTTP session helpers
and build scripts come from betweentwomidnights' audiocraft.cpp and sa3.cpp (MIT).
