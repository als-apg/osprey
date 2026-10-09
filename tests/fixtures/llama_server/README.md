# Recorded llama-server responses

Raw `/v1/embeddings` and `/v1/models` responses from a real llama-server, replayed
by `tests/models/test_providers_llama_cpp.py` (the `replay` tests) so the
`llama-cpp` adapter is checked against the server's actual dialect rather than a
hand-written response.

## Server

Built from a git checkout of llama.cpp tag `b11277` (not a tarball, not Homebrew),
with Metal, on an Apple-silicon Mac:

```
git clone --branch b11277 https://github.com/ggml-org/llama.cpp.git
cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server
```

- commit: `eae11d2217fe9225d1aaba48773b6cca45ae4de9`
- `llama-server --version`:

  ```
  version: 0.5.0-dev (build 11277, commit eae11d221)
  built with AppleClang 21.0.0.21000334 for Darwin arm64
  ```

Command the responses were recorded against (the documented command; only the
port differs, and `--n-gpu-layers 99` is the GPU-build flag):

```
llama-server --embedding --pooling last -m Qwen.Qwen3-VL-Embedding-2B.Q4_K_M.gguf --mmproj mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf --alias qwen3-vl-embedding-2b --host 127.0.0.1 --port 18080 --no-webui --no-slots -c 8192 -b 2048 -ub 2048 -np 2 --n-gpu-layers 99 --image-max-tokens 256
```

`--image-max-tokens 256` caps each picture at 256 vision tokens, about 512x512 px
of area for Qwen3-VL (one token per 32x32 px). It keeps a query that arrives while
a picture is being embedded fast on a CPU-only server, and it changes every
picture vector, so these recordings are only valid with it. At start-up (with
`-lv 4`) the server confirms it:

```
load_hparams: image_max_pixels:   262144 (custom value)
```

Each probe picture costs 245 prompt tokens under the cap.

`--alias qwen3-vl-embedding-2b` is `LLAMA_CPP_DEFAULT_MODEL`, which is what
`/v1/models` reports as `data[0].id`.

## Weights

Both from Hugging Face repository `DevQuasar/Qwen.Qwen3-VL-Embedding-2B-GGUF`,
revision `6a1b927414664e0e17dd379913e3416a1ae1b48d`. The vectors depend on both.

- model: `Qwen.Qwen3-VL-Embedding-2B.Q4_K_M.gguf` (1107410528 bytes),
  sha256 `42a4ebc629ecc6514649e12b1529b857f54900273bb854f853c970fb90edd09d`
- mmproj: `mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf` (819395136 bytes),
  sha256 `3f89a7768ffa6606935319f71bf56bb71871249ba549bf1080a0caea7a088613`

## Files

- `orbit_kick.png`, `tunnel_temp.png` — the two probe pictures (840x420 px), the
  exact bytes that were embedded. Drawn by `make_probe_plots.py OUT_DIR`.
- `models.json` — `GET /v1/models`, verbatim (plus a final newline).
- `embeddings/<key>.json` — one `POST /v1/embeddings` response each, verbatim
  (plus a final newline).
  `<key>` is the sha256 of the request's single content part serialised as
  `json.dumps(part, sort_keys=True, separators=(",", ":"))` (`record.part_key`).
  The request bodies are the adapter's own: `{"model": "qwen3-vl-embedding-2b",
  "input": [{"content": [part]}]}`.
- `manifest.json` — maps each picture and query to its key.
- `record.py BASE_URL` — re-records everything against a running server.

## Cosines

Each query against each picture, after L2 normalisation, at the full 2048
dimensions and after truncation to 1024 and
renormalisation (what the adapter returns for `dimensions=1024`):

| query | orbit_kick 2048-d | tunnel_temp 2048-d | orbit_kick 1024-d | tunnel_temp 1024-d |
|---|---|---|---|---|
| orbit kick near BPM 7 | **0.525** | 0.251 | **0.540** | 0.227 |
| horizontal orbit distortion after fill | **0.475** | 0.338 | **0.491** | 0.330 |
| tunnel air temperature drift | 0.327 | **0.717** | 0.333 | **0.715** |
| RF cavity trip strip chart (near tie) | 0.399 | 0.398 | 0.418 | 0.398 |

The three discriminative 1024-d cosines (0.540, 0.491, 0.715) are inputs to the
fusion `min_similarity` calibration. The RF row is a near tie, so the replay
test does not assert its nearest picture.
