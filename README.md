# qwen2.5-ane — Qwen2.5-1.5B on the Apple Neural Engine

A native Swift server that runs **Qwen2.5-1.5B-Instruct** on the
**Apple Neural Engine** (ANE) and exposes an **OpenAI-compatible HTTP API**.
There is no Python at runtime, no external Swift packages, and no network
access: the model runs entirely on your Mac.

- Monolithic stateful LUT4 Core ML model, KV cache resident in `MLState`
- Transformer compute 100% on the ANE (only the final argmax tail runs on CPU)
- Batched prompt prefill, 64 tokens per ANE dispatch (multi-function `infer` + `prefill` artifact)
- `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`,
  SSE streaming, optional bearer auth

## What it is for

The Neural Engine is a dedicated machine-learning processor in every Apple
Silicon chip. Most apps leave it idle while the CPU and GPU do the work.
This server puts a 1.5-billion-parameter chat model on it. The model
answers with low latency, uses little power, and leaves the CPU and GPU
free for whatever else you are running.

Typical uses:

- **Private local assistant.** Chat, drafting, summarizing and rewriting
  without sending text to a cloud service.
- **Drop-in backend for existing tools.** Any app, script or editor plugin
  that accepts a custom OpenAI base URL can use it unchanged.
- **Background text tasks.** Classification, tagging, short extraction and
  routing jobs that run continuously on a laptop without heating it up or
  slowing it down.
- **Offline and air-gapped work.** Once the model is downloaded, nothing
  leaves the machine.
- **A reference for ANE deployment.** A working, measured example of a
  stateful transformer on the Neural Engine, including the fix that lets
  batched prefill compile for it.

### This is one example

This repository packages and measures one model, Qwen2.5-1.5B-Instruct
with 4-bit LUT weights and a 512-token context. The method is not specific
to that model:

- stateful KV cache in `MLState`
- palettized (LUT) weights
- a separate batched `prefill` function
- the KV-write rewrite in `tools/kvselect`

The vendored converter (`conversion/anemll`) supports the Llama, Qwen,
Gemma 3 and DeepSeek families at other sizes and context lengths. Porting
another model takes three steps:

1. Convert it.
2. Run `kvselect` on its `prefill` function.
3. Adapt the tokenizer and the model's tensor shapes in the engine.

Only the model above has been measured here; nothing in this repository
claims numbers for other models.

## Performance

Measured on a MacBook with an Apple A18 Pro (8 GB, macOS 27.2): warm
server, greedy decoding, through the HTTP API with `./bench.sh` (5 warm-up
requests discarded, 20 measured requests of 128 tokens each):

| Metric | ms / token | tok/s |
|---|---|---|
| Decode, fastest request (min-of-burst) | 21.20 | 47.2 |
| Decode, median request (p50) | 22.68 | 44.1 |
| Decode, slow tail (p90) | 24.26 | 41.2 |
| Time to first token, 413-token prompt | 0.28 s | — |

**Raw ANE dispatch floor.** A bare min-of-120 dispatch burst (no server),
measured on the previous artifact, whose `infer` graph has the same shape
and placement, gives:

- 18.8 ms per token minimum
- 19.4–19.9 ms per token median (~51 tok/s)

The engine adds about 1 ms per token on top of that floor. The rest of the
spread comes from the chip's thermal state between requests.

The server reports decode speed in every non-streaming response under
`timings.decode_tok_per_sec`. Prompt prefill is excluded from that figure.

## Getting started, step by step

No prior experience with Swift or machine learning is needed. Each step
is a command you paste into the Terminal app.

1. **Check your Mac.** You need Apple Silicon (an M-series or A-series
   chip) and macOS 15 or later. Open  → *About This Mac* and look at
   *Chip*.
2. **Install Xcode** from the Mac App Store (version 16 or later), open it
   once, and accept the license. This provides the Swift compiler.
3. **Open Terminal** (Applications → Utilities → Terminal).
4. **Download this project:**

   ```sh
   git clone https://github.com/naytewilson/qwen2.5-ane.git
   cd qwen2.5-ane
   ```

5. **Build the server and install the model.** `setup.sh` compiles the
   server and installs the model from the path or URL you give it. It
   then checks that the model loads correctly on the Neural Engine. The
   model is about 855 MB.

   ```sh
   QWEN_MODEL_URL=<model download URL> ./setup.sh
   ```

6. **Start the server.** The first start takes a minute or two while
   macOS prepares the model for the Neural Engine; later starts are
   faster. Leave this Terminal window open.

   ```sh
   .build/release/qwen-ane --serve
   ```

7. **Check that it works.** In a second Terminal window, from the same
   folder:

   ```sh
   ./smoke.sh --url http://127.0.0.1:45282
   ```

   Every line should read `PASS`.

8. **Use it.** In any app that supports a custom OpenAI-compatible
   server, set:
   - base URL: `http://127.0.0.1:45282/v1`
   - model: `qwen2.5-1.5b-ane`
   - API key: any value, unless you set one (below)

To stop the server, press `Ctrl-C` in its Terminal window.

## Using it from code

Any OpenAI client works. Point it at the local base URL:

```sh
curl http://127.0.0.1:45282/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b-ane","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:45282/v1", api_key="unused")
print(client.chat.completions.create(
    model="qwen2.5-1.5b-ane",
    messages=[{"role": "user", "content": "Hello!"}],
).choices[0].message.content)
```

### Configuration

| Variable | Default | Effect |
|---|---|---|
| `QWEN_SSE_PORT` | `45282` | HTTP port |
| `QWEN_SSE_HOST` | `127.0.0.1` | Bind address. Only local programs can connect unless you change it. |
| `QWEN_API_KEY` | unset | When set, requests must send `Authorization: Bearer <key>` |
| `QWEN_MODEL_PATH` | `models/qwen25-1.5b-ane-ctx512-lut4/…` | Use a different compiled model |

`setup.sh` refuses an artifact that does not load as multi-function,
because its prompt prefill would fall back to one ANE dispatch per token.
Set `QWEN_ALLOW_SINGLE_FUNCTION=1` to accept one anyway.

## Tools

| Command | What it does |
|---|---|
| `./smoke.sh` | Starts a server, checks every endpoint (incl. SSE), exits non-zero on any failure |
| `./bench.sh` | The measurement protocol above; prints min / p50 / p90 / mean and TTFT |
| `qwen-ane --probe-model <mlmodelc>` | Lists functions and reports MULTI-FUNCTION / SINGLE-FUNCTION |
| `qwen-ane --compute-plan <mlmodelc>` | Per-op device placement (ANE / CPU / GPU) for each function |

## How it works

| Path | Role |
|---|---|
| `Sources/qwen_infer/CortexEngine.swift` | The ANE runner. It loads `infer` and `prefill` from one package and shares one `MLState` KV cache between them. It prefills the prompt in batches of 64, then decodes one token per ANE dispatch with preallocated inputs. |
| `Sources/qwen_infer/NativeTokenizer.swift` | Qwen BPE tokenizer in Swift |
| `Sources/qwen_infer/HTTPSseServer.swift` | The OpenAI-compatible server |
| `tools/kvselect/` | Rewrites the `prefill` KV-cache writes so the batch-64 prefill compiles for the ANE (see [PROVENANCE.md](PROVENANCE.md#model-artifact)). Build-time only; the server does not use it. |
| `conversion/` | The ANEMLL pipeline the model artifact was produced with, kept for reference. Neither the runtime nor `setup.sh` uses it. |

## Provenance

Packaged from a private research repository at a pinned commit. Every
change from that source, and how the model artifact was built and
verified, is listed in [PROVENANCE.md](PROVENANCE.md).

## License

MIT for this package. Qwen2.5 weights © Alibaba Cloud, Apache-2.0.
