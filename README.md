# qwen2.5-ane — Qwen2.5-1.5B on the Apple Neural Engine

A complete transformer language model running on the **Apple Neural
Engine** (ANE). All 28 transformer layers run on the ANE: attention, the
KV cache, the MLPs and the LM head. Nothing runs on the GPU. The CPU
handles only the token-embedding lookup at the input and the argmax at the
output.

The model is deliberately small: **Qwen2.5-1.5B-Instruct**, 512-token
context. It is not chosen for size or context length. It shows that a
stateful LLM can be placed fully on the Neural Engine, which is often
assumed not to be possible. A native Swift server exposes it through an
**OpenAI-compatible HTTP API**, with no Python at runtime and no network
access.

- Monolithic stateful LUT4 Core ML model, KV cache resident in `MLState`
- Batched prompt prefill, 64 tokens per ANE dispatch (multi-function `infer` + `prefill` artifact)
- `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`,
  SSE streaming, optional bearer auth

## ANE placement, and how to check it yourself

Core ML reports, per operation, which device it will run on. For this
model:

| Function | Neural Engine | CPU | GPU |
|---|---|---|---|
| `infer` (one token per call) | 2147 ops | 82 ops: the embedding lookup (1 `gather`), the argmax over the 16 LM-head chunks, and small position/mask index ops | 0 |
| `prefill` (64 tokens per call) | 2360 ops | 13 ops: the embedding lookup and causal-mask/position setup | 0 |

Every transformer layer is on the Neural Engine, including the attention,
the KV-cache reads and writes and the LM-head projection. Run the same audit on your own Mac:

```sh
.build/release/qwen-ane --compute-plan models/qwen25-1.5b-ane-ctx512-lut4/qwen25_monolithic_full_lut4.mlmodelc
```

It loads one compute plan per function and lists every op that is not on
the Neural Engine.

## What it is for

The Neural Engine is a dedicated machine-learning processor in every Apple
Silicon chip. Most apps leave it idle while the CPU and GPU do the work.
This server puts a 1.5-billion-parameter chat model on it. Because the
work runs on the Neural Engine, the CPU and GPU stay free for whatever else
you are running. Power draw has not been measured for this release.

Typical uses:

- **Private local assistant.** Chat, drafting, summarizing and rewriting
  without sending text to a cloud service.
- **Drop-in backend for existing tools.** Any app, script or editor plugin
  that accepts a custom OpenAI base URL can use it unchanged.
- **Background text tasks.** Classification, tagging, short extraction and
  routing jobs that run in the background without occupying the CPU or
  GPU.
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
| Decode, fastest request (min-of-burst) | 18.95 | 52.8 |
| Decode, median request (p50) | 19.32 | 51.8 |
| Decode, slow tail (p90) | 21.95 | 45.6 |
| Time to first token, 413-token prompt | 0.24 s | — |

**Compared with the raw hardware.** A bare min-of-120 dispatch burst, with
no server, measured 18.8 ms per token minimum and 19.4–19.9 ms median on
the previous artifact, whose `infer` graph has the same shape and
placement. The server matches that floor: its own work between ANE calls
measures about 0.05 ms per token. The slow tail is variation inside the
ANE call itself, not in the server.

The server reports decode speed in every non-streaming response under
`timings.decode_tok_per_sec`. Prompt prefill is excluded from that figure.

## Getting started, step by step

No prior experience with Swift or machine learning is needed. Each step
is a command you paste into the Terminal app.

1. **Check your Mac.** You need Apple Silicon (an M-series or A-series
   chip) and macOS 15 or later. Open  → *About This Mac* and look at
   *Chip*.
2. **Install Xcode** from the Mac App Store (version 16.3 or later, for
   Swift 6.1), open it once, and accept the license. This provides the
   Swift compiler.
3. **Open Terminal** (Applications → Utilities → Terminal).
4. **Download this project:**

   ```sh
   git clone https://github.com/naytewilson/qwen2.5-ane.git
   cd qwen2.5-ane
   ```

5. **Build the server and install the model.** `setup.sh` compiles the
   server and installs the model from the path or URL you give it. It
   then checks that the model loads correctly on the Neural Engine. The
   model is about 855 MB. It is hosted on Hugging Face at
   [NayteThaGreat/Qwen2.5-1.5B-Instruct-ANE](https://huggingface.co/NayteThaGreat/Qwen2.5-1.5B-Instruct-ANE).

   ```sh
   QWEN_MODEL_URL=https://huggingface.co/NayteThaGreat/Qwen2.5-1.5B-Instruct-ANE/resolve/main/qwen25_monolithic_full_lut4.mlmodelc.zip ./setup.sh
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

**Tested on:** MacBook with Apple A18 Pro, 8 GB, macOS 27.2, Swift 6.4.
The package targets macOS 15 and later on any Apple Silicon Mac, but other
chips and OS versions have not been measured. Speeds will differ by chip.

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
