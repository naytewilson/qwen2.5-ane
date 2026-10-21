# qwen-ane — Qwen2.5-1.5B on the Apple Neural Engine

A native Swift inference server that runs **Qwen2.5-1.5B-Instruct** on the
**Apple Neural Engine** and exposes an **OpenAI-compatible HTTP API**. No
Python at runtime, no external Swift packages, no cloud.

- Monolithic stateful LUT4 Core ML model, KV cache resident in `MLState`
- Transformer compute 100% on the ANE (only the final argmax tail runs on CPU)
- Batched prompt prefill, 64 tokens per ANE dispatch (multi-function `infer` + `prefill` artifact)
- `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`,
  SSE streaming, optional bearer auth

## Performance

Measured on a MacBook (A18 Pro, 8 GB, macOS 27.2), warm server, greedy
decoding, through the HTTP API with `./bench.sh` (5 warm-up requests
discarded, 20 measured requests of 128 tokens each):

| Metric | ms / token | tok/s |
|---|---|---|
| Decode, fastest request (min-of-burst) | 21.20 | 47.2 |
| Decode, median request (p50) | 22.68 | 44.1 |
| Decode, slow tail (p90) | 24.26 | 41.2 |
| Time to first token, 413-token prompt | 0.28 s | — |

Raw ANE dispatch floor, measured with a bare min-of-120 dispatch burst
(no server) on the previous artifact, whose `infer` graph has the same
shape and placement: **18.8 ms minimum, 19.4–19.9 ms
median per token (~51 tok/s)**. The engine adds ~1 ms/token on top of
that; the rest of the spread is the chip's thermal state between
requests.

Decode speed is reported by the server in every non-streaming response
under `timings.decode_tok_per_sec` (prompt prefill excluded).

## Requirements

- Apple Silicon Mac, macOS 15 or later
- Xcode 16+ / Swift 6.1+ toolchain
- The compiled model (`qwen25_monolithic_full_lut4.mlmodelc`, ~855 MB),
  multi-function with `infer` + `prefill` (batch 64). How it is built and
  verified: [PROVENANCE.md § Model artifact](PROVENANCE.md#model-artifact)

## Quickstart

```sh
git clone https://github.com/naytewilson/qwen2.5-ane.git && cd qwen2.5-ane

# Build the engine and install the model (pick one)
QWEN_MODEL_URL=https://…/qwen25_monolithic_full_lut4.mlmodelc.zip ./setup.sh
QWEN_MODEL_PATH=/path/to/qwen25_monolithic_full_lut4.mlmodelc ./setup.sh

# Start the server (port 45282 by default; QWEN_SSE_PORT to change)
.build/release/qwen-ane --serve

# Verify the endpoint end to end
./smoke.sh --url http://127.0.0.1:45282
```

`setup.sh` refuses an artifact that does not load as multi-function (its
prompt prefill would fall back to one ANE dispatch per token); set
`QWEN_ALLOW_SINGLE_FUNCTION=1` to accept one anyway.

## Using it

Any OpenAI client works — point it at the local base URL:

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

Set `QWEN_API_KEY` before starting the server to require
`Authorization: Bearer <key>`.

## Tools

| Command | What it does |
|---|---|
| `./smoke.sh` | Starts a server, checks every endpoint (incl. SSE), exits non-zero on any failure |
| `./bench.sh` | The measurement protocol above; prints min / p50 / p90 / mean and TTFT |
| `qwen-ane --probe-model <mlmodelc>` | Lists functions and reports MULTI-FUNCTION / SINGLE-FUNCTION |
| `qwen-ane --compute-plan <mlmodelc>` | Per-op device placement (ANE / CPU / GPU) for each function |

## How it works

- `Sources/qwen_infer/CortexEngine.swift` — the ANE runner: loads `infer`
  and `prefill` from one package, shares one `MLState` KV cache, prefills
  the prompt in batches, then decodes one token per ANE dispatch with
  preallocated inputs.
- `Sources/qwen_infer/NativeTokenizer.swift` — Qwen BPE tokenizer in Swift.
- `Sources/qwen_infer/HTTPSseServer.swift` — the OpenAI-compatible server.
- `tools/kvselect/` — rewrites the `prefill` KV-cache writes so the
  batch-64 prefill compiles for the ANE (see PROVENANCE.md). Build-time
  only; the server does not use it.
- `conversion/` — the ANEMLL pipeline the model artifact was produced
  with, kept for reference. Neither the runtime nor `setup.sh` uses it.

## Provenance

Packaged from a private research repository at a pinned commit. Every change from upstream is listed with its reason
in [PROVENANCE.md](PROVENANCE.md).

## License

MIT for this package. Qwen2.5 weights © Alibaba Cloud, Apache-2.0.
