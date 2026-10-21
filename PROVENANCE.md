# Provenance

This release is a packaging of a specific upstream state. Everything that
differs from that state is listed here, with the reason.

## Engine source

- Repo: `naytewilson/qwen2.5-ane-optimization` (private research repository)
- Branch: `fix/b4-singlebatch-graphs`
- Commit: `d448f15453cfe0beb312806dfa1355983c21f3a8`
- `qwen_infer` tree: `bb9fc71ed8ea94ee939ec032b60cfd1c8066f202`

## Deltas from upstream

The packaging is no longer byte-identical to upstream. Each change below was
made to turn a research tree into a shippable server, and each one is
verifiable by diffing against the upstream tree.

| File | Change | Why |
|---|---|---|
| `Package.swift` | Removed the MLX and swift-transformers package dependencies; product renamed `qwen_infer` → `qwen-ane` | No target imports those products (`MLXServerBackend` is a URLSession proxy, not an MLX client), and they pulled ~1.6 GB of checkouts. The build is now zero-dependency and finishes in ~10 s. |
| `Sources/qwen_infer/main.swift` | Portable `PROJECT_ROOT`; `--serve`, `--compute-plan`, `--probe-model`; model-cache path moved to `~/.cache/qwen-ane/models`; routing prefers a multi-function artifact; the global default model path (used by `--serve` and `--compute-plan`) is `models/qwen25-1.5b-ane-ctx512-lut4/` | `PROJECT_ROOT` was hardcoded to the author's machine. The new flags expose the placement audit and the artifact probe. The cache path was a leftover personal directory. |
| `Sources/qwen_infer/CortexEngine.swift` | Real-time/QoS policy applied on the ANE-blocked thread, on by default (`QWEN_RT_THREAD=0` disables); per-token percentile stats; decode-loop wall time recorded for `decode_tok_per_sec`; warning when the artifact has no `prefill` function; detected prefill batch logged | The RT policy was applied on a throwaway queue block and had no effect. With it applied to the decode thread, median decode improved and the p90 dispatch tail tightened from 21.0–23.5 ms to 19.3–20.2 ms in an interleaved A/B through the server. The warning makes the single-function degradation visible instead of silently costing 4× prompt latency. |
| `Sources/qwen_infer/HTTPSseServer.swift` | Rewritten as an OpenAI-compatible server; non-streaming responses add a `timings` object (`decode_tok_per_sec`, `tok_per_sec`, `elapsed_sec`) | Upstream exposed a bespoke JSON-RPC socket. The release serves `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, with SSE streaming and optional bearer auth. |
| `Sources/qwen_infer/DaemonServer.swift` | Mode label reflects the path actually used; serve-mode startup reports the real HTTP port; per-token timing on by default; cache path de-branded; `decode_tok_per_sec` computed from the engine's decode-loop time | The log claimed "Speculative Decode" for the non-speculative ANE path, and printed a port that was not the one listening. Timing collection costs ~0.02 ms/token and is needed for `decode_tok_per_sec`. Upstream reported whole-request speed (prompt prefill included) under that name. |
| `Sources/qwen_infer/PromptInjector.swift` | Lessons injection is opt-in via `QWEN_LESSONS_PATH`; no default path | Upstream injected a local knowledge base from a fixed personal path into every prompt. Off by default now. |
| `Sources/qwen_infer/IPCServer.swift` | Dispatch-queue label renamed | De-branding only; no behavioural change. |
| `Sources/qwen_infer/BatchServing.swift` | Two comment lines annotate upstream decision IDs | Comment-only; no code change. |
| `Sources/qwen_infer/ComputePlanAudit.swift` | New file: `--compute-plan` placement audit (one compute plan per function) and `--probe-model` | Placement must be shown empirically, per function: a compute plan loaded without `functionName` only assigns devices to the default function, so `prefill` would read as 100% unassigned. The probe is the release gate for batched prefill. |
| `conversion/anemll/ane_converter/qwen2_5_converter.py` | `ct.target.iOS26` replaced by `deploy_target()`, which reads `ANEMLL_DEPLOY_TARGET` and defaults to `ios18` | The shipped artifact is an ios18 export. (An earlier version of this note blamed the batch-64 `prefill` load failure on the ios19 opset. That was falsified: re-stamping the package to ios18, and recompiling with coremlc 3520.5.1, both still failed. The real cause is the KV-cache write pattern — see "Model artifact".) `ANEMLL_DEPLOY_TARGET=ios26` restores the old behaviour. |
| `conversion/anemll/utils/check_dependencies.sh` | Accepts `python3` when no bare `python` exists | Stock macOS ships `python3` only, so the upstream dependency check aborted every fresh conversion. |
| `setup.sh`, `smoke.sh`, `bench.sh` | Release scripts | Upstream's driver assumed the research repo layout. `setup.sh` builds the engine, installs a prebuilt compiled model (local path or URL) and probes it — it never runs the Python converter; `smoke.sh` gates the HTTP surface; `bench.sh` runs the documented measurement protocol. |

| `tools/kvselect/` | New tool (separate Swift package, depends on swift-protobuf): rewrites the KV-cache writes in a package's `prefill` function | Required to make the batch-64 `prefill` compile for the ANE — see "Model artifact". Not part of the server build. |

## Model artifact

The shipped `qwen25_monolithic_full_lut4.mlmodelc` (`infer` + `prefill`, batch 64) is produced as follows:

1. **Convert** with the vendored pipeline (on a machine with enough RAM; the converter peaks above 8 GB):
   `conversion/anemll/utils/convert_monolith.sh --model Qwen/Qwen2.5-1.5B-Instruct --context 512 --batch 64 --lut 4,8 --lut-embeddings 4 --lut-lmhead 4 --argmax`. The shipped artifact was converted on 2026-10-21 with the then-default `ios26` target. Its opset tag was then re-stamped to `ios18` (`CoreML8`) in `model.mlmodel`; no op or weight changed. Converting with today's default (`ios18`) should give the same program, but that has not been verified byte-for-byte.
2. **Repair `prefill`** with `tools/kvselect`:
   `kvselect <in.mlpackage> <out.mlpackage> prefill`.
   The converter writes each layer's K/V into the `MLState` cache with a `slice_update` whose sequence `begin` is the runtime position and whose width is the batch (64). The ANE compiler rejects that pattern (`ANECCompile` status -14, reported by Core ML as "functionName must be nil unless ML Program"). kvselect replaces each of the 56 writes with an equivalent static-bounds form: it reads the layer row, places the new K/V with a one-hot matmul built from `equal(position − k, current_pos)`, blends that with the old row, and writes the row back with constant bounds. The blend is exact, because every cache element is either kept or replaced, never mixed. Output names are unchanged, so nothing downstream moves.
3. **Compile** with `xcrun coremlcompiler compile` (Xcode 27.1 toolchain).

Steps 1–2 are portable CPU work (steps 2–3 take seconds), but step 1 is heavy, so it belongs on a large Linux box. Only step 3 and everything after it need a Mac.

kvselect writes its output deterministically. Re-running it on the archived source package reproduces the shipped program exactly: same `model.mil` and `weights/weight.bin`. Only `coremldata.bin`, which holds model metadata, differs between compiles.

| File (in the `.mlmodelc`) | SHA-256 |
|---|---|
| `model.mil` | `9e8630d4fa90baddc7eb5555b3a16decfc3dd4e6419de62c75ba0b7918e3fef5` |
| `weights/weight.bin` | `d465d50b4ed75c1a0869145006c6319e4ed5f55f6086c3ee2bf198c5a662a10d` |

Verification, on the hardware below:

- `--probe-model`: MULTI-FUNCTION, prefill batch 64.
- Greedy parity, 5 prompts × 48 tokens:
  - batch-64 prefill vs the same artifact with `QWEN_NO_PREFILL=1` (token-by-token prefill through `infer`): 5/5 identical.
  - vs the June 2026 artifact's token-by-token output: 5/5 identical.
- Placement (`--compute-plan`, per function): `infer` 2147 ANE / 82 CPU (the embedding lookup and the argmax tail); `prefill` 2360 ANE / 13 CPU (the embedding lookup and mask/position setup, the same count as the previous batch-4 artifact). All 56 rewritten KV writes run on the ANE.

The previous release artifact (June 2026, batch-4 `prefill`) does **not** pass the same parity check: its batch-4 prefill matched its own token-by-token prefill on only 1 of 5 prompts. On one prompt it switched the answer's language. This artifact replaces it.

## Tokenizer

- `tokenizer/` is byte-identical to the same upstream commit
  (tree `e96a29428264cb52b9ea64c58b85827db5837c1c`).

## Conversion pipeline

- `conversion/anemll/` is [Anemll](https://github.com/Anemll/Anemll) at
  commit `cc818aeb15a084e36b54981c81602a5597823b9c`, vendored with the two
  deltas listed above.
- `conversion/requirements.txt` pins the dependency set the conversion was
  verified with. The conversion pipeline is kept as a reference for how the shipped artifact was produced; the release runtime and `setup.sh` do not use it.

## Verification environment

Every performance number in the README was measured on:

- MacBook Neo, `Mac17,5`, Apple A18 Pro (6 cores), 8 GB unified memory
- macOS 27.2 (build 26B118), AC power, `lowpowermode 0`
- Swift 6.4 release toolchain, `swift build -c release`
- Model: monolithic LUT4, context 512, `infer` + `prefill` (batch 64) functions — see "Model artifact"

Numbers recorded elsewhere (a different host, a different artifact, or a
different harness) are labelled as such; they are not presented as this
release's measurements.

## What is left out on purpose

The research repo contains experiments that never beat the shipped path:
speculative decoding (EAGLE/Medusa) variants, a fused-attention attempt, and
alternative model lines. None of them is in the served path, and none of
their projected throughput figures are quoted as measurements here.
