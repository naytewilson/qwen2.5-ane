#!/bin/bash
# smoke.sh — end-to-end check of the OpenAI-compatible ANE server.
#
# Exercises the real HTTP surface (not the CLI path): health, model listing,
# a non-streaming chat completion, and an SSE streaming completion. Exits
# non-zero if any endpoint misbehaves, so it is usable as a release gate.
#
# Usage:
#   ./smoke.sh                          # start a server on :45290, test, stop
#   ./smoke.sh --port 45300
#   ./smoke.sh --url http://127.0.0.1:45290   # test an already-running server
set -euo pipefail
cd "$(dirname "$0")"

BIN=".build/release/qwen-ane"
PORT="${QWEN_SSE_PORT:-45290}"
URL=""
SERVER_PID=""
PASS=0
FAIL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --url)  URL="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$URL" ] || URL="http://127.0.0.1:${PORT}"

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ok()   { PASS=$((PASS+1)); printf '  PASS  %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL  %s\n' "$1"; }

if [ ! -x "$BIN" ]; then
  echo "Engine not built. Run: swift build -c release" >&2
  exit 1
fi

if [ ! -d "models" ]; then
  echo "No models/ directory. Run: ./setup.sh" >&2
  exit 1
fi
if ! find models -name '*.mlmodelc' -maxdepth 3 2>/dev/null | grep -q .; then
  echo "No compiled model under models/. Run: ./setup.sh" >&2
  exit 1
fi

if ! curl -fsS --max-time 2 "${URL}/health" >/dev/null 2>&1; then
  echo "Starting $BIN --serve on :${PORT} (first load compiles the ANE program; be patient)…"
  QWEN_SSE_PORT="$PORT" "$BIN" --serve > smoke-server.log 2>&1 &
  SERVER_PID=$!
  deadline=$(( $(date +%s) + 300 ))
  until curl -fsS --max-time 2 "${URL}/health" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "server did not become healthy within 300s — see smoke-server.log" >&2
      exit 1
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited during startup — see smoke-server.log" >&2
      exit 1
    fi
    sleep 1
  done
fi

echo
echo "Testing ${URL}"

# 1. /health
HEALTH=$(curl -sS --max-time 10 "${URL}/health" || true)
case "$HEALTH" in
  *'"status"'*'ok'*) ok "/health responds" ;;
  *) bad "/health — got: $(printf '%s' "$HEALTH" | head -c 120)" ;;
esac

# 2. /v1/models
MODELS=$(curl -sS --max-time 10 "${URL}/v1/models" || true)
# JSON key order is not guaranteed; require a data array holding a model object.
if [[ "$MODELS" == *'"data":['* && "$MODELS" == *'"object":"model"'* && "$MODELS" == *'"object":"list"'* ]]; then
  ok "/v1/models lists a model"
else
  bad "/v1/models — got: $(printf '%s' "$MODELS" | head -c 120)"
fi

# 3. non-streaming chat completion
CHAT=$(curl -sS --max-time 300 -X POST "${URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b-ane","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":24,"temperature":0,"stream":false}' || true)
if printf '%s' "$CHAT" | grep -q '"content"[[:space:]]*:[[:space:]]*"[^"]'; then
  ok "/v1/chat/completions returns content"
  printf '        %s\n' "$(printf '%s' "$CHAT" | sed -n 's/.*"content":"\([^"]*\)".*/\1/p' | head -c 160)"
else
  bad "/v1/chat/completions — got: $(printf '%s' "$CHAT" | head -c 200)"
fi
if printf '%s' "$CHAT" | grep -q '"prompt_tokens"'; then
  ok "response reports token usage"
else
  bad "response is missing token usage"
fi

# 4. streaming chat completion (SSE)
STREAM=$(curl -sS -N --max-time 300 -X POST "${URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b-ane","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":24,"temperature":0,"stream":true}' || true)
if printf '%s' "$STREAM" | grep -q 'data: '; then
  ok "SSE stream emits data: frames"
else
  bad "SSE stream produced no data frames"
fi
if printf '%s' "$STREAM" | grep -q '\[DONE\]'; then
  ok "SSE stream terminates with [DONE]"
else
  bad "SSE stream did not terminate with [DONE]"
fi

# 5. legacy completions endpoint
COMPL=$(curl -sS --max-time 300 -X POST "${URL}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5-1.5b-ane","prompt":"The capital of France is","max_tokens":16,"temperature":0,"stream":false}' || true)
if printf '%s' "$COMPL" | grep -q '"text"'; then
  ok "/v1/completions returns text"
else
  bad "/v1/completions — got: $(printf '%s' "$COMPL" | head -c 200)"
fi

echo
echo "Result: ${PASS} passed, ${FAIL} failed"
if [ -n "$SERVER_PID" ]; then echo "Server log: smoke-server.log"; fi
[ "$FAIL" -eq 0 ]
