#!/bin/bash
# bench.sh — repeatable decode + prefill measurement against this build.
#
# Protocol (stated explicitly so the numbers mean something):
#   * one server process, warmed by WARMUP discarded requests
#   * N measured requests, greedy decoding, fixed output length
#   * two decode numbers are reported, because they answer different questions:
#       min-of-burst — fastest single request (best case; this is the statistic
#                      the upstream "19.4 ms/token" reference was recorded with)
#       p50          — typical request in a steady stream (what a user feels)
#   * prefill is reported separately as time-to-first-byte on a streaming
#     request; decode speed says nothing about prompt latency.
#
# Usage:
#   ./bench.sh                       # start a server on :45290, bench it, stop it
#   ./bench.sh --port 45300
#   ./bench.sh --runs 30 --tokens 128
#   ./bench.sh --url http://127.0.0.1:45290   # bench an already-running server
set -euo pipefail
cd "$(dirname "$0")"

BIN="${QWEN_ANE_BIN:-.build/release/qwen-ane}"
PORT="${QWEN_SSE_PORT:-45290}"
WARMUP=5
RUNS=20
TOKENS=128
URL=""
PROMPT="Explain in detail how a neural engine accelerates transformer inference."
LONG_PROMPT=""
for _ in $(seq 1 12); do
  LONG_PROMPT="${LONG_PROMPT}The neural engine is a fixed-function matrix accelerator beside the CPU and GPU that executes quantized transformer blocks with a very small per-dispatch overhead. "
done
LONG_PROMPT="${LONG_PROMPT}Summarize the above in one sentence."

while [ $# -gt 0 ]; do
  case "$1" in
    --port)    PORT="$2"; shift 2 ;;
    --warmup)  WARMUP="$2"; shift 2 ;;
    --runs)    RUNS="$2"; shift 2 ;;
    --tokens)  TOKENS="$2"; shift 2 ;;
    --url)     URL="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$URL" ] || URL="http://127.0.0.1:${PORT}"
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_health() {
  local deadline=$(( $(date +%s) + 300 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS --max-time 2 "${URL}/health" >/dev/null 2>&1; then return 0; fi
    if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited during startup — see bench-server.log" >&2
      return 1
    fi
    sleep 1
  done
  echo "server did not become healthy within 300s" >&2
  return 1
}

if [ -z "${BENCH_REUSE_SERVER:-}" ] && ! curl -fsS --max-time 2 "${URL}/health" >/dev/null 2>&1; then
  if [ ! -x "$BIN" ]; then
    echo "Engine not built. Run: swift build -c release" >&2
    exit 1
  fi
  echo "Starting $BIN --serve on :${PORT} (model load can take a couple of minutes)…"
  QWEN_SSE_PORT="$PORT" "$BIN" --serve > bench-server.log 2>&1 &
  SERVER_PID=$!
  wait_for_health
fi

echo
echo "endpoint: ${URL}"
curl -fsS "${URL}/v1/models" || { echo "models endpoint unavailable" >&2; exit 1; }
echo

request() { # $1 = prompt, $2 = max_tokens, $3 = stream(true|false)
  local body
  body=$(cat <<JSON
{"model":"qwen2.5-1.5b-ane","messages":[{"role":"user","content":"$1"}],"max_tokens":$2,"temperature":0,"stream":$3}
JSON
)
  curl -sS --max-time 600 -X POST "${URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$body"
}

echo "Warming up (${WARMUP} discarded requests)…"
for _ in $(seq 1 "$WARMUP"); do
  request "$PROMPT" "$TOKENS" false >/dev/null
done

echo "Measuring ${RUNS} requests × ${TOKENS} tokens…"
TPS_LIST=""
for i in $(seq 1 "$RUNS"); do
  RESP=$(request "$PROMPT" "$TOKENS" false)
  TPS=$(printf '%s' "$RESP" | sed -n 's/.*"decode_tok_per_sec":\([0-9.]*\).*/\1/p')
  if [ -z "$TPS" ]; then
    echo "request $i returned no decode_tok_per_sec; body:" >&2
    printf '%s\n' "$RESP" | head -c 400 >&2
    echo >&2
    exit 1
  fi
  TPS_LIST="${TPS_LIST}${TPS}"$'\n'
  printf '  %2d/%d  %6.2f tok/s  %6.2f ms/token\n' "$i" "$RUNS" "$TPS" "$(awk -v t="$TPS" 'BEGIN{printf "%.2f", 1000/t}')"
done

echo
# POSIX awk (macOS): functions are top-level. Percentiles are latency percentiles,
# so p90 is the slow tail = the 10th percentile of tok/s.
printf '%s' "$TPS_LIST" | awk '
  function q(pct,   idx) { idx=int(pct*n + 0.999); if (idx<1) idx=1; if (idx>n) idx=n; return v[idx] }
  { v[NR]=$1; s+=$1 }
  END {
    n=NR
    # ascending tok/s sort
    for (i=1;i<=n;i++) for (j=i+1;j<=n;j++) if (v[j]<v[i]) { t=v[i]; v[i]=v[j]; v[j]=t }
    best=v[n]; p50=q(0.50); p90=q(0.10); mean=s/n
    printf "Decode (%d requests, %d tokens each)\n", n, '"$TOKENS"'
    printf "  min-of-burst : %6.2f tok/s   %6.2f ms/token\n", best, 1000/best
    printf "  p50          : %6.2f tok/s   %6.2f ms/token\n", p50, 1000/p50
    printf "  p90          : %6.2f tok/s   %6.2f ms/token\n", p90, 1000/p90
    printf "  mean         : %6.2f tok/s   %6.2f ms/token\n", mean, 1000/mean
  }'

echo
echo "Prefill / TTFT (streaming, 413-token prompt)…"
# TTFT = request start → first SSE frame carrying generated text. curl's
# time_starttransfer is NOT used: the server flushes SSE headers immediately,
# so it would report header latency (~0.01 s), not prefill + first token.
TTFT_BODY=$(cat <<JSON
{"model":"qwen2.5-1.5b-ane","messages":[{"role":"user","content":"$LONG_PROMPT"}],"max_tokens":16,"temperature":0,"stream":true}
JSON
)
TTFT=$(/usr/bin/perl -MTime::HiRes=time -e '
  my $t0 = time;
  open(my $fh, "-|", @ARGV) or exit 1;
  while (my $l = <$fh>) {
    if ($l =~ /^data: / && $l =~ /"(content|text)":"[^"]/) { printf "%.3f\n", time - $t0; exit 0 }
  }
  exit 1' curl -s -N --max-time 600 -X POST "${URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$TTFT_BODY" || true)
if [ -n "$TTFT" ]; then
  awk -v t="$TTFT" 'BEGIN{printf "  time to first token: %.2f s\n", t}'
else
  echo "  TTFT measurement failed" >&2
fi
echo
echo "Done. Server log: bench-server.log"
