#!/usr/bin/env bash
# 9B 서버 watchdog. 30초마다 health 확인 → 죽었으면 즉시 재시작.
# 죽은 횟수 누적 로그 남겨서 실제로 얼마나 죽었는지 추적.
#
# 사용:
#   nohup tools/9b_watchdog.sh > /tmp/9b_watchdog.log 2>&1 &
#   tail -f /tmp/9b_watchdog.log

set -u
PORT=8001
MODEL=/home/paru/models/gguf/Qwopus3.5-9B-v3.5-Q8_0.gguf
PROFILE=/home/paru/qwen-engine/profiles/9B_v3.bin
ENGINE=/home/paru/qwen-engine/build/qwen-engine
LOGFILE=/home/paru/qwen-engine/main_9b.log
DEATHS_FILE=/tmp/9b_deaths.log

cd /home/paru/qwen-engine

start_9b() {
  echo 9717 | sudo -S sysctl -w vm.drop_caches=3 > /dev/null 2>&1
  sleep 2
  rm -f "$LOGFILE"
  MINF_SPARSE_ATTN=1 MINF_BUDGET=0.20 MINF_MIN_SEQ=4096 \
  MINF_PROFILE_PATH="$PROFILE" \
  QWEN_SLOTS=4 nohup "$ENGINE" "$MODEL" \
    --serve "$PORT" --max-seq 32768 \
    >> "$LOGFILE" 2>&1 &
  PID=$!
  echo "$(date -Is) [watchdog] starting 9B PID=$PID"
  # Wait until either ready or it dies.
  for _ in $(seq 1 60); do
    if grep -q "API Server listening" "$LOGFILE" 2>/dev/null; then
      echo "$(date -Is) [watchdog] 9B ready (PID=$PID)"
      return 0
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "$(date -Is) [watchdog] 9B died during boot (PID=$PID)"
      return 1
    fi
    sleep 5
  done
  echo "$(date -Is) [watchdog] 9B boot timeout (PID=$PID)"
  return 1
}

is_alive() {
  # Either the process is running OR the API answers /health.
  pgrep -f "qwen-engine.*serve $PORT " > /dev/null && \
    curl -s -m 5 "http://127.0.0.1:$PORT/health" > /dev/null
}

# Boot once at start if not already running.
if ! is_alive; then
  start_9b || true
fi

while true; do
  sleep 30
  if ! is_alive; then
    DEATHS=$(wc -l < "$DEATHS_FILE" 2>/dev/null || echo 0)
    DEATHS=$((DEATHS + 1))
    echo "$(date -Is) deaths=$DEATHS" >> "$DEATHS_FILE"
    echo "$(date -Is) [watchdog] 9B unreachable (death #$DEATHS) — restarting"
    # Capture last 50 lines of the corpse log for forensics.
    if [ -f "$LOGFILE" ]; then
      cp "$LOGFILE" "/tmp/9b_corpse_${DEATHS}.log"
    fi
    # Reap any orphaned process before restart.
    pkill -f "qwen-engine.*serve $PORT" 2>/dev/null || true
    sleep 5
    start_9b || sleep 30
  fi
done
