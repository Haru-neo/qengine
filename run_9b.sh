#!/bin/bash
cd /home/paru/qwen-engine
ulimit -c unlimited
echo "[wrapper] start $(date -Is) parent_pid=$$" >> main_9b.exit
CUDA_VISIBLE_DEVICES=0,1 QWEN_SLOTS=4 ./build/qwen-engine \
  /home/paru/models/gguf/Qwopus3.5-9B-v3.5-Q8_0.gguf \
  --serve 8001 --max-seq 32768 \
  > main_9b.log 2> main_9b.err
RC=$?
echo "[wrapper] exit code=$RC time=$(date -Is)" >> main_9b.exit
case $RC in
  0)   echo "[wrapper] clean exit" >> main_9b.exit ;;
  130) echo "[wrapper] SIGINT (Ctrl-C, code 130)" >> main_9b.exit ;;
  137) echo "[wrapper] SIGKILL (kernel OOM-killer or kill -9, code 137)" >> main_9b.exit ;;
  143) echo "[wrapper] SIGTERM (kill, code 143)" >> main_9b.exit ;;
  139) echo "[wrapper] SIGSEGV (segfault, code 139) — check core/apport" >> main_9b.exit ;;
  134) echo "[wrapper] SIGABRT (abort, code 134) — usually CUDA assertion or std::terminate" >> main_9b.exit ;;
  *)   echo "[wrapper] unusual exit code=$RC" >> main_9b.exit ;;
esac
