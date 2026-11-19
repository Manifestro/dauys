#!/usr/bin/env bash
# DY-32 track A: evaluation rounds on a timer (UTC). Each round: bash dy32_round.sh round-HHMM; the final ones add HF.
#   nohup bash scripts/dy32_timer.sh > /data/dy32_timer.log 2>&1 &
repo=$(cd "$(dirname "$0")/.." && pwd)
for t in 0300 0400 0500 0600 0700 0750:hf 0805:hf; do
  hm=${t%%:*}; hf=; [[ $t == *:hf ]] && hf=hf
  target=$(date -u -d "today ${hm:0:2}:${hm:2:2}" +%s)
  now=$(date -u +%s)
  (( now > target + 1800 )) && { echo "skip $hm (past)"; continue; }
  (( target > now )) && sleep $((target - now))
  [[ -e /data/dy32/a/STOP_TIMER ]] && { echo "stopped"; exit 0; }
  echo "=== round-$hm start $(date -u +%H:%M:%S)"
  bash "$repo/scripts/dy32_round.sh" round-$hm $hf > /data/dy32_round_$hm.log 2>&1
  grep -E "ROUND|FAILED" /data/dy32_round_$hm.log
done
echo "TIMER DONE"
