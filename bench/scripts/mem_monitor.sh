#!/bin/bash
# usage: mem_monitor.sh PID OUTFILE ; logs every second: time rss_MB free% swap_used
PID=$1; OUT=$2; echo "time rss_mb phys_footprint_mb mem_free_pct swap_used" > "$OUT"
while kill -0 $PID 2>/dev/null; do
  rss=$(ps -o rss= -p $PID | awk '{print int($1/1024)}')
  fp=$(footprint -p $PID 2>/dev/null | awk '/phys_footprint:/{print $2,$3; exit}')
  free=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/{print $2}')
  sw=$(sysctl -n vm.swapusage | awk '{print $6}')
  echo "$(date +%H:%M:%S) $rss \"$fp\" $free $sw" >> "$OUT"; sleep 1
done
