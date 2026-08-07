#!/usr/bin/env bash
# Run every shard as a background process on one machine, no scheduler needed.
# Use this if the cluster has no SLURM, or to run the whole thing on a single
# node. Same sharding and same resume behaviour as the array job.
#
#   export REVIEWDELTA_CONTACT='you@example.edu'
#   ./run_local.sh 8
#
# Reattach later with: tail -f logs/shard*.log
set -euo pipefail

NSHARDS=${1:-8}

if [ -z "${REVIEWDELTA_CONTACT:-}" ]; then
  echo "REVIEWDELTA_CONTACT unset. export it first:" >&2
  echo "  export REVIEWDELTA_CONTACT='you@example.edu'" >&2
  exit 1
fi

mkdir -p logs
for i in $(seq 0 $((NSHARDS - 1))); do
  shard=$(printf "%02d" "$i")
  nohup python3 fetch_pairs.py --shard "$i" --nshards "$NSHARDS" \
    > "logs/shard${shard}.log" 2>&1 &
  echo "launched shard $i (pid $!)"
done

cat <<EOF

${NSHARDS} shards running in the background. Safe to close this terminal.

  progress : tail -f logs/shard*.log
  count    : cat pairs_shard*.jsonl | wc -l
  stop all : pkill -f fetch_pairs.py

When every shard has exited:
  python3 fetch_pairs.py --merge
  python3 report.py
EOF
wait
