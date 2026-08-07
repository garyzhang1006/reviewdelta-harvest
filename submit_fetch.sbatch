#!/bin/bash
#SBATCH --job-name=rd-fetch
#SBATCH --array=0-7
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fetch_%A_%a.out
#SBATCH --error=logs/fetch_%A_%a.err

# Source fetching for ReviewDelta, one array task per shard.
#
# This job is network-bound, not compute-bound: each task sleeps 4 seconds
# between arXiv requests and uses roughly 1% of one core. Ask for 1 CPU and
# 2 GB; asking for more only makes the job queue longer for no gain.
#
# Eight shards is deliberate. Each task is a separate arXiv client, so the
# effective request rate is 8 requests per 4 seconds. Going much wider risks
# arXiv throttling the whole cluster's egress IP, which would cost you the
# corpus rather than time. If your cluster NATs everything through one address,
# drop --array to 0-3.
#
#   mkdir -p logs
#   export REVIEWDELTA_CONTACT='you@example.edu'
#   sbatch --export=ALL submit_fetch.sbatch
#
# Then, once every task finishes:
#   python3 fetch_pairs.py --merge && python3 report.py

set -euo pipefail

if [ -z "${REVIEWDELTA_CONTACT:-}" ]; then
  echo "REVIEWDELTA_CONTACT unset; submit with --export=ALL after exporting it" >&2
  exit 1
fi

NSHARDS=${NSHARDS:-8}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

echo "shard ${SHARD}/${NSHARDS} starting on $(hostname) at $(date -Is)"
python3 fetch_pairs.py --shard "${SHARD}" --nshards "${NSHARDS}"
echo "shard ${SHARD} finished at $(date -Is)"
