#!/bin/bash
# run.sh — convenience launcher for the experiment runner.
#
# Usage:
#   ./run.sh w1                     # smoke-test on wikitext
#   ./run.sh w2                     # HotpotQA
#   ./run.sh w3                     # agentic tool use
#   ./run.sh w1 --budgets 32 64 96  # pass extra args
#
# All arguments after the workload tag are forwarded to experiment_runner.py.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

WORKLOAD="${1:-w1}"
shift || true

case "$WORKLOAD" in
    w1|w2|w3)
        ;;
    *)
        echo "usage: $0 {w1|w2|w3} [extra args]" >&2
        exit 1
        ;;
esac

exec python "$HERE/scripts/experiment_runner.py" \
    --workloads "$WORKLOAD" \
    "$@"
