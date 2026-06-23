#!/usr/bin/env bash
# Run the FULL OSPREY e2e suite across a model × seed matrix with bounded
# parallelism (issue #259). Each (model,seed) is an isolated process writing
# results/<safe>__seed<seed>.{xml,json}. RESUMABLE: combos whose summary json
# already exists are skipped, so a re-launch picks up where it left off.
#
#   scripts/benchmark/matrix_driver.sh           # defaults below
#   MATRIX_PARALLEL=6 scripts/benchmark/matrix_driver.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
REPO="$PWD"

# Model list is overridable via MATRIX_MODELS (space-separated) so an extra
# model (e.g. google/qwen-3-coder) can be appended as a SEPARATE driver run
# without disturbing an already-launched matrix — the resumable skip-logic
# below protects any combos whose summary json already exists.
read -ra MODELS <<< "${MATRIX_MODELS:-gpt-oss-20b gemma-4 cborg-coder gpt-oss-120b}"
# Seeds also overridable via MATRIX_SEEDS — e.g. a single-seed Anthropic
# reference run: MATRIX_MODELS="google/claude-haiku-4-5" MATRIX_SEEDS="1".
read -ra SEEDS <<< "${MATRIX_SEEDS:-1 2 3}"
PARALLEL="${MATRIX_PARALLEL:-4}"
export E2E_TEST_TIMEOUT="${E2E_TEST_TIMEOUT:-1800}"   # hang-breaker only (timeouts scored inconclusive, not failures)

mkdir -p "$REPO/results"
LOG="$REPO/results/matrix.log"
export REPO LOG
echo "=== matrix start $(date) models=[${MODELS[*]}] seeds=[${SEEDS[*]}] parallel=$PARALLEL cap=${E2E_TEST_TIMEOUT}s ===" | tee -a "$LOG"

# SEED-MAJOR order: run every model for seed 1 first, then seed 2, then seed 3,
# so a complete cross-model comparison lands after the first pass (not 3 seeds of
# one model before any other model is touched).
combos=()
for s in "${SEEDS[@]}"; do
  for m in "${MODELS[@]}"; do
    safe="${m//\//__}"
    if [ -f "$REPO/results/${safe}__seed${s}.json" ]; then
      echo "skip (already done): $m seed$s" | tee -a "$LOG"
    else
      combos+=("$m $s")
    fi
  done
done
echo "to run: ${#combos[@]} combos (of $(( ${#MODELS[@]} * ${#SEEDS[@]} )) total)" | tee -a "$LOG"

if [ "${#combos[@]}" -gt 0 ]; then
  printf '%s\n' "${combos[@]}" | xargs -P "$PARALLEL" -I{} bash -c '
    combo="$1"; m="${combo% *}"; s="${combo##* }"; safe="${m//\//__}"
    echo ">> START $m seed$s $(date +%H:%M:%S)" >> "$LOG"
    bash "$REPO/scripts/benchmark/run_e2e_for_model.sh" "$m" "$s" \
      >> "$REPO/results/${safe}__seed${s}.run.log" 2>&1
    echo ">> END   $m seed$s rc=$? $(date +%H:%M:%S)" >> "$LOG"
  ' _ {}
fi

echo "=== matrix complete $(date) ===" | tee -a "$LOG"
echo "MATRIX_EXIT_0" >> "$LOG"
