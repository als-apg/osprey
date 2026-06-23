#!/usr/bin/env bash
# Drive the 3-model als-apg reference bracket (issue #259): full e2e kit per
# model via NATIVE als-apg, writing the same results/matrix.log markers the
# dashboard updater watches for completion. Resumable (skips combos whose
# summary json already exists). 2-wide by default to keep als-apg load low so
# the reference ceiling isn't confounded by rate-limit scatter.
#
#   ALS_APG_API_KEY=sk-... scripts/benchmark/run_alsapg_refs.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
REPO="$PWD"
[ -n "${ALS_APG_API_KEY:-}" ] || { echo "ALS_APG_API_KEY unset" >&2; exit 2; }

read -ra MODELS <<< "${ALSREF_MODELS:-claude-haiku-4-5-20251001 claude-sonnet-4-6 claude-opus-4-6}"
SEED="${ALSREF_SEED:-1}"
PARALLEL="${ALSREF_PARALLEL:-2}"
export E2E_TEST_TIMEOUT="${E2E_TEST_TIMEOUT:-1800}"
export ALS_APG_API_KEY

mkdir -p "$REPO/results"
LOG="$REPO/results/matrix.log"
export REPO LOG
echo "=== alsapg-refs start $(date) models=[${MODELS[*]}] seed=$SEED parallel=$PARALLEL ===" | tee -a "$LOG"

combos=()
for m in "${MODELS[@]}"; do
  safe="${m//\//__}"
  if [ -f "$REPO/results/${safe}__seed${SEED}.json" ]; then
    echo "skip (already done): $m seed$SEED" | tee -a "$LOG"
  else
    combos+=("$m")
  fi
done
echo "to run: ${#combos[@]} combos" | tee -a "$LOG"

if [ "${#combos[@]}" -gt 0 ]; then
  printf '%s\n' "${combos[@]}" | SEED="$SEED" xargs -P "$PARALLEL" -I{} bash -c '
    m="$1"; safe="${m//\//__}"
    echo ">> START $m seed'"$SEED"' $(date +%H:%M:%S)" >> "$LOG"
    bash "$REPO/scripts/benchmark/run_e2e_alsapg.sh" "$m" "'"$SEED"'" \
      >> "$REPO/results/${safe}__seed'"$SEED"'.run.log" 2>&1
    echo ">> END   $m seed'"$SEED"' rc=$? $(date +%H:%M:%S)" >> "$LOG"
  ' _ {}
fi

echo "=== alsapg-refs complete $(date) ===" | tee -a "$LOG"
echo "MATRIX_EXIT_0" >> "$LOG"
