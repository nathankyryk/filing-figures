#!/usr/bin/env bash
# Run the GPU stages on the pod (after scripts/setup_pod.sh).
#
#   bash scripts/run_gpu_jobs.sh check    # attended, ~50 min: every path once, on tiny inputs, plus timings
#   bash scripts/run_gpu_jobs.sh bakeoff  # ~1 h: 4 candidates zero-shot on all valid filings; pick the base
#   bash scripts/run_gpu_jobs.sh train    # 1.5-4 h: LoRA fine-tune; score every checkpoint on valid; pick one
#   bash scripts/run_gpu_jobs.sh serve    # ~1.5 h: merge, quantize (FP8, W4A16, NVFP4), accuracy + speed of each
#   bash scripts/run_gpu_jobs.sh all      # bakeoff, then train on the winner, then serve, unattended
#   bash scripts/run_gpu_jobs.sh test     # ~20 min, ONCE, after the hand-check: the final test-split runs
#
# Every stage but check re-launches itself inside tmux (a dropped SSH connection can't kill it),
# uploads its outputs to the private HF repo kyryk/filing-figures-runs, then stops the pod, so
# nothing bills overnight. If that last upload failed, the pod waits HOLD_MINUTES first (outputs
# stay on its disk until then). Every step is timed; a failed step doesn't stop the others.
#
# Settings (environment variables):
#   BASE_MODEL  base to fine-tune (default: the bake-off winner, outputs/bakeoff/winner.json)
#   RUN         training run name (default ft-<base>, e.g. ft-qwen3.5-4b)
#   EPOCHS=3 RANK=16 ALPHA=32 LR=2e-4    training settings
#   CANDIDATES  bake-off models (default: the four in the plan)
#   GPU_PRICE   this pod's $/hour, for the cost per 1,000 filings (serve)
#   QUANT=w4a16 the 4-bit format for the test run (pick it from serve's results)
#   HANDCHECK_DONE=1  required by test: the answer key is final and pushed to HF
#   CHECK_STEPS=5 NO_STOP=1 (keep the pod running) HOLD_MINUTES=120

set -euo pipefail

stage="${1:-}"
case "$stage" in
  check | bakeoff | train | serve | all | test) ;;
  *) echo "usage: $0 check|bakeoff|train|serve|all|test" >&2; exit 2 ;;
esac

# shellcheck source=scripts/pod_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pod_env.sh"
cd "$REPO_DIR"

CANDIDATES="${CANDIDATES:-Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B google/gemma-4-E4B-it Qwen/Qwen3-8B}"
BASE_MODEL="${BASE_MODEL:-}"
RUN="${RUN:-}"
EPOCHS="${EPOCHS:-3}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-32}"
LR="${LR:-2e-4}"
GPU_PRICE="${GPU_PRICE:-}"
QUANT="${QUANT:-w4a16}"
HANDCHECK_DONE="${HANDCHECK_DONE:-0}"
CHECK_STEPS="${CHECK_STEPS:-5}"
NO_STOP="${NO_STOP:-0}"
HOLD_MINUTES="${HOLD_MINUTES:-120}"

# Long stages: re-launch inside tmux, passing the settings explicitly (a running tmux
# server would otherwise hand the new session its own, older environment).
if [ "$stage" != check ] && [ -z "${INSIDE_JOB:-}" ]; then
  if tmux has-session -t "$stage" 2>/dev/null; then
    echo "'$stage' is already running. Watch it: tmux attach -t $stage"
    exit 1
  fi
  tmux new-session -d -s "$stage" \
    "env INSIDE_JOB=1 CANDIDATES='$CANDIDATES' BASE_MODEL='$BASE_MODEL' RUN='$RUN' EPOCHS=$EPOCHS RANK=$RANK \
ALPHA=$ALPHA LR=$LR GPU_PRICE='$GPU_PRICE' QUANT=$QUANT HANDCHECK_DONE=$HANDCHECK_DONE CHECK_STEPS=$CHECK_STEPS \
NO_STOP=$NO_STOP HOLD_MINUTES=$HOLD_MINUTES bash $REPO_DIR/scripts/run_gpu_jobs.sh $stage; exec bash"
  echo "Started '$stage' in tmux. Watch it: tmux attach -t $stage   (leave it running: Ctrl-b then d)"
  echo "Status so far: cat $LOG_DIR/${stage}_status.txt"
  exit 0
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${stage}_$(date +%Y%m%d_%H%M%S).log"
STATUS="$LOG_DIR/${stage}_status.txt"
exec > >(tee -a "$LOG") 2>&1
: >"$STATUS"
failures=0
uploaded=0

# GPU use every 30 s: shows afterwards whether the GPU sat idle
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 30 \
  >>"$LOG_DIR/${stage}_gpu.csv" 2>/dev/null &
sampler=$!

run() { # run <label> <command...>: time it, record ok/FAILED, never abort the stage
  local label=$1 start=$SECONDS
  shift
  printf '\n== %s ==\n$ %s\n' "$label" "$*"
  if "$@"; then
    printf 'ok      %-50s %6ds\n' "$label" $((SECONDS - start)) | tee -a "$STATUS"
  else
    printf 'FAILED  %-50s %6ds\n' "$label" $((SECONDS - start)) | tee -a "$STATUS"
    failures=$((failures + 1))
    return 1
  fi
}

finish() {
  kill "$sampler" 2>/dev/null || true
  printf '\n%s finished %s with %d failed step(s):\n' "$stage" "$(date)" "$failures"
  cat "$STATUS"
  printf 'log: %s\n' "$LOG"
  [ "$stage" = check ] && return
  if [ "$NO_STOP" = 1 ]; then
    echo "NO_STOP=1: the pod keeps running (and billing)."
    return
  fi
  if [ "$uploaded" != 1 ]; then
    # Outputs exist only on this pod's disk, and a stopped pod may not start again (its GPU
    # taken), so leave time to copy them by hand; then stop anyway.
    echo "!! final upload failed: waiting $HOLD_MINUTES min before stopping (touch $LOG_DIR/${stage}_copied to stop now)"
    local waited=0
    while [ ! -f "$LOG_DIR/${stage}_copied" ] && [ "$waited" -lt $((HOLD_MINUTES * 60)) ]; do
      sleep 30
      waited=$((waited + 30))
    done
  fi
  if [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "!! RUNPOD_POD_ID unknown: stop the pod in the Runpod console"
    return
  fi
  sync
  echo "Stopping pod $RUNPOD_POD_ID."
  if command -v runpodctl >/dev/null; then
    # The pod image ships runpodctl v1 ("stop pod"); the pod's own key may stop its pod.
    { runpodctl stop pod "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID"; } && return
  fi
  curl -fsS -X POST -H "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
    "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" && return
  echo "!! automatic stop failed: stop the pod in the Runpod console"
}
trap finish EXIT

can_stop_self() { # the pieces `runpodctl stop pod` needs (a real stop can't be tested without stopping)
  [ -n "${RUNPOD_POD_ID:-}" ] || { echo "RUNPOD_POD_ID is not set"; return 1; }
  [ -n "${RUNPOD_API_KEY:-}" ] || { echo "RUNPOD_API_KEY is not set"; return 1; }
  command -v runpodctl >/dev/null || { echo "runpodctl is missing"; return 1; }
  echo "pod $RUNPOD_POD_ID: runpodctl and the pod key are present"
}

short() { basename "$1" | tr '[:upper:]' '[:lower:]'; } # Qwen/Qwen3.5-4B -> qwen3.5-4b

upload() { # upload <message>: every results folder that exists, to kyryk/filing-figures-runs
  local msg=$1 paths=() p
  shift
  mkdir -p "$FIGURES_OUT/logs/pod" && cp -r "$LOG_DIR"/. "$FIGURES_OUT/logs/pod/" 2>/dev/null || true
  for p in "$@" "$FIGURES_OUT"/{preds,scores,serve,bench,bakeoff,logs}; do
    if [ -e "$p" ]; then paths+=("$p"); fi
  done
  in_env serve figures upload "${paths[@]}" --message "$msg"
}

final_upload() { # the stage's last upload: only its success lets the pod stop at once
  if run "upload results ($1)" upload "$@"; then uploaded=1; fi
}

winner() { # the bake-off winner (fetched from HF if this pod didn't run the bake-off)
  [ -f "$FIGURES_OUT/bakeoff/winner.json" ] || in_env serve figures fetch-runs "$FIGURES_OUT/bakeoff/*" >/dev/null
  in_env serve python -c "import json; print(json.load(open('$FIGURES_OUT/bakeoff/winner.json'))['winner'])"
}

resolve_run() { # sets BASE_MODEL and RUN; fails if there is no base model to use
  if [ -z "$BASE_MODEL" ]; then
    BASE_MODEL=$(winner) || { echo "!! no BASE_MODEL and no bake-off winner"; failures=$((failures + 1)); return 1; }
  fi
  [ -n "$RUN" ] || RUN="ft-$(short "$BASE_MODEL")"
  echo "base model: $BASE_MODEL   run: $RUN"
}

bakeoff_steps() {
  local m
  for m in $CANDIDATES; do
    run "zero-shot $(short "$m"): valid" in_env serve figures vllm-predict --model "$m" --run "base-$(short "$m")" || true
  done
  local names=${CANDIDATES// /,}
  run "pick the base model" in_env serve figures pick-base --models "$names" || return 0
  local w
  w=$(winner) || return 0
  run "thinking on: $(short "$w") valid" in_env serve figures vllm-predict --model "$w" --run "base-$(short "$w")-thinking" \
    --thinking || true
  run "JSON schema on: $(short "$w") valid" in_env serve figures vllm-predict --model "$w" \
    --run "base-$(short "$w")-schema" --schema || true
}

train_steps() {
  resolve_run || return 0
  run "train $RUN ($EPOCHS epochs, r=$RANK, lr $LR)" in_env train figures train --base "$BASE_MODEL" --name "$RUN" \
    --epochs "$EPOCHS" --rank "$RANK" --alpha "$ALPHA" --lr "$LR" || return 0
  local ckpt step merged
  for ckpt in $(in_env train figures checkpoints --name "$RUN"); do
    step=${ckpt##*-}
    merged="$FIGURES_OUT/merged/$RUN-step$step"
    run "merge step $step" in_env train figures merge --adapter "$ckpt" --out "$merged" || continue
    run "score step $step: valid" in_env serve figures vllm-predict --model "$merged" --run "$RUN-step$step" || true
    rm -rf "$merged"  # remade for the chosen step in serve (disk: 9-19 GB each)
  done
  run "pick the checkpoint" in_env serve figures pick-checkpoint --name "$RUN" || true
}

best_checkpoint() {
  [ -f "$FIGURES_OUT/train/$RUN/best.json" ] || in_env serve figures fetch-runs "$FIGURES_OUT/train/$RUN/*" >/dev/null
  in_env serve python -c "import json; print(json.load(open('$FIGURES_OUT/train/$RUN/best.json'))['checkpoint'])"
}

serve_steps() {
  resolve_run || return 0
  local ckpt merged="$FIGURES_OUT/merged/$RUN" q price=()
  ckpt=$(best_checkpoint) || { echo "!! no chosen checkpoint for $RUN"; failures=$((failures + 1)); return 0; }
  if [ -n "$GPU_PRICE" ]; then price=(--gpu-price "$GPU_PRICE"); fi
  run "merge the chosen checkpoint ($ckpt)" in_env train figures merge --adapter "$ckpt" --out "$merged" || return 0
  for q in fp8 w4a16 nvfp4; do
    run "quantize $q" in_env quant figures quantize --model "$merged" --scheme "$q" || continue
    run "score $q: valid" in_env serve figures vllm-predict --model "$FIGURES_OUT/quant/$RUN-$q" --run "$RUN-$q" || true
  done
  run "JSON schema on: $RUN valid" in_env serve figures vllm-predict --model "$merged" --run "$RUN-schema" --schema || true
  run "speed: base $(short "$BASE_MODEL")" in_env serve figures bench --model "$BASE_MODEL" \
    --run "base-$(short "$BASE_MODEL")" "${price[@]}" || true
  run "speed: $RUN bf16" in_env serve figures bench --model "$merged" --run "$RUN-bf16" "${price[@]}" || true
  for q in fp8 w4a16 nvfp4; do
    [ -d "$FIGURES_OUT/quant/$RUN-$q" ] || continue
    run "speed: $RUN $q" in_env serve figures bench --model "$FIGURES_OUT/quant/$RUN-$q" --run "$RUN-$q" "${price[@]}" || true
  done
}

test_steps() {
  if [ "$HANDCHECK_DONE" != 1 ]; then
    echo "!! test needs the final answer key: finish the hand-check, run figures export + push, then HANDCHECK_DONE=1"
    failures=$((failures + 1))
    return 0
  fi
  resolve_run || return 0
  run "fetch the hand-checked dataset" in_env serve figures fetch-data || return 0
  local merged="$FIGURES_OUT/merged/$RUN" quant="$FIGURES_OUT/quant/$RUN-$QUANT" ckpt
  if [ ! -d "$merged" ]; then
    ckpt=$(best_checkpoint) || { echo "!! no chosen checkpoint for $RUN"; failures=$((failures + 1)); return 0; }
    run "merge the chosen checkpoint" in_env train figures merge --adapter "$ckpt" --out "$merged" || true
  fi
  if [ ! -d "$quant" ]; then run "fetch $RUN-$QUANT" in_env serve figures fetch-runs "$quant/*" || true; fi
  run "TEST: base $(short "$BASE_MODEL")" in_env serve figures vllm-predict --model "$BASE_MODEL" \
    --run "base-$(short "$BASE_MODEL")" --split test --final || true
  run "TEST: $RUN bf16" in_env serve figures vllm-predict --model "$merged" --run "$RUN-bf16" --split test --final || true
  run "TEST: $RUN $QUANT" in_env serve figures vllm-predict --model "$quant" --run "$RUN-$QUANT" --split test --final || true
}

export FIGURES_OUT=outputs
run "update code" git pull --ff-only || true
for extra in train quant serve; do # also re-creates an environment a pod restart wiped
  run "sync $extra environment" env UV_PROJECT_ENVIRONMENT="$(venv_dir "$extra")" \
    uv sync --frozen --no-dev --extra "$extra" || true
done

case "$stage" in
  check)
    # Separate output folder: these tiny runs must never look like finished work
    export FIGURES_OUT=outputs/check
    run "doctor" in_env serve figures doctor || true
    run "pod can stop itself" can_stop_self || true
    mkdir -p "$FIGURES_OUT" && date >"$FIGURES_OUT/hello.txt"
    run "HF upload" in_env serve figures upload "$FIGURES_OUT/hello.txt" --message "check: upload works" || true
    for m in $CANDIDATES; do
      run "train $CHECK_STEPS steps: $(short "$m")" in_env train figures train --base "$m" --name "check-$(short "$m")" \
        --max-steps "$CHECK_STEPS" || true
    done
    first=${CANDIDATES%% *}
    adapter=$(in_env train figures checkpoints --name "check-$(short "$first")" | tail -n1)
    merged="$FIGURES_OUT/merged/check-$(short "$first")"
    if [ -n "$adapter" ] && run "merge: $(short "$first")" in_env train figures merge --adapter "$adapter" --out "$merged"; then
      run "vLLM on the merged model: 5 valid" in_env serve figures vllm-predict --model "$merged" --run check-merged --sample 5 || true
      for q in fp8 w4a16 nvfp4; do
        run "quantize $q (8 filings)" in_env quant figures quantize --model "$merged" --scheme "$q" --calib 8 || continue
        run "vLLM on $q: 5 valid" in_env serve figures vllm-predict --model "$FIGURES_OUT/quant/check-$(short "$first")-$q" \
          --run "check-$q" --sample 5 || true
      done
      run "speed (quick)" in_env serve figures bench --model "$merged" --run check-merged --quick || true
    fi
    # Training time for the real run: steps per epoch x epochs x seconds per step
    in_env serve python - "$FIGURES_OUT" "$EPOCHS" <<'EOF' || true
import json, math, pathlib, sys
out, epochs = pathlib.Path(sys.argv[1]), float(sys.argv[2])
steps = math.ceil(940 / 16) * epochs
for meta in sorted(out.glob("train/check-*/train_meta.json")):
    m = json.loads(meta.read_text())
    s = m["seconds_per_step"]
    print(f"{meta.parent.name:24s} {s:6.1f} s/step, peak {m['peak_gpu_gib']:5.1f} GiB -> {epochs:g} epochs ~ {steps * s / 3600:.1f} h + scoring")
EOF
    printf '\nIf everything above is ok:  GPU_PRICE=<$/h> bash scripts/run_gpu_jobs.sh all\n'
    ;;
  bakeoff)
    bakeoff_steps
    final_upload "bake-off"
    ;;
  train)
    train_steps
    final_upload "train $RUN" "$FIGURES_OUT/train/$RUN"
    ;;
  serve)
    serve_steps
    final_upload "serve $RUN" "$FIGURES_OUT/quant"
    ;;
  all)
    bakeoff_steps
    run "upload bake-off" upload "bake-off" || true
    train_steps
    run "upload train" upload "train $RUN" "$FIGURES_OUT/train/$RUN" || true
    serve_steps
    final_upload "all: serve $RUN" "$FIGURES_OUT/train/$RUN" "$FIGURES_OUT/quant"
    ;;
  test)
    test_steps
    final_upload "TEST $RUN"
    ;;
esac
