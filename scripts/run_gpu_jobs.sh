#!/usr/bin/env bash
# Run the GPU stages on the pod (after scripts/setup_pod.sh).
#
# Task v1 (six fields):
#   bash scripts/run_gpu_jobs.sh check    # attended, ~50 min: every path once, on tiny inputs, plus timings
#   bash scripts/run_gpu_jobs.sh bakeoff  # ~1 h: 4 candidates zero-shot on all valid filings; pick the base
#   bash scripts/run_gpu_jobs.sh train    # 1.5-4 h: LoRA fine-tune; score every checkpoint on valid; pick one
#   bash scripts/run_gpu_jobs.sh serve    # ~1.5 h: merge, quantize (FP8, W4A16, NVFP4), accuracy + speed of each
#   bash scripts/run_gpu_jobs.sh all      # bakeoff, then train on the winner, then serve, unattended
#   bash scripts/run_gpu_jobs.sh test     # ~20 min, ONCE, after the hand-check: the final test-split runs
#
# Task v2 (twelve fields; does GRPO generalize better than SFT out of distribution?):
#   TASK=v2 bash scripts/run_gpu_jobs.sh check   # attended, ~1.25 h: pod tests, then SFT, the filter and GRPO
#                                                #   once each on tiny inputs; go/no-go lines and the runs' cost
#   bash scripts/run_gpu_jobs.sh sft2            # ~1 h: base zero-shot, SFT with v1's recipe, every checkpoint
#                                                #   scored on valid + ood-valid, one picked
#   INIT=base bash scripts/run_gpu_jobs.sh grpo  # ~4.5 h: filter the train filings, GRPO, every checkpoint
#                                                #   scored, one picked. INIT=sft starts from sft2's pick
#   CHAIN="sft2 grpo" bash scripts/run_gpu_jobs.sh chain  # several v2 stages on one pod, then stop
#   bash scripts/run_gpu_jobs.sh test2           # ONCE, after the v2 hand-check: every model on test + ood-test
#   DRY_RUN=1 ...                                # print a v2 stage's commands, run none (works off the pod)
#
# Every stage but v1's check re-launches itself inside tmux (a dropped SSH connection can't kill it).
# Every stage but check uploads its outputs to the private HF repo kyryk/filing-figures-runs, then
# stops the pod, so nothing bills overnight. If that last upload failed, the pod waits HOLD_MINUTES
# first (outputs stay on its disk until then). Every step is timed; a failed step doesn't stop the others.
#
# Settings (environment variables):
#   BASE_MODEL  base to fine-tune (default: the bake-off winner, outputs/bakeoff/winner.json)
#   RUN         training run name (default ft-<base>, e.g. ft-qwen3.5-4b)
#   EPOCHS=3 RANK=16 ALPHA=32 LR=2e-4    training settings (SFT, both tasks)
#   CANDIDATES  bake-off models (default: the four in the plan)
#   GPU_PRICE   this pod's $/hour, for the cost per 1,000 filings (serve) and the runs' cost (v2 check)
#   QUANT=w4a16 the 4-bit format for the test run (pick it from serve's results)
#   HANDCHECK_DONE=1  required by test and test2: the answer key is final and pushed to HF
#   CHECK_STEPS=5 NO_STOP=1 (keep the pod running) HOLD_MINUTES=120
# Task v2:
#   V2_BASE=Qwen/Qwen3.5-4B  SFT2_RUN=ft2-<base>  INIT=base|sft  STACK=trl|unsloth (the fallback)
#   GRPO_STEPS=150  CHECK_GRPO_STEPS=8  MIN_KEPT=100 (grpo INIT=sft stops if the filter keeps fewer)
#   VLLM_MEM    vLLM's share of GPU memory while GRPO trains (default 0.3, or 0.9 for unsloth)

set -euo pipefail

stage="${1:-}"
case "$stage" in
  check | bakeoff | train | serve | all | test | sft2 | grpo | chain | test2) ;;
  *) echo "usage: $0 check|bakeoff|train|serve|all|test|sft2|grpo|chain|test2" >&2; exit 2 ;;
esac

DRY_RUN="${DRY_RUN:-0}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/pod_env.sh
source "$SCRIPTS/pod_env.sh"
if [ "$DRY_RUN" = 1 ]; then cd "$SCRIPTS/.."; else cd "$REPO_DIR"; fi

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
TASK="${TASK:-v1}"
V2_BASE="${V2_BASE:-Qwen/Qwen3.5-4B}"
SFT2_RUN="${SFT2_RUN:-}"
INIT="${INIT:-base}"
STACK="${STACK:-trl}"
GRPO_STEPS="${GRPO_STEPS:-150}"
CHECK_GRPO_STEPS="${CHECK_GRPO_STEPS:-8}"
MIN_KEPT="${MIN_KEPT:-100}"
VLLM_MEM="${VLLM_MEM:-}"
CHAIN="${CHAIN:-}"

short() { basename "$1" | tr '[:upper:]' '[:lower:]'; } # Qwen/Qwen3.5-4B -> qwen3.5-4b
if [ -z "$SFT2_RUN" ]; then SFT2_RUN="ft2-$(short "$V2_BASE")"; fi
case "$stage" in
  sft2 | grpo | chain | test2) TASK=v2 ;;
esac
case "$TASK" in v1 | v2) ;; *) echo "TASK is v1 or v2" >&2; exit 2 ;; esac
case "$INIT" in base | sft) ;; *) echo "INIT is base or sft" >&2; exit 2 ;; esac
case "$STACK" in trl | unsloth) ;; *) echo "STACK is trl or unsloth" >&2; exit 2 ;; esac
if [ "$stage" = chain ]; then
  [ -n "$CHAIN" ] || { echo 'CHAIN is empty: e.g. CHAIN="sft2 grpo"' >&2; exit 2; }
  for s in $CHAIN; do
    case "$s" in sft2 | grpo) ;; *) echo "CHAIN takes sft2 and grpo, not $s" >&2; exit 2 ;; esac
  done
fi
if [ "$DRY_RUN" = 1 ] && [ "$TASK" != v2 ]; then
  echo "DRY_RUN covers the v2 stages: TASK=v2 check, sft2, grpo, chain, test2" >&2
  exit 2
fi
MEM_ARGS=()
if [ -n "$VLLM_MEM" ]; then MEM_ARGS=(--vllm-mem "$VLLM_MEM"); fi

# Long stages: re-launch inside tmux, passing the settings explicitly (a running tmux
# server would otherwise hand the new session its own, older environment).
if { [ "$stage" != check ] || [ "$TASK" = v2 ]; } && [ -z "${INSIDE_JOB:-}" ] && [ "$DRY_RUN" != 1 ]; then
  if tmux has-session -t "$stage" 2>/dev/null; then
    echo "'$stage' is already running. Watch it: tmux attach -t $stage"
    exit 1
  fi
  tmux new-session -d -s "$stage" \
    "env INSIDE_JOB=1 CANDIDATES='$CANDIDATES' BASE_MODEL='$BASE_MODEL' RUN='$RUN' EPOCHS=$EPOCHS RANK=$RANK \
ALPHA=$ALPHA LR=$LR GPU_PRICE='$GPU_PRICE' QUANT=$QUANT HANDCHECK_DONE=$HANDCHECK_DONE CHECK_STEPS=$CHECK_STEPS \
NO_STOP=$NO_STOP HOLD_MINUTES=$HOLD_MINUTES TASK=$TASK V2_BASE='$V2_BASE' SFT2_RUN='$SFT2_RUN' INIT=$INIT \
STACK=$STACK GRPO_STEPS=$GRPO_STEPS CHECK_GRPO_STEPS=$CHECK_GRPO_STEPS MIN_KEPT=$MIN_KEPT VLLM_MEM='$VLLM_MEM' \
CHAIN='$CHAIN' bash $REPO_DIR/scripts/run_gpu_jobs.sh $stage; exec bash"
  echo "Started '$stage' in tmux. Watch it: tmux attach -t $stage   (leave it running: Ctrl-b then d)"
  echo "Status so far: cat $LOG_DIR/${stage}_status.txt"
  exit 0
fi

if [ "$DRY_RUN" = 1 ]; then LOG_DIR=$(mktemp -d); fi
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${stage}_$(date +%Y%m%d_%H%M%S).log"
STATUS="$LOG_DIR/${stage}_status.txt"
exec > >(tee -a "$LOG") 2>&1
: >"$STATUS"
failures=0
uploaded=0

# GPU use every 30 s: shows afterwards whether the GPU sat idle
sampler=
if [ "$DRY_RUN" != 1 ]; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 30 \
    >>"$LOG_DIR/${stage}_gpu.csv" 2>/dev/null &
  sampler=$!
fi

run() { # run <label> <command...>: time it, record ok/FAILED, never abort the stage
  local label=$1 start=$SECONDS
  shift
  printf '\n== %s ==\n$ %s\n' "$label" "$*"
  if [ "$DRY_RUN" = 1 ]; then
    printf 'dry     %s\n' "$label" >>"$STATUS"
    return 0
  fi
  if "$@"; then
    printf 'ok      %-50s %6ds\n' "$label" $((SECONDS - start)) | tee -a "$STATUS"
  else
    printf 'FAILED  %-50s %6ds\n' "$label" $((SECONDS - start)) | tee -a "$STATUS"
    failures=$((failures + 1))
    return 1
  fi
}

ask() { # ask <stand-in> <command...>: output a later step needs (DRY_RUN prints the stand-in instead)
  local stand_in=$1
  shift
  if [ "$DRY_RUN" = 1 ]; then echo "$stand_in"; else "$@"; fi
}

finish() {
  if [ -n "$sampler" ]; then kill "$sampler" 2>/dev/null || true; fi
  printf '\n%s finished %s with %d failed step(s):\n' "$stage" "$(date)" "$failures"
  cat "$STATUS"
  printf 'log: %s\n' "$LOG"
  if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN=1: nothing ran."
    return
  fi
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

# ---- task v2 ------------------------------------------------------------------------------

rl_env() { if [ "$STACK" = unsloth ]; then echo rl-unsloth; else echo rl; fi; }

grpo_name() { echo "grpo-$INIT-$(short "$V2_BASE")"; } # grpo-base-qwen3.5-4b, grpo-sft-qwen3.5-4b

last_checkpoint() { in_env train figures checkpoints --name "$1" | tail -n1; }

kept_count() { # kept_count <filter folder>: the train filings the difficulty filter kept
  in_env serve python -c "import json, sys; print(json.load(open(sys.argv[1]))['kept'])" "$1/filter_summary.json"
}

hello_upload() {
  mkdir -p "$FIGURES_OUT" && date >"$FIGURES_OUT/hello.txt" &&
    in_env serve figures upload "$FIGURES_OUT/hello.txt" --message "check: upload works"
}

score2() { # score2 <model> <run>: valid and ood-valid, one vLLM start
  run "score $2: valid + ood-valid" in_env serve figures vllm-predict --task v2 --model "$1" --run "$2" \
    --splits valid,ood-valid
}

score_checkpoints() { # score_checkpoints <run>: merge, score and delete each saved adapter, then pick one
  local name=$1 ckpt step merged
  for ckpt in $(ask "$FIGURES_OUT/train/$name/checkpoint-N" in_env train figures checkpoints --name "$name"); do
    step=${ckpt##*-}
    merged="$FIGURES_OUT/merged/$name-step$step"
    run "merge $name step $step" in_env train figures merge --adapter "$ckpt" --out "$merged" || continue
    score2 "$merged" "$name-step$step" || true
    if [ "$DRY_RUN" != 1 ]; then rm -rf "$merged"; fi # ~9 GB each; the chosen one is merged again when needed
  done
  run "pick the $name checkpoint (valid + ood-valid pooled)" in_env serve figures pick-checkpoint --task v2 \
    --name "$name"
}

best_of() { # best_of <run>: its chosen checkpoint's folder (fetched from HF if this pod didn't train it)
  local name=$1 best="$FIGURES_OUT/train/$1/best.json"
  if [ "$DRY_RUN" = 1 ]; then
    echo "$FIGURES_OUT/train/$name/checkpoint-BEST"
    return
  fi
  [ -f "$best" ] || in_env serve figures fetch-runs "$FIGURES_OUT/train/$name/*" >/dev/null || return 1
  [ -f "$best" ] || return 1
  in_env serve python -c "import json, sys; print(json.load(open(sys.argv[1]))['checkpoint'])" "$best"
}

merged_best() { # merged_best <run>: prints the folder of its chosen checkpoint merged into its base (kept)
  local name=$1 merged="$FIGURES_OUT/merged/$1" ckpt
  if [ "$DRY_RUN" = 1 ] || [ ! -f "$merged/filing_figures.json" ]; then # merge.py writes it last
    ckpt=$(best_of "$name") || { echo "!! no chosen checkpoint for $name" >&2; return 1; }
    run "merge the chosen $name checkpoint" in_env train figures merge --adapter "$ckpt" --out "$merged" >&2 ||
      return 1
  fi
  echo "$merged"
}

check2_steps() {
  local sft=check2-sft grpo=check2-grpo filt=check2-filter adapter merged kept keep=() price=()
  run "doctor (v2)" in_env serve figures doctor --task v2 || true
  run "pod can stop itself" can_stop_self || true
  run "HF upload" hello_upload || true
  run "pod tests (rl environment)" in_env rl python -m pytest -q tests/test_rl_pod.py tests/test_rl.py \
    tests/test_reward.py || true
  # SFT v2 end to end: a few steps, merge, the 12-field answer on 5 valid filings in vLLM
  if run "SFT v2: $CHECK_STEPS steps" in_env train figures train --task v2 --base "$V2_BASE" --name "$sft" \
    --max-steps "$CHECK_STEPS" --max-seq-length 9216; then
    adapter=$(ask "$FIGURES_OUT/train/$sft/checkpoint-$CHECK_STEPS" last_checkpoint "$sft")
    merged="$FIGURES_OUT/merged/$sft"
    if run "merge SFT v2" in_env train figures merge --adapter "$adapter" --out "$merged"; then
      run "vLLM on SFT v2: 5 valid" in_env serve figures vllm-predict --task v2 --model "$merged" --run "$sft" \
        --sample 5 || true
      if [ "$DRY_RUN" != 1 ]; then rm -rf "$merged"; fi
    fi
  fi
  # The difficulty filter on 32 train filings: its time per filing, and how many the base model leaves to learn
  run "filter: 32 train filings" in_env serve figures rl-filter --task v2 --init "$V2_BASE" --name "$filt" \
    --sample 32 || true
  kept=$(ask 16 kept_count "$FIGURES_OUT/rl-filter/$filt") || kept=0
  if [ "$kept" -ge 8 ] 2>/dev/null; then keep=(--keep "$FIGURES_OUT/rl-filter/$filt/filter.jsonl"); fi # else unfiltered
  # GRPO for a few steps: speed, memory, and the vLLM/trainer agreement the gates read
  if run "GRPO: $CHECK_GRPO_STEPS steps ($STACK)" in_env "$(rl_env)" figures train-rl --task v2 --init "$V2_BASE" \
    --name "$grpo" --sample 32 "${keep[@]}" --steps "$CHECK_GRPO_STEPS" --saves 1 --stack "$STACK" "${MEM_ARGS[@]}"; then
    adapter=$(ask "$FIGURES_OUT/train/$grpo/checkpoint-$CHECK_GRPO_STEPS" last_checkpoint "$grpo")
    merged="$FIGURES_OUT/merged/$grpo"
    if run "merge GRPO step $CHECK_GRPO_STEPS" in_env train figures merge --adapter "$adapter" --out "$merged"; then
      run "vLLM on GRPO: 5 ood-valid" in_env serve figures vllm-predict --task v2 --model "$merged" --run "$grpo" \
        --split ood-valid --sample 5 || true
      if [ "$DRY_RUN" != 1 ]; then rm -rf "$merged"; fi
    fi
  fi
  if [ -n "$GPU_PRICE" ]; then price=(--price "$GPU_PRICE"); fi
  run "go/no-go for GRPO, and the runs' cost" in_env serve figures check-report --grpo-steps "$GRPO_STEPS" \
    "${price[@]}" || true
  printf '\nIf every line above is GO:  CHAIN="sft2 grpo" INIT=base GPU_PRICE=<$/h> bash scripts/run_gpu_jobs.sh chain\n'
  printf 'If GRPO is NO-GO:          STACK=unsloth TASK=v2 bash scripts/run_gpu_jobs.sh check\n'
}

sft2_steps() {
  score2 "$V2_BASE" "base-$(short "$V2_BASE")" || true # zero-shot: the floor
  run "train $SFT2_RUN (v2; $EPOCHS epochs, r=$RANK, lr $LR)" in_env train figures train --task v2 \
    --base "$V2_BASE" --name "$SFT2_RUN" --epochs "$EPOCHS" --rank "$RANK" --alpha "$ALPHA" --lr "$LR" \
    --max-seq-length 9216 || return 0
  score_checkpoints "$SFT2_RUN" || return 0
  merged_best "$SFT2_RUN" >/dev/null || true # kept on this pod for grpo INIT=sft
}

grpo_steps() {
  local name init filt kept
  name=$(grpo_name)
  filt="$FIGURES_OUT/rl-filter/$name"
  if [ "$INIT" = sft ]; then
    init=$(merged_best "$SFT2_RUN") || { echo "!! no SFT v2 model to start from"; failures=$((failures + 1)); return 0; }
  else
    init=$V2_BASE
  fi
  echo "GRPO run $name from $init (stack $STACK)"
  run "filter: 8 answers per train filing ($INIT)" in_env serve figures rl-filter --task v2 --init "$init" \
    --name "$name" || return 0
  kept=$(ask 500 kept_count "$filt") || { failures=$((failures + 1)); return 0; }
  echo "the filter kept $kept train filings"
  if [ "$INIT" = sft ] && [ "$kept" -lt "$MIN_KEPT" ]; then
    echo "SFT v2 already solves its training set: the filter kept $kept train filings (fewer than $MIN_KEPT)," \
      "so GRPO from SFT was not run." | tee "$filt/finding.txt"
    return 0
  fi
  run "GRPO $name: $GRPO_STEPS steps ($STACK)" in_env "$(rl_env)" figures train-rl --task v2 --init "$init" \
    --name "$name" --keep "$filt/filter.jsonl" --steps "$GRPO_STEPS" --stack "$STACK" "${MEM_ARGS[@]}" || return 0
  score_checkpoints "$name" || true
}

final2() { # final2 <run> <model>: test and ood-test, once (the run is logged)
  run "TEST: $1 on test + ood-test" in_env serve figures vllm-predict --task v2 --model "$2" --run "$1" \
    --splits test,ood-test --final || true
}

pair2() { # pair2 <a> <b> <split>: b - a on the filings both answered, company-bootstrap interval
  if [ "$DRY_RUN" = 1 ] || [ -f "$FIGURES_OUT/preds/v2/$3/$2.jsonl" ]; then
    run "compare $2 - $1 ($3)" in_env serve figures compare --task v2 --a "$1" --b "$2" --split "$3" || true
  fi
}

test2_steps() {
  if [ "$HANDCHECK_DONE" != 1 ]; then
    echo "!! test2 needs the final v2 answer key: finish the hand-check, run figures export --task v2 and" \
      "figures push --task v2, then HANDCHECK_DONE=1"
    failures=$((failures + 1))
    return 0
  fi
  run "fetch the hand-checked v2 dataset" in_env serve figures fetch-data --task v2 || return 0
  local base_run sft model init name split
  base_run="base-$(short "$V2_BASE")"
  final2 "$base_run" "$V2_BASE"
  if sft=$(merged_best "$SFT2_RUN"); then final2 "$SFT2_RUN" "$sft"; fi # first: SFT->GRPO merges onto it
  for init in base sft; do
    name="grpo-$init-$(short "$V2_BASE")"
    if model=$(merged_best "$name"); then final2 "$name" "$model"; else echo "no $name run: skipped"; fi
  done
  for split in ood-test test; do
    pair2 "$base_run" "$SFT2_RUN" "$split"
    pair2 "$SFT2_RUN" "grpo-base-$(short "$V2_BASE")" "$split" # the headline: GRPO - SFT
    pair2 "$SFT2_RUN" "grpo-sft-$(short "$V2_BASE")" "$split"
  done
}

# ---- the stages ---------------------------------------------------------------------------

export FIGURES_OUT=outputs
run "update code" git pull --ff-only || true
if [ "$TASK" = v2 ]; then
  envs="train serve rl"
  if [ "$STACK" = unsloth ]; then envs="$envs rl-unsloth"; fi
else
  envs="train quant serve"
fi
for extra in $envs; do # also re-creates an environment a pod restart wiped
  # shellcheck disable=SC2046 - dev_flag's output is meant to split
  run "sync $extra environment" env UV_PROJECT_ENVIRONMENT="$(venv_dir "$extra")" \
    uv sync --frozen $(dev_flag "$extra") --extra "$extra" || true
done

case "$stage" in
  check)
    # Separate output folder: these tiny runs must never look like finished work
    export FIGURES_OUT=outputs/check
    if [ "$TASK" = v2 ]; then
      check2_steps
    else
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
    fi
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
  sft2)
    sft2_steps
    final_upload "sft2 $SFT2_RUN" "$FIGURES_OUT/train/$SFT2_RUN"
    ;;
  grpo)
    grpo_steps
    final_upload "grpo $(grpo_name)" "$FIGURES_OUT/train/$(grpo_name)" "$FIGURES_OUT/rl-filter/$(grpo_name)"
    ;;
  chain)
    paths=()
    for s in $CHAIN; do
      if [ "$s" = sft2 ]; then
        sft2_steps
        paths+=("$FIGURES_OUT/train/$SFT2_RUN")
      else
        grpo_steps
        paths+=("$FIGURES_OUT/train/$(grpo_name)" "$FIGURES_OUT/rl-filter/$(grpo_name)")
      fi
      run "upload after $s" upload "chain: $s" "${paths[@]}" || true # a later failure keeps this one's results
    done
    final_upload "chain: $CHAIN" "${paths[@]}"
    ;;
  test2)
    test2_steps
    final_upload "TEST v2" "$FIGURES_OUT/compare"
    ;;
esac
