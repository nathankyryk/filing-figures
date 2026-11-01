#!/usr/bin/env bash
# Set up a fresh Runpod pod for FilingFigures. From the laptop (Git Bash, in the repo folder):
#
#   1. Send the Hugging Face token (only that line of .env; it is never printed):
#        grep '^HF_TOKEN=' .env | ssh -p <PORT> root@<IP> 'umask 077; mkdir -p /workspace; cat > /workspace/.env'
#   2. Set up:
#        ssh -p <PORT> root@<IP> 'curl -fsSL https://raw.githubusercontent.com/nathankyryk/filing-figures/master/scripts/setup_pod.sh | bash'
#
# Safe to re-run: each step checks what is already there. ENVS="serve" sets up fewer environments
# (default: all three). Ends with `figures doctor`; a non-zero exit means something needs attention.

set -euo pipefail

REPO_URL="https://github.com/nathankyryk/filing-figures.git"
UV_VERSION="0.13.1"  # same uv that wrote uv.lock
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$WORKSPACE/filing-figures"
ENVS="${ENVS:-train quant serve}"

mkdir -p "$WORKSPACE/logs"
LOG="$WORKSPACE/logs/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n== %s ==\n' "$*"; }
fail() { printf '!! %s\n' "$*" >&2; exit 1; }

step "GPU and driver"
command -v nvidia-smi >/dev/null || fail "no nvidia-smi: this is not a GPU pod"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n1)
echo "driver supports CUDA $cuda"
# torch 2.13/2.14 wheels from PyPI are built for CUDA 13.0
awk -v v="$cuda" 'BEGIN { exit !(v + 0 >= 13.0) }' ||
  fail "PyTorch here needs CUDA 13.0+: redeploy the pod with the CUDA 13.0 filter"

step "System tools"
missing=()
for tool in git curl tmux; do command -v "$tool" >/dev/null || missing+=("$tool"); done
if ((${#missing[@]})); then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
fi
echo "git, curl and tmux present"

step "Repository"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" log --oneline -1
# shellcheck source=scripts/pod_env.sh
source "$REPO_DIR/scripts/pod_env.sh"
cd "$REPO_DIR"

step "Hugging Face token"
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN missing: send it from the laptop first (step 1 at the top of this script)"
echo "HF_TOKEN is set (${#HF_TOKEN} characters)"

step "uv $UV_VERSION"
if ! command -v uv >/dev/null || [ "$(uv --version | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

for extra in $ENVS; do
  step "Environment: $extra (exactly as uv.lock)"
  UV_PROJECT_ENVIRONMENT="$(venv_dir "$extra")" uv sync --frozen --no-dev --extra "$extra"
  in_env "$extra" python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)}")
EOF
done

step "Dataset (from Hugging Face)"
first=${ENVS%% *}
in_env "$first" figures fetch-data
df -h "$WORKSPACE" | tail -n 1

step "figures doctor"
if in_env "$first" figures doctor; then
  printf '\nReady. Next: bash scripts/run_gpu_jobs.sh check   (log: %s)\n' "$LOG"
else
  printf '\nNot ready: fix the !! lines above, then re-run this script. (log: %s)\n' "$LOG"
  exit 1
fi
