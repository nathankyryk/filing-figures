# Shared settings for the pod scripts (source it, don't run it).

# SSH sessions may not inherit the container's environment (RUNPOD_POD_ID, RUNPOD_API_KEY);
# Runpod keeps a copy in /etc/rp_environment.
if [ -z "${RUNPOD_POD_ID:-}" ] && [ -f /etc/rp_environment ]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/rp_environment
  set +a
fi

export WORKSPACE="${WORKSPACE:-/workspace}"
export REPO_DIR="$WORKSPACE/filing-figures"
export LOG_DIR="$WORKSPACE/logs"

# The HF token, copied from the laptop (never in git, never printed): see setup_pod.sh
if [ -f "$WORKSPACE/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$WORKSPACE/.env"
  set +a
fi

export UV_INSTALL_DIR="$WORKSPACE/.cache/uv-bin"
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.cache/uv-python"
export HF_HOME="$WORKSPACE/.cache/huggingface"
export TRITON_CACHE_DIR="$WORKSPACE/.cache/triton"
export PATH="$UV_INSTALL_DIR:$PATH"
export HF_HUB_DISABLE_PROGRESS_BARS=1  # keeps the logs readable
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Each GPU extra has its own environment (vLLM and Unsloth pin different torch versions), on
# the container disk: /workspace can be a network filesystem, where importing vLLM took 50 s
# instead of 11 s (measured 2026-11-01, EUR-NO-1). The container disk is wiped when the pod
# stops; run_gpu_jobs.sh re-creates missing environments from the uv cache (~1 min each).
export VENV_ROOT="${VENV_ROOT:-/root/venvs}"
venv_dir() { echo "$VENV_ROOT/$1"; }

#   in_env train figures train ...
in_env() {
  local extra=$1
  shift
  UV_PROJECT_ENVIRONMENT="$(venv_dir "$extra")" uv run --no-sync "$@"
}
