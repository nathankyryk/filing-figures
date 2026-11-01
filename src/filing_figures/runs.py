"""Pod results on Hugging Face. There is no network volume: each stage uploads its outputs to a
private model repo before the pod stops itself, and the next pod (or the laptop) downloads them.
"""

from __future__ import annotations

import os
from pathlib import Path

RUNS_REPO = "kyryk/filing-figures-runs"
DATASET_REPO = "kyryk/filing-figures"


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def upload(paths: list[Path], message: str, repo: str = RUNS_REPO) -> list[str]:
    """Each path keeps its place in the repo (outputs/scores/... stays outputs/scores/...)."""
    api = _api()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    done = []
    for p in paths:
        if not p.exists():
            continue
        rel = p.as_posix()
        if p.is_dir():
            api.upload_folder(folder_path=str(p), path_in_repo=rel, repo_id=repo, commit_message=message)
        else:
            api.upload_file(path_or_fileobj=str(p), path_in_repo=rel, repo_id=repo, commit_message=message)
        done.append(rel)
    return done


def download(patterns: list[str], repo: str = RUNS_REPO, local_dir: Path = Path(".")) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, allow_patterns=patterns, local_dir=str(local_dir), token=os.environ.get("HF_TOKEN"))


def fetch_dataset(out: Path = Path("data/export")) -> list[Path]:
    """The exported splits, as last pushed by `figures push`."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        allow_patterns=["*.jsonl"],
        local_dir=str(out),
        token=os.environ.get("HF_TOKEN"),
    )
    return sorted(out.glob("*.jsonl"))
