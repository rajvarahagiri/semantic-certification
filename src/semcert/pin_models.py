from __future__ import annotations

import json
from pathlib import Path
from huggingface_hub import HfApi

from .models import PROMPT_VERSION


def pin_models(full_repo: str, gate_repo: str, out_path: str | Path) -> dict:
    api = HfApi()
    full = api.model_info(full_repo)
    gate = api.model_info(gate_repo)
    lock = {
        "full_model_repo": full_repo,
        "full_model_revision": full.sha,
        "gate_model_repo": gate_repo,
        "gate_model_revision": gate.sha,
        "prompt_version": PROMPT_VERSION,
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock
