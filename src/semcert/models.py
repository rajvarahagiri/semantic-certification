from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


PROMPT_VERSION = "lead-reference-need-v1"
SYSTEM_PROMPT = (
    "You are a deterministic binary classifier for a research experiment. "
    "Classify only the supplied Wikipedia lead text. Output exactly 0 or 1 and nothing else."
)
USER_PROMPT = """Task: lead_reference_need

Return 1 if at least one material factual claim in this lead appears to require an additional supporting citation based on the citation markers present in the supplied lead text. Return 0 otherwise.

Interpret [REF] as evidence that a nearby claim has a citation. Interpret [CITATION_NEEDED] as strong evidence for 1.
Do not assume that citations in the unseen article body support uncited lead claims; classify only from the supplied lead.

LEAD:
{lead}

Output exactly one character: 0 or 1."""


class Labeler(Protocol):
    def label(self, lead: str) -> int: ...


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...


@dataclass
class ModelLock:
    full_model_repo: str
    full_model_revision: str
    gate_model_repo: str
    gate_model_revision: str
    prompt_version: str = PROMPT_VERSION

    @classmethod
    def load(cls, path: str | Path) -> "ModelLock":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class QwenBinaryLabeler:
    def __init__(self, repo_id: str, revision: str, device: str = "auto", dtype: str = "auto", max_new_tokens: int = 4):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError("Install model dependencies: pip install -e '.[models]'") from e
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            revision=revision,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def label(self, lead: str) -> int:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(lead=lead)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output[0][inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        m = re.search(r"[01]", raw)
        if not m:
            raise ValueError(f"Unexpected classifier output: {raw!r}")
        return int(m.group(0))


class BGEEmbedder:
    def __init__(self, repo_id: str, revision: str, device: str = "auto"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("Install model dependencies: pip install -e '.[models]'") from e
        kwargs = {"revision": revision}
        if device != "auto":
            kwargs["device"] = device
        self.model = SentenceTransformer(repo_id, **kwargs)

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vec, dtype=np.float32)


class MockLabeler:
    """Deterministic test-only classifier; never use for Phase 0 results."""

    def label(self, lead: str) -> int:
        x = lead.lower()
        return int("[citation_needed]" in x or ("largest" in x and "[ref]" not in x))


class MockEmbedder:
    def embed(self, text: str) -> np.ndarray:
        # Tiny deterministic character histogram for tests only.
        v = np.zeros(32, dtype=np.float32)
        for b in text.encode("utf-8"):
            v[b % 32] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v
