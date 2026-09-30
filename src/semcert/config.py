from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    path: Path

    @property
    def experiment(self) -> dict[str, Any]:
        return self.raw["experiment"]

    @property
    def wikimedia(self) -> dict[str, Any]:
        return self.raw["wikimedia"]

    @property
    def extraction(self) -> dict[str, Any]:
        return self.raw["extraction"]

    @property
    def models(self) -> dict[str, Any]:
        return self.raw["models"]

    @property
    def probe(self) -> dict[str, Any]:
        return self.raw["probe"]


def load_config(path: str | Path) -> Config:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    required = ["experiment", "wikimedia", "extraction", "models", "probe"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    return Config(raw=raw, path=p)


def ensure_user_agent(cfg: Config) -> None:
    ua = str(cfg.wikimedia.get("user_agent", ""))
    if not ua or "REPLACE_WITH_YOUR_EMAIL" in ua:
        raise ValueError(
            "Set wikimedia.user_agent in config/phase0.yaml to a descriptive User-Agent "
            "with your contact information before calling Wikimedia APIs."
        )
