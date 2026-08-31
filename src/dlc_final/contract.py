"""Frozen method and input contracts for the final test run."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_SCHEMA = "dlc2026-final-base-n16-cascade.v1"
EXPECTED_METHOD = {
    "method": "base_majority_n16",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "model_artifact_sha256": "358b5b70a11ed13a57002d1468336d63bb9019ee3cdd4ca4363c85e8a6f3ab5a",
    "expected_precision": "bf16",
    "prompt_version": "baseline-v2",
    "rollouts": 16,
    "seeds": list(range(20260811, 20260827)),
    "temperature": 0.6,
    "top_p": 0.95,
    "batch_size": 128,
    "queue_size": 128,
    "prefill_batch_size": 4,
    "generation_budgets": [1024, 4096, 8192],
    "fallback_answer": 0,
    "retry_policy": "capped-no-explicit-answer",
    "aggregation": "valid-answer-majority-stable-first-tie",
}


@dataclass(frozen=True)
class Problem:
    id: str
    question: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_config(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"Expected config schema {CONFIG_SCHEMA}")
    allowed = {"schema_version", *EXPECTED_METHOD}
    if set(value) != allowed:
        raise ValueError(
            f"Config keys changed: expected {sorted(allowed)}, got {sorted(value)}"
        )
    for key, expected in EXPECTED_METHOD.items():
        if value.get(key) != expected:
            raise ValueError(
                f"Frozen method value changed for {key}: "
                f"expected {expected!r}, got {value.get(key)!r}"
            )
    return dict(value)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Config must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    return validate_config(value)


def _header_map(fieldnames: list[str] | None) -> dict[str, str]:
    if fieldnames is None:
        raise ValueError("CSV header is missing")
    normalized = [name.strip().casefold() for name in fieldnames]
    if len(normalized) != len(set(normalized)):
        raise ValueError("CSV columns collide after normalization")
    return dict(zip(normalized, fieldnames, strict=True))


def load_input(path: Path) -> list[Problem]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = _header_map(reader.fieldnames)
        if not {"id", "question"}.issubset(headers):
            raise ValueError("Input must contain id and question columns")
        return [
            Problem(
                str(row.get(headers["id"]) or ""),
                str(row.get(headers["question"]) or ""),
            )
            for row in reader
        ]
