"""Pinned BF16 MLX batch-generation engine."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlc_final.contract import Problem, sha256
from dlc_final.prompting import build_messages


@dataclass(frozen=True)
class BatchOutput:
    texts: list[str]
    output_tokens: list[int]
    generation_tps: float
    peak_memory_gib: float


def model_fingerprint(model_path: Path) -> tuple[str, str]:
    path = model_path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError("--model must be a local directory pinned to one revision")
    revision = path.name if path.parent.name == "snapshots" else path.name
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix in {".json", ".safetensors", ".model"}
    )
    if not files:
        raise ValueError("Model directory contains no recognized model artifacts")
    for candidate in files:
        digest.update(str(candidate.relative_to(path)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256(candidate)))
    return revision, digest.hexdigest()


def _batch_seed(base_seed: int, problems: list[Problem]) -> int:
    material = "\0".join([str(base_seed), *(problem.id for problem in problems)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


class MLXEngine:
    def __init__(
        self,
        model_path: Path,
        *,
        expected_revision: str,
        expected_artifact_sha256: str,
        completion_batch_size: int,
        prefill_batch_size: int,
    ) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm import batch_generate, load
        from mlx_lm.sample_utils import make_sampler

        self.mx = mx
        self.batch_generate = batch_generate
        self.make_sampler = make_sampler
        self.model_revision, self.model_artifact_sha256 = model_fingerprint(model_path)
        if self.model_revision != expected_revision:
            raise ValueError(
                f"Model revision changed: {self.model_revision} != {expected_revision}"
            )
        if self.model_artifact_sha256 != expected_artifact_sha256:
            raise ValueError("Model artifact SHA-256 changed")
        self.model, self.tokenizer = load(str(model_path.expanduser().resolve()))
        dtypes = Counter(
            str(value.dtype)
            for _, value in tree_flatten(self.model.parameters())
            if hasattr(value, "dtype")
        )
        quantized_modules = sum(
            1
            for _, module in self.model.named_modules()
            if hasattr(module, "bits") and hasattr(module, "group_size")
        )
        if set(dtypes) != {"mlx.core.bfloat16"} or quantized_modules:
            raise RuntimeError(
                f"Final model must be adapter-free BF16: dtypes={dict(dtypes)} "
                f"quantized_modules={quantized_modules}"
            )
        self.precision = "bf16"
        self.completion_batch_size = completion_batch_size
        self.prefill_batch_size = prefill_batch_size

    def _prompt(self, problem: Problem) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            build_messages(problem.question),
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate(
        self,
        problems: list[Problem],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> BatchOutput:
        self.mx.random.seed(_batch_seed(seed, problems))
        prompts = [self._prompt(problem) for problem in problems]
        response = self.batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=max_tokens,
            sampler=self.make_sampler(temp=temperature, top_p=top_p),
            completion_batch_size=min(self.completion_batch_size, len(problems)),
            prefill_batch_size=min(self.prefill_batch_size, len(problems)),
            verbose=False,
        )
        texts = list(response.texts)
        return BatchOutput(
            texts=texts,
            output_tokens=[
                len(self.tokenizer.encode(text, add_special_tokens=False))
                for text in texts
            ],
            generation_tps=float(response.stats.generation_tps),
            peak_memory_gib=float(response.stats.peak_memory),
        )

    def memory(self) -> tuple[float, float]:
        gib = 1024**3
        return self.mx.get_active_memory() / gib, self.mx.get_peak_memory() / gib
