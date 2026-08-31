"""Resumable Base N16 rollout, cascading retry, vote, and submission pipeline."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from dlc_final.answers import extract_final_answer, needs_retry
from dlc_final.contract import Problem, canonical_sha256, sha256
from dlc_final.engine import BatchOutput
from dlc_final.output import validate_submission, write_submissions
from dlc_final.state import StateStore
from dlc_final.voting import majority_vote, merge_retry_records


class Engine(Protocol):
    model_revision: str
    model_artifact_sha256: str
    precision: str

    def generate(
        self,
        problems: list[Problem],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> BatchOutput: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = str(row["id"])
            if problem_id in records:
                raise ValueError(f"Duplicate prediction ID at line {line_number}: {problem_id}")
            records[problem_id] = row
    return records


class _Heartbeat:
    def __init__(self, store: StateStore, engine: Engine, interval: float = 10.0) -> None:
        self.store = store
        self.engine = engine
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="final-heartbeat", daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(self.interval):
            memory = getattr(self.engine, "memory", None)
            active, peak = memory() if callable(memory) else (0.0, 0.0)
            self.store.heartbeat(active_memory_gib=active, peak_memory_gib=peak)

    def __enter__(self) -> "_Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=self.interval + 1.0)


class FinalPipeline:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        input_path: Path,
        problems: list[Problem],
        output_dir: Path,
        engine: Engine,
    ) -> None:
        self.config = config
        self.input_path = input_path.resolve()
        self.problems = problems
        self.output_dir = output_dir.resolve()
        self.engine = engine
        self.store = StateStore(self.output_dir / "state.json")
        self.run_id = self.output_dir.name

    @property
    def generation_budgets(self) -> list[int]:
        budgets = [int(value) for value in self.config["generation_budgets"]]
        if not budgets or any(value <= 0 for value in budgets):
            raise ValueError("Generation budgets must be positive integers")
        if budgets != sorted(set(budgets)):
            raise ValueError("Generation budgets must be strictly increasing")
        return budgets

    @staticmethod
    def _stage_name(index: int, budget: int) -> str:
        return f"primary{budget}" if index == 0 else f"retry{budget}"

    def _stage_contract(
        self,
        *,
        stage: str,
        seed: int,
        max_tokens: int,
        problems: list[Problem],
    ) -> dict[str, Any]:
        return {
            "schema_version": "dlc2026-final-stage.v1",
            "stage": stage,
            "seed": seed,
            "max_tokens": max_tokens,
            "temperature": float(self.config["temperature"]),
            "top_p": float(self.config["top_p"]),
            "queue_size": int(self.config["queue_size"]),
            "ids_sha256": canonical_sha256([problem.id for problem in problems]),
            "input_sha256": sha256(self.input_path),
            "model_revision": self.engine.model_revision,
            "model_artifact_sha256": self.engine.model_artifact_sha256,
            "precision": self.engine.precision,
        }

    def _run_stage(
        self,
        *,
        rollout_number: int,
        seed: int,
        stage: str,
        max_tokens: int,
        problems: list[Problem],
        directory: Path,
    ) -> dict[str, dict[str, Any]]:
        predictions = directory / f"{stage}.jsonl"
        metadata_path = directory / f"{stage}.run.json"
        contract = self._stage_contract(
            stage=stage,
            seed=seed,
            max_tokens=max_tokens,
            problems=problems,
        )
        saved: dict[str, Any] | None = None
        if metadata_path.exists():
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            mismatches = {
                key: {"saved": saved.get(key), "current": value}
                for key, value in contract.items()
                if saved.get(key) != value
            }
            if mismatches:
                raise ValueError(f"Refusing incompatible stage resume: {mismatches}")
        else:
            _atomic_json(metadata_path, {**contract, "status": "running", "completed": 0})

        completed = _load_records(predictions)
        ordered_ids = [problem.id for problem in problems]
        if list(completed) != ordered_ids[: len(completed)]:
            raise ValueError("Safe resume requires an ordered input prefix")
        if saved is not None and saved.get("status") == "completed":
            if len(completed) != len(problems):
                raise ValueError("Completed stage metadata has incomplete predictions")
            recorded_sha = saved.get("predictions_sha256")
            if recorded_sha != sha256(predictions):
                raise ValueError("Completed stage predictions SHA-256 changed")
            return completed
        queue_size = int(self.config["queue_size"])
        if len(completed) < len(problems) and len(completed) % queue_size:
            raise ValueError("Safe resume refuses an incomplete persisted batch")

        pending = problems[len(completed) :]
        for start in range(0, len(pending), queue_size):
            batch = pending[start : start + queue_size]
            batch_started = time.perf_counter()
            generated = self.engine.generate(
                batch,
                max_tokens=max_tokens,
                temperature=float(self.config["temperature"]),
                top_p=float(self.config["top_p"]),
                seed=seed,
            )
            if len(generated.texts) != len(batch) or len(generated.output_tokens) != len(batch):
                raise RuntimeError("MLX generation returned a different batch size")
            new_rows: list[dict[str, Any]] = []
            for problem, response, output_tokens in zip(
                batch, generated.texts, generated.output_tokens, strict=True
            ):
                try:
                    answer = extract_final_answer(response)
                    value, method, status = answer.value, answer.method, "ok"
                except ValueError:
                    value, method, status = (
                        int(self.config["fallback_answer"]),
                        "configured_fallback",
                        "fallback",
                    )
                new_rows.append(
                    {
                        "id": problem.id,
                        "answer": value,
                        "status": status,
                        "extraction_method": method,
                        "output_tokens": int(output_tokens),
                        "hit_max_tokens": int(output_tokens) >= max_tokens,
                        "response": response,
                    }
                )
            _append_jsonl(predictions, new_rows)
            completed.update({row["id"]: row for row in new_rows})
            active_memory, peak_memory = (
                self.engine.memory()
                if callable(getattr(self.engine, "memory", None))
                else (0.0, generated.peak_memory_gib)
            )
            self.store.update(
                status="running",
                stage=stage,
                completed_rollouts=rollout_number - 1,
                current_rollout=rollout_number,
                stage_completed=len(completed),
                stage_total=len(problems),
                generation_tps=generated.generation_tps,
                active_memory_gib=active_memory,
                peak_memory_gib=max(peak_memory, generated.peak_memory_gib),
            )
            _atomic_json(
                metadata_path,
                {
                    **contract,
                    "status": "running",
                    "completed": len(completed),
                    "last_batch_seconds": time.perf_counter() - batch_started,
                    "generation_tps": generated.generation_tps,
                    "peak_memory_gib": generated.peak_memory_gib,
                },
            )
            print(
                f"[progress] rollout={rollout_number}/{len(self.config['seeds'])} "
                f"stage={stage} rows={len(completed)}/{len(problems)} "
                f"batch_seconds={time.perf_counter() - batch_started:.1f} "
                f"generation_tps={generated.generation_tps:.1f} "
                f"peak_memory_gib={generated.peak_memory_gib:.2f}",
                flush=True,
            )
        _atomic_json(
            metadata_path,
            {
                **contract,
                "status": "completed",
                "completed": len(completed),
                "completed_at": _now(),
                "predictions_sha256": sha256(predictions),
            },
        )
        return completed

    def _vote(
        self, rollouts: list[dict[str, dict[str, Any]]]
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        answers: dict[str, int] = {}
        cases: list[dict[str, Any]] = []
        for problem in self.problems:
            vote = majority_vote(
                [rollout[problem.id] for rollout in rollouts],
                fallback=int(self.config["fallback_answer"]),
            )
            answers[problem.id] = vote.answer
            cases.append(
                {
                    "id": problem.id,
                    "answer": vote.answer,
                    "counts": vote.counts,
                    "tied": vote.tied,
                    "valid_rollouts": vote.valid_rollouts,
                }
            )
        return answers, cases

    def _exclude_exhausted(
        self, records: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], int]:
        audited = {problem_id: dict(row) for problem_id, row in records.items()}
        exhausted = 0
        for row in audited.values():
            if not needs_retry(
                str(row.get("response", "")),
                hit_max_tokens=bool(row.get("hit_max_tokens")),
            ):
                continue
            exhausted += 1
            parsed_answer = row.get("answer") if row.get("status") == "ok" else None
            parsed_method = (
                row.get("extraction_method") if row.get("status") == "ok" else None
            )
            # A physically capped response without a terminal answer tag is
            # incomplete by the same contract that selected it for retry.
            # Preserve any heuristic parse for audit, but never let it vote.
            row.update(
                {
                    "answer": int(self.config["fallback_answer"]),
                    "status": "exhausted",
                    "extraction_method": "configured_fallback_after_max_budget",
                    "parsed_answer_before_exhaustion": parsed_answer,
                    "parsed_method_before_exhaustion": parsed_method,
                }
            )
        return audited, exhausted

    def run(self) -> dict[str, Any]:
        final_summary = self.output_dir / "summary.json"
        config_sha256 = canonical_sha256(self.config)
        input_sha256 = sha256(self.input_path)
        previous_elapsed_seconds = 0.0
        if final_summary.is_file():
            summary = json.loads(final_summary.read_text(encoding="utf-8"))
            if (
                summary.get("status") == "completed"
                and summary.get("config_sha256") == config_sha256
            ):
                if summary.get("input_sha256") != input_sha256:
                    raise ValueError(
                        "Completed result input does not match the current input"
                    )
                return summary
            if summary.get("status") == "completed":
                previous_elapsed_seconds = float(summary.get("elapsed_seconds") or 0.0)

        seeds = [int(seed) for seed in self.config["seeds"]]
        budgets = self.generation_budgets
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.store.initialize(run_id=self.run_id, total_rollouts=len(seeds))
        self.store.update(
            status="running",
            stage="ready",
            completed_rollouts=0,
            current_rollout=0,
            stage_completed=0,
            stage_total=0,
            failure=None,
        )
        self.store.event(
            "run_started",
            method="base_majority_n16",
            generation_budgets=budgets,
            config_sha256=config_sha256,
        )
        rollout_records: list[dict[str, dict[str, Any]]] = []
        retry_totals = {str(budget): 0 for budget in budgets[1:]}
        exhausted_total = 0
        started = time.perf_counter()
        try:
            with _Heartbeat(self.store, self.engine):
                for index, seed in enumerate(seeds):
                    number = index + 1
                    directory = self.output_dir / f"rollout-{index:02d}-seed{seed}"
                    effective = self._run_stage(
                        rollout_number=number,
                        seed=seed,
                        stage=self._stage_name(0, budgets[0]),
                        max_tokens=budgets[0],
                        problems=self.problems,
                        directory=directory,
                    )
                    retry_candidates_by_budget: dict[str, int] = {}
                    for stage_index, budget in enumerate(budgets[1:], start=1):
                        retry_ids = {
                            problem_id
                            for problem_id, row in effective.items()
                            if needs_retry(
                                str(row.get("response", "")),
                                hit_max_tokens=bool(row.get("hit_max_tokens")),
                            )
                        }
                        retry_candidates_by_budget[str(budget)] = len(retry_ids)
                        retry_totals[str(budget)] += len(retry_ids)
                        if not retry_ids:
                            continue
                        retry_problems = [
                            problem for problem in self.problems if problem.id in retry_ids
                        ]
                        retry = self._run_stage(
                            rollout_number=number,
                            seed=seed,
                            stage=self._stage_name(stage_index, budget),
                            max_tokens=budget,
                            problems=retry_problems,
                            directory=directory,
                        )
                        if set(retry) != retry_ids:
                            raise ValueError("Retry IDs differ from the selected candidates")
                        effective = merge_retry_records(
                            effective,
                            retry,
                            retry_budget=budget,
                        )
                    effective, exhausted = self._exclude_exhausted(effective)
                    exhausted_total += exhausted
                    if list(effective) != [problem.id for problem in self.problems]:
                        raise ValueError("Effective rollout IDs or order changed")
                    effective_path = directory / "effective.jsonl"
                    _atomic_jsonl(effective_path, list(effective.values()))
                    _atomic_json(
                        directory / "result.json",
                        {
                            "status": "completed",
                            "rollout": number,
                            "seed": seed,
                            "rows": len(effective),
                            "generation_budgets": budgets,
                            "retry_candidates_by_budget": retry_candidates_by_budget,
                            "exhausted_after_max_budget": exhausted,
                            "effective_sha256": sha256(effective_path),
                        },
                    )
                    rollout_records.append(effective)
                    answers, cases = self._vote(rollout_records)
                    full = self.output_dir / f"majority-n{number:02d}.submission.csv"
                    compact = self.output_dir / f"majority-n{number:02d}.answers.csv"
                    write_submissions(
                        self.input_path,
                        self.problems,
                        answers,
                        full,
                        compact,
                    )
                    validate_submission(full, self.problems, preserve_questions=True)
                    validate_submission(compact, self.problems, preserve_questions=False)
                    _atomic_json(
                        self.output_dir / "progressive.json",
                        {
                            "status": "running",
                            "n": number,
                            "submission": full.name,
                            "submission_sha256": sha256(full),
                            "ties": sum(case["tied"] for case in cases),
                            "all_invalid": sum(
                                case["valid_rollouts"] == 0 for case in cases
                            ),
                        },
                    )
                    self.store.update(
                        status="running",
                        stage="vote",
                        completed_rollouts=number,
                        current_rollout=number,
                        stage_completed=len(self.problems),
                        stage_total=len(self.problems),
                    )
                    self.store.event(
                        "rollout_completed",
                        rollout=number,
                        seed=seed,
                        retry_candidates_by_budget=retry_candidates_by_budget,
                        exhausted_after_max_budget=exhausted,
                    )

            answers, cases = self._vote(rollout_records)
            submission = self.output_dir / "submission.csv"
            compact = self.output_dir / "answers.csv"
            write_submissions(
                self.input_path,
                self.problems,
                answers,
                submission,
                compact,
            )
            validation = validate_submission(
                submission, self.problems, preserve_questions=True
            )
            compact_validation = validate_submission(
                compact, self.problems, preserve_questions=False
            )
            summary = {
                "schema_version": "dlc2026-final-result.v2",
                "status": "completed",
                "method": "base_majority_n16",
                "config_sha256": config_sha256,
                "generation_budgets": budgets,
                "retry_candidates_by_budget": retry_totals,
                "exhausted_after_max_budget": exhausted_total,
                "rollouts": len(seeds),
                "seeds": seeds,
                "input_sha256": input_sha256,
                "model_revision": self.engine.model_revision,
                "model_artifact_sha256": self.engine.model_artifact_sha256,
                "precision": self.engine.precision,
                "elapsed_seconds": previous_elapsed_seconds + time.perf_counter() - started,
                "latest_execution_seconds": time.perf_counter() - started,
                "submission": submission.name,
                "submission_sha256": sha256(submission),
                "answers": compact.name,
                "answers_sha256": sha256(compact),
                "validation": validation,
                "compact_validation": compact_validation,
                "ties": sum(case["tied"] for case in cases),
                "all_invalid": sum(case["valid_rollouts"] == 0 for case in cases),
                "cases": cases,
                "completed_at": _now(),
            }
            _atomic_json(final_summary, summary)
            self.store.update(
                status="completed",
                stage="finalize",
                completed_rollouts=len(seeds),
                current_rollout=len(seeds),
                stage_completed=len(self.problems),
                stage_total=len(self.problems),
            )
            self.store.event(
                "run_completed",
                submission_sha256=summary["submission_sha256"],
            )
            return summary
        except BaseException as exc:
            self.store.update(
                status="failed",
                failure={"type": type(exc).__name__, "message": str(exc)},
            )
            self.store.event(
                "run_failed", failure_type=type(exc).__name__, message=str(exc)
            )
            raise
