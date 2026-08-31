"""Conditional retry merge and stable majority voting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Vote:
    answer: int
    counts: dict[int, int]
    tied: bool
    valid_rollouts: int


def majority_vote(records: list[dict[str, Any]], fallback: int) -> Vote:
    valid = [row for row in records if row.get("status") == "ok"]
    if not valid:
        # The submission contract requires one integer for every row. Zero is
        # used only when all 16 generations lack an explicit final integer; it
        # never competes with or outweighs a parsed model answer.
        return Vote(fallback, {}, True, 0)
    counts = Counter(int(row["answer"]) for row in valid)
    best_count = max(counts.values())
    tied_answers = {answer for answer, count in counts.items() if count == best_count}
    answer = next(
        int(row["answer"])
        for row in valid
        if int(row["answer"]) in tied_answers
    )
    return Vote(answer, dict(counts), len(tied_answers) > 1, len(valid))


def merge_retry_records(
    primary: dict[str, dict[str, Any]],
    retry: dict[str, dict[str, Any]],
    *,
    retry_budget: int,
) -> dict[str, dict[str, Any]]:
    if not set(retry) <= set(primary):
        raise ValueError("Retry contains IDs outside the primary rollout")
    merged = {problem_id: dict(row) for problem_id, row in primary.items()}
    for problem_id, row in retry.items():
        merged[problem_id] = {
            **row,
            "effective_retry_budget": retry_budget,
        }
    return merged
