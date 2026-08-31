"""Submission writers and validators."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Mapping, Sequence

from dlc_final.contract import Problem, sha256


_INTEGER = re.compile(r"^[+-]?\d+$")


def write_submissions(
    input_path: Path,
    problems: Sequence[Problem],
    answers: Mapping[str, int],
    full_path: Path,
    compact_path: Path,
) -> None:
    missing = [problem.id for problem in problems if problem.id not in answers]
    if missing:
        raise ValueError(f"Cannot write submission: {len(missing)} answers are missing")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    compact_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != len(problems):
        raise ValueError("Input row count changed while writing submission")
    with full_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "question", "answer"])
        writer.writeheader()
        for problem, row in zip(problems, rows, strict=True):
            if row.get("id") != problem.id or row.get("question") != problem.question:
                raise ValueError("Input order or question text changed")
            writer.writerow(
                {
                    "id": problem.id,
                    "question": problem.question,
                    "answer": int(answers[problem.id]),
                }
            )

    with compact_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer"])
        writer.writeheader()
        for problem in problems:
            writer.writerow({"id": problem.id, "answer": int(answers[problem.id])})


def validate_submission(
    path: Path,
    problems: Sequence[Problem],
    *,
    preserve_questions: bool,
) -> dict[str, int]:
    expected_header = ["id", "question", "answer"] if preserve_questions else ["id", "answer"]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise ValueError(f"Invalid submission header: {reader.fieldnames}")
        rows = list(reader)
    if [row["id"] for row in rows] != [problem.id for problem in problems]:
        raise ValueError("Submission IDs, order, or row count changed")
    if preserve_questions and [row["question"] for row in rows] != [
        problem.question for problem in problems
    ]:
        raise ValueError("Submission question text changed")
    invalid = [
        row["id"]
        for row in rows
        if not _INTEGER.fullmatch((row.get("answer") or "").strip())
    ]
    if invalid:
        raise ValueError(f"Submission contains blank or non-integer answers: {invalid[:5]}")
    if len(set(row["id"] for row in rows)) != len(rows):
        raise ValueError("Submission contains duplicate IDs")
    return {"rows": len(rows), "unique_ids": len(rows), "invalid_answers": 0}


def _id_answer_pairs(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            (str(row["id"]), str(row["answer"]).strip())
            for row in csv.DictReader(handle)
        ]


def verify_completed_outputs(
    submission_path: Path,
    answers_path: Path,
    problems: Sequence[Problem],
    *,
    expected_submission_sha256: str,
    expected_answers_sha256: str,
) -> dict[str, object]:
    submission = validate_submission(
        submission_path, problems, preserve_questions=True
    )
    answers = validate_submission(answers_path, problems, preserve_questions=False)
    submission_sha256 = sha256(submission_path)
    answers_sha256 = sha256(answers_path)
    if submission_sha256 != expected_submission_sha256:
        raise ValueError("Submission SHA-256 changed after completion")
    if answers_sha256 != expected_answers_sha256:
        raise ValueError("Answers SHA-256 changed after completion")
    if _id_answer_pairs(submission_path) != _id_answer_pairs(answers_path):
        raise ValueError("Submission and answers id,answer pairs differ")
    return {
        "status": "verified",
        "submission": submission,
        "answers": answers,
        "submission_sha256": submission_sha256,
        "answers_sha256": answers_sha256,
    }
