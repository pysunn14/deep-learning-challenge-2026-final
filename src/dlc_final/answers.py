"""Strict integer extraction and conditional retry selection."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedAnswer:
    value: int
    method: str


_INTEGER = r"[+\-−]?\d[\d,]*"
_INTEGER_VALUE = rf"{_INTEGER}(?:\.0+)?"
_TAGGED = re.compile(rf"<answer>\s*({_INTEGER_VALUE})\s*</answer>", re.IGNORECASE)
_BOXED = re.compile(rf"\\boxed\{{\s*({_INTEGER_VALUE})\s*\}}")
_FINAL = re.compile(
    rf"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*({_INTEGER_VALUE})(?!\.\d)",
    re.IGNORECASE,
)
_TAGGED_WHOLE_HOUR = re.compile(
    rf"<answer>\s*({_INTEGER}):00\s*</answer>", re.IGNORECASE
)
_FINAL_WHOLE_HOUR = re.compile(
    rf"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*({_INTEGER}):00(?!\d)",
    re.IGNORECASE,
)
_TERMINAL_CONCLUSION = re.compile(
    rf"(?:therefore|thus|hence|in conclusion)[^\n]*?"
    rf"(?:[,;:]\s*|(?:is|equals|=|for|be)\s+)"
    rf"(?<![\d.])({_INTEGER})(?![\d.]|\\|/|π)[^0-9\n]*[.!]?\s*$",
    re.IGNORECASE,
)
_TERMINAL_ANSWER = re.compile(
    rf"<answer>\s*(?:{_INTEGER_VALUE}|{_INTEGER}:00)\s*</answer>\s*$",
    re.IGNORECASE,
)
_THINK_TAG = re.compile(r"<\s*(/?)\s*think\s*>", re.IGNORECASE)


def _integer(raw: str) -> int:
    return int(raw.replace(",", "").replace("−", "-").split(".", maxsplit=1)[0])


def extract_final_answer(response: str) -> ExtractedAnswer:
    for method, pattern in (
        ("answer_tag", _TAGGED),
        ("boxed", _BOXED),
        ("final_answer", _FINAL),
    ):
        matches = pattern.findall(response)
        if matches:
            return ExtractedAnswer(_integer(matches[-1]), method)
    for method, pattern in (
        ("answer_tag_whole_hour", _TAGGED_WHOLE_HOUR),
        ("final_answer_whole_hour", _FINAL_WHOLE_HOUR),
    ):
        matches = pattern.findall(response)
        if matches:
            return ExtractedAnswer(_integer(matches[-1]), method)
    matches = _TERMINAL_CONCLUSION.findall(response)
    if matches:
        return ExtractedAnswer(_integer(matches[-1]), "terminal_conclusion")
    raise ValueError("No explicit final integer answer found")


def has_terminal_answer_outside_think(response: str) -> bool:
    match = _TERMINAL_ANSWER.search(response)
    if match is None:
        return False
    think_depth = 0
    for tag in _THINK_TAG.finditer(response, 0, match.start()):
        think_depth = max(0, think_depth - 1) if tag.group(1) else think_depth + 1
    return think_depth == 0


def needs_retry(response: str, *, hit_max_tokens: bool) -> bool:
    return hit_max_tokens and not has_terminal_answer_outside_think(response)
