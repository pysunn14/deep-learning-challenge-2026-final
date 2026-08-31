"""Frozen prompt used for every rollout."""

from __future__ import annotations


PROMPT_VERSION = "baseline-v2"
SYSTEM_PROMPT = """You are solving a math competition problem.
Reason carefully and check your arithmetic.
The official final answer is always an integer.
End with the computed integer inside XML tags, for example: <answer>42</answer>.
Replace 42 with your computed answer.
Do not write anything after the closing </answer> tag."""


def build_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
