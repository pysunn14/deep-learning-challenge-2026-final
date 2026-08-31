"""Atomic state, event, and heartbeat storage for long-running inference."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.events_path = path.with_name("events.jsonl")
        self._lock = threading.RLock()

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def read(self) -> dict[str, Any]:
        with self._lock:
            return self._read_unlocked()

    def _write_unlocked(self, value: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    def event(self, event: str, **fields: Any) -> None:
        record = {"at": _now(), "event": event, **fields}
        with self._lock:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def initialize(self, *, run_id: str, total_rollouts: int) -> dict[str, Any]:
        with self._lock:
            existing = self._read_unlocked()
            if existing:
                if existing.get("run_id") != run_id or existing.get("total_rollouts") != total_rollouts:
                    raise ValueError("Existing run state belongs to another contract")
                return existing
            state = {
                "schema_version": "dlc2026-final-run-state.v1",
                "run_id": run_id,
                "status": "ready",
                "stage": "ready",
                "started_at": _now(),
                "updated_at": _now(),
                "heartbeat_at": _now(),
                "total_rollouts": total_rollouts,
                "completed_rollouts": 0,
                "current_rollout": 0,
                "stage_completed": 0,
                "stage_total": 0,
                "overall_progress_percent": 0.0,
                "generation_tps": 0.0,
                "active_memory_gib": 0.0,
                "peak_memory_gib": 0.0,
                "failure": None,
            }
            self._write_unlocked(state)
        self.event("run_initialized", run_id=run_id, total_rollouts=total_rollouts)
        return state

    @staticmethod
    def _stage_fraction(stage: str, completed: int, total: int) -> float:
        fraction = min(1.0, max(0.0, completed / total)) if total else 0.0
        if stage == "primary1024":
            return 0.70 * fraction
        if stage == "retry4096":
            return 0.70 + 0.20 * fraction
        if stage == "retry8192":
            return 0.90 + 0.09 * fraction
        if stage in {"vote", "finalize"}:
            # The rollout has already moved into completed_rollouts. Counting a
            # stage fraction here would make progress jump forward and regress
            # again when the next rollout starts.
            return 0.0
        return 0.0

    def update(self, **fields: Any) -> dict[str, Any]:
        with self._lock:
            state = self._read_unlocked()
            if not state:
                raise ValueError("Run state is not initialized")
            previous_peak = float(state.get("peak_memory_gib") or 0.0)
            state.update(fields)
            state["updated_at"] = _now()
            state["heartbeat_at"] = _now()
            state["peak_memory_gib"] = max(
                previous_peak,
                float(fields.get("peak_memory_gib") or 0.0),
            )
            total_rollouts = int(state["total_rollouts"])
            completed_rollouts = int(state.get("completed_rollouts") or 0)
            current_fraction = self._stage_fraction(
                str(state.get("stage") or ""),
                int(state.get("stage_completed") or 0),
                int(state.get("stage_total") or 0),
            )
            state["overall_progress_percent"] = min(
                100.0,
                100.0 * (completed_rollouts + current_fraction) / total_rollouts,
            )
            if state.get("status") == "completed":
                state["overall_progress_percent"] = 100.0
                state["completed_at"] = fields.get("completed_at", _now())
            self._write_unlocked(state)
            return state

    def heartbeat(self, *, active_memory_gib: float, peak_memory_gib: float) -> None:
        self.update(
            active_memory_gib=active_memory_gib,
            peak_memory_gib=peak_memory_gib,
        )
