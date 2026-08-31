"""Command-line interface for the frozen final inference method."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dlc_final.contract import canonical_sha256, load_config, load_input
from dlc_final.engine import MLXEngine
from dlc_final.output import verify_completed_outputs
from dlc_final.pipeline import FinalPipeline


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config/final.json"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/test_submission.csv")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/final-base-n16")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    _common(validate)

    run = subparsers.add_parser("run")
    _common(run)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--execute", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/final-base-n16")

    verify = subparsers.add_parser("verify")
    _common(verify)
    return parser.parse_args(argv)


def _validate(args: argparse.Namespace) -> tuple[dict[str, Any], list[Any]]:
    config = load_config(args.config.resolve())
    problems = load_input(args.input.resolve())
    return config, problems


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "status":
        path = args.output_dir.resolve() / "state.json"
        if not path.is_file():
            raise ValueError(f"Run state does not exist: {path}")
        print(path.read_text(encoding="utf-8"), end="")
        return

    config, problems = _validate(args)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "status": "ready",
                    "method": config["method"],
                    "rows": len(problems),
                    "rollouts": config["rollouts"],
                    "batch_size": config["batch_size"],
                    "generation_budgets": config["generation_budgets"],
                },
                indent=2,
            )
        )
        return

    output_dir = args.output_dir.resolve()
    if args.command == "verify":
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file():
            raise ValueError("Completed summary is missing")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise ValueError("Run is not completed")
        if summary.get("config_sha256") != canonical_sha256(config):
            raise ValueError("Completed summary does not match the current config")
        full = output_dir / str(summary["submission"])
        compact = output_dir / str(summary["answers"])
        result = verify_completed_outputs(
            full,
            compact,
            problems,
            expected_submission_sha256=str(summary["submission_sha256"]),
            expected_answers_sha256=str(summary["answers_sha256"]),
        )
        print(json.dumps(result, indent=2))
        return

    if not args.execute:
        raise ValueError("Full inference requires explicit --execute")
    engine = MLXEngine(
        args.model,
        expected_revision=str(config["model_revision"]),
        expected_artifact_sha256=str(config["model_artifact_sha256"]),
        completion_batch_size=int(config["batch_size"]),
        prefill_batch_size=int(config["prefill_batch_size"]),
    )
    result = FinalPipeline(
        config=config,
        input_path=args.input,
        problems=problems,
        output_dir=output_dir,
        engine=engine,
    ).run()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
