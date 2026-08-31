# DLC 2026 Final Inference

[한국어](README_KO.md) | English

This repository contains the inference code used to produce the DLC 2026 final
submission with the unmodified BF16 `Qwen/Qwen2.5-3B-Instruct` base model. It
samples 16 solutions per problem. Outputs that reach the token limit without a
terminal answer tag are retried at 4,096 tokens and then, if needed, at 8,192
tokens. The final integer is selected by majority vote, with generation order
used to break ties.

## Requirements

- Apple silicon Mac
- macOS 14 or later, as required by [MLX](https://ml-explore.github.io/mlx/build/html/install.html)
- At least 10 GB of free disk space for the environment, model, and outputs
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

The reference submission was generated on macOS 26.5.1 with an M5 Pro, 64 GB
of unified memory, Python 3.12.13, `uv 0.11.28`, and `mlx-lm==0.31.3`. Keep the
batch size at 128 to reproduce the same outputs. The full run has not been
tested on Macs with less memory, and changing the batch size changes the
generated results.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
git clone https://github.com/pysunn14/deep-learning-challenge-2026-final.git
cd deep-learning-challenge-2026-final
uv sync --frozen

mkdir -p data
cp /path/to/test_submission.csv data/test_submission.csv

MODEL_DIR="$(uv run hf download Qwen/Qwen2.5-3B-Instruct \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --quiet)"
```

The model download is about 6.2 GB. The command follows the official
[`hf download` interface](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
and prints the exact snapshot directory required by `--model`.
Repeat the `MODEL_DIR=...` command after opening a new terminal. The cached
snapshot is reused, so the model is not downloaded again.

## Check the input

```bash
uv run dlc-final validate
```

`validate` loads the input and reports its row count. The input may use any
filename and row count; it needs `id` and `question` columns. The `run` command
checks the model revision, artifact SHA-256, BF16 precision, and quantization
status before generation.

## Run

```bash
uv run dlc-final run \
  --model "$MODEL_DIR" \
  --execute
```

Inspect machine-readable progress in another terminal:

```bash
uv run dlc-final status
```

To pause, press `Ctrl-C`. Repeating the identical `run` command resumes after
the last fully persisted batch. Do not change the input file, model snapshot,
configuration, or output directory between runs.

The retry cascade is fixed to `1,024 -> 4,096 -> 8,192`. A longer stage runs
only when the preceding output physically reaches its token limit and has no
terminal `<answer>integer</answer>` outside an open thinking block. Completed
shorter stages are verified by their recorded contracts and SHA-256 values and
are reused without model generation.

## Verify and submit

```bash
uv run dlc-final verify
```

Upload only:

```text
runs/final-base-n16/submission.csv
```

The authoritative submission preserves the input row order in
`id,question,answer` format and contains integer answers. `answers.csv` is a
compact `id,answer` cross-check copy and is not the official upload. `verify`
checks both files, their recorded SHA-256 values, and equality of every
`id,answer` pair.

## Reproducibility note

The frozen method uses temperature `0.6`, `top_p=0.95`, batch size `128`, and
16 fixed seeds. The batch seed is derived from the base seed and the IDs in
that batch. Changing batch size or problem order therefore changes random
seeds, generated solutions, and potentially the majority-vote answers even
when every other option remains identical. Exact reproduction requires the
committed `config/final.json` without modification.

The completed reference run took 9 hours 2 minutes 22 seconds and reached
17.87 GiB peak active Metal memory. The resumed 8,192-token extension processed
210 generations in 1 hour 5 minutes 3 seconds; 42 still-capped generations were
audited and excluded from voting.
