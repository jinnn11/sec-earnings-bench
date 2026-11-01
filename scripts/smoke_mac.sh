#!/usr/bin/env bash
# Smoke test of stage 2 on a laptop, end to end, with no GPU-only package: Qwen3-0.6B
# (configs/train/smoke-qwen3-0.6b.toml) on 5 short examples. It downloads the model,
# prepares a run, trains 2 steps and stops (--stop-after 2), resumes to the last step,
# and scores the base model and the 3 checkpoints on 2 validation releases through
# secbench.hfserve, the laptop stand-in for vLLM. train checks every example against
# the model's chat template, and score checks every prompt's tokens against the
# server's /tokenize. Scores from a 0.6B model after 5 steps mean nothing; the test is
# that every step runs. Run it at a clean commit (prepare records the commit).
#
# Usage: bash scripts/smoke_mac.sh   (RUN=... to name the run; default smoke-<time>)
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/train/smoke-qwen3-0.6b.toml
RUN=${RUN:-smoke-$(date +%Y%m%dT%H%M%S)}

step() { printf '\n== %s %s\n' "$(date +%H:%M:%S)" "$*"; }

step "Install the train group"
uv sync --group train

step "Download the model"
uv run --no-sync python -m secbench.train download --config "$CONFIG"

step "Prepare training/$RUN"
uv run --no-sync python -m secbench.train prepare --config "$CONFIG" --run-name "$RUN"

step "Train 2 steps, then stop"
uv run --no-sync python -m secbench.train train "training/$RUN" --stop-after 2

step "Resume from the checkpoint at step 2"
uv run --no-sync python -m secbench.train train "training/$RUN"

step "Score the base model and each checkpoint on 2 validation releases"
uv run --no-sync python -m secbench.score "training/$RUN" --server hf --limit 2 \
    --concurrency 1 --runs-dir "training/$RUN/runs"

step "Done: training/$RUN/validation.md"
