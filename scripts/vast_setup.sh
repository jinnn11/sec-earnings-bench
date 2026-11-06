#!/usr/bin/env bash
# Stage 2 on a fresh Vast.ai GPU instance, in one of two modes.
#
# Training (bash vast_setup.sh BUNDLE [CONFIG]), in order: check the GPU, unpack the
# bundle, install, download the model, prepare, score the base model on validation,
# train, score the checkpoints on validation, pack the results. The base model is
# scored before training, on the server setup the checkpoints get (vLLM with LoRA
# on), so a bad flag, a broken install or a prompt vLLM tokenizes differently from
# training (the /tokenize check) stops the script about 30 minutes in, before any
# training. Running it again continues where it stopped: the install and download
# are no-ops once done, a prepared run is kept, training resumes from its last
# checkpoint, and models already scored are kept (a rerun during training also
# scores the checkpoints saved so far before it resumes).
#
# A test split (bash vast_setup.sh BUNDLE --split SPLIT --gpu-hourly PRICE), from a
# bundle made with secbench.bundle --test-config ... --split SPLIT: check the GPU,
# unpack (the training run's config and checkpoint too), install, download the base
# model, then secbench.score's test mode: the base model's and the test config's
# checkpoint's predictions on the frozen split, not graded (the laptop grades). No
# prepare, no training. Whatever score does, the runs and training/<run>/test/ are
# then packed into results-<run>-<split>.tar, and the script exits with score's
# status. PRICE is the instance's price in dollars per hour, recorded as the runs'
# cost basis. A run that failed is never overwritten: score says to move it to
# <name>--attempt-N first (results/test_protocol.md).
#
# Pick an instance with one A100 80GB or H100, a driver for CUDA 13.0 or later
# (torch 2.13.0 and vLLM 0.32.0 are CUDA 13 builds) and 100 GB of disk.
#
# On the laptop, at a clean commit:
#   uv run python -m secbench.bundle                        # gpu-bundle-<commit>.tar
#   (or: uv run python -m secbench.bundle --test-config configs/test/qwen3-8b-2k.toml
#        --split company_test                               # gpu-bundle-<commit>-<split>.tar)
#   scp -P <port> gpu-bundle-<commit>.tar scripts/vast_setup.sh root@<host>:/workspace/
# On the instance:
#   cd /workspace && nohup bash vast_setup.sh gpu-bundle-<commit>.tar > setup.log 2>&1 &
#   (or: nohup bash vast_setup.sh gpu-bundle-<commit>-<split>.tar --split <split>
#        --gpu-hourly <price> > setup.log 2>&1 &)
#   tail -f setup.log
# Back on the laptop, when it has finished:
#   scp -P <port> root@<host>:/workspace/results-<run>.tar .   (or results-<run>-<split>.tar)
#
# Usage: bash vast_setup.sh BUNDLE [CONFIG]   (CONFIG: configs/train/qwen3-8b-2k.toml)
#        bash vast_setup.sh BUNDLE --split SPLIT --gpu-hourly PRICE
# In training mode the run is named after CONFIG's file (RUN=... to choose another
# name); in test mode the run, the test config and the split come from the bundle.
set -euo pipefail

step() { printf '\n== %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
fail() { echo "error: $*" >&2; exit 1; }

BUNDLE=$(realpath "${1:?usage: bash vast_setup.sh BUNDLE [CONFIG] | BUNDLE --split SPLIT --gpu-hourly PRICE}")
shift
WORK=$(dirname "$BUNDLE")
REPO="$WORK/sec-earnings-bench"
UV_VERSION=0.13.0
MIN_DISK_GB=80
# BUNDLE.json's value for a key ("commit", or the test entry's "run").
info=$(tar -xOf "$BUNDLE" BUNDLE.json)
field() { printf '%s\n' "$info" | sed -n "s/.*\"$1\": \"\([^\"]*\)\".*/\1/p" | head -1; }

case "${1:-}" in
    --split | --gpu-hourly)
        MODE=test
        SPLIT= PRICE=
        while [ $# -gt 0 ]; do
            case "$1" in
                --split) SPLIT=${2:?--split needs a value}; shift 2 ;;
                --gpu-hourly) PRICE=${2:?--gpu-hourly needs a value}; shift 2 ;;
                *) fail "unknown argument $1" ;;
            esac
        done
        [ -n "$SPLIT" ] && [ -n "$PRICE" ] || fail "a test split needs --split and --gpu-hourly"
        [ "$(field split)" = "$SPLIT" ] ||
            fail "$BUNDLE is for split '$(field split)', not $SPLIT (secbench.bundle --test-config ... --split $SPLIT)"
        RUN=$(field run)
        TEST_CONFIG=$(field config)
        ;;
    *)
        MODE=train
        CONFIG=${1:-configs/train/qwen3-8b-2k.toml}
        RUN=${RUN:-$(basename "$CONFIG" .toml)}
        ;;
esac

step "GPU, driver and disk"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1/p' | head -1)
[ -n "$cuda" ] && [ "$cuda" -ge 13 ] ||
    fail "the driver supports CUDA ${cuda:-?}; torch 2.13.0 needs 13.0 or later (choose another instance)"
free_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "$MIN_DISK_GB" ] || fail "$free_gb GB free in $WORK; need $MIN_DISK_GB"
command -v git > /dev/null || { apt-get update -qq && apt-get install -y -qq git; }

step "Unpack $BUNDLE"
if [ ! -d "$REPO/.git" ]; then
    staging="$WORK/bundle"
    rm -rf "$staging" && mkdir -p "$staging"
    tar -xf "$BUNDLE" -C "$staging"
    (cd "$staging" && sha256sum --quiet -c SHA256SUMS)
    commit=$(field commit)
    branch=$(field branch)
    git clone -q -b "$branch" "$staging/repo.bundle" "$REPO"
    [ "$(git -C "$REPO" rev-parse HEAD)" = "$commit" ] || fail "the bundle's branch isn't at $commit"
    for dir in data training configs; do
        [ ! -d "$staging/$dir" ] || cp -a "$staging/$dir" "$REPO/"
    done
    [ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] ||
        fail "the bundle's files differ from tracked files at $commit"
    rm -rf "$staging"
fi
cd "$REPO"
echo "$(git rev-parse --abbrev-ref HEAD) at $(git rev-parse HEAD)"

step "Install uv $UV_VERSION, Python and the train and serve groups"
export PATH="$HOME/.local/bin:$PATH"
command -v uv > /dev/null || curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
uv sync --frozen --group train --group serve
uv run --no-sync python -c '
import torch, transformers, peft, vllm
assert torch.cuda.is_available(), "torch sees no GPU"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name()}; "
      f"transformers {transformers.__version__}, peft {peft.__version__}, vLLM {vllm.__version__}")'

if [ "$MODE" = test ]; then
    step "Download the base model at training/$RUN's revision"
    uv run --no-sync python -c '
import sys
from pathlib import Path
from secbench import train
_, config, _ = train.read_run(Path(sys.argv[1]))
print(f"{config.model} at {config.revision}: {train.download(config, train.DEFAULT_HF_CACHE)}")' \
        "training/$RUN"

    step "Predictions on $SPLIT: the base model and $TEST_CONFIG's checkpoint (not graded)"
    status=0
    uv run --no-sync python -m secbench.score "training/$RUN" --test-config "$TEST_CONFIG" \
        --split "$SPLIT" --gpu-hourly "$PRICE" || status=$?

    step "Pack the results (score exited $status)"
    results="$WORK/results-$RUN-$SPLIT.tar"
    paths=()
    for path in runs/"$RUN--$SPLIT"--* "training/$RUN/test"; do
        [ ! -e "$path" ] || paths+=("$path")
    done
    if [ ${#paths[@]} -gt 0 ]; then
        tar -cf "$results" "${paths[@]}"
        echo "wrote $results: ${paths[*]}"
    else
        echo "nothing to pack"
    fi
    exit "$status"
fi

step "Download the model at the config's revision"
uv run --no-sync python -m secbench.train download --config "$CONFIG"

step "Prepare training/$RUN"
if [ -f "training/$RUN/config.json" ]; then
    echo "training/$RUN is prepared; kept"
else
    uv run --no-sync python -m secbench.train prepare --config "$CONFIG" --run-name "$RUN"
fi

step "Score the base model on validation, before training"
uv run --no-sync python -m secbench.score "training/$RUN"
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
for _ in $(seq 30); do [ "$(gpu_used)" -lt 2048 ] && break; sleep 2; done
[ "$(gpu_used)" -lt 2048 ] || fail "$(gpu_used) MiB of GPU memory still in use after vLLM stopped"

step "Train (from the last checkpoint, if any)"
uv run --no-sync python -m secbench.train train "training/$RUN"

step "Score every checkpoint on validation (the base model's run is kept)"
uv run --no-sync python -m secbench.score "training/$RUN"

step "Pack the results"
results="$WORK/results-$RUN.tar"
tar -cf "$results" --exclude=training_state.pt "training/$RUN" runs/"$RUN"--*
echo "wrote $results (adapters, logs, validation; optimizer states stay in $REPO/training/$RUN)"
