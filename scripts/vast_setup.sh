#!/usr/bin/env bash
# Stage 2 on a fresh Vast.ai GPU instance, in order: check the GPU, unpack the bundle,
# install, download the model, prepare, train, score on validation, pack the results.
# Running it again continues where it stopped: the install and download are no-ops
# once done, a prepared run is kept, training resumes from its last checkpoint, and
# models already scored are kept.
#
# Pick an instance with one A100 80GB or H100, a driver for CUDA 13.0 or later
# (torch 2.13.0 and vLLM 0.32.0 are CUDA 13 builds) and 100 GB of disk.
#
# On the laptop, at a clean commit:
#   uv run python -m secbench.bundle                        # gpu-bundle-<commit>.tar
#   scp -P <port> gpu-bundle-<commit>.tar scripts/vast_setup.sh root@<host>:/workspace/
# On the instance:
#   cd /workspace && nohup bash vast_setup.sh gpu-bundle-<commit>.tar > setup.log 2>&1 &
#   tail -f setup.log
# Back on the laptop, when it has finished:
#   scp -P <port> root@<host>:/workspace/results-<run>.tar .
#
# Usage: bash vast_setup.sh BUNDLE [CONFIG]   (CONFIG: configs/train/qwen3-8b-2k.toml)
# The run is named after CONFIG's file (RUN=... to choose another name).
set -euo pipefail

BUNDLE=$(realpath "${1:?usage: bash vast_setup.sh BUNDLE [CONFIG]}")
CONFIG=${2:-configs/train/qwen3-8b-2k.toml}
RUN=${RUN:-$(basename "$CONFIG" .toml)}
WORK=$(dirname "$BUNDLE")
REPO="$WORK/sec-earnings-bench"
UV_VERSION=0.13.0
MIN_DISK_GB=80

step() { printf '\n== %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
fail() { echo "error: $*" >&2; exit 1; }

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
    commit=$(sed -n 's/.*"commit": "\([0-9a-f]*\)".*/\1/p' "$staging/BUNDLE.json")
    branch=$(sed -n 's/.*"branch": "\([^"]*\)".*/\1/p' "$staging/BUNDLE.json")
    git clone -q -b "$branch" "$staging/repo.bundle" "$REPO"
    [ "$(git -C "$REPO" rev-parse HEAD)" = "$commit" ] || fail "the bundle's branch isn't at $commit"
    cp -a "$staging/data" "$REPO/"
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

step "Download the model at the config's revision"
uv run --no-sync python -m secbench.train download --config "$CONFIG"

step "Prepare training/$RUN"
if [ -f "training/$RUN/config.json" ]; then
    echo "training/$RUN is prepared; kept"
else
    uv run --no-sync python -m secbench.train prepare --config "$CONFIG" --run-name "$RUN"
fi

step "Train (from the last checkpoint, if any)"
uv run --no-sync python -m secbench.train train "training/$RUN"

step "Score the base model and every checkpoint on validation"
uv run --no-sync python -m secbench.score "training/$RUN"

step "Pack the results"
results="$WORK/results-$RUN.tar"
tar -cf "$results" --exclude=training_state.pt "training/$RUN" runs/"$RUN"--*
echo "wrote $results (adapters, logs, validation; optimizer states stay in $REPO/training/$RUN)"
