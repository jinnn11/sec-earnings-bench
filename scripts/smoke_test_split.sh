#!/usr/bin/env bash
# Smoke test of a test split's predictions on a laptop, from an unpacked bundle, with
# no GPU-only package: write a test config for a run's checkpoint (training/<run>/
# smoke-test.toml, not tracked), pack it with secbench.bundle --test-config, unpack
# the bundle into a new temporary folder as scripts/vast_setup.sh does (git clone of
# repo.bundle, then the bundle's data, training and configs files), run
# secbench.score's test mode there with that copy's code (PYTHONPATH) through
# secbench.hfserve on the split's first 2 releases, and grade both runs on the
# laptop with the split's files. The split must be frozen (secbench.splits), the
# commit clean (bundle refuses otherwise), and the run's model already downloaded
# (data/raw/hf-hub). Scores from a smoke run mean nothing; the test is that every
# step runs from the bundle's files alone.
#
# Usage: bash scripts/smoke_test_split.sh [RUN] [CHECKPOINT] [SPLIT]
#        (default: smoke-b5e9990 step-00004 validation, Qwen3-0.6B from
#        scripts/smoke_mac.sh)
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${1:-smoke-b5e9990}
CHECKPOINT=${2:-step-00004}
SPLIT=${3:-validation}
REPO=$(pwd)
PYTHON="$REPO/.venv/bin/python"  # the train group is installed; no uv sync
CONFIG="training/$RUN/smoke-test.toml"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/smoke-test-split.XXXXXX")
COPY="$WORK/sec-earnings-bench"

step() { printf '\n== %s %s\n' "$(date +%H:%M:%S)" "$*"; }

step "Test config for $RUN $CHECKPOINT: $CONFIG (concurrency 1: hfserve answers one at a time)"
"$PYTHON" - "$RUN" "$CHECKPOINT" "$CONFIG" <<'EOF'
import sys
from pathlib import Path

from secbench import chat, labels, score, train

run, checkpoint, out = sys.argv[1:]
_, config, _ = train.read_run(Path("training", run))
weights = Path("training", run, train.CHECKPOINTS, checkpoint, train.ADAPTER_WEIGHTS)
Path(out).write_text(f"""# Laptop smoke test only (scripts/smoke_test_split.sh).
[model]
run = "{run}"
base = "{config.model}"
revision = "{config.revision}"
checkpoint = "{checkpoint}"
adapter_sha256 = "{score.file_sha256(weights)}"

[server]
max_model_len = {chat.MAX_MODEL_LEN}
max_lora_rank = {score.lora_rank(config.r)}
max_loras = 1
max_cpu_loras = 1

[request]
prompt = "{labels.PROMPT}"
temperature = 0.0
enable_thinking = false
max_tokens = {chat.MAX_TOKENS}
concurrency = 1
""")
print(open(out).read(), end="")
EOF

step "Bundle $SPLIT for $RUN $CHECKPOINT"
"$PYTHON" -m secbench.bundle --test-config "$CONFIG" --split "$SPLIT" --out "$WORK/bundle.tar"

step "Unpack into $COPY, as vast_setup.sh does"
mkdir "$WORK/bundle"
tar -xf "$WORK/bundle.tar" -C "$WORK/bundle"
(cd "$WORK/bundle" && shasum -a 256 -c SHA256SUMS > /dev/null)
branch=$(sed -n 's/.*"branch": "\([^"]*\)".*/\1/p' "$WORK/bundle/BUNDLE.json")
git clone -q -b "$branch" "$WORK/bundle/repo.bundle" "$COPY"
for dir in data training configs; do
    [ ! -d "$WORK/bundle/$dir" ] || cp -a "$WORK/bundle/$dir" "$COPY/"
done
[ -z "$(git -C "$COPY" status --porcelain --untracked-files=no)" ]
(cd "$COPY" && find data training configs -type f | sort | sed 's/^/  /')

step "Predictions on $SPLIT's first 2 releases, from the copy (its code; hfserve)"
cd "$COPY"
export PYTHONPATH="$COPY/src"
"$PYTHON" -c 'import secbench, sys; assert secbench.__file__.startswith(sys.argv[1]), secbench.__file__' "$COPY/src"
"$PYTHON" -m secbench.score "training/$RUN" --test-config "$CONFIG" --split "$SPLIT" \
    --server hf --limit 2 --gpu-hourly 1 --hf-cache "$REPO/data/raw/hf-hub"
unset PYTHONPATH
cd "$REPO"

step "Grade both runs on the laptop, with the split's files"
releases=$("$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["releases_dir"])' \
    "data/splits/$SPLIT/manifest.json")
for run in "$COPY/runs/$RUN--$SPLIT"--*; do
    "$PYTHON" -m secbench.grade "$run" --links "data/splits/$SPLIT/links.csv" \
        --answers "data/splits/$SPLIT/answers.csv" --releases "$releases"
done

step "Done: $COPY (runs/, training/$RUN/test/)"
