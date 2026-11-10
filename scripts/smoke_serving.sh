#!/usr/bin/env bash
# Smoke test of the stage 3 serving benchmark on a laptop, with no GPU-only package:
# Qwen3-0.6B with the stage 2 smoke run's step-00004 adapter (scripts/smoke_mac.sh
# made training/smoke-b5e9990), served by secbench.hfserve, the stand-in for vLLM.
# It writes a serving config for that run (training/<run>/serving/smoke.toml, not
# tracked), draws its calibration examples (4 of at most 4,096 tokens), and runs
# secbench.servebench: adapter and merged end to end (the merge, the server, the
# start-up log, the /tokenize check of every validation and time-test prompt, the
# load test at 1 and 2 in flight with 4 time-test prompts per level after 1 warm-up,
# drawn from the time test's first 8 releases, and the accuracy runs on validation's
# first 2 and the time test's first 4 releases); fp8 and int4-awq need vLLM on a GPU
# and are skipped. Then secbench.servereport grades the runs and writes the report,
# and scripts/plot_serving.py the chart, into training/<run>/serving/report/. If
# .venv-quantize exists (UV_PROJECT_ENVIRONMENT=.venv-quantize uv sync --group train
# --group quantize), it then quantizes the merged 0.6B model with the AWQ recipe on
# the 4 calibration examples, to try it on a real Qwen3 (not served: vLLM only).
# Numbers from a 0.6B model on a laptop mean nothing; the test is that every step
# runs. Outputs stay in training/<run>/serving/ and data/serving/<run>/; a rerun
# keeps what is complete (delete those folders to start over).
#
# Usage: bash scripts/smoke_serving.sh [RUN] [CHECKPOINT]
#        (default: smoke-b5e9990 step-00004, Qwen3-0.6B from scripts/smoke_mac.sh)
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${1:-smoke-b5e9990}
CHECKPOINT=${2:-step-00004}
PYTHON="$PWD/.venv/bin/python"  # the train group is installed; no uv sync
SERVING="training/$RUN/serving"
CONFIG="$SERVING/smoke.toml"
RUNS="$SERVING/runs"
REPORT="$SERVING/report"

step() { printf '\n== %s %s\n' "$(date +%H:%M:%S)" "$*"; }

step "Serving config for $RUN $CHECKPOINT: $CONFIG"
mkdir -p "$SERVING"
"$PYTHON" - "$RUN" "$CHECKPOINT" "$CONFIG" <<'EOF'
import sys
from pathlib import Path

from secbench import chat, labels, score, train

run, checkpoint, out = sys.argv[1:]
_, config, _ = train.read_run(Path("training", run))
weights = Path("training", run, train.CHECKPOINTS, checkpoint, train.ADAPTER_WEIGHTS)
reference = Path("training", run, "runs", f"{run}--{checkpoint}")
Path(out).write_text(f"""# Laptop smoke test only (scripts/smoke_serving.sh).
[model]
run = "{run}"
base = "{config.model}"
revision = "{config.revision}"
checkpoint = "{checkpoint}"
adapter_sha256 = "{score.file_sha256(weights)}"

[server]
variants = ["adapter", "merged", "fp8", "int4-awq"]
max_model_len = {chat.MAX_MODEL_LEN}
prefix_caching = false

[request]
prompt = "{labels.PROMPT}"
temperature = 0.0
enable_thinking = false
max_tokens = {chat.MAX_TOKENS}

[load]
split = "time_test"
levels = [1, 2]
prompts = 4
warmup = 1
seed = 20261109
pool = 8  # the time test's first 8 releases (10,000-15,000 tokens): a laptop's memory

[accuracy]
splits = ["validation", "time_test"]
concurrency = 1  # hfserve answers one at a time
limit = {{ validation = 2, time_test = 4 }}

[calibration]
examples = 4
max_tokens = 4096
seed = 20261109
scheme = "W4A16_ASYM"
duo_scaling = "both"
ignore = ["lm_head"]

[report]
""" + (f'validation_reference = "{reference}"\n' if reference.is_dir() else ""))
print(open(out).read(), end="")
EOF

step "Calibration examples: data/serving/$RUN/"
"$PYTHON" -m secbench.quantize calibration --config "$CONFIG"

step "secbench.servebench through secbench.hfserve (fp8 and int4-awq need vLLM)"
"$PYTHON" -m secbench.servebench --config "$CONFIG" --gpu-hourly 1 --server hf \
    --runs-dir "$RUNS"

step "Report: $REPORT"
"$PYTHON" -m secbench.servereport --config "$CONFIG" --runs-dir "$RUNS" --out-dir "$REPORT"
"$PYTHON" scripts/plot_serving.py --results "$REPORT/serving_results.json" \
    --out "$REPORT/serving_throughput_latency.png"

if [ -x .venv-quantize/bin/python ]; then
    step "AWQ on the merged model, in .venv-quantize (not served: vLLM only)"
    awq="$SERVING/models/int4-awq"
    if [ -f "$awq/manifest.json" ]; then
        echo "$awq is made; kept"
    else
        .venv-quantize/bin/python -m secbench.quantize awq --model "$SERVING/models/merged" \
            --calibration "data/serving/$RUN/calibration.jsonl" --out "$awq" --max-tokens 4096
    fi
    "$PYTHON" - "$awq/manifest.json" <<'EOF'
import json
import sys

manifest = json.load(open(sys.argv[1]))
config = manifest["quantization_config"]
weights = next(iter(config["config_groups"].values()))["weights"]
print(
    f"{config['format']}: {weights['num_bits']}-bit weights, group "
    f"{weights['group_size']}, symmetric {weights['symmetric']}; "
    f"{manifest['weight_bytes'] / 2**20:,.0f} MiB; seconds {manifest['timings_s']}"
)
EOF
else
    step "No .venv-quantize: AWQ not tried (UV_PROJECT_ENVIRONMENT=.venv-quantize uv sync --group train --group quantize)"
fi

step "Done: $REPORT/serving_results.md"
