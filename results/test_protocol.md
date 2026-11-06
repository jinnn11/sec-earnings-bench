# Test protocol: stage 2

Fixed on 2026-11-05, before any call on a test set. This file is the one canonical
copy; PLAN.md's Decision of 2026-11-05 gives its sha256 and the commit that added it
("this commit" below). Changes are allowed only as dated amendments at the end, and
none after the first test call.

## Model under test

- Qwen3-8B at revision b968826d9c46dd6066d109eabc6255188de91218 with the LoRA adapter
  step-00025 of the training run qwen3-8b-2k (PLAN.md Decisions 2026-11-01), chosen on
  validation (Progress 2026-11-02): adapter_model.safetensors, 174,655,536 bytes,
  sha256 b93376d91af812159f4bd3a0fb80be090808e6d3539b251dc53ed26b9d27041b. Frozen: no
  more training, no other checkpoint.
- Its validation accuracy, 98.0% (287/293), is always labelled "chosen on
  validation": the checkpoint was picked for that score, so it overstates; the test
  sets give the estimate.
- Served and called as on validation (configs/test/qwen3-8b-2k.toml; `secbench.score
  --test-config` refuses any other setting): vLLM 0.32.0 with --max-model-len 40960,
  LoRA on (--max-lora-rank 16, --max-loras 1, --max-cpu-loras 1), the adapter served
  as qwen3-8b-2k--step-00025; prompt extract_v2, temperature 0, thinking off,
  max_tokens 1024, 16 requests at a time.

## References

- Qwen3-8B base: the same server and settings, thinking off, in the same session as
  the adapter. Validation reference: runs/qwen3-8b-2k--base (262/293, 89.4%).
- API models, each with the settings of its stage 1 validation run and prompt
  extract_v2:
  - GPT-6 Luna: openai provider, reasoning_effort medium, no temperature, $0.1 in and
    $0.5 out per million tokens. Validation reference:
    runs/2026-10-27T001339_gpt-6-luna_extract_v2.
  - Claude Haiku 5.5: anthropic provider, effort medium, max_tokens 16000, no
    temperature, $0.1 in and $0.5 out. Validation reference:
    runs/2026-10-17T222907_claude-haiku-5-5_extract_v2.
  - GPT-6.1 Sol: openai provider, reasoning_effort medium, no temperature, $2 in and
    $10 out. Its stage 1 run was on extract_v1
    (runs/2026-10-27T001744_gpt-6.1-sol_extract_v1), so it gets an extract_v2
    validation run with those settings before any test call; that run is its
    validation reference.
- The report checks each test run's provider, model, request_parameters,
  prompt_sha256 and price_per_million_tokens (gpu_hourly_usd aside, see Cost) against
  its model's validation reference run, and shows any difference.

## Test sets

- Company test: the frozen 148 releases of 37 companies, each company's latest 4
  linked releases in the stage 2 data merged by 51a63ce (data/frozen_test_releases.csv,
  data/splits/company_test/). Its answer key has 700 figures, 662 of them printed in
  the input; those 662 are graded for every model.
- Time test: train companies' releases filed from 2026-07-01 to a cutoff set in
  November, when the Q3 2026 releases are added, and frozen the same way
  (`secbench.splits --freeze time_test`) before any time-test call; scored then. The
  cutoff and the time test's split file sha256s go in a PLAN.md Decision when it is
  frozen (part of this protocol, not an amendment). Today's provisional time test
  (346 releases kept) is not scored.
- Context limit (PLAN.md Decisions 2026-11-01): a release whose prompt is over 39,936
  tokens is left out for every model, API models included, and listed
  (data/splits/<split>/left_out.csv). None in the company test; in the time test the
  Decision's 9 so far, and any Q3 2026 release the rule leaves out.
- Validation (data/splits/validation/, also frozen) is for checkpoint choice and
  reference runs only.

## Primary result

On the company test: step-00025's overall accuracy minus the base model's, in
percentage points, over all kept releases, with its 95% interval (Uncertainty).
Accuracy = correct / graded figures; not_in_input and no_answer are left out, as
`secbench.grade` does. Everything else is descriptive: fields, period types, error
types, the API references, the time test, cost and latency.

## Sensitivity

The same difference, and each model's accuracy, without the releases whose linked
filing reports nonzero preferred dividends (data/splits/<split>/preferred.csv, status
kept): whole releases are dropped, 11 in the company test and 20 in the time test
(today's count; fixed when it is frozen). Shown beside each model's net income
accuracy on those releases alone. Why: training broke Oracle's net income on the
"available to common shareholders" line (Findings 2026-11-03), and the prompt doesn't
say which line to take (prompt candidate 1, not applied).

## Uncertainty

- First, every run on a split must have the same graded (release_accession, field,
  period_type) keys; if not, the report stops and says which differ.
- 95% percentile intervals from a bootstrap that resamples companies (tickers) with
  replacement: 10,000 draws from numpy's default_rng(20261105), one set of draws
  shared by every model, subset and difference. Each draw's accuracy is Σcorrect /
  Σgraded over the drawn companies' figures (a company drawn twice counts twice); a
  difference is computed within each draw.
- With each difference: the number of figures where the two models disagree (one
  correct, the other not) and how many companies they come from.
- On the time test nearly every company has one release (346 kept releases of 344
  companies today), so there the bootstrap is in effect over releases; the report
  says so beside its intervals.

## Runs

- Exactly one reported run per model and split, named in the report.
- A run with an infrastructure API error (an api_error outcome other than a refusal:
  HTTP errors, timeouts and connection failures after extract's retries) or a stop
  (status aborted) is moved to <name>--attempt-N, listed in the report, and rerun in
  full once. If the rerun fails too, scoring stops and the failure is reported.
- Refusals and invalid replies are the model's answers: graded as missing, never
  rerun. extract records a refusal as an API error with a reason starting "refusal";
  the report counts it as the model's answer.
- GPU runs: `secbench.score --test-config` on the instance (predictions only),
  graded on the laptop with the split's links.csv and answers.csv. API runs:
  `secbench.extract --prompt extract_v2 --links data/splits/<split>/links.csv
  --answers data/splits/<split>/answers.csv --releases <the split's releases_dir>`.

## Frozen from this commit

- The code that builds inputs or answers, parses replies or grades:
  src/secbench/inputs.py, link.py, answers.py, extract.py (reply parsing and
  conversion), grade.py and splits.py; prompts/extract_v2.txt (sha256
  2e9e989384563ebf94db29e78604c0d78187767bcb688645fbafa3777c9ece29).
- The split files, by sha256:

  | file | validation | company_test |
  | --- | --- | --- |
  | links.csv | 9613ce78eaec205cbeba07b8faf9252acd92e868aea4cf4451d83bc4a24bff75 | 382791e2ab011a8669a53268eb474bcf26f60107070ce110191f36391f893af0 |
  | answers.csv | d56762363f6abdd224ca35ed0acb261411ed279f03f3e59e75c8f556b489de15 | 07001d58c10057f8e0168f8102f4a8badec247e8f802851c70146c6b42488cab |
  | tokens.csv | e1aaa95fa1a8c0704dde37cafe2c2fb62a1cf4483010e017cf522323cc79cf43 | eba3ab4b4873b572b8a67a8a4e58b1c8ee99c78dab36366019035d14f4fb0513 |
  | left_out.csv | 0f3dc7f1f52c6e1a5ab00038cbffbf7683995b674ba492bd889bbed7070d1ab1 | 0f3dc7f1f52c6e1a5ab00038cbffbf7683995b674ba492bd889bbed7070d1ab1 |
  | preferred.csv | ee97a30a04bc7d07b5d89f886e9caa3ec1e5f6c2737ed624ef2b4b42a92ba832 | 31b53b1d4f6a137107d3949229a22a31350c5892d3baa925a82b4c0a517a8e71 |

  and data/frozen_test_releases.csv, sha256
  6cba5eb290e21e6a0c34c08115b72ece94e8901ceae5e73c23eb91ae5e048101 (its validation
  and company_test rows; the time test's rows are added when it is frozen).
- The test config configs/test/qwen3-8b-2k.toml, sha256
  e211fe0da1e99fc02483d4c961cb96d23366d29c0f0b0756f40bc03df6b30155, and the adapter
  above.
- A change to any of these after test scoring starts is reported both ways, with the
  result under this commit's code and files as the primary one.

## Cost and latency

- GPU runs at the price paid (the GPU type from training/<run>/test/<split>.json and
  the $/h given to vast_setup.sh, run.json's gpu_hourly_usd) times the run's wall
  time (run.json's wall_time_s: first call to last, server start excluded), and at
  $1/h as results/comparison.md shows GPU runs. API runs at their token prices
  (above).
- Latency is reported per run but not compared across providers, which ran at
  different concurrency (16 on the GPU, 4 for the APIs); stage 3 measures speed.

## Amendments

None.
