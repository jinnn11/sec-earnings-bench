# Model comparison

From `uv run python -m secbench.compare`: 9 runs in runs/, graded against data/answers.csv; data/links.csv has 60 linked releases.

## Runs on all 60 linked releases

| run | model | provider | prompt | settings sent | concurrency | releases | status | accuracy | revenue | net_income | eps_diluted | correct | missing | sign | scale | wrong_period | noncontrolling_interest | rounded | wrong_line | other | no_answer | format retries | cost | cost basis | cost / 1,000 releases | latency p50 | latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-14T023027_claude-opus-5-5_extract_v1 | claude-opus-5-5 | anthropic | extract_v1 | max_tokens 16000, effort medium, no temperature | 4 | 60 | graded | 293/293 (100.0%) | 102/102 (100.0%) | 102/102 (100.0%) | 89/89 (100.0%) | 293 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 67 | 0 | $3.8026 | tokens ($4 in, $20 out per M) | $63.38 | 3.29 s | 6.46 s |
| 2026-10-14T023302_claude-haiku-5-5_extract_v1 | claude-haiku-5-5 | anthropic | extract_v1 | max_tokens 16000, effort medium, no temperature | 4 | 60 | graded | 292/293 (99.7%) | 102/102 (100.0%) | 101/102 (99.0%) | 89/89 (100.0%) | 292 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 67 | 0 | $0.0965 | tokens ($0.1 in, $0.5 out per M) | $1.61 | 1.23 s | 3.15 s |
| 2026-10-14T023330_claude-sonnet-5-5_extract_v1 | claude-sonnet-5-5 | anthropic | extract_v1 | max_tokens 16000, effort medium, no temperature | 4 | 60 | graded | 293/293 (100.0%) | 102/102 (100.0%) | 102/102 (100.0%) | 89/89 (100.0%) | 293 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 67 | 4 | $2.0369 | tokens ($2 in, $10 out per M) | $33.95 | 1.39 s | 2.38 s |
| 2026-10-17_qwen3-8b | Qwen/Qwen3-8B | openai | extract_v1 | no temperature | 16 | 60 | graded | 205/293 (70.0%) | 53/102 (52.0%) | 63/102 (61.8%) | 89/89 (100.0%) | 205 | 0 | 0 | 68 | 0 | 2 | 6 | 7 | 5 | 67 | 0 | $0.0187 | GPU ($0.167/h x 402.2 s) | $0.31 | 78.62 s | 229.55 s |
| 2026-10-17_qwen3-8b-nothink | Qwen/Qwen3-8B | openai | extract_v1 | temperature 0, chat_template_kwargs {"enable_thinking": false} | 16 | 60 | graded | 199/293 (67.9%) | 61/102 (59.8%) | 63/102 (61.8%) | 75/89 (84.3%) | 199 | 0 | 0 | 35 | 3 | 1 | 4 | 6 | 45 | 67 | 0 | $0.0041 | GPU ($0.167/h x 89.4 s) | $0.07 | 22.24 s | 36.28 s |
| 2026-10-17_gemma-12b | google/gemma-3-12b-it | openai | extract_v1 | temperature 0 | 16 | 60 | graded | 210/293 (71.7%) | 60/102 (58.8%) | 62/102 (60.8%) | 88/89 (98.9%) | 210 | 0 | 0 | 59 | 0 | 4 | 8 | 10 | 2 | 41 | 0 | $0.0053 | GPU ($0.167/h x 113.5 s) | $0.09 | 27.66 s | 47.41 s |
| 2026-10-17_gpt-oss-20b | openai/gpt-oss-20b | openai | extract_v1 | no temperature | 16 | 60 | graded | 277/293 (94.5%) | 96/102 (94.1%) | 96/102 (94.1%) | 85/89 (95.5%) | 277 | 2 | 0 | 4 | 6 | 1 | 2 | 0 | 1 | 67 | 0 | $0.0040 | GPU ($0.167/h x 87.1 s) | $0.07 | 18.91 s | 52.95 s |
| 2026-10-17_llama-8b | meta-llama/Llama-3.1-8B-Instruct | openai | extract_v1 | temperature 0 | 16 | 60 | graded | 130/293 (44.4%) | 22/102 (21.6%) | 24/102 (23.5%) | 84/89 (94.4%) | 130 | 0 | 4 | 103 | 0 | 2 | 2 | 5 | 47 | 67 | 0 | $0.0031 | GPU ($0.167/h x 66.6 s) | $0.05 | 16.20 s | 26.44 s |

## Other runs: fewer releases, stopped or not graded

| run | model | provider | prompt | settings sent | releases | status | accuracy | format retries | cost | cost basis | latency p50 | latency p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-14T022812_claude-opus-5-5_extract_v1 | claude-opus-5-5 | anthropic | extract_v1 | max_tokens 16000, effort medium, no temperature | 1 | graded | 6/6 (100.0%) | 0 | $0.0244 | tokens ($4 in, $20 out per M) | 2.06 s | 2.06 s |

Notes:
- Accuracy is correct figures over figures graded; not_in_input and no_answer are left out, as in each run's summary.txt.
- Verdicts 0 in every run in the first table: adjusted, not_in_input.
- Format retries: releases whose first reply didn't validate and were asked again; "(n invalid)" were still invalid and graded as missing.
- Cost is run.json's estimate: tokens at the run's prices per million, or GPU dollars per hour times the run's wall-clock time. Latency is per call.
- stale: re-grade: graded.csv doesn't match data/answers.csv; run `uv run python -m secbench.grade runs/<run>/`.
