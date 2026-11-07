# Stage 2 test results

From `uv run python -m secbench.testreport` with configs/report/stage2.toml; the analysis is fixed by results/test_protocol.md (sha256 99767156…, commit cf77acb). Anything beyond the protocol is marked descriptive (not pre-registered). Figures are graded figures: answers printed in the input (not_in_input and no_answer are left out). Intervals are 95% bootstrap intervals over companies (Bootstrap below).

## Primary result

On the company test, step-00025 minus Qwen3-8B base: **+10.12 points [+5.08, +15.58]**, from 89.12% to 99.24% of 662 figures; 73 discordant figures (70 fixed by training, 3 broken) from 16 companies. Everything else is descriptive under the protocol.

step-00025's validation score, 98.0% (287/293), is chosen on validation (the checkpoint was picked for that score, so it may overstate; the test sets give the estimate).

## Checks

Each would stop the report. No frozen item changed:

- protocol results/test_protocol.md sha256 99767156…
- 15 split files under data/splits/ by sha256 (the protocol's table; the time test's from PLAN.md Decisions 2026-11-07)
- data/frozen_test_releases.csv sha256 46c185e2…, its validation and company_test rows as at cf77acb (6cba5eb2…)
- prompts/extract_v2.txt 2e9e9893…, configs/test/qwen3-8b-2k.toml e211fe0d…
- training/qwen3-8b-2k/test/<split>.json: that config, adapter b93376d9…, prompts identical to training's layout
- no change since cf77acb (git diff, committed and working tree) to src/secbench/inputs.py, src/secbench/link.py, src/secbench/answers.py, src/secbench/extract.py, src/secbench/grade.py, src/secbench/splits.py, prompts/extract_v2.txt, configs/test/qwen3-8b-2k.toml

Company test:

- runs: all 5 complete with no selection, covering the 148 kept releases, no stop and no API error but refusals (0 refusals), on the frozen links and answers, at commit cf77acb, clean
- graded.csv: current in all 5 (not stale; predicted values are predictions.jsonl's; re-graded in memory, byte for byte)
- settings: provider, model, request_parameters, prompt_sha256 and price_per_million_tokens equal to each model's validation reference
- keys: the same 662 graded and 38 not_in_input figures in all 5 runs

Time test:

- runs: all 5 complete with no selection, covering the 346 kept releases, no stop and no API error but refusals (0 refusals), on the frozen links and answers, at commit e352c14, clean
- graded.csv: current in all 5 (not stale; predicted values are predictions.jsonl's; re-graded in memory, byte for byte)
- settings: provider, model, request_parameters, prompt_sha256 and price_per_million_tokens equal to each model's validation reference
- keys: the same 1,740 graded and 52 not_in_input figures in all 5 runs

## Model under test

- Qwen/Qwen3-8B at revision b968826d, with LoRA r 16 (alpha 32) on all 7 projections (q, k, v, o, gate, up, down), thinking off.
- 125 steps of 16 examples over 2,000 of the 3,200 labelled train releases (drawn from the 3,101 under the 32,768-token cap). The chosen checkpoint, step-00025, had seen 400 examples; adapter sha256 b93376d9…041b.
- Served with vLLM 0.32.0 on one Vast NVIDIA H100 NVL per test split, as configs/test/qwen3-8b-2k.toml sets (temperature 0, max_tokens 1024, 16 requests at a time); the base model in the same session.

## Company test

148 kept releases of 37 companies (data/splits/company_test/), runs made on 2026-11-07 from 11:16 to 11:31 UTC at commit cf77acb.

### Runs

| model | run | model ID | provider | settings sent | concurrency | outcomes | validation reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| step-00025 | qwen3-8b-2k--company_test--step-00025 | qwen3-8b-2k--step-00025 | openai | temperature 0, chat_template_kwargs {"enable_thinking": false}, max_tokens 1024 | 16 | 148 ok | qwen3-8b-2k--step-00025 |
| Qwen3-8B base | qwen3-8b-2k--company_test--base | Qwen/Qwen3-8B | openai | temperature 0, chat_template_kwargs {"enable_thinking": false}, max_tokens 1024 | 16 | 147 ok, 1 ok after a format retry | qwen3-8b-2k--base |
| GPT-6.1 Sol | gpt-6.1-sol--company_test | gpt-6.1-sol | openai | reasoning_effort medium, no temperature | 4 | 148 ok | 2026-11-07T105947_gpt-6.1-sol_extract_v2 |
| Claude Haiku 5.5 | claude-haiku-5-5--company_test | claude-haiku-5-5 | anthropic | max_tokens 16000, effort medium, no temperature | 4 | 147 ok, 1 ok after a format retry | 2026-10-17T222907_claude-haiku-5-5_extract_v2 |
| GPT-6 Luna | gpt-6-luna--company_test | gpt-6-luna | openai | reasoning_effort medium, no temperature | 4 | 148 ok | 2026-10-27T001339_gpt-6-luna_extract_v2 |

Settings differing from the validation reference, besides those checked equal (notes):

- step-00025: env_file training/qwen3-8b-2k/test/company_test--step-00025.env (reference training/qwen3-8b-2k/validation/step-00025.env)
- step-00025: gpu_hourly_usd 2.776 (reference 1.0)
- Qwen3-8B base: env_file training/qwen3-8b-2k/test/company_test--base.env (reference training/qwen3-8b-2k/validation/base.env)
- Qwen3-8B base: gpu_hourly_usd 2.776 (reference 1.0)
- Claude Haiku 5.5: env_file .env.claude-haiku (reference none)

### Coverage

| answer key over the kept releases | figures |
| --- | --- |
| expected (answered + missing, data/stage2/releases.csv) | 728 |
| answered (an XBRL answer) | 700 |
| missing (no XBRL answer) | 28 |
| in the input | 662 |
| graded | 662 |

Releases with no answer at all: 4 (by sector, of the kept releases: Health Care 1 of 28; Industrials 1 of 24; Utilities 2 of 8).

Releases with at least one graded figure: 143 of 148 kept; companies: 37 of 37. Kept releases without one: ETR 0000065984-26-000279 (no answer); CAH 0000721371-26-000017 (no answer); JCI 0000833444-26-000083 (no answer); PPL 0000922224-26-000042 (no answer); CCI 0001051470-26-000052 (none of its 3 answers in the input).

Left out by the context limit: none.

Attempts (runs moved aside and rerun): none.

### Accuracy

Correct / graded figures (%), 95% interval.

| model | all | revenue | net_income | eps_diluted |
| --- | --- | --- | --- | --- |
| step-00025 | 657/662 (99.24%) [98.05, 100.00] | 223/223 (100.00%) [100.00, 100.00] | 236/241 (97.93%) [94.69, 100.00] | 198/198 (100.00%) [100.00, 100.00] |
| Qwen3-8B base | 590/662 (89.12%) [83.64, 94.19] | 207/223 (92.83%) [86.08, 98.21] | 193/241 (80.08%) [69.11, 89.87] | 190/198 (95.96%) [90.10, 99.51] |
| GPT-6.1 Sol | 662/662 (100.00%) [100.00, 100.00] | 223/223 (100.00%) [100.00, 100.00] | 241/241 (100.00%) [100.00, 100.00] | 198/198 (100.00%) [100.00, 100.00] |
| Claude Haiku 5.5 | 662/662 (100.00%) [100.00, 100.00] | 223/223 (100.00%) [100.00, 100.00] | 241/241 (100.00%) [100.00, 100.00] | 198/198 (100.00%) [100.00, 100.00] |
| GPT-6 Luna | 654/662 (98.79%) [96.84, 100.00] | 223/223 (100.00%) [100.00, 100.00] | 236/241 (97.93%) [94.63, 100.00] | 195/198 (98.48%) [95.96, 100.00] |

| model | quarter | year_to_date | full_year |
| --- | --- | --- | --- |
| step-00025 | 364/367 (99.18%) [97.48, 100.00] | 190/192 (98.96%) [96.57, 100.00] | 103/103 (100.00%) [100.00, 100.00] |
| Qwen3-8B base | 323/367 (88.01%) [81.47, 93.77] | 174/192 (90.62%) [85.37, 95.72] | 93/103 (90.29%) [83.65, 96.12] |
| GPT-6.1 Sol | 367/367 (100.00%) [100.00, 100.00] | 192/192 (100.00%) [100.00, 100.00] | 103/103 (100.00%) [100.00, 100.00] |
| Claude Haiku 5.5 | 367/367 (100.00%) [100.00, 100.00] | 192/192 (100.00%) [100.00, 100.00] | 103/103 (100.00%) [100.00, 100.00] |
| GPT-6 Luna | 363/367 (98.91%) [96.99, 100.00] | 192/192 (100.00%) [100.00, 100.00] | 99/103 (96.12%) [90.20, 100.00] |

### Differences

step-00025 minus each reference, in points, with the discordant figures (one model correct, the other not) and the companies they come from. The first row is the primary result.

| difference | points | 95% interval | discordant | of which | companies |
| --- | --- | --- | --- | --- | --- |
| minus Qwen3-8B base | +10.12 | [+5.08, +15.58] | 73 | 70 fixed, 3 broken | 16 |
| minus GPT-6.1 Sol | −0.76 | [−1.95, 0.00] | 5 | 0 step-00025 right, 5 GPT-6.1 Sol right | 2 |
| minus Claude Haiku 5.5 | −0.76 | [−1.95, 0.00] | 5 | 0 step-00025 right, 5 Claude Haiku 5.5 right | 2 |
| minus GPT-6 Luna | +0.45 | [−1.41, +2.67] | 13 | 8 step-00025 right, 5 GPT-6 Luna right | 4 |

### Sensitivity: without the preferred-dividend releases

Without the 11 kept releases in preferred.csv (whole releases dropped: 50 graded figures; 612 left).

| model | accuracy |
| --- | --- |
| step-00025 | 610/612 (99.67%) [98.95, 100.00] |
| Qwen3-8B base | 551/612 (90.03%) [84.19, 95.35] |
| GPT-6.1 Sol | 612/612 (100.00%) [100.00, 100.00] |
| Claude Haiku 5.5 | 612/612 (100.00%) [100.00, 100.00] |
| GPT-6 Luna | 604/612 (98.69%) [96.60, 100.00] |

| difference | points | 95% interval | discordant | of which | companies |
| --- | --- | --- | --- | --- | --- |
| minus Qwen3-8B base | +9.64 | [+4.24, +15.54] | 63 | 61 fixed, 2 broken | 13 |
| minus GPT-6.1 Sol | −0.33 | [−1.05, 0.00] | 2 | 0 step-00025 right, 2 GPT-6.1 Sol right | 1 |
| minus Claude Haiku 5.5 | −0.33 | [−1.05, 0.00] | 2 | 0 step-00025 right, 2 Claude Haiku 5.5 right | 1 |
| minus GPT-6 Luna | +0.98 | [−0.66, +3.19] | 10 | 8 step-00025 right, 2 GPT-6 Luna right | 3 |

Net income on the preferred-dividend releases alone, counts only, no interval: the 15 figures come from 3 companies (4.7% of draws contain none of them). step-00025 12/15; Qwen3-8B base 5/15; GPT-6.1 Sol 15/15; Claude Haiku 5.5 15/15; GPT-6 Luna 15/15.

### Verdicts and unit slips

| verdict | step-00025 | Qwen3-8B base | GPT-6.1 Sol | Claude Haiku 5.5 | GPT-6 Luna |
| --- | --- | --- | --- | --- | --- |
| correct | 657 | 590 | 662 | 662 | 654 |
| scale | 0 | 2 | 0 | 0 | 0 |
| zeros | 0 | 10 | 0 | 0 | 0 |
| wrong_period | 2 | 3 | 0 | 0 | 1 |
| noncontrolling_interest | 0 | 24 | 0 | 0 | 0 |
| rounded | 0 | 6 | 0 | 0 | 0 |
| adjusted | 0 | 15 | 0 | 0 | 6 |
| wrong_line | 3 | 12 | 0 | 0 | 1 |
| not_in_input | 38 | 38 | 38 | 38 | 38 |
| no_answer | 188 | 188 | 187 | 187 | 187 |
| unit slips | 0 | 12 | 0 | 0 | 0 |

Unit slips: rows matched with a power-of-ten factor (graded.csv's factor column). not_in_input and no_answer rows aren't graded.

### Cost and latency

| model | cost basis | cost | per 1,000 releases | per 1,000 at $1/h | latency p50 | latency p95 |
| --- | --- | --- | --- | --- | --- | --- |
| step-00025 | $2.538/h × 559.0 s | $0.3941 | $2.66 | $1.05 | 11.49 s | 135.95 s (freeze) |
| Qwen3-8B base | $2.538/h × 113.7 s | $0.0802 | $0.54 | $0.21 | 11.25 s | 18.45 s |
| GPT-6.1 Sol | tokens ($2 in, $10 out per M) | $3.5193 | $23.78 | – | 1.79 s | 3.68 s |
| Claude Haiku 5.5 | tokens ($0.1 in, $0.5 out per M) | $0.2737 | $1.85 | – | 1.58 s | 3.05 s |
| GPT-6 Luna | tokens ($0.1 in, $0.5 out per M) | $0.1814 | $1.23 | – | 2.29 s | 4.97 s |

GPU runs at the price billed ($2.538/h) times the run's wall time (first call to last), and at $1/h as results/comparison.md shows GPU runs; they leave out each session's setup, model loading and idle time. API runs at their token prices (run.json's estimate). Latency is per call, reported but not compared across providers (16 calls at a time on the GPU, 4 or fewer on the APIs).

Price: both GPU runs record gpu_hourly_usd 2.776, an example price passed to vast_setup.sh by mistake; the instance was billed $2.538/h (Deviations, 2). At the recorded price, the protocol's letter: step-00025 $0.4311, $2.91 per 1,000 releases; Qwen3-8B base $0.0877, $0.59 per 1,000 releases.

Instance freeze (descriptive): qwen3-8b-2k--company_test--step-00025 took 559.0 s, but the whole instance froze from about 11:22:37 to 11:29:58 UTC (441 s). training/qwen3-8b-2k/test/server-company_test.log has no timestamped line between 11:22:31 and 11:29:58, while vLLM otherwise logs its throughput every 10 s; the 16 requests in flight each took 134-137 s (the next longest 23.5 s), and the run's p95 and max latency are among them. As in training's 6 slow steps (PLAN.md Progress 2026-11-02), the wall clock ran ahead of the process's monotonic clock: the run took about 240.9 s by its own clock (the sum of call latencies / 16), so the wall clock gained about 317 s (after the base run's usual 1.1 s of wall time beyond its latencies / 16).

Freeze-excluded estimate: 559.0 − 441 = 118.0 s, about $0.56 per 1,000 releases at $2.538/h ($0.22 at $1/h). Cross-check: (the other 132 latencies + 16 × their median) / 16 + 1.1 s = 117.4 s. The base run took 113.7 s.

### step-00025's misses (descriptive, not pre-registered)

The rule: (1) net income whose predicted value is printed on an input row whose label names common shareholders, stockholders, shares or stock, whatever its verdict (full labels read from the input, not graded.csv's shortened evidence; table rows and prose lines); (2) a 10-Q release's year-to-date prediction equal to its quarter prediction; (3) the rest, by verdict. A miss meeting (1) and (2) goes to (2), since its value is the quarter's whichever line prints it.

5 misses: (1) 3, (2) 2, (3) none.

Group (1): 3 of the 5 misses, from 1 company (BA); verdicts wrong_line 3. 3 are in preferred.csv releases and 0 outside it, so the sensitivity analysis drops all of them. Group (1) includes 1 of the 3 figures broken by training (the base model's correct).

Group (2): 2 from WY: the quarter copied into year to date in second- or third-quarter releases whose income statement has quarter columns only, with the year to date printed elsewhere (the row where the input prints the year-to-date answer is listed).

| group | ticker | release | field | period | answer | predicted | verdict | Qwen3-8B base verdict | row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| common shareholders' line (preferred.csv) | BA | 0001628280-26-003518 | net_income | quarter | 8220000000 | 1890000000 | wrong_line | correct | Net earnings/(loss) attributable to Boeing common shareholders |
| common shareholders' line (preferred.csv) | BA | 0001628280-26-026391 | net_income | quarter | -4000000 | -90000000 | wrong_line | noncontrolling_interest | Net loss attributable to Boeing common shareholders |
| common shareholders' line (preferred.csv) | BA | 0001628280-26-049929 | net_income | quarter | -444000000 | -620000000 | wrong_line | noncontrolling_interest | Net loss attributable to Boeing common shareholders |
| quarter copied into year to date | WY | 0001193125-25-258510 | net_income | year_to_date | 250000000 | 80000000 | wrong_period | correct | The table below reconciles Adjusted EBITDA for the year-to-date period ended September 30, 2025: › Adjusted EBITDA by Segment: › Net earnings |
| quarter copied into year to date | WY | 0001193125-26-326124 | net_income | year_to_date | 318000000 | 162000000 | wrong_period | correct | The table below reconciles Adjusted EBITDA for the year-to-date period ended June 30, 2026: › Adjusted EBITDA by Segment: › Net earnings |

### Abstention (descriptive, not pre-registered)

| model | not_in_input figures answered | null figures of all predicted |
| --- | --- | --- |
| step-00025 | 38 of 38 | 0 of 888 |
| Qwen3-8B base | 38 of 38 | 0 of 888 |
| GPT-6.1 Sol | 19 of 38 | 20 of 888 |
| Claude Haiku 5.5 | 26 of 38 | 13 of 888 |
| GPT-6 Luna | 26 of 38 | 13 of 888 |

A not_in_input figure has an XBRL answer that the grader doesn't find in the input in any of its printed forms. Where the release doesn't report the figure (a year to date in a release whose tables have quarter columns only), the prompt's "null when the release doesn't report a figure" applies. But many such releases do print it, in a form the grader doesn't search (an EPS of $.13 for 0.13) or as another value on the line the prompt asks for (rounded, revised, or a total that differs from XBRL's), and then answering is right. So the answered count is an upper bound on missed abstentions, not a count of them. Accuracy leaves these figures out, so it doesn't reward abstaining. Training had no null labels (PLAN.md, Limitation: no nulls in training).

## Time test

346 kept releases of 344 companies (data/splits/time_test/), runs made on 2026-11-07 from 12:32 to 12:57 UTC at commit e352c14.

### Runs

| model | run | model ID | provider | settings sent | concurrency | outcomes | validation reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| step-00025 | qwen3-8b-2k--time_test--step-00025 | qwen3-8b-2k--step-00025 | openai | temperature 0, chat_template_kwargs {"enable_thinking": false}, max_tokens 1024 | 16 | 346 ok | qwen3-8b-2k--step-00025 |
| Qwen3-8B base | qwen3-8b-2k--time_test--base | Qwen/Qwen3-8B | openai | temperature 0, chat_template_kwargs {"enable_thinking": false}, max_tokens 1024 | 16 | 342 ok, 4 ok after a format retry | qwen3-8b-2k--base |
| GPT-6.1 Sol | gpt-6.1-sol--time_test | gpt-6.1-sol | openai | reasoning_effort medium, no temperature | 2 | 346 ok | 2026-11-07T105947_gpt-6.1-sol_extract_v2 |
| Claude Haiku 5.5 | claude-haiku-5-5--time_test | claude-haiku-5-5 | anthropic | max_tokens 16000, effort medium, no temperature | 4 | 346 ok | 2026-10-17T222907_claude-haiku-5-5_extract_v2 |
| GPT-6 Luna | gpt-6-luna--time_test | gpt-6-luna | openai | reasoning_effort medium, no temperature | 4 | 346 ok | 2026-10-27T001339_gpt-6-luna_extract_v2 |

Settings differing from the validation reference, besides those checked equal (notes):

- step-00025: env_file training/qwen3-8b-2k/test/time_test--step-00025.env (reference training/qwen3-8b-2k/validation/step-00025.env)
- step-00025: gpu_hourly_usd 2.589 (reference 1.0)
- Qwen3-8B base: env_file training/qwen3-8b-2k/test/time_test--base.env (reference training/qwen3-8b-2k/validation/base.env)
- Qwen3-8B base: gpu_hourly_usd 2.589 (reference 1.0)
- GPT-6.1 Sol: max_parallel 2 (reference 4)
- GPT-6.1 Sol: env_file .env.gpt-sol-c2 (reference .env.gpt-sol)
- Claude Haiku 5.5: env_file .env.claude-haiku (reference none)

### Coverage

| answer key over the kept releases | figures |
| --- | --- |
| expected (answered + missing, data/stage2/releases.csv) | 1,916 |
| answered (an XBRL answer) | 1,792 |
| missing (no XBRL answer) | 124 |
| in the input | 1,740 |
| graded | 1,740 |

Releases with no answer at all: 29 (by sector, of the kept releases: Consumer Staples 3 of 28; Health Care 4 of 51; Industrials 5 of 75; Information Technology 1 of 66; Materials 2 of 19; Real Estate 3 of 17; Utilities 11 of 23).

Releases with at least one graded figure: 317 of 346 kept; companies: 315 of 344. Kept releases without one: ABT 0001628280-26-048377 (no answer); CLX 0000021076-26-000028 (no answer); FRT 0000034903-26-000040 (no answer); GD 0000040533-26-000029 (no answer); HUM 0000049071-26-000045 (no answer); SO 0000092122-26-000056 (no answer); LNT 0000352541-26-000042 (no answer); D 0001193125-26-326812 (no answer); NEE 0000753308-26-000058 (no answer); EIX 0000827052-26-000058 (no answer); HCA 0001193125-26-315151 (no answer); INCY 0000879169-26-000053 (no answer); REG 0001193125-26-323784 (no answer); MAA 0001193125-26-323756 (no answer); DTE 0000936340-26-000144 (no answer); FE 0001031296-26-000122 (no answer); SRE 0001032208-26-000043 (no answer); ED 0001047862-26-000143 (no answer); CTSH 0001058290-26-000030 (no answer); URI 0001067701-26-000028 (no answer); MDLZ 0001628280-26-050157 (no answer); CNP 0001104659-26-087279 (no answer); DUK 0001326160-26-000037 (no answer); VMC 0001140361-26-029958 (no answer); VRSK 0001437749-26-024748 (no answer); DOW 0001751788-26-000144 (no answer); CARR 0001783180-26-000030 (no answer); VLTO 0001967680-26-000043 (no answer); BG 0001628280-26-050334 (no answer).

Left out by the context limit (prompt over 39,936 tokens; left_out.csv), for every model: UDR 0000074208-26-000070 (50,970 tokens); FCX 0000831259-26-000033 (50,171 tokens); ARE 0001035443-26-000067 (87,822 tokens); BXP 0001037540-26-000031 (106,137 tokens); SPG 0001104659-26-093361 (104,334 tokens); NEM 0001164727-26-000034 (46,825 tokens); DLR 0001104659-26-086270 (70,776 tokens); PSX 0001534701-26-000030 (52,257 tokens); INVH 0001687229-26-000041 (44,808 tokens).

Attempt runs/gpt-6.1-sol--time_test--attempt-1 (run.json still says run_name gpt-6.1-sol--time_test; only its folder was renamed): 2 API errors at 4 calls at a time (NCLH 0001171843-26-005050, DIS 0001744489-26-000056), each HTTP 429 after 6 attempts, on the 1,000,000 tokens-per-minute limit. As the protocol says (Runs), it was moved aside and rerun in full once: gpt-6.1-sol--time_test, with .env.gpt-sol-c2, Sol's settings plus MODEL_CONCURRENCY=2 (the same provider, model, request_parameters, prompt_sha256, price_per_million_tokens, reply_format; max_parallel 2). On the 344 releases both runs answered, every predicted figure is identical; 1 period_end differs (TRMB 0000864749-26-000106: 2026-07-03 in the attempt, after a format retry, and 2026-06-30 in the rerun). The attempt cost $8.3046, listed here and left out of the per-release cost.

### Accuracy

Correct / graded figures (%), 95% interval. On the time test 317 releases with graded figures come from 315 companies, so the bootstrap is in effect over releases (protocol, Uncertainty).

| model | all | revenue | net_income | eps_diluted |
| --- | --- | --- | --- | --- |
| step-00025 | 1,699/1,740 (97.64%) [96.64, 98.57] | 583/591 (98.65%) [97.29, 99.67] | 545/577 (94.45%) [91.86, 96.86] | 571/572 (99.83%) [99.46, 100.00] |
| Qwen3-8B base | 1,516/1,740 (87.13%) [84.73, 89.49] | 505/591 (85.45%) [81.62, 89.00] | 456/577 (79.03%) [74.39, 83.48] | 555/572 (97.03%) [95.15, 98.61] |
| GPT-6.1 Sol | 1,733/1,740 (99.60%) [99.14, 99.94] | 584/591 (98.82%) [97.49, 99.83] | 577/577 (100.00%) [100.00, 100.00] | 572/572 (100.00%) [100.00, 100.00] |
| Claude Haiku 5.5 | 1,730/1,740 (99.43%) [98.91, 99.83] | 584/591 (98.82%) [97.49, 99.83] | 574/577 (99.48%) [98.60, 100.00] | 572/572 (100.00%) [100.00, 100.00] |
| GPT-6 Luna | 1,731/1,740 (99.48%) [99.02, 99.88] | 584/591 (98.82%) [97.49, 99.83] | 575/577 (99.65%) [99.12, 100.00] | 572/572 (100.00%) [100.00, 100.00] |

| model | quarter | year_to_date | full_year |
| --- | --- | --- | --- |
| step-00025 | 867/885 (97.97%) [97.00, 98.86] | 758/777 (97.55%) [96.45, 98.58] | 74/78 (94.87%) [85.54, 100.00] |
| Qwen3-8B base | 764/885 (86.33%) [83.65, 88.90] | 684/777 (88.03%) [85.36, 90.59] | 68/78 (87.18%) [76.74, 95.71] |
| GPT-6.1 Sol | 881/885 (99.55%) [99.09, 99.89] | 774/777 (99.61%) [99.11, 100.00] | 78/78 (100.00%) [100.00, 100.00] |
| Claude Haiku 5.5 | 879/885 (99.32%) [98.75, 99.78] | 773/777 (99.49%) [98.95, 99.88] | 78/78 (100.00%) [100.00, 100.00] |
| GPT-6 Luna | 880/885 (99.44%) [98.88, 99.89] | 773/777 (99.49%) [98.95, 99.88] | 78/78 (100.00%) [100.00, 100.00] |

### Differences

step-00025 minus each reference, in points, with the discordant figures (one model correct, the other not) and the companies they come from. On the time test 317 releases with graded figures come from 315 companies, so the bootstrap is in effect over releases (protocol, Uncertainty).

| difference | points | 95% interval | discordant | of which | companies |
| --- | --- | --- | --- | --- | --- |
| minus Qwen3-8B base | +10.52 | [+8.15, +12.93] | 211 | 197 fixed, 14 broken | 94 |
| minus GPT-6.1 Sol | −1.95 | [−2.94, −1.04] | 38 | 2 step-00025 right, 36 GPT-6.1 Sol right | 21 |
| minus Claude Haiku 5.5 | −1.78 | [−2.73, −0.91] | 35 | 2 step-00025 right, 33 Claude Haiku 5.5 right | 19 |
| minus GPT-6 Luna | −1.84 | [−2.82, −0.96] | 36 | 2 step-00025 right, 34 GPT-6 Luna right | 19 |

### Sensitivity: without the preferred-dividend releases

Without the 20 kept releases in preferred.csv (whole releases dropped: 98 graded figures; 1,642 left). On the time test 317 releases with graded figures come from 315 companies, so the bootstrap is in effect over releases (protocol, Uncertainty).

| model | accuracy |
| --- | --- |
| step-00025 | 1,623/1,642 (98.84%) [98.08, 99.51] |
| Qwen3-8B base | 1,441/1,642 (87.76%) [85.29, 90.17] |
| GPT-6.1 Sol | 1,637/1,642 (99.70%) [99.28, 100.00] |
| Claude Haiku 5.5 | 1,637/1,642 (99.70%) [99.28, 100.00] |
| GPT-6 Luna | 1,636/1,642 (99.63%) [99.21, 99.94] |

| difference | points | 95% interval | discordant | of which | companies |
| --- | --- | --- | --- | --- | --- |
| minus Qwen3-8B base | +11.08 | [+8.67, +13.54] | 192 | 187 fixed, 5 broken | 86 |
| minus GPT-6.1 Sol | −0.85 | [−1.60, −0.18] | 18 | 2 step-00025 right, 16 GPT-6.1 Sol right | 10 |
| minus Claude Haiku 5.5 | −0.85 | [−1.60, −0.18] | 18 | 2 step-00025 right, 16 Claude Haiku 5.5 right | 10 |
| minus GPT-6 Luna | −0.79 | [−1.54, −0.12] | 17 | 2 step-00025 right, 15 GPT-6 Luna right | 9 |

Net income on the preferred-dividend releases alone, counts only, no interval: the 28 figures come from 15 companies (0.0% of draws contain none of them). step-00025 8/28; Qwen3-8B base 12/28; GPT-6.1 Sol 28/28; Claude Haiku 5.5 25/28; GPT-6 Luna 27/28.

### Verdicts and unit slips

| verdict | step-00025 | Qwen3-8B base | GPT-6.1 Sol | Claude Haiku 5.5 | GPT-6 Luna |
| --- | --- | --- | --- | --- | --- |
| correct | 1,699 | 1,516 | 1,733 | 1,730 | 1,731 |
| missing | 0 | 0 | 0 | 0 | 1 |
| scale | 0 | 21 | 0 | 0 | 0 |
| zeros | 0 | 48 | 0 | 0 | 0 |
| wrong_period | 4 | 11 | 0 | 0 | 0 |
| noncontrolling_interest | 1 | 52 | 0 | 0 | 0 |
| rounded | 0 | 29 | 0 | 0 | 0 |
| adjusted | 6 | 23 | 1 | 2 | 2 |
| wrong_line | 30 | 36 | 6 | 8 | 6 |
| other | 0 | 4 | 0 | 0 | 0 |
| not_in_input | 52 | 52 | 52 | 52 | 52 |
| no_answer | 284 | 284 | 271 | 275 | 271 |
| unit slips | 0 | 77 | 0 | 0 | 0 |

Unit slips: rows matched with a power-of-ten factor (graded.csv's factor column). not_in_input and no_answer rows aren't graded.

### Cost and latency

| model | cost basis | cost | per 1,000 releases | per 1,000 at $1/h | latency p50 | latency p95 |
| --- | --- | --- | --- | --- | --- | --- |
| step-00025 | $2.589/h × 281.8 s | $0.2027 | $0.59 | $0.23 | 12.76 s | 16.59 s |
| Qwen3-8B base | $2.589/h × 265.8 s | $0.1912 | $0.55 | $0.21 | 12.00 s | 15.52 s |
| GPT-6.1 Sol | tokens ($2 in, $10 out per M) | $8.3417 | $24.11 | – | 1.71 s | 2.73 s |
| Claude Haiku 5.5 | tokens ($0.1 in, $0.5 out per M) | $0.6433 | $1.86 | – | 1.66 s | 3.08 s |
| GPT-6 Luna | tokens ($0.1 in, $0.5 out per M) | $0.4273 | $1.24 | – | 2.14 s | 3.59 s |

GPU runs at the price billed ($2.589/h) times the run's wall time (first call to last), and at $1/h as results/comparison.md shows GPU runs; they leave out each session's setup, model loading and idle time. API runs at their token prices (run.json's estimate). Latency is per call, reported but not compared across providers (16 calls at a time on the GPU, 4 or fewer on the APIs).

### step-00025's misses (descriptive, not pre-registered)

The rule: (1) net income whose predicted value is printed on an input row whose label names common shareholders, stockholders, shares or stock, whatever its verdict (full labels read from the input, not graded.csv's shortened evidence; table rows and prose lines); (2) a 10-Q release's year-to-date prediction equal to its quarter prediction; (3) the rest, by verdict. A miss meeting (1) and (2) goes to (2), since its value is the quarter's whichever line prints it.

41 misses: (1) 29, (2) 1, (3) adjusted 3, noncontrolling_interest 1, wrong_line 4, wrong_period 3.

Group (1): 29 of the 41 misses, from 16 companies (ALB, CMS, DOC, GOOGL, HPE, KDP, KIM, LHX, MCHP, NRG, PSA, SMCI, T, TDG, VMRK, WMB); verdicts adjusted 3, wrong_line 26. 20 are in preferred.csv releases and 9 outside it: HPE, KDP, LHX print preferred-stock lines (a row naming preferred) that the XBRL preferred-dividend check missed; DOC, TDG report income after participating securities (a row naming them). Group (1) includes 11 of the 14 figures broken by training (the base model's correct). The sensitivity analysis drops only preferred.csv's releases, so it understates this failure.

Group (2): 1 from TWLO: the quarter copied into year to date in second- or third-quarter releases whose income statement has quarter columns only, with the year to date printed elsewhere (the row where the input prints the year-to-date answer is listed).

| group | ticker | release | field | period | answer | predicted | verdict | Qwen3-8B base verdict | row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| common shareholders' line (preferred.csv) | ALB | 0000915913-26-000101 | net_income | quarter | 479959000 | 438272000 | wrong_line | correct | Net income (loss) attributable to Albemarle Corporation common shareholders |
| common shareholders' line (preferred.csv) | ALB | 0000915913-26-000101 | net_income | year_to_date | 799050000 | 715675000 | wrong_line | correct | Net income (loss) attributable to Albemarle Corporation common shareholders |
| common shareholders' line (preferred.csv) | CMS | 0001104659-26-087288 | net_income | quarter | 120000000 | 117000000 | wrong_line | correct | Net Income Available to Common Stockholders |
| common shareholders' line (preferred.csv) | CMS | 0001104659-26-087288 | net_income | year_to_date | 460000000 | 455000000 | wrong_line | correct | Net Income Available to Common Stockholders |
| common shareholders' line | DOC | 0001628280-26-052608 | net_income | quarter | 52818000 | 52668000 | wrong_line | wrong_line | Net income (loss) applicable to common shares |
| common shareholders' line | DOC | 0001628280-26-052608 | net_income | year_to_date | 246451000 | 246152000 | wrong_line | wrong_line | Net income (loss) applicable to common shares |
| common shareholders' line (preferred.csv) | GOOGL | 0001652044-26-000066 | net_income | quarter | 112193000000 | 112107000000 | wrong_line | wrong_line | Net income available to common stockholders |
| common shareholders' line (preferred.csv) | GOOGL | 0001652044-26-000066 | net_income | year_to_date | 174771000000 | 174685000000 | wrong_line | wrong_line | Net income available to common stockholders |
| common shareholders' line | HPE | 0001645590-26-000078 | net_income | quarter | 1540000000 | 1511000000 | wrong_line | correct | Net earnings attributable to common stockholders |
| common shareholders' line | HPE | 0001645590-26-000078 | net_income | year_to_date | 2616000000 | 2529000000 | wrong_line | wrong_line | Net earnings (loss) attributable to common stockholders |
| common shareholders' line | KDP | 0001418135-26-000048 | net_income | quarter | 142000000 | 60000000 | adjusted | adjusted | GAAP net income attributable to common shareholders decreased 89.0% to $60 million, or $0.04 per diluted share, primarily driven by an unfavorable year-over-year impact of items a… |
| common shareholders' line | KDP | 0001418135-26-000048 | net_income | year_to_date | 412000000 | 330000000 | wrong_line | wrong_line | Net income attributable to common shareholders |
| common shareholders' line (preferred.csv) | KIM | 0001193125-26-331498 | net_income | quarter | 153294000 | 145758000 | wrong_line | wrong_line | Net income available to the company's common shareholders |
| common shareholders' line (preferred.csv) | KIM | 0001193125-26-331498 | net_income | year_to_date | 318192000 | 303120000 | wrong_line | wrong_line | Net income available to the company's common shareholders |
| common shareholders' line | LHX | 0000202058-26-000054 | net_income | quarter | 600000000 | 586000000 | wrong_line | correct | Net income available to common shareholders |
| common shareholders' line | LHX | 0000202058-26-000054 | net_income | year_to_date | 1112000000 | 1098000000 | wrong_line | correct | Net income available to common shareholders |
| common shareholders' line (preferred.csv) | MCHP | 0000827054-26-000037 | net_income | quarter | 229800000 | 202000000.0 | adjusted | adjusted | •On a GAAP basis: gross profit of 63.2%; operating income of $336.8 million and 22.7% of net sales; net income attributable to common stockholders of $202.0 million; and EPS of $0… |
| common shareholders' line (preferred.csv) | NRG | 0001013871-26-000018 | net_income | quarter | 506000000 | 489000000 | wrong_line | correct | Net Income/(Loss) Available for Common Stockholders |
| common shareholders' line (preferred.csv) | NRG | 0001013871-26-000018 | net_income | year_to_date | 631000000 | 597000000 | adjusted | correct | Net Income/(Loss) Available for Common Stockholders |
| common shareholders' line (preferred.csv) | PSA | 0001628280-26-050608 | net_income | quarter | 499960000 | 450258000 | wrong_line | correct | Net income allocable to common shareholders |
| common shareholders' line (preferred.csv) | PSA | 0001628280-26-050608 | net_income | year_to_date | 1026233000 | 927046000 | wrong_line | correct | Net income allocable to common shareholders |
| common shareholders' line (preferred.csv) | SMCI | 0001375365-26-000021 | net_income | quarter | 1178217000 | 1165129000 | wrong_line | wrong_line | Net income attributable to common stockholders |
| common shareholders' line (preferred.csv) | SMCI | 0001375365-26-000021 | net_income | full_year | 2230453000 | 2217365000 | wrong_line | wrong_line | Net income attributable to common stockholders |
| common shareholders' line (preferred.csv) | T | 0000732717-26-000294 | net_income | quarter | 4627000000 | 4591000000 | wrong_line | noncontrolling_interest | Net Income Attributable to Common Stock |
| common shareholders' line (preferred.csv) | T | 0000732717-26-000294 | net_income | year_to_date | 8456000000 | 8384000000 | wrong_line | noncontrolling_interest | Net Income Attributable to Common Stock |
| common shareholders' line | TDG | 0001260221-26-000052 | net_income | year_to_date | 1519000000 | 1460000000 | wrong_line | noncontrolling_interest | NET INCOME APPLICABLE TO TD GROUP COMMON STOCKHOLDERS |
| common shareholders' line (preferred.csv) | VMRK | 0001193125-26-312445 | net_income | quarter | 114135000 | 113780000 | wrong_line | noncontrolling_interest | Net income available to Common Shares |
| common shareholders' line (preferred.csv) | VMRK | 0001193125-26-312445 | net_income | year_to_date | 204214000 | 203503000 | wrong_line | noncontrolling_interest | Net income available to Common Shares |
| common shareholders' line (preferred.csv) | WMB | 0000107263-26-000024 | net_income | year_to_date | 1692000000 | 1691000000 | wrong_line | wrong_line | Net income (loss) available to common stockholders |
| quarter copied into year to date | TWLO | 0001447669-26-000088 | net_income | year_to_date | 1157348000 | 1067209000 | wrong_period | wrong_period | (Unaudited) › CASH FLOWS FROM OPERATING ACTIVITIES: › Net income |
| adjusted | ELV | 0001156039-26-000058 | revenue | quarter | 50474000000 | 49826000000 | adjusted | adjusted | 'Total operating revenue' (heading '•Reconciliation of Non-GAAP Financial Measures'): 49,826 (x1,000,000) |
| adjusted | ELV | 0001156039-26-000058 | revenue | year_to_date | 100655000000 | 99320000000 | adjusted | adjusted | 'Total operating revenue' (heading '•Reconciliation of Non-GAAP Financial Measures'): 99,320 (x1,000,000) |
| adjusted | NTAP | 0001002047-26-000003 | revenue | quarter | 1819000000 | 2025000000 | adjusted | adjusted | 'Net revenues' (section 'Non-GAAP Results'): $2,025 (x1,000,000) |
| noncontrolling_interest | HLT | 0001585689-26-000042 | net_income | year_to_date | 867000000 | 865000000 | noncontrolling_interest | noncontrolling_interest | ProfitLoss 2026-01-01..2026-06-30: 865000000 |
| wrong_line | MPC | 0001510295-26-000060 | revenue | quarter | 51994000000 | 52337000000 | wrong_line | correct | 'Total revenues and other income': 52,337 (x1,000,000) |
| wrong_line | MPC | 0001510295-26-000060 | revenue | year_to_date | 86194000000 | 86905000000 | wrong_line | correct | 'Total revenues and other income': 86,905 (x1,000,000) |
| wrong_line (preferred.csv) | OXY | 0001628280-26-053377 | revenue | quarter | 8065000000 | 5738000000 | wrong_line | correct | 'Total': 5,738 (x1,000,000) |
| wrong_line (preferred.csv) | OXY | 0001628280-26-053377 | revenue | year_to_date | 13295000000 | 13436000000 | wrong_line | wrong_line | 'Total': 13,436 (x1,000,000) |
| wrong_period | LRCX | 0000707549-26-000033 | revenue | full_year | 23232690000 | 6722238000 | wrong_period | wrong_period | 'Three Months Ended June 28, 2026' column of 'Revenue': $6,722,238 (x1,000) |
| wrong_period | LRCX | 0000707549-26-000033 | net_income | full_year | 7265396000 | 2277282000 | wrong_period | wrong_period | 'Three Months Ended June 28, 2026' column of 'Net income': $2,277,282 (x1,000) |
| wrong_period | LRCX | 0000707549-26-000033 | eps_diluted | full_year | 5.76 | 1.81 | wrong_period | wrong_period | 'Three Months Ended June 28, 2026' column of 'Diluted': $1.81 |

### Abstention (descriptive, not pre-registered)

| model | not_in_input figures answered | null figures of all predicted |
| --- | --- | --- |
| step-00025 | 52 of 52 | 0 of 2,076 |
| Qwen3-8B base | 52 of 52 | 0 of 2,076 |
| GPT-6.1 Sol | 30 of 52 | 35 of 2,076 |
| Claude Haiku 5.5 | 34 of 52 | 27 of 2,076 |
| GPT-6 Luna | 36 of 52 | 30 of 2,076 |

A not_in_input figure has an XBRL answer that the grader doesn't find in the input in any of its printed forms. Where the release doesn't report the figure (a year to date in a release whose tables have quarter columns only), the prompt's "null when the release doesn't report a figure" applies. But many such releases do print it, in a form the grader doesn't search (an EPS of $.13 for 0.13) or as another value on the line the prompt asks for (rounded, revised, or a total that differs from XBRL's), and then answering is right. So the answered count is an upper bound on missed abstentions, not a count of them. Accuracy leaves these figures out, so it doesn't reward abstaining. Training had no null labels (PLAN.md, Limitation: no nulls in training).

## Validation, for reference

Counts only, no intervals: the 60 validation releases' graded figures, then without ORCL's 3 preferred-dividend releases.

| model | run | all | without preferred |
| --- | --- | --- | --- |
| step-00025 | qwen3-8b-2k--step-00025 | 287/293, chosen on validation (the checkpoint was picked for that score, so it may overstate; the test sets give the estimate) | 278/279 |
| Qwen3-8B base | qwen3-8b-2k--base | 262/293 | 248/279 |
| GPT-6.1 Sol | 2026-11-07T105947_gpt-6.1-sol_extract_v2 | 293/293 | 279/279 |
| Claude Haiku 5.5 | 2026-10-17T222907_claude-haiku-5-5_extract_v2 | 290/293 | 277/279 |
| GPT-6 Luna | 2026-10-27T001339_gpt-6-luna_extract_v2 | 290/293 | 279/279 |

The qwen3-8b-2k validation runs record $1/h; they ran on a Vast H100 NVL billed $2.776/h (PLAN.md Progress 2026-11-02), not on stage 1's A100.

GPT-6.1 Sol's first validation attempt, 2026-11-07T104935_gpt-6.1-sol_extract_v2, stopped (aborted) on HTTP 401 (a revoked key) with 0 replies; it was not a test call. The validation reference is the run after it.

## Bootstrap

As the protocol states (Uncertainty): 10,000 draws of companies with replacement from numpy's default_rng(20261105), one set per split shared by every model, subset and difference; a difference is computed within each draw. These details were chosen on 2026-11-07, after the test runs were graded (Deviations, 4): the companies are the split's companies with at least one graded figure, sorted by ticker; draws = rng.integers(0, n, size=(10000, n)); each draw's accuracy = Σ(times drawn × correct) / Σ(times drawn × graded); intervals = np.percentile(values, [2.5, 97.5]). In a subset, a company with no figures left adds 0 to both sums; a draw with no graded figure would stop the report (none did), never be dropped.

Robustness (descriptive): the primary difference's interval with the companies in other orders, company test: in links.csv order or by CIK [+5.00, +15.67] (by ticker [+5.08, +15.58]); time test: in links.csv order or by CIK [+8.16, +13.01] (by ticker [+8.15, +12.93]). The bounds move by at most 0.09 points.

## Deviations from the protocol

No amendment was made (none are allowed after the first test call); these are recorded here instead.

1. The time test. The protocol set its cutoff for when the Q3 2026 releases are added, and said the provisional 346-release time test would not be scored. Decision 2026-11-07 (commit e352c14) froze that same set (releases filed 2026-07-01 to 2026-09-30) at 12:20 UTC on 2026-11-07: after the company-test runs (11:16-11:31 UTC) and before any time-test call (12:32 UTC). No amendment records it, and none were allowed. Every time-test result was descriptive under the protocol anyway.
2. The company test's GPU price. The protocol prices GPU runs at the $/h given to vast_setup.sh (run.json's gpu_hourly_usd). Both company-test GPU runs record 2.776, an example price passed to vast_setup.sh by mistake, which is the price by the protocol's letter; the instance was billed $2.538/h, which the costs here use (both GPU runs' costs at the recorded price are also given, under Cost and latency). The time test's price, $2.589/h, was recorded correctly.
3. GPT-6.1 Sol's time-test rerun ran 2 calls at a time, not the protocol's 4 (Cost and latency): its first attempt hit the 1,000,000 tokens-per-minute limit at 4 (Time test, Coverage).
4. The bootstrap's details were fixed on 2026-11-07, after the test runs were graded (Bootstrap, with its robustness line).

## Notes

- CCI 0001051470-26-000052, a frozen company-test release, is a divestiture and outlook announcement, not results; none of its 3 answers is in its input. Its period_end warning appears in 4 of the 5 runs, not step-00025's.
- Logs: company test training/qwen3-8b-2k/test/server-company_test.log and training/qwen3-8b-2k/test/setup-company_test.log; time test training/qwen3-8b-2k/test/server.log and training/qwen3-8b-2k/test/setup-time_test.log.
