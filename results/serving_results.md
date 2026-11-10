# Serving benchmark: qwen3-8b-2k step-00025

Qwen/Qwen3-8B with the step-00025 adapter, served by vLLM 0.32.0 on NVIDIA H100 NVL, 95830 MiB, 580.173.02, at $3.429 per GPU-hour; session 2026-11-10T02:08:39+00:00 to 2026-11-10T04:06:59+00:00, commit 3624f3c62c7e. Config configs/serving/qwen3-8b-2k.toml (sha256 3f8153a70aca). Prefix caching off, --max-model-len 40,960; vLLM's defaults otherwise.

## Variants

| variant | status | served as | model | note |
| --- | --- | --- | --- | --- |
| adapter | complete | qwen3-8b-2k--step-00025 | base + adapter |  |
| merged | complete | qwen3-8b-2k--step-00025--merged | merged |  |
| fp8 | complete | qwen3-8b-2k--step-00025--fp8 | merged |  |
| int4-awq | complete | qwen3-8b-2k--step-00025--int4-awq | int4-awq |  |

## Memory (vLLM's start-up log)

| variant | weights (GiB) | KV cache (GiB) | KV cache (tokens) | max concurrency at 40,960 tokens | model on disk (GiB) | launch to healthy (s) |
| --- | --- | --- | --- | --- | --- | --- |
| adapter | 15.36 | 66.84 | 486,720 | 11.88x | – | 172 |
| merged | 15.27 | 67.28 | 489,952 | 11.96x | 15.26 | 64 |
| fp8 | 9.07 | 73.48 | 535,088 | 13.06x | 15.26 | 74 |
| int4-awq | 5.78 | 76.71 | 558,608 | 13.64x | 5.68 | 64 |

## Throughput and latency (time_test prompts, 128 per level after 8 warm-up)

Per level: the same prompts, that many requests in flight. Cost per 1,000 releases: the GPU's price for the makespan ($3.429/h, and $1/h).

| variant | in flight | ok | makespan (s) | releases/h | output tok/s | prompt tok/s | latency p50 / p95 / p99 (s) | first token p50 / p95 (s) | decode tok/s (median) | $/1,000 at $3.429/h | $/1,000 at $1/h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adapter | 1 | 128/128 | 228.7 | 2,015 | 74 | 7,257 | 1.73 / 2.47 / 2.85 | 0.59 / 1.32 | 118 | 1.702 | 0.496 |
| adapter | 2 | 128/128 | 159.5 | 2,888 | 107 | 10,403 | 2.38 / 3.50 / 4.30 | 0.66 / 1.62 | 79 | 1.187 | 0.346 |
| adapter | 4 | 128/128 | 129.1 | 3,570 | 132 | 12,859 | 3.91 / 5.70 / 5.94 | 0.79 / 2.25 | 44 | 0.961 | 0.280 |
| adapter | 8 | 128/128 | 113.5 | 4,061 | 150 | 14,628 | 6.95 / 9.96 / 12.81 | 1.16 / 3.00 | 23 | 0.844 | 0.246 |
| adapter | 16 | 128/128 | 106.1 | 4,345 | 161 | 15,649 | 12.93 / 18.64 / 21.16 | 2.15 / 8.44 | 12 | 0.789 | 0.230 |
| adapter | 32 | 128/128 | 102.6 | 4,491 | 166 | 16,177 | 25.32 / 41.09 / 45.49 | 3.06 / 18.95 | 6 | 0.763 | 0.223 |
| adapter | 64 | 128/128 | 102.7 | 4,489 | 166 | 16,168 | 49.09 / 71.60 / 75.55 | 24.35 / 44.74 | 5 | 0.764 | 0.223 |
| merged | 1 | 128/128 | 189.0 | 2,438 | 90 | 8,782 | 1.43 / 2.10 / 2.49 | 0.54 / 1.18 | 151 | 1.406 | 0.410 |
| merged | 2 | 128/128 | 138.5 | 3,326 | 123 | 11,982 | 2.08 / 3.07 / 3.80 | 0.63 / 1.49 | 94 | 1.031 | 0.301 |
| merged | 4 | 128/128 | 114.3 | 4,032 | 149 | 14,523 | 3.49 / 5.32 / 5.38 | 0.69 / 1.92 | 51 | 0.850 | 0.248 |
| merged | 8 | 128/128 | 102.4 | 4,501 | 166 | 16,212 | 6.33 / 8.57 / 9.37 | 1.10 / 2.50 | 26 | 0.762 | 0.222 |
| merged | 16 | 128/128 | 95.9 | 4,804 | 178 | 17,303 | 11.58 / 16.28 / 19.09 | 1.87 / 7.63 | 13 | 0.714 | 0.208 |
| merged | 32 | 128/128 | 93.2 | 4,945 | 183 | 17,811 | 23.13 / 37.10 / 41.38 | 3.09 / 16.98 | 7 | 0.693 | 0.202 |
| merged | 64 | 128/128 | 93.8 | 4,915 | 182 | 17,703 | 44.14 / 65.37 / 71.86 | 21.33 / 42.11 | 5 | 0.698 | 0.203 |
| fp8 | 1 | 128/128 | 139.5 | 3,303 | 122 | 11,899 | 1.04 / 1.62 / 1.95 | 0.41 / 0.94 | 210 | 1.038 | 0.303 |
| fp8 | 2 | 128/128 | 105.3 | 4,375 | 162 | 15,759 | 1.57 / 2.43 / 2.75 | 0.53 / 1.34 | 130 | 0.784 | 0.229 |
| fp8 | 4 | 128/128 | 89.3 | 5,160 | 191 | 18,587 | 2.68 / 4.24 / 4.33 | 0.57 / 1.74 | 66 | 0.665 | 0.194 |
| fp8 | 8 | 128/128 | 81.0 | 5,691 | 210 | 20,500 | 4.95 / 7.09 / 8.42 | 0.86 / 3.04 | 33 | 0.602 | 0.176 |
| fp8 | 16 | 128/128 | 76.7 | 6,010 | 222 | 21,648 | 9.32 / 13.02 / 15.16 | 1.46 / 6.01 | 17 | 0.571 | 0.166 |
| fp8 | 32 | 128/128 | 75.0 | 6,143 | 227 | 22,127 | 18.82 / 28.43 / 31.52 | 2.26 / 13.73 | 8 | 0.558 | 0.163 |
| fp8 | 64 | 128/128 | 75.3 | 6,118 | 226 | 22,037 | 35.82 / 54.96 / 58.94 | 15.11 / 34.18 | 6 | 0.560 | 0.163 |
| int4-awq | 1 | 128/128 | 170.3 | 2,706 | 100 | 9,747 | 1.27 / 1.98 / 2.36 | 0.52 / 1.21 | 180 | 1.267 | 0.370 |
| int4-awq | 2 | 128/128 | 128.8 | 3,578 | 132 | 12,889 | 1.92 / 3.17 / 3.37 | 0.58 / 1.58 | 104 | 0.958 | 0.279 |
| int4-awq | 4 | 128/128 | 109.8 | 4,197 | 155 | 15,118 | 3.28 / 4.99 / 5.44 | 0.79 / 2.17 | 55 | 0.817 | 0.238 |
| int4-awq | 8 | 128/128 | 100.0 | 4,609 | 171 | 16,601 | 6.09 / 8.57 / 11.50 | 0.99 / 3.44 | 27 | 0.744 | 0.217 |
| int4-awq | 16 | 128/128 | 95.6 | 4,822 | 178 | 17,367 | 11.65 / 16.16 / 19.19 | 1.96 / 7.78 | 14 | 0.711 | 0.207 |
| int4-awq | 32 | 128/128 | 93.3 | 4,938 | 183 | 17,788 | 23.42 / 36.38 / 41.01 | 3.23 / 17.44 | 7 | 0.694 | 0.202 |
| int4-awq | 64 | 128/128 | 93.8 | 4,915 | 182 | 17,704 | 44.15 / 70.89 / 75.05 | 17.89 / 43.20 | 5 | 0.698 | 0.203 |

## Best throughput

| variant | in flight | releases/h | latency p95 (s) | $/1,000 releases at $3.429/h | at $1/h |
| --- | --- | --- | --- | --- | --- |
| adapter | 32 | 4,491 | 41.09 | 0.763 | 0.223 |
| merged | 32 | 4,945 | 37.10 | 0.693 | 0.202 |
| fp8 | 32 | 6,143 | 28.43 | 0.558 | 0.163 |
| int4-awq | 32 | 4,938 | 36.38 | 0.694 | 0.202 |

## Accuracy on validation

| variant | correct | revenue | net_income | eps_diluted | not correct | fallbacks | same as adapter | same as stage 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adapter | 287/293 (98.0%) | 102/102 | 96/102 | 89/89 | noncontrolling_interest 1, wrong_line 5 | 0 | (itself) | 360/360 figures, 60/60 releases |
| merged | 287/293 (98.0%) | 102/102 | 96/102 | 89/89 | noncontrolling_interest 1, wrong_line 5 | 0 | 360/360 figures, 60/60 releases | 360/360 figures, 60/60 releases |
| fp8 | 285/293 (97.3%) | 100/102 | 96/102 | 89/89 | noncontrolling_interest 1, wrong_line 7 | 0 | 358/360 figures, 59/60 releases | 358/360 figures, 59/60 releases |
| int4-awq | 280/293 (95.6%) | 96/102 | 95/102 | 89/89 | noncontrolling_interest 2, wrong_line 11 | 0 | 353/360 figures, 56/60 releases | 353/360 figures, 56/60 releases |

Stage 2's run of the checkpoint: 287/293 (98.0%).
Merge check: merged's predictions equal adapter's in 360 of 360 figures (60 of 60 releases identical).

## The time test (descriptive)

Its stage 2 results are reported and no variant is chosen on it; this shows how far each variant's predictions move from stage 2's.

| variant | correct | revenue | net_income | eps_diluted | not correct | fallbacks | same as adapter | same as stage 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adapter | 1700/1740 (97.7%) | 584/591 | 545/577 | 571/572 | adjusted 6, noncontrolling_interest 1, wrong_line 29, wrong_period 4 | 0 | (itself) | 2069/2076 figures, 343/346 releases |
| merged | 1698/1740 (97.6%) | 581/591 | 547/577 | 570/572 | adjusted 5, noncontrolling_interest 1, wrong_line 32, wrong_period 4 | 0 | 2066/2076 figures, 340/346 releases | 2065/2076 figures, 341/346 releases |
| fp8 | 1698/1740 (97.6%) | 582/591 | 546/577 | 570/572 | adjusted 6, noncontrolling_interest 1, wrong_line 31, wrong_period 4 | 1 | 2061/2076 figures, 340/346 releases | 2062/2076 figures, 340/346 releases |
| int4-awq | 1699/1740 (97.6%) | 579/591 | 550/577 | 570/572 | adjusted 5, noncontrolling_interest 1, wrong_line 31, wrong_period 4 | 1 | 2034/2076 figures, 328/346 releases | 2032/2076 figures, 328/346 releases |

Stage 2's run of the checkpoint: 1699/1740 (97.6%).

## Session overhead

| step (setup log) | minutes |
| --- | --- |
| GPU, driver and disk | 0.0 |
| Unpack /workspace/gpu-bundle-3624f3c62c7e-serving.tar | 0.0 |
| Install uv 0.13.0, Python and the train and serve groups | 0.5 |
| Install the train and quantize groups in .venv-quantize | 1.2 |
| Download the base model at training/qwen3-8b-2k's revision | 7.5 |
| Serving benchmark (secbench.servebench): every variant of the config | 118.6 |
| Pack the results (exit status 0) | – |

Server starts (launch to healthy): adapter 172 s, merged 64 s, fp8 74 s, int4-awq 64 s. Model builds: merged 34 s, int4-awq 2050 s. Together 41.0 min, $2.34 at $3.429/h; the install and download are in the setup log's steps.
