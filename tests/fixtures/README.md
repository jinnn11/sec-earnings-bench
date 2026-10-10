# Fixtures

Trimmed copies of real SEC responses, fetched 2026-10-10. The JSON is re-indented;
keys, values and order are as SEC served them.

| File | Source | Trimmed to |
| --- | --- | --- |
| `company_tickers.json` | https://www.sec.gov/files/company_tickers.json | 6 of 10,435 entries: NVDA, AAPL, GOOGL, BRK-B, GOOG, BRK-A (original keys kept) |
| `CIK0000320193.json` | https://data.sec.gov/submissions/CIK0000320193.json | all metadata; `filings.recent` cut from 1,001 to 11 rows, every column kept |
| `CIK0000320193-submissions-001.json` | https://data.sec.gov/submissions/CIK0000320193-submissions-001.json | 6 of 1,264 rows, every column kept |

`filings.files` in `CIK0000320193.json` is untouched, so its `filingCount` (1264)
describes the full older file, not the trimmed one.

Rows were chosen to cover: Item 2.02 8-Ks, 10-Q, 10-K, 8-K/A, Form 4, Form 144 and
424B2 with a blank `reportDate`, a DEF 14A whose `reportDate` is after its
`filingDate`, an S-8 with a blank `primaryDocument`, and a 1999 8-K with old-style
item `"2"`.
