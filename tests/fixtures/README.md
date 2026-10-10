# Fixtures

Trimmed copies of real SEC responses, fetched 2026-10-10. The JSON is re-indented;
keys, values and order are as SEC served them.

| File | Source | Trimmed to |
| --- | --- | --- |
| `company_tickers.json` | https://www.sec.gov/files/company_tickers.json | 7 of 10,435 entries: NVDA, AAPL, GOOGL, BRK-B, XOM, GOOG, BRK-A (original keys kept) |
| `CIK0000320193.json` | https://data.sec.gov/submissions/CIK0000320193.json | all metadata; `filings.recent` cut from 1,001 to 11 rows, every column kept |
| `CIK0000320193-submissions-001.json` | https://data.sec.gov/submissions/CIK0000320193-submissions-001.json | 6 of 1,264 rows, every column kept |

`filings.files` in `CIK0000320193.json` is untouched, so its `filingCount` (1264)
describes the full older file, not the trimmed one.

Filing documents, fetched 2026-10-10 and saved byte for byte as SEC served them
(including the `<DOCUMENT>...<TEXT>` wrapper and the script tag SEC appends):

| File | Source | Why |
| --- | --- | --- |
| `0000320193-26-000018-index.htm` | https://www.sec.gov/Archives/edgar/data/320193/000032019326000018/0000320193-26-000018-index.htm | Filing index for AAPL's Q3 FY26 release (an Item 2.02 8-K in the submissions fixture) |
| `a8-kex991q3202606272026.htm` | https://www.sec.gov/Archives/edgar/data/320193/000032019326000018/a8-kex991q3202606272026.htm | Its EX-99.1: three- and nine-month columns |
| `a8-kex991q4202509272025.htm` | https://www.sec.gov/Archives/edgar/data/320193/000032019325000077/a8-kex991q4202509272025.htm | EX-99.1 of AAPL's Q4 FY25 release (0000320193-25-000077): three-month and full-year columns side by side |

Submission rows were chosen to cover: Item 2.02 8-Ks, 10-Q, 10-K, 8-K/A, Form 4, Form 144 and
424B2 with a blank `reportDate`, a DEF 14A whose `reportDate` is after its
`filingDate`, an S-8 with a blank `primaryDocument`, and a 1999 8-K with old-style
item `"2"`.
