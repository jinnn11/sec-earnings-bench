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

Linking fixtures (`tests/test_link.py`), built 2026-10-10 from the cache (`data/raw/`)
and the saved releases (`data/releases/`) of the 10-company release run:

| File | Contents |
| --- | --- |
| `pressrelease062326.htm` | https://www.sec.gov/Archives/edgar/data/320187/000032018726000070/pressrelease062326.htm, byte for byte: NKE's 2026-06-23 Item 2.02 8-K exhibit, a CFO-transition announcement with no tables |
| `link_releases.json` | All 61 saved releases: each `record` is the saved `<accession>.json` unchanged; `markdown` is the first table of the chosen exhibit's markdown that `link.check_earnings` accepts on its own (`markdown_is_excerpt: true`), or the whole markdown when it accepts none (only NKE 0000320187-26-000070) |
| `link_submissions.json` | Responses by URL. `company_tickers.json`: the 10 tickers' entries (original keys). One submissions JSON per CIK (the 10 tickers plus XOM's predecessor, 34088): every top-level field kept; `filings.recent` holds every row from `recent` and the older-filings files whose form contains "10-Q" or "10-K" and that was filed on or after 2025-01-01 (every column kept); `filings.files` is set to `[]` |

The filter accepts a release if any one of its tables qualifies, so trimming a
release to the table it accepted gives the same verdict as the whole markdown.
`tests/fixtures/links_handchecked.csv` is hand-checked and maintained by hand.
