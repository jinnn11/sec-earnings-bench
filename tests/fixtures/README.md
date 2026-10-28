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
| `0000065984-23-000090-index.htm` | https://www.sec.gov/Archives/edgar/data/65984/000006598423000090/0000065984-23-000090-index.htm (fetched 2026-10-28) | Filing index of ETR's Q3 2023 release, filed jointly with its subsidiaries: every document is linked under a co-registrant's CIK (7323), same accession |

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

Answer-key fixtures (`tests/test_answers.py`), built 2026-10-11 from the cache
(company facts fetched 2026-10-11) and `data/links.csv`:

| File | Contents |
| --- | --- |
| `answers_links.csv` | `data/links.csv` from the linking run, unchanged (61 rows, 60 linked) |
| `answers_companyfacts.json` | Responses by URL. `company_tickers.json`: the 10 tickers' entries (original keys). One company facts JSON (https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json) per CIK (the 10 tickers plus XOM's predecessor, 34088): top-level fields kept; `facts` cut to `us-gaap`, and within it to the answer tags (Revenues, RevenueFromContractWithCustomerExcludingAssessedTax, NetIncomeLoss, NetIncomeLossAvailableToCommonStockholdersBasic, ProfitLoss, EarningsPerShareDiluted, EarningsPerShareBasicAndDiluted), NetIncomeLossAttributableToNoncontrollingInterest (ProfitLoss's zero condition), the preferred-dividend tags (PreferredStockDividendsIncomeStatementImpact, DividendsPreferredStock, DividendsPreferredStockCash, PreferredStockDividendsAndOtherAdjustments; available-to-common's condition) and tags named `Revenue...`/`SalesRevenue...`; each tag keeps its label and description and only the facts whose `accn` is a linked filing in `answers_links.csv` or the third-quarter 10-Q of a linked 10-K (every fact field kept, SEC's order). For an answer tag or NetIncomeLossAttributableToNoncontrollingInterest, each unit with none keeps its first fact so it is still known (CAT's NetIncomeLoss is only in a DEF 14A; NFLX's NetIncomeLossAvailableToCommonStockholdersBasic is in no linked filing). Rebuilt 2026-10-15 from the same cache to add the third-quarter 10-Qs that are no linked filing, ORCL 0000950170-25-037143 and NKE 0000320187-25-000016 (25 facts added, nothing else changed; the same rules on the linked filings alone give the 2026-10-11 file byte for byte). Rebuilt 2026-10-27 from the same cache to add the preferred-dividend tags: 10 ORCL facts (PreferredStockDividendsIncomeStatementImpact 9, DividendsPreferredStockCash 1); the other tags have none in these filings, and nothing else changed (additions only) |

The answers tests also serve `link_submissions.json` (see linking fixtures), from
which `secbench.answers` finds each 10-K's third-quarter 10-Q. With both,
`secbench.answers` gives the same report and the same `answers.csv` on these fixtures
as on the full cache (rechecked 2026-10-15, 293 rows). Values are written back as Python floats; every kept value was
checked to parse to the same decimal as SEC's text.
`tests/fixtures/answers_handchecked.csv` is hand-checked and maintained by hand.

Release-input fixtures (`tests/test_inputs.py`, `tests/test_grade.py`), copied
2026-10-11 byte for byte from `data/releases/`: `input_releases/{cik10}/` holds five
saved releases, each record (`<accession>.json`) with every exhibit's markdown.

| Release | Why |
| --- | --- |
| XOM `0002115436-26-000006` (Q2 2026) | EX-99.1 has no income statement (only a summary table that passes the link filter); its statements are in EX-99.2, which the input appends |
| AAPL `0000320193-25-000055` (Q2 FY25) | Statements in EX-99.1; three- and six-month columns |
| CAT `0000018230-25-000037` (Q2 2025) | Statements in EX-99.1, plus an EX-99.2 that is not appended |
| MSFT `0000950170-25-100226` (FY25) | A fourth-quarter release linked to a 10-K |
| WMT `0000104169-25-000069` (Q1 FY26) | A real noncontrolling interest: "Consolidated net income" 4,639 (ProfitLoss) beside 4,487 attributable to Walmart |

Company list fixture (`tests/test_companies.py`), made 2026-10-27:
`sp500_snapshot.html` is Wikipedia's "List of S&P 500 companies" at revision
1382097112 (2026-10-26T16:05:24Z,
https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies&oldid=1382097112),
cut to its `constituents` table, with the header row and 23 rows copied byte for byte:
the 10 validation companies, three share-class pairs (GOOGL/GOOG, FOXA/FOX,
NWSA/NWS), three Financials (BRK.B, JPM, V) and ABT, AMD, APD and LIN. The tests also
read the tracked `data/companies.csv` to check that its split reproduces.
