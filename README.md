# sec-earnings-bench

A benchmark that measures how accurately large language models extract GAAP headline
figures (revenue, net income and diluted EPS) from US earnings press releases, the
EX-99.1 exhibits of 8-K filings under Item 2.02. Answers are graded automatically
against the XBRL facts the company later reports in the linked 10-Q or 10-K, with
error types such as scale, sign, wrong period and non-GAAP confusion tracked
separately. The project then fine-tunes a small open model with LoRA and compares its
accuracy, speed and cost with frontier APIs.
