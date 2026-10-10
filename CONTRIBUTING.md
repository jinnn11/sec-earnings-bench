# Contributing

## Data access
- SEC requests send a User-Agent with a name and email (SEC_USER_AGENT in .env).
- Stay under 10 requests per second to SEC endpoints.
- Cache every download under data/raw/; never fetch the same URL twice.
- Tests never touch the network; they use saved fixtures in tests/fixtures/.

## Code
- Python 3.11 with uv. Run tests with `uv run pytest`.
- Changes to parsing, linking, matching or grading need tests.
- Never change the grader to raise a score. Grader changes get their own commit with a reason.
- Save raw model outputs for every evaluation run under runs/.