"""Forward vs trailing P/E, fetched from Yahoo Finance -- flags when a
stock's forward multiple is priced for earnings growth its trailing
results don't show yet (or worse, priced for earnings to SHRINK), a
valuation-stretch signal distinct from anything price-action based
elsewhere in this project.

Neither Tiingo's price API (used everywhere else in this project) nor its
base fundamentals tier carries forward-looking consensus EPS estimates --
that's specialist analyst-estimate data, which Yahoo's quoteSummary
endpoint happens to expose for free. Unlike the chart API this project
already relies on for OHLCV (data.py, cross_validate.py, split_guard.py),
quoteSummary now requires a session cookie + CSRF "crumb" (Yahoo locked
this endpoint down since the chart API's usage here was first
established) -- fetched once per session/batch and reused, not re-fetched
per ticker. This is a more fragile dependency than the chart API: an
undocumented auth flow that could change or start rate-limiting a 400+
ticker nightly batch without notice. Missing/failed data is always
treated as "no data" (neutral), never as a bad valuation, precisely
because of that fragility.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_CRUMB_COOKIE_URL = "https://fc.yahoo.com"
_CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
_QUOTE_SUMMARY_URL = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
_FETCH_WORKERS = 8


def _new_session() -> tuple[requests.Session, str]:
    """A fresh session + crumb, needed once per batch (not per ticker) --
    quoteSummary rejects requests without a valid crumb tied to the
    session's own cookies."""
    session = requests.Session()
    session.headers.update(_HEADERS)
    session.get(_CRUMB_COOKIE_URL, timeout=20)
    crumb = session.get(_CRUMB_URL, timeout=20).text
    return session, crumb


def _extract_pe(field: dict | None) -> float | None:
    """Yahoo returns {} (no 'raw' key) when a stat doesn't apply -- e.g. an
    ETF has no forward EPS estimate, a loss-making company has no
    meaningful trailing P/E. A P/E computed on negative earnings is also
    not a usable multiple, so non-positive values are treated as missing
    too, same as an absent field."""
    if not field:
        return None
    value = field.get("raw")
    if value is None or value <= 0:
        return None
    return float(value)


def fetch_valuation_one(ticker: str, session: requests.Session, crumb: str) -> dict | None:
    """{'trailing_pe': float|None, 'forward_pe': float|None} for one ticker,
    or None if the REQUEST itself failed (network/HTTP error) -- missing
    individual fields is a valid, common result, not a failure."""
    response = session.get(
        _QUOTE_SUMMARY_URL.format(ticker=ticker),
        params={"modules": "summaryDetail", "crumb": crumb},
        timeout=20,
    )
    if response.status_code != 200:
        return None
    results = response.json().get("quoteSummary", {}).get("result") or []
    if not results:
        return None
    summary = results[0].get("summaryDetail", {})
    return {
        "trailing_pe": _extract_pe(summary.get("trailingPE")),
        "forward_pe": _extract_pe(summary.get("forwardPE")),
    }


def fetch_valuation_batch(tickers: list[str]) -> dict[str, dict]:
    """One session/crumb shared across the whole batch, fetched concurrently.
    Tickers whose request fails outright are simply absent from the result
    dict -- callers already treat "no valuation data" as neutral, not
    penalized."""
    session, crumb = _new_session()
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch_valuation_one, ticker, session, crumb): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                data = future.result()
            except Exception:
                data = None
            if data is not None:
                results[ticker] = data
    return results


def fetch_valuation(ticker: str) -> dict | None:
    """Single-ticker convenience wrapper -- fetches its own session/crumb.
    For the live single-ticker path (setup_score.compute_setup_score); the
    batch function above is for the nightly refresh across the whole
    tracking universe."""
    session, crumb = _new_session()
    return fetch_valuation_one(ticker, session, crumb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trailing/forward P/E for a ticker, from Yahoo Finance.")
    parser.add_argument("ticker")
    args = parser.parse_args()

    result = fetch_valuation(args.ticker)
    if result is None:
        print(f"{args.ticker.upper()}: request failed")
        return
    print(f"{args.ticker.upper()}: trailing P/E {result['trailing_pe']}, forward P/E {result['forward_pe']}")


if __name__ == "__main__":
    main()
