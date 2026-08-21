"""Cross-vendor validation: run the exact same strategy code
(smc_regime.strategies / smc_regime.backtest, completely unmodified) against
Yahoo Finance data instead of Tiingo, then diff the results against what's
already in the SQLite store.

This is deliberately NOT routed through the TradingView MCP server's
backtest tools -- those run a different, fixed menu of generic strategies
(no rsi_dip_recovery, no rsi_macd_reversal) on Yahoo data with commission/
slippage baked in, so a "match" there wouldn't actually confirm anything
about this system's strategies. Running our own strategy functions against
an independent data vendor is the test that actually says something: if a
strategy's edge only shows up on Tiingo's specific bars, that's a red flag;
if it shows up on Yahoo's independently-sourced (and dividend/split
-adjusted) bars too, that's real cross-vendor confirmation.

Fetches Yahoo's chart API directly with `requests` rather than via the
`yfinance` package -- yfinance's newer transport (curl_cffi, for TLS
fingerprint impersonation) doesn't route through this environment's HTTPS
proxy the way plain `requests` does; Yahoo's public chart endpoint itself
works fine over plain requests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from . import db as db_module
from .backtest import backtest_strategy

_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_yahoo_ohlcv(ticker: str, start_date: str = "2019-01-01") -> pd.DataFrame:
    """Daily OHLCV from Yahoo's chart API, adjusted for splits/dividends.

    Yahoo returns raw OHLC plus a separate `adjclose` series (adjusted close
    only). Scaling every raw OHLC field by the same per-bar adjclose/close
    ratio keeps the bar internally consistent (High >= Close >= Low, etc.)
    while making it comparable to Tiingo's adjClose -- the field this
    system's own data.py should also be using (see the known raw-close bug
    this cross-check helps confirm the scope of).
    """
    period1 = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    period2 = int(pd.Timestamp.now(tz="UTC").timestamp())
    resp = requests.get(
        _YAHOO_URL.format(ticker=ticker),
        params={"period1": period1, "period2": period2, "interval": "1d", "events": "div,splits"},
        headers=_HEADERS,
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()["chart"]["result"]
    if not payload:
        raise ValueError(f"No Yahoo data returned for {ticker!r}")
    result = payload[0]

    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adjclose = result["indicators"]["adjclose"][0]["adjclose"]

    df = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Volume": quote["volume"],
            "AdjClose": adjclose,
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True),
    ).dropna(subset=["Close", "AdjClose"])

    ratio = df["AdjClose"] / df["Close"]
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col] * ratio
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _compounded_return_pct(returns: pd.Series) -> float:
    return (((1 + returns / 100).prod()) - 1) * 100


def _yahoo_stats(ticker: str, strategy_name: str, start_date: str) -> dict | None:
    df = fetch_yahoo_ohlcv(ticker, start_date)
    trades = backtest_strategy(df, strategy_name)
    if not trades:
        return None
    returns = pd.Series([t.return_pct for t in trades])
    return {
        "trades": len(trades),
        "win_rate": float((returns > 0).mean() * 100),
        "avg_return_pct": float(returns.mean()),
        "compounded_return_pct": float(_compounded_return_pct(returns)),
    }


def _tiingo_stats(conn, ticker: str, strategy_name: str, interval: str) -> dict | None:
    run_at = conn.execute("SELECT MAX(run_at) FROM runs WHERE interval = ?", (interval,)).fetchone()[0]
    if run_at is None:
        return None
    df = pd.read_sql_query(
        """SELECT t.entry_date, t.return_pct
           FROM trades t JOIN runs r ON r.id = t.run_id JOIN strategies s ON s.id = t.strategy_id
           WHERE r.run_at = ? AND r.interval = ? AND t.ticker = ? AND s.name = ?""",
        conn,
        params=(run_at, interval, ticker, strategy_name),
    )
    if df.empty:
        return None
    returns = df.sort_values("entry_date")["return_pct"]
    return {
        "trades": len(returns),
        "win_rate": float((returns > 0).mean() * 100),
        "avg_return_pct": float(returns.mean()),
        "compounded_return_pct": float(_compounded_return_pct(returns)),
    }


def cross_validate(
    tickers: list[str],
    strategy_names: list[str],
    db_path: str = str(db_module.DEFAULT_DB_PATH),
    interval: str = "1d",
    start_date: str = "2019-01-01",
) -> pd.DataFrame:
    conn = db_module.connect(db_path)
    rows = []
    for ticker in tickers:
        for strategy_name in strategy_names:
            tiingo = _tiingo_stats(conn, ticker, strategy_name, interval)
            try:
                yahoo = _yahoo_stats(ticker, strategy_name, start_date)
            except Exception as exc:
                yahoo = None
                print(f"  Yahoo fetch failed for {ticker}/{strategy_name}: {exc}")

            row = {"ticker": ticker, "strategy": strategy_name}
            row.update({f"tiingo_{k}": v for k, v in (tiingo or {}).items()})
            row.update({f"yahoo_{k}": v for k, v in (yahoo or {}).items()})
            if tiingo and yahoo:
                row["compounded_agree_sign"] = (tiingo["compounded_return_pct"] > 0) == (yahoo["compounded_return_pct"] > 0)
                row["win_rate_diff"] = yahoo["win_rate"] - tiingo["win_rate"]
                row["trade_count_diff"] = yahoo["trades"] - tiingo["trades"]
            rows.append(row)
    conn.close()
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    """Aggregate confidence metrics across the whole cross-validation run."""
    required = {"tiingo_compounded_return_pct", "yahoo_compounded_return_pct"}
    if not required.issubset(results.columns):
        return {"comparable_rows": 0, "total_rows": int(len(results)), "error": "no comparable tiingo/yahoo rows"}
    both = results.dropna(subset=list(required))
    if both.empty:
        return {"comparable_rows": 0}
    return {
        "comparable_rows": int(len(both)),
        "total_rows": int(len(results)),
        "sign_agreement_pct": float(both["compounded_agree_sign"].mean() * 100),
        "mean_abs_win_rate_diff": float(both["win_rate_diff"].abs().mean()),
        "mean_abs_trade_count_diff": float(both["trade_count_diff"].abs().mean()),
        "compounded_return_correlation": float(
            both["tiingo_compounded_return_pct"].corr(both["yahoo_compounded_return_pct"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate strategy results against Yahoo Finance data.")
    parser.add_argument("--tickers", required=True, help="comma-separated ticker list")
    parser.add_argument("--strategies", default="rsi_dip_recovery,rsi,rsi_macd_reversal", help="comma-separated strategy names")
    parser.add_argument("--db-file", default=str(db_module.DEFAULT_DB_PATH))
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--log-file", default="backtest_logs/cross_validation_log.jsonl")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]

    results = cross_validate(tickers, strategy_names, args.db_file, args.interval, args.start_date)

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps({"run_at": pd.Timestamp.now(tz="UTC").isoformat(), "results": results.to_dict(orient="records")}) + "\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    print(results.round(2).to_string(index=False))
    print()
    print("Summary:", json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
