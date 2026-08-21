"""Cross-validation: run the exact same strategy signals (smc_regime.strategies,
completely unmodified) through two independent checks and diff the results
against what's already in the SQLite store:

  1. An independent DATA VENDOR -- our own backtest.py engine, fed Yahoo
     Finance data instead of Tiingo. Isolates data-source-specific bugs
     (e.g. a stale/erroneous bar, or the raw-vs-adjusted-close issue this
     also helped confirm the scope of).
  2. An independent ENGINE -- the real, widely-used `Backtesting.py`
     library (PyPI package `backtesting`), fed the SAME Yahoo data used in
     check 1. Isolates bugs in our own hand-rolled backtest.py simulation
     loop itself, since the data is held constant and only the execution
     engine changes.

Both checks reuse strategies.py's entry/exit signal functions unmodified --
the thing being validated is "does this signal + this data + this engine
combination agree with the others," not a reimplementation of the
strategies themselves.

This is deliberately NOT routed through the TradingView MCP server's
backtest tools -- those run a different, fixed menu of generic strategies
(no rsi_dip_recovery, no rsi_macd_reversal) on Yahoo data with commission/
slippage baked in, so a "match" there wouldn't confirm anything about this
system's actual strategies.

Yahoo's chart API is fetched directly with `requests` rather than via the
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
from backtesting import Backtest
from backtesting import Strategy as BTStrategy

from . import db as db_module
from .backtest import Trade, run_backtest
from .strategies import STRATEGIES

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


def _finalized_trades(df: pd.DataFrame, strategy_name: str) -> list[Trade]:
    """Same as backtest_strategy(), but if a position is still open at the
    last bar, close it there at the final available price instead of
    dropping it.

    backtest_strategy()/run_backtest() intentionally drop a dangling
    open-at-data-end position everywhere else in this system (regime
    tagging only makes sense for a trade with a real entry AND exit).
    Backtesting.py's default behavior is the opposite: it marks an open
    position to market in Return[%]/Equity Final but excludes it from the
    trade list, which made an early version of this cross-check compare
    a closed-trades-only number (ours) against a number that secretly
    included one open position's unrealized P&L (Backtesting.py's) --
    not actually the same measurement. Finalizing on both sides here (this
    function, plus `finalize_trades=True` below) makes the comparison
    apples-to-apples without changing backtest.py's production behavior.
    """
    signals = STRATEGIES[strategy_name](df)
    trades = run_backtest(df, signals)

    in_position = False
    entry_date = entry_price = None
    for date, row in signals.iterrows():
        if not in_position and row["entry"]:
            in_position, entry_date, entry_price = True, date, df.loc[date, "Close"]
        elif in_position and row["exit"]:
            in_position = False
    if in_position:
        last_date = df.index[-1]
        trades = [*trades, Trade(entry_date, last_date, entry_price, df.loc[last_date, "Close"])]
    return trades


def _own_engine_stats(df: pd.DataFrame, strategy_name: str) -> dict | None:
    trades = _finalized_trades(df, strategy_name)
    if not trades:
        return None
    returns = pd.Series([t.return_pct for t in trades])
    return {
        "trades": len(trades),
        "win_rate": float((returns > 0).mean() * 100),
        "avg_return_pct": float(returns.mean()),
        "compounded_return_pct": float(_compounded_return_pct(returns)),
    }


class _SignalStrategy(BTStrategy):
    """Generic Backtesting.py Strategy driven entirely by precomputed
    entry/exit arrays -- signal generation stays in strategies.py, this
    class only exercises Backtesting.py's own order/position/equity
    machinery against it."""

    entry_signal = None
    exit_signal = None

    def init(self):
        self.entry = self.I(lambda: self.entry_signal, name="entry", overlay=False)
        self.exit = self.I(lambda: self.exit_signal, name="exit", overlay=False)

    def next(self):
        if not self.position and self.entry[-1]:
            self.buy(size=0.999)
        elif self.position and self.exit[-1]:
            self.position.close()


def _backtesting_py_stats(df: pd.DataFrame, strategy_name: str) -> dict | None:
    signals = STRATEGIES[strategy_name](df)
    if not signals["entry"].any():
        return None

    # Backtesting.py doesn't want a tz-aware index.
    df_naive = df.copy()
    if df_naive.index.tz is not None:
        df_naive.index = df_naive.index.tz_localize(None)

    bt = Backtest(df_naive, _SignalStrategy, cash=100_000, commission=0.0, trade_on_close=True, finalize_trades=True)
    stats = bt.run(entry_signal=signals["entry"].to_numpy(), exit_signal=signals["exit"].to_numpy())

    n_trades = int(stats["# Trades"])
    if n_trades == 0:
        return None
    trade_returns_pct = stats._trades["ReturnPct"] * 100
    return {
        "trades": n_trades,
        "win_rate": float(stats["Win Rate [%]"]),
        "avg_return_pct": float(trade_returns_pct.mean()),
        "compounded_return_pct": float(stats["Return [%]"]),
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
        try:
            yahoo_df = fetch_yahoo_ohlcv(ticker, start_date)
        except Exception as exc:
            yahoo_df = None
            print(f"  Yahoo fetch failed for {ticker}: {exc}")

        for strategy_name in strategy_names:
            tiingo = _tiingo_stats(conn, ticker, strategy_name, interval)

            yahoo = None
            btpy = None
            if yahoo_df is not None:
                yahoo = _own_engine_stats(yahoo_df, strategy_name)
                try:
                    btpy = _backtesting_py_stats(yahoo_df, strategy_name)
                except Exception as exc:
                    print(f"  Backtesting.py failed for {ticker}/{strategy_name}: {exc}")

            row = {"ticker": ticker, "strategy": strategy_name}
            row.update({f"tiingo_{k}": v for k, v in (tiingo or {}).items()})
            row.update({f"yahoo_{k}": v for k, v in (yahoo or {}).items()})
            row.update({f"btpy_{k}": v for k, v in (btpy or {}).items()})
            if tiingo and yahoo:
                row["tiingo_yahoo_agree_sign"] = (tiingo["compounded_return_pct"] > 0) == (yahoo["compounded_return_pct"] > 0)
                row["tiingo_yahoo_win_rate_diff"] = yahoo["win_rate"] - tiingo["win_rate"]
                row["tiingo_yahoo_trade_count_diff"] = yahoo["trades"] - tiingo["trades"]
            if yahoo and btpy:
                row["yahoo_btpy_agree_sign"] = (yahoo["compounded_return_pct"] > 0) == (btpy["compounded_return_pct"] > 0)
                row["yahoo_btpy_win_rate_diff"] = btpy["win_rate"] - yahoo["win_rate"]
                row["yahoo_btpy_trade_count_diff"] = btpy["trades"] - yahoo["trades"]
            rows.append(row)
    conn.close()
    return pd.DataFrame(rows)


def _pair_summary(results: pd.DataFrame, a_col: str, b_col: str, agree_col: str, win_rate_diff_col: str, trade_count_diff_col: str) -> dict:
    required = {a_col, b_col}
    if not required.issubset(results.columns):
        return {"comparable_rows": 0, "error": f"no comparable {a_col}/{b_col} rows"}
    both = results.dropna(subset=list(required))
    if both.empty:
        return {"comparable_rows": 0}
    return {
        "comparable_rows": int(len(both)),
        "sign_agreement_pct": float(both[agree_col].mean() * 100),
        "mean_abs_win_rate_diff": float(both[win_rate_diff_col].abs().mean()),
        "mean_abs_trade_count_diff": float(both[trade_count_diff_col].abs().mean()),
        "compounded_return_correlation": float(both[a_col].corr(both[b_col])),
    }


def summarize(results: pd.DataFrame) -> dict:
    """Aggregate confidence metrics for both cross-checks: Tiingo-engine vs
    Yahoo-engine (data-vendor check) and Yahoo-engine vs Yahoo-Backtesting.py
    (engine check, data held constant)."""
    return {
        "total_rows": int(len(results)),
        "data_vendor_check_tiingo_vs_yahoo": _pair_summary(
            results, "tiingo_compounded_return_pct", "yahoo_compounded_return_pct",
            "tiingo_yahoo_agree_sign", "tiingo_yahoo_win_rate_diff", "tiingo_yahoo_trade_count_diff",
        ),
        "engine_check_yahoo_vs_backtestingpy": _pair_summary(
            results, "yahoo_compounded_return_pct", "btpy_compounded_return_pct",
            "yahoo_btpy_agree_sign", "yahoo_btpy_win_rate_diff", "yahoo_btpy_trade_count_diff",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-validate strategy results against Yahoo Finance data and the Backtesting.py engine.")
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

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", None)
    print(results.round(2).to_string(index=False))
    print()
    print("Summary:", json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
