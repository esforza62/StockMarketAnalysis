"""Backtest against TradingView bars when no Tiingo key is present.

WHY THIS EXISTS. data.fetch_ohlcv needs TIINGO_API_KEY. The nightly has it
as a GitHub Actions secret, but a dev container started without it cannot
fetch a single bar, so there is no way to try a strategy change against
real price action -- only synthetic random walks, which are exactly the
wrong thing to judge a liquidity-sweep pattern on. A swing failure needs
real swing points, real wicks and real stop clusters; a Gaussian random
walk has none of those in any meaningful sense.

HOW IT WORKS, and the constraint that shapes it: the TradingView data
arrives over MCP, and MCP tools are callable only by the agent driving the
session, never from inside a Python process. So this module cannot fetch
anything itself. The flow is:

    1. the agent calls get_ohlcv over MCP for each symbol
    2. the agent calls save() with what came back
    3. backtests read the cache through load(), which returns a frame in
       exactly the shape data.fetch_ohlcv produces

That makes the cache the interface, not the network.

WHAT THIS IS NOT FOR. It is an iteration tool, not a second data source:

  * TradingView is a different vendor. Tiingo's daily bars are SPLIT- AND
    DIVIDEND-ADJUSTED (data.py uses adjClose deliberately -- the raw close
    shows NVDA's 2024 10:1 split as a 90% crash); these bars are whatever
    TradingView serves for that symbol. Numbers from here will not match
    the nightly and are not comparable with anything on the desk.
  * MCP does not exist inside GitHub Actions, so nothing here can ever
    feed the nightly. Fixing that properly means setting TIINGO_API_KEY on
    the environment.

Use it to see whether a signal fires sensibly on real structure, and to
compare two variants against each other on identical bars. Do not use it
to decide whether a strategy has edge.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("backtest_logs/tv_bars")

# TradingView wants EXCHANGE:TICKER. The exchange strings in
# ticker_metadata are close but not identical to TradingView's.
_EXCHANGE_MAP = {
    "NASDAQ": "NASDAQ",
    "NYSE": "NYSE",
    "BATS": "BATS",
    "AMEX": "AMEX",
    "NYSE ARCA": "AMEX",   # TradingView files ARCA-listed ETFs under AMEX
}

# data.py's interval vocabulary -> TradingView's.
INTERVAL_MAP = {"1d": "1D", "1h": "1h", "15m": "15m", "1w": "1W"}


def tv_symbol(ticker: str, exchange: str | None) -> str:
    """EXCHANGE:TICKER for TradingView, defaulting to NASDAQ when unknown."""
    return f"{_EXCHANGE_MAP.get((exchange or '').upper(), 'NASDAQ')}:{ticker}"


def cache_path(ticker: str, interval: str) -> Path:
    return CACHE_DIR / interval / f"{ticker}.json"


def save(ticker: str, interval: str, bars: list[dict], symbol: str | None = None) -> Path:
    """Persist one get_ohlcv response.

    `bars` is the MCP payload's own list of {t, o, h, l, c, v} dicts, stored
    as-is alongside its provenance so a cached file can always be traced
    back to the symbol and vendor it came from.
    """
    path = cache_path(ticker, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "ticker": ticker,
        "symbol": symbol,
        "interval": interval,
        "source": "tradingview-mcp",
        "bar_count": len(bars),
        "bars": bars,
    }))
    return path


def load(ticker: str, interval: str = "1d") -> pd.DataFrame:
    """Cached bars in exactly data.fetch_ohlcv's shape.

    Same columns, same capitalisation, same tz-aware UTC index, so anything
    that takes a fetch_ohlcv frame takes this one unchanged.
    """
    path = cache_path(ticker, interval)
    if not path.is_file():
        raise FileNotFoundError(
            f"no cached {interval} bars for {ticker!r}. The agent populates this "
            f"over MCP -- ask it to fetch {ticker}, or run "
            f"`python -m smc_regime.tv_bars --list` to see what is cached."
        )
    payload = json.loads(path.read_text())
    bars = payload.get("bars") or []
    if not bars:
        raise ValueError(f"cached file for {ticker!r} has no bars: {path}")

    df = pd.DataFrame(bars)
    df["date"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.set_index("date").rename(
        columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
    )
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    # Defensive: the engine assumes chronological, unique bars.
    return df[~df.index.duplicated(keep="first")].sort_index()


def ingest(result_file: str | Path, ticker: str, interval: str) -> Path:
    """Load a get_ohlcv response that the MCP layer spilled to a file.

    A full history is far too large to return inline, so the MCP writes it
    to disk and hands back the path. Reading it here rather than through
    the agent's context is not just an optimisation -- a thousand bars of
    JSON would crowd out the actual work, and the file is the same payload
    either way.
    """
    payload = json.loads(Path(result_file).read_text())
    if not payload.get("success", True):
        raise ValueError(f"MCP result reports failure: {payload.get('error', payload)}")
    bars = payload.get("bars") or []
    if not bars:
        raise ValueError(f"no bars in {result_file}")
    return save(ticker, interval, bars, symbol=payload.get("symbol"))


def cached(interval: str | None = None) -> list[tuple[str, str, int]]:
    """(ticker, interval, bar_count) for everything in the cache."""
    out = []
    for path in sorted(CACHE_DIR.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if interval and payload.get("interval") != interval:
            continue
        out.append((payload.get("ticker", path.stem),
                    payload.get("interval", path.parent.name),
                    payload.get("bar_count", 0)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="show what is cached")
    parser.add_argument("--interval", default=None)
    parser.add_argument("--show", default=None, metavar="TICKER", help="print a cached frame's head and tail")
    parser.add_argument("--ingest", nargs=3, metavar=("RESULT_FILE", "TICKER", "INTERVAL"),
                        help="load a spilled get_ohlcv result file into the cache")
    args = parser.parse_args()

    if args.ingest:
        path = ingest(*args.ingest)
        print(f"cached -> {path}")
        return

    if args.show:
        df = load(args.show, args.interval or "1d")
        print(f"{args.show}  {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
        print(df.head(3))
        print("...")
        print(df.tail(3))
        return

    rows = cached(args.interval)
    if not rows:
        print("cache is empty -- ask the agent to fetch bars over MCP")
        return
    for ticker, interval, n in rows:
        print(f"  {ticker:<8}{interval:<6}{n:>6} bars")
    print(f"\n{len(rows)} cached series in {CACHE_DIR}")


if __name__ == "__main__":
    main()
