"""Ticker metadata: exchange + GICS sector + GICS industry group, used to
support a fallback inference tier (symbol-specific -> industry-level ->
sector-level -> full-population pooled) for tickers that don't yet have
enough of their own trade history.

Sector and industry group come from the scraped `smc_regime/sectors/*.txt`
and `smc_regime/industry_groups/*.txt` GICS files, which only cover S&P 500
constituents (industry group is GICS's 25-bucket "Industry Group" level --
one step finer than the 11 GICS sectors, e.g. "Semiconductors &
Semiconductor Equipment" or "Software & Services" within Information
Technology -- derived from Wikipedia's S&P 500 GICS Sub-Industry column
mapped up through the standard GICS Sub-Industry -> Industry Group table).
The tracking universe also includes non-S&P names (foreign large caps,
ETFs, recent IPOs, and a few tickers that have since dropped out of the
S&P 500) -- those are covered by SECTOR_OVERRIDES / INDUSTRY_OVERRIDES
below, hand-mapped.

Exchange comes from Tiingo's `/tiingo/daily/{ticker}` metadata endpoint
(no `/prices` suffix), cached to disk since it almost never changes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

_SECTORS_DIR = Path(__file__).parent / "sectors"
_INDUSTRY_GROUPS_DIR = Path(__file__).parent / "industry_groups"
_CACHE_PATH = Path(__file__).parent / "data" / "exchange_cache.json"
_META_URL = "https://api.tiingo.com/tiingo/daily/{ticker}"

# smc_regime/industry_groups/*.txt filenames are slugified (lowercase,
# "&"->"and", non-alphanumerics->"_") GICS Industry Group names -- this
# maps each slug back to the real display name.
_INDUSTRY_GROUP_DISPLAY_NAMES = {
    "automobiles_and_components": "Automobiles & Components",
    "banks": "Banks",
    "capital_goods": "Capital Goods",
    "commercial_and_professional_services": "Commercial & Professional Services",
    "consumer_discretionary_distribution_and_retail": "Consumer Discretionary Distribution & Retail",
    "consumer_durables_and_apparel": "Consumer Durables & Apparel",
    "consumer_services": "Consumer Services",
    "consumer_staples_distribution_and_retail": "Consumer Staples Distribution & Retail",
    "energy": "Energy",
    "equity_real_estate_investment_trusts_reits": "Equity Real Estate Investment Trusts (REITs)",
    "financial_services": "Financial Services",
    "food_beverage_and_tobacco": "Food, Beverage & Tobacco",
    "health_care_equipment_and_services": "Health Care Equipment & Services",
    "household_and_personal_products": "Household & Personal Products",
    "insurance": "Insurance",
    "materials": "Materials",
    "media_and_entertainment": "Media & Entertainment",
    "pharmaceuticals_biotechnology_and_life_sciences": "Pharmaceuticals, Biotechnology & Life Sciences",
    "real_estate_management_and_development": "Real Estate Management & Development",
    "semiconductors_and_semiconductor_equipment": "Semiconductors & Semiconductor Equipment",
    "software_and_services": "Software & Services",
    "technology_hardware_and_equipment": "Technology Hardware & Equipment",
    "telecommunication_services": "Telecommunication Services",
    "transportation": "Transportation",
    "utilities": "Utilities",
}

# ETFs and non-S&P-500 names not covered by the scraped GICS sector files,
# hand-mapped to their GICS sector.
SECTOR_OVERRIDES = {
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "AAL": "Industrials",
    "ALAB": "Information Technology",
    "ALNY": "Health Care",
    "ARM": "Information Technology",
    "ASML": "Information Technology",
    "CCEP": "Consumer Staples",
    "CRWV": "Information Technology",
    "FER": "Industrials",
    "MELI": "Consumer Discretionary",
    "MSTR": "Information Technology",
    "NBIS": "Information Technology",
    "NIO": "Consumer Discretionary",
    "PDD": "Consumer Discretionary",
    "QS": "Industrials",
    "RKLB": "Industrials",
    "SHOP": "Information Technology",
    "SPCX": "Industrials",
    "TRI": "Industrials",
    "BRK-B": "Financials",
    # Added alongside the XLY discretionary sweep -- turned out to have
    # already dropped out of the current S&P 500 (confirmed live against
    # Wikipedia's constituent table), so the scraped sectors/*.txt files
    # don't carry them either.
    "POOL": "Consumer Discretionary",
    "ETSY": "Consumer Discretionary",
    "LKQ": "Consumer Discretionary",
    "KMX": "Consumer Discretionary",
    "CZR": "Consumer Discretionary",
}

# Same idea as SECTOR_OVERRIDES, one level finer (GICS Industry Group).
INDUSTRY_OVERRIDES = {
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "AAL": "Transportation",
    "ALAB": "Semiconductors & Semiconductor Equipment",
    "ALNY": "Pharmaceuticals, Biotechnology & Life Sciences",
    "ARM": "Semiconductors & Semiconductor Equipment",
    "ASML": "Semiconductors & Semiconductor Equipment",
    "CCEP": "Food, Beverage & Tobacco",
    "CRWV": "Software & Services",
    "FER": "Capital Goods",
    "MELI": "Consumer Discretionary Distribution & Retail",
    "MSTR": "Software & Services",
    "NBIS": "Software & Services",
    "NIO": "Automobiles & Components",
    "PDD": "Consumer Discretionary Distribution & Retail",
    "QS": "Capital Goods",
    "RKLB": "Capital Goods",
    "SHOP": "Software & Services",
    "SPCX": "Capital Goods",
    "TRI": "Commercial & Professional Services",
    "BRK-B": "Insurance",
    "POOL": "Consumer Discretionary Distribution & Retail",
    "ETSY": "Consumer Discretionary Distribution & Retail",
    "LKQ": "Automobiles & Components",
    "KMX": "Consumer Discretionary Distribution & Retail",
    "CZR": "Consumer Services",
}


def load_sector_map() -> dict[str, str]:
    sector_map: dict[str, str] = {}
    for path in _SECTORS_DIR.glob("*.txt"):
        sector = path.stem.replace("_", " ").title()
        for line in path.read_text().splitlines():
            ticker = line.strip()
            if ticker:
                sector_map[ticker] = sector
    return sector_map


def get_sector(ticker: str, sector_map: dict[str, str] | None = None) -> str:
    sector_map = sector_map if sector_map is not None else load_sector_map()
    ticker = ticker.upper()
    return sector_map.get(ticker) or SECTOR_OVERRIDES.get(ticker) or "Unknown"


def load_industry_map() -> dict[str, str]:
    industry_map: dict[str, str] = {}
    for path in _INDUSTRY_GROUPS_DIR.glob("*.txt"):
        industry = _INDUSTRY_GROUP_DISPLAY_NAMES.get(path.stem, path.stem.replace("_", " ").title())
        for line in path.read_text().splitlines():
            ticker = line.strip()
            if ticker:
                industry_map[ticker] = industry
    return industry_map


def get_industry(ticker: str, industry_map: dict[str, str] | None = None) -> str:
    industry_map = industry_map if industry_map is not None else load_industry_map()
    ticker = ticker.upper()
    return industry_map.get(ticker) or INDUSTRY_OVERRIDES.get(ticker) or "Unknown"


def _load_cache() -> dict[str, str]:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def get_exchange(ticker: str, cache: dict[str, str] | None = None) -> str:
    """Fetch a ticker's exchange code from Tiingo, cached to disk since it
    essentially never changes. Pass a shared `cache` dict when resolving
    many tickers in a loop to batch the disk write."""
    ticker = ticker.upper()
    owns_cache = cache is None
    if owns_cache:
        cache = _load_cache()

    if ticker not in cache:
        token = os.environ.get("TIINGO_API_KEY")
        if not token:
            raise RuntimeError("TIINGO_API_KEY environment variable is not set")
        response = requests.get(_META_URL.format(ticker=ticker), params={"token": token}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        cache[ticker] = payload.get("exchangeCode") or "Unknown"
        if owns_cache:
            _save_cache(cache)

    return cache[ticker]


def build_ticker_metadata(tickers: list[str]) -> list[dict[str, str]]:
    """Resolve {ticker, exchange, sector, industry} for a list of tickers,
    one API call per not-yet-cached ticker. `industry` here is the GICS
    Industry Group level (25 buckets), not the finer 74-bucket GICS
    Industry level."""
    sector_map = load_sector_map()
    industry_map = load_industry_map()
    cache = _load_cache()
    rows = []
    for ticker in tickers:
        rows.append(
            {
                "ticker": ticker.upper(),
                "exchange": get_exchange(ticker, cache=cache),
                "sector": get_sector(ticker, sector_map=sector_map),
                "industry": get_industry(ticker, industry_map=industry_map),
            }
        )
    _save_cache(cache)
    return rows
