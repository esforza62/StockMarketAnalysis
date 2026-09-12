"""Ticker news + headline sentiment via Tiingo's News API.

Tiingo's /tiingo/news endpoint returns articles (title, description, source,
tags, tickers, dates) but no sentiment score of its own -- confirmed against
an independent source (QuantConnect's Lean data model for this exact feed)
before building this, rather than assuming. Sentiment here is computed from
the headline/description text with VADER (vaderSentiment), a lightweight
rule-based analyzer built for short, informal text -- no API call, no heavy
model, deterministic.

VADER's stock lexicon is general-purpose and misses common finance headline
vocabulary by default (e.g. "beats earnings, raises guidance" scores as
neutral out of the box). _FINANCE_LEXICON layers a small set of recurring
finance-headline terms on top of VADER's base lexicon to fix that -- not a
replacement, just an extension of the same rule-based approach.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_NEWS_URL = "https://api.tiingo.com/tiingo/news"

_FINANCE_LEXICON = {
    "beats": 2.5, "beat": 2.0, "misses": -2.5, "miss": -2.0, "missed": -2.0,
    "raises": 1.5, "raised": 1.5, "cuts": -1.8, "cut": -1.5, "slashes": -2.5, "slashed": -2.5,
    "upgrade": 2.0, "upgraded": 2.0, "downgrade": -2.0, "downgraded": -2.0,
    "outperform": 1.8, "underperform": -1.8,
    "tariff": -1.5, "tariffs": -1.5, "recall": -2.0, "recalls": -2.0,
    "lawsuit": -2.0, "sues": -1.8, "sued": -1.8, "probe": -1.8, "investigation": -1.5,
    "bankruptcy": -3.0, "layoffs": -2.2, "layoff": -2.2, "buyback": 1.5, "buybacks": 1.5,
    "surge": 2.0, "surges": 2.0, "soar": 2.2, "soars": 2.2, "plunge": -2.5, "plunges": -2.5,
    "tumbles": -2.2, "tumble": -2.2, "rally": 1.8, "rallies": 1.8,
}

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(_FINANCE_LEXICON)


def score_headline(text: str) -> float:
    """VADER compound score in [-1, 1] for a single headline/description."""
    return _analyzer.polarity_scores(text)["compound"]


# CALIBRATED AGAINST THIS FEED, NOT VADER'S GENERIC THRESHOLDS.
#
# The first live run showed why that is necessary. VADER's own convention --
# +/-0.05 separating sentiment from neutral, +/-0.50 marking it strong --
# put 95% of the universe in "positive" or "very positive": 309 and 78
# tickers against 11 negative, 10 neutral and ZERO very negative, the last
# being a band that could not fire at all because the minimum score observed
# was -0.494. A label that reads the same nineteen times in twenty carries
# no information.
#
# The cause is not the finance lexicon below -- checked directly, plain and
# tuned VADER score identical on neutral corporate headlines, and the lexicon
# is the more NEGATIVE-weighted of the two (26 negative terms to 15). It is
# VADER's base lexicon reading ordinary newswire language as positive
# ("names new CFO" scores +0.477, "declares quarterly dividend" +0.296) on a
# feed that is mostly exactly that. The corpus genuinely sits at a median of
# +0.344, mean +0.358, spread -0.494 to +0.917 over 408 tickers.
#
# So the thresholds are percentiles of that observed spread, rounded, giving
# roughly 5 / 10 / 68 / 12 / 5 percent. Most tickers are unremarkable, which
# is the honest reading of a newswire, and each tail is small enough to be
# worth looking at.
#
# THE LABELS SAY "WEAKER" AND "STRONGER", NOT "NEGATIVE" AND "POSITIVE",
# because on this feed those would be false: a ticker at +0.15 is using mildly
# positive language and is merely below a very positive median. Calling it
# negative would invent a claim the score does not make. The bottom band is
# the exception that proves the calibration honest -- its highest member
# scores +0.045, so "much weaker" really is at or below neutral in absolute
# terms as well as relative ones.
#
# Re-check these if the feed, the window or the article limit changes; they
# describe THIS corpus, the same caveat the grade cuts carry.
_BANDS = [
    (0.72, "much stronger"),
    (0.52, "stronger"),
    (0.18, "typical"),
    (0.05, "weaker"),
]
_FLOOR_LABEL = "much weaker"


def _label(compound: float) -> str:
    """One of the five bands above, from a VADER compound score in [-1, 1].

    Relative to this feed's own spread -- see the calibration note above.
    """
    for threshold, label in _BANDS:
        if compound >= threshold:
            return label
    return _FLOOR_LABEL


def fetch_news(ticker: str, days: int = 7, limit: int = 20) -> list[dict]:
    token = os.environ.get("TIINGO_API_KEY")
    if not token:
        raise RuntimeError("TIINGO_API_KEY environment variable is not set")

    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    response = requests.get(
        _NEWS_URL,
        params={
            "token": token,
            "tickers": ticker.upper(),
            "startDate": start_date,
            "sortBy": "publishedDate",
            "limit": limit,
            "onlyWithTickers": True,
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def ticker_sentiment(ticker: str, days: int = 7, limit: int = 20) -> dict:
    """Fetch recent news for a ticker and score each headline's sentiment.

    Returns {article_count, avg_compound, label, counts, articles} where
    `label` is the overall positive/negative/neutral call from avg_compound
    and `articles` is the per-headline detail (title, url, published_date,
    compound, label), most recent first.
    """
    raw_articles = fetch_news(ticker, days=days, limit=limit)
    if not raw_articles:
        return {"article_count": 0, "avg_compound": None, "label": "no data", "counts": {}, "articles": []}

    articles = []
    for a in raw_articles:
        text = a.get("title", "")
        if a.get("description"):
            text = f"{text}. {a['description']}"
        compound = score_headline(text)
        articles.append(
            {
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("source", ""),
                "published_date": a.get("publishedDate", ""),
                "compound": round(compound, 3),
                "label": _label(compound),
            }
        )

    compounds = pd.Series([a["compound"] for a in articles])
    counts = pd.Series([a["label"] for a in articles]).value_counts().to_dict()
    avg_compound = float(compounds.mean())
    return {
        "article_count": len(articles),
        "avg_compound": round(avg_compound, 3),
        "label": _label(avg_compound),
        "counts": counts,
        "articles": articles,
    }


_FETCH_WORKERS = 8


def fetch_sentiment_batch(tickers: list[str], days: int = 7, limit: int = 20) -> dict[str, dict]:
    """Headline sentiment for a whole universe, concurrently.

    One request per ticker rather than one batched request over all of
    them: Tiingo's news endpoint takes a comma-separated `tickers` list,
    but a shared `limit` across 400+ names would silently starve the
    thinly-covered ones -- the heavily-covered tickers would fill the
    response and a quiet small-cap would come back with nothing, which
    reads identically to "no news" and is not the same thing.

    A ticker whose request fails is absent from the result, exactly like
    valuation: callers treat missing news as no news, never as bad news.
    The returned dicts drop the per-article detail and keep the summary --
    the nightly stores one row per ticker, not an article archive.
    """
    if not os.environ.get("TIINGO_API_KEY"):
        # Checked once here rather than discovered 400 times inside the
        # pool: every worker would raise the same RuntimeError and every
        # one would be swallowed, so the caller gets the same empty dict
        # either way -- just after firing a pool's worth of doomed work.
        return {}

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        futures = {pool.submit(ticker_sentiment, t, days, limit): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                data = future.result()
            except Exception:
                continue
            results[ticker] = {
                "article_count": data["article_count"],
                "avg_compound": data["avg_compound"],
                "label": data["label"],
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Recent news + headline sentiment for a ticker.")
    parser.add_argument("ticker")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    result = ticker_sentiment(args.ticker, days=args.days, limit=args.limit)
    if result["article_count"] == 0:
        print(f"{args.ticker.upper()}: no news in the last {args.days} day(s).")
        return

    print(
        f"{args.ticker.upper()}: {result['article_count']} articles (last {args.days}d) -> "
        f"{result['label']} (avg compound {result['avg_compound']:+.3f}), "
        f"counts: {result['counts']}"
    )
    for a in result["articles"]:
        print(f"  [{a['label']:>8} {a['compound']:+.3f}] {a['published_date']}  {a['title']}  ({a['source']})")


if __name__ == "__main__":
    main()
