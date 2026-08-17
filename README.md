# StockMarketAnalysis

## SMC regime classifier

`smc_regime` classifies a stock's recent price action into one of three regimes:

- **choppy** -- no persistent direction, price churns back and forth
- **trending** -- steady, efficient move in one direction
- **parabolic** -- trending *and* accelerating (volatility expanding, price stretched
  far from its moving average, path curving rather than straight)

This is step one of a two-step workflow: classify the regime, then evaluate which
trading strategy performs best conditioned on that regime (backtesting comes next,
once the classifier's labels look right against real charts).

### Usage

```bash
pip install -r requirements.txt
python -m smc_regime.cli AAPL --period 6mo
```

Or from Python:

```python
from smc_regime import fetch_ohlcv, classify_regime

df = fetch_ohlcv("AAPL", period="6mo")
result = classify_regime(df)
print(result[["regime", "direction"]].tail())
```

### How classification works

Rule-based, not ML -- deliberately, so the thresholds can be inspected and tuned
against real charts before anything more complex is layered on top.

| Signal | Metric | Role |
|---|---|---|
| Trendiness | Kaufman Efficiency Ratio, ADX | separates choppy from trending |
| Volatility acceleration | ATR% and its rate of change | flags expanding volatility |
| Extension | Distance from EMA in ATR units | flags overextended price |
| Curvature | R² gain of quadratic vs. linear fit | flags an accelerating (convex) path vs. a straight trend |

A bar is `parabolic` only if it's already `trending` AND stretched from its EMA AND
volatility is accelerating AND the price path is curving -- all four, not just one,
to avoid mislabeling an ordinary strong trend as parabolic.

All thresholds live in `smc_regime.regime.RegimeThresholds` and can be overridden:

```python
from smc_regime import classify_regime, RegimeThresholds

result = classify_regime(df, RegimeThresholds(er_trend_min=0.25, extension_parabolic_min=2.5))
```

### Next steps

- Validate labels by eye against a chart for a few tickers with known regimes, tune thresholds
- Add SMC structure signals (BOS/CHoCH frequency, order block respect rate) as a confirming layer
- Build a backtest harness that runs candidate strategies filtered by regime label and compares performance
