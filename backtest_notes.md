# Validation notes

The model predicts whether the next ~1 hour has an ATR-normalized upward move.

Metrics shown:
- Accuracy: all holdout observations.
- ROC-AUC: ranking quality.
- Coverage: percentage of holdout observations where model confidence >= 70%.
- High-confidence accuracy: accuracy only on those >=70% confidence observations.

A high-confidence accuracy of 70–80% is not guaranteed. If coverage is very low, an impressive percentage may simply mean the model only speaks on a small number of historical cases.

For serious research, add:
- strict walk-forward retraining,
- purged/embargoed time-series validation,
- realistic spread/fees/slippage,
- separate bull/bear/range regimes,
- timestamped news to prevent lookahead,
- out-of-sample periods,
- probability calibration,
- paper-trade logs.
