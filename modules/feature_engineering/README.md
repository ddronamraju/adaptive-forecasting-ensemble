# Feature Engineering Toolkit (Retail Forecasting)

## Objective

Provide a reusable, production-inspired feature engineering function for weekly retail demand forecasting.
This module standardizes how features are generated across modeling, monitoring, and decision tools.

## Features Included

- Time features: week-of-year, month, year
- Lag features: 1, 2, 4, 8, 13, 26, 52 weeks
- Rolling windows (shifted to prevent leakage): mean, std, min, max
- Momentum features: first difference, percent change
- Holiday proximity: centered holiday window flag
- YoY change: year-over-year demand ratio
- Fourier terms: seasonal basis functions for boosting models

## Key Design Principle: Prevent Leakage

Rolling features are generated using `shift(1)` before the rolling operation to ensure the current
week's target is not used to compute features for that same week.

## Outputs

- Notebook: `feature_engineering_toolkit.ipynb`
- Parquet feature set for downstream modeling: `features_store1_dept1.parquet`
- Diagnostic plot: `images/feature_correlation_selected.png`

## Role in the Overall System

- Serves as the standardized feature layer for:
  - Global forecasting model (LightGBM)
  - Residual-based anomaly detection enhancements
  - Inventory risk uncertainty estimation
  - Scenario simulation inputs

## Production & Scaling Notes

- At scale, this logic should be implemented in a Feature Store pattern:
  - Same transformations for training and serving to avoid training-serving skew.
  - Distributed computation (Spark/Dask) for lag/rolling features across many SKUs.
- Persist features to Parquet and partition by `Store`, `Dept`, and/or time for efficient retrieval.

## Files

- `feature_engineering_toolkit.ipynb`
- `features_store1_dept1.parquet`
- `images/feature_correlation_selected.png`