# SKU Segmentation (Trend, Volatility, Seasonality)

## Objective

Group SKUs (Store–Department combinations) into behaviorally similar segments based on:
- Demand level
- Volatility
- Trend
- Seasonality strength

This enables differentiated strategies for forecasting, inventory, and promotions.

## Functionality

- Defines SKU as `Store_Dept`.
- Computes SKU-level summary metrics:
  - `mean`, `std`, `median`, `max`, `min`, `cv` (coefficient of variation).
- Aggregates weekly demand to derive:
  - `trend` (late weeks vs early weeks).
  - `seasonality_strength` (variance across weeks of the year).
- Standardizes features and applies K-Means clustering.
- Produces:
  - `images/sku_clusters_trend_cv.png` – scatterplot (trend vs volatility, colored by cluster).
  - `images/sku_cluster_profiles.png` – heatmap of average features per cluster.
  - `sku_segmentation.ipynb` – full analysis.

## Role in the Overall System

- Provides a **segmentation layer** that downstream components can use:
  - Different forecasting strategies by cluster (e.g., robust models for volatile seasonal SKUs).
  - Different inventory strategies (higher safety stock for volatile clusters).
  - Targeted analysis for declining or high-potential SKUs.
- Integrates with:
  - Inventory risk modeling (cluster-based safety stock policies).
  - Feature engineering and model selection (e.g., more complex models for certain clusters).

## Production & Scaling Notes

- At scale, segmentation is typically run:
  - On a scheduled basis (e.g., monthly or quarterly).
  - Across thousands to millions of SKUs using distributed compute (Spark, Dask).
- Practical considerations:
  - Store clusters in a **Feature Store** as a categorical feature (`sku_cluster`) for forecasting models.
  - Refresh segments when demand patterns shift (e.g., after major assortment or pricing changes).
  - Use more advanced clustering (e.g., Gaussian Mixture Models, time-series clustering) as needed.

## Files

- `sku_segmentation.ipynb`
- `images/sku_clusters_trend_cv.png`
- `images/sku_cluster_profiles.png`