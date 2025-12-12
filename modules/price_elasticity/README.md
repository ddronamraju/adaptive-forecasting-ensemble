# Price Elasticity & Promo Impact Modeling

## Objective

Estimate how sensitive demand is to price changes and promotions for a specific
store–department time series. This module supports pricing and merchandising decisions
by quantifying elasticity and promo uplift.

## Functionality

- Reuses Walmart-like sales data from the baseline forecasting module.
- Simulates a realistic price series around a base price with random weekly noise.
- Flags "promo" weeks based on significantly lower price levels.
- Fits a log-linear regression model:

  \[
  \log(\text{sales}) = \beta_0 + \beta_1 \log(\text{price}) + \beta_2 \cdot \text{promo} + \epsilon
  \]

- Computes:
  - **Price elasticity** (β₁)
  - **Promo uplift** as `exp(β₂) - 1`
- Produces:
  - `images/demand_curve_store1_dept1.png`
  - `images/demand_curve_promo_store1_dept1.png`
  - `price_elasticity.ipynb`

## Role in the Overall System

- Provides a **pricing and promo lens** on top of pure forecasting.
- Informs:
  - How aggressive promotions should be.
  - Where price increases are likely to be feasible without large volume loss.
- Connects directly to:
  - Scenario simulation (price/promo what-if tools).
  - Revenue, margin, and profit-to-serve analysis.

## Production & Scaling Notes

- In a real retail environment, price and promo data comes from:
  - Pricing systems
  - Promotion calendars
  - Competitive price feeds

- At scale:
  - Elasticity models are run in batch across thousands of SKUs/categories.
  - Results are stored in a **feature store** and reused by:
    - Forecasting models (as features)
    - Optimization engines (pricing, promo, assortment)

- Implementation details at scale:
  - Use Spark or Dask to run log-linear regressions across many SKUs.
  - Group SKUs into **elasticity families** (e.g., similar categories / brands).
  - Monitor elasticity over time as consumer behavior changes.

## Files

- `price_elasticity.ipynb`
- `images/demand_curve_store1_dept1.png`
- `images/demand_curve_promo_store1_dept1.png`
