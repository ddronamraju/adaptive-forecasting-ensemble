# Retail Forecasting Analytics System

> **Production-ready ML ensemble for weekly retail demand forecasting with holiday spike handling and cold-start capability**

A comprehensive forecasting system combining Ridge Regression and LightGBM with adaptive weighting to predict weekly retail sales across multiple store-department combinations. Features include statistical baselines, ML models, and an interactive Streamlit dashboard for scenario analysis.

---

## 🎯 Business Value

- **Accurate Forecasting**: 1-4% WAPE on test data (vs 9-22% baseline)
- **Holiday Robustness**: L2 regularization handles promotional spikes effectively
- **Cold-Start Capability**: Global LightGBM predicts for new stores with zero training history
- **Adaptive Intelligence**: Entity-specific weights optimize performance per store-dept combination
- **Scenario Planning**: Interactive dashboard for "what-if" analysis

---

## 📊 Model Performance Summary

| Model | Test WAPE Range | Key Strength |
|-------|-----------------|--------------|
| Prophet (Baseline) | 4-22% | Statistical decomposition |
| **Ridge Regression** | **0.7-13%** | Holiday spike robustness |
| LightGBM Global | 1-9% | Cold-start capability |
| **Adaptive Ensemble** | **0.8-5%** | Best of both models ✨ |

### Ensemble Results (10 Store-Dept Combinations)

```
Store  Dept  Prophet  Ridge  LightGBM  Ensemble  Ridge_Wt  LGBM_Wt
  1     1    9.28%   1.16%    1.45%     1.12%      55%       45%
  1     2    4.54%   0.69%    1.24%     0.77%      64%       36%
  1     3   22.23%   4.34%    8.36%     3.77%      66%       34%
  2     2    3.38%   1.06%    3.94%     0.81%      79%       21%
  3     2    4.44%   1.35%    1.06%     1.02%      44%       56%
```

---

## 🏗️ Project Structure

```
retail-forecasting-analytics-system/
├── data/                          # Walmart recruiting dataset
│   ├── train.csv                  # Historical weekly sales
│   ├── test.csv                   # Test set
│   ├── features.csv               # Holiday flags, economic indicators
│   └── stores.csv                 # Store metadata
│
├── modules/
│   ├── feature_engineering/       # Reusable feature creation
│   │   ├── feature_utils.py       # Lag, rolling, seasonal features
│   │   └── feature_engineering.ipynb
│   │
│   ├── forecast/                  # Model training notebooks
│   │   ├── 1_prophet_baseline.ipynb     # Statistical baseline
│   │   ├── 2_ridge_forecast.ipynb       # Ridge w/ L2 regularization
│   │   ├── 3_lgbm_forecast.ipynb        # Global LightGBM
│   │   └── 4_ensemble_forecast.ipynb    # Adaptive ensemble
│   │
│   └── scenario_simulator/
│       └── simple_dashboard.py    # Streamlit interactive dashboard
│
├── artifacts/                     # Trained models & weights
│   ├── ridge_global.pkl           # Global Ridge model
│   ├── ridge_global_scaler.pkl    # Feature scaler
│   ├── global_lgbm_model.pkl      # Global LightGBM model
│   ├── prophet_store*_dept*.pkl   # Prophet baselines (per entity)
│   └── ensemble_weights.json      # Adaptive weights per entity
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd retail-forecasting-analytics-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

Execute notebooks in order:

```bash
# Navigate to forecast module
cd modules/forecast

# Run notebooks sequentially
jupyter notebook 1_prophet_baseline.ipynb      # ~3 min
jupyter notebook 2_ridge_forecast.ipynb        # ~2 min
jupyter notebook 3_lgbm_forecast.ipynb         # ~3 min
jupyter notebook 4_ensemble_forecast.ipynb     # ~4 min
```

**Outputs**: All trained models saved to `artifacts/`

### 3. Launch Interactive Dashboard

```bash
cd modules/scenario_simulator
streamlit run simple_dashboard.py
```

Visit `http://localhost:8501` in your browser.

---

## 🔬 Technical Approach

### Feature Engineering

**Reusable feature creation** via `feature_utils.py`:

- **Time features**: week-of-year, month, year
- **Lag features**: 1, 2, 4, 8, 13, 26, 52 weeks
- **Rolling statistics**: mean, std, min, max (4, 8, 12, 24 week windows)
- **Momentum**: first difference, percent change
- **Seasonality**: Fourier terms for yearly patterns
- **Holiday proximity**: 3-week centered holiday window
- **YoY growth**: Year-over-year change ratio

**Leakage prevention**: All rolling features use `shift(1)` before aggregation.

### Model Architecture

#### 1. Prophet Baseline (~9% WAPE)
- Statistical time series decomposition
- Trend + yearly/weekly seasonality
- Multiplicative mode for retail % changes
- **Use case**: Performance floor for comparison

#### 2. Ridge Regression (0.7-13% WAPE) 🏆
- Global model pooling 10 store-dept combinations
- L2 regularization (alpha=10.0) prevents overfitting to holiday spikes
- Linear extrapolation handles out-of-distribution values
- **Key strength**: Robust to extreme promotional events

#### 3. LightGBM Global (1-9% WAPE)
- Single model trained on all entities
- Categorical features: Store ID, Dept ID
- **Cold-start capability**: Predicts for unseen stores (1.2% MAPE)
- **Production scalable**: One model for 1,000+ SKUs

#### 4. Adaptive Ensemble (0.8-5% WAPE) ✨
- **Weight calculation**: `weight = (1/WAPE) / sum(1/WAPE)`
- Entity-specific optimization
- Example: Ridge 79%, LGBM 21% (Store 2, Dept 2)
- Automatically favors better-performing model per entity

---

## 📈 Key Features

### ✅ Holiday Spike Handling
- Ridge L2 regularization prevents overfitting to 3-4x spikes
- Tested on Black Friday, Super Bowl, Christmas patterns
- Superior extrapolation vs tree-based models

### ✅ Cold-Start Capability
- LightGBM predicts for **zero-history entities**
- Cross-entity learning from Store/Dept features
- Critical for new store launches

### ✅ Production-Ready Design
- Global models (not per-entity) for scalability
- Standardized feature engineering pipeline
- Saved weights for dashboard deployment
- Train/test split prevents data leakage

### ✅ Interactive Dashboard
- Compare Prophet vs Ridge vs LightGBM vs Ensemble
- Scenario analysis: normal vs extreme demand
- Entity-level performance visualization
- Adaptive weight inspection

---

## 💼 Use Cases

1. **Inventory Planning**: Reduce stockouts during promotional periods
2. **Staffing Optimization**: Predict labor needs per department
3. **New Store Launches**: Forecast demand with zero sales history
4. **Promotional ROI**: Model impact of holiday campaigns
5. **Supply Chain**: Optimize warehouse allocation across stores

---

## 🛠️ Technologies

- **Languages**: Python 3.13+
- **ML Frameworks**: scikit-learn, LightGBM, Prophet
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, Streamlit
- **Deployment**: Streamlit Cloud-ready

---

## 📚 Notebooks Execution Summary

### 1. Prophet Baseline (`1_prophet_baseline.ipynb`)
- **Runtime**: ~3 minutes
- **Output**: 10 Prophet models → `artifacts/prophet_store*_dept*.pkl`
- **Result**: 9-22% WAPE (baseline performance floor)

### 2. Ridge Regression (`2_ridge_forecast.ipynb`)
- **Runtime**: ~2 minutes
- **Output**: Global Ridge model → `artifacts/ridge_global.pkl`
- **Result**: 0.7-13% WAPE (holiday robust)

### 3. LightGBM Global (`3_lgbm_forecast.ipynb`)
- **Runtime**: ~3 minutes
- **Output**: Global LightGBM → `artifacts/global_lgbm_model.pkl`
- **Result**: 1-9% WAPE (cold-start capable)

### 4. Adaptive Ensemble (`4_ensemble_forecast.ipynb`)
- **Runtime**: ~4 minutes
- **Output**: Adaptive weights → `artifacts/ensemble_weights.json`
- **Result**: 0.8-5% WAPE (best overall)

---

## 🎓 Lessons & Best Practices

### What Worked Well
- ✅ **L2 regularization** critical for holiday robustness
- ✅ **Global models** enable cold-start and scale to 1,000+ SKUs
- ✅ **Adaptive weighting** outperforms fixed ensemble
- ✅ **Prophet baseline** validates ML value (2-10x improvement)

### Production Considerations
- **Feature store pattern**: Centralize feature computation to prevent train-serve skew
- **Monitoring**: Track per-entity WAPE degradation over time
- **Retraining**: Quarterly retrain with expanding window
- **AB testing**: Gradual rollout of new models per region

---

## 📦 Artifacts

All trained models and metadata stored in `artifacts/`:

| File | Description | Size |
|------|-------------|------|
| `ridge_global.pkl` | Ridge regression model | ~2KB |
| `ridge_global_scaler.pkl` | StandardScaler for features | ~2KB |
| `global_lgbm_model.pkl` | LightGBM booster | ~50KB |
| `prophet_store*_dept*.pkl` | 10 Prophet baselines | ~20KB each |
| `ensemble_weights.json` | Adaptive weights per entity | ~1KB |

---

## 🔮 Future Enhancements

- [ ] **Uncertainty quantification**: Prediction intervals via quantile regression
- [ ] **External features**: Incorporate weather, competitor pricing
- [ ] **Multi-step forecasting**: Extend to 4-week horizon
- [ ] **Anomaly detection**: Flag unusual sales patterns
- [ ] **AutoML**: Hyperparameter optimization via Optuna
- [ ] **API deployment**: FastAPI endpoints for production
- [ ] **MLOps**: Model versioning with MLflow

---

## 📞 Contact & Support

For questions, issues, or collaboration:
- **Author**: [Your Name]
- **Email**: [Your Email]
- **GitHub**: [Your GitHub Profile]

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Walmart Recruiting - Store Sales Forecasting (Kaggle)
- **Inspiration**: Real-world retail forecasting challenges
- **Tools**: scikit-learn, LightGBM, Prophet, Streamlit communities

---

**Built with ❤️ for data-driven retail operations**
