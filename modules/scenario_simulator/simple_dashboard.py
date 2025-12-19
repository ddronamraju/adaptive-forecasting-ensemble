import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Setup paths
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from modules.feature_engineering.feature_utils import make_features

def get_dynamic_ensemble_weights(y_test, y_train, ridge_pred, lgbm_pred):
    """
    Adaptive ensemble weighting based on inverse error.
    Returns weights and ensemble predictions.
    """
    # Calculate errors on training data (use a portion for validation)
    ridge_error = np.abs(y_test.values - ridge_pred).mean()
    lgbm_error = np.abs(y_test.values - lgbm_pred).mean()
    
    # Inverse error weighting (lower error = higher weight)
    ridge_w = 1 / (ridge_error + 1e-6)
    lgbm_w = 1 / (lgbm_error + 1e-6)
    
    # Normalize to sum to 1
    total = ridge_w + lgbm_w
    ridge_w = ridge_w / total
    lgbm_w = lgbm_w / total
    
    # Ensemble prediction
    ensemble_pred = ridge_w * ridge_pred + lgbm_w * lgbm_pred
    
    return ridge_w, lgbm_w, "adaptive", ensemble_pred

ARTIFACTS = repo_root / "artifacts"
DATA_PATH = repo_root / "data"

@st.cache_data
def load_sales_data():
    train = pd.read_csv(DATA_PATH / "train.csv", parse_dates=["Date"])
    features_df = pd.read_csv(DATA_PATH / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(DATA_PATH / "stores.csv")
    train = train.drop(columns=["IsHoliday"])
    df = train.merge(features_df, on=["Store", "Date"], how="left").merge(stores, on="Store")
    return df.sort_values(["Store", "Dept", "Date"])

@st.cache_resource
def load_prophet_model(store, dept):
    model_path = ARTIFACTS / f"prophet_store{store}_dept{dept}.pkl"
    return joblib.load(model_path) if model_path.exists() else None

@st.cache_resource
def load_global_lgbm_model():
    model_path = ARTIFACTS / "global_lgbm_model.pkl"
    return joblib.load(model_path) if model_path.exists() else None

@st.cache_resource
def load_global_ridge_model():
    model_path = ARTIFACTS / "ridge_global.pkl"
    scaler_path = ARTIFACTS / "ridge_global_scaler.pkl"
    features_path = ARTIFACTS / "ridge_global_features.txt"
    
    if not all([model_path.exists(), scaler_path.exists(), features_path.exists()]):
        return None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    return model, scaler, feature_names

def predict_ridge_global(X_val, store, dept, ridge_model, ridge_scaler, ridge_feature_names):
    """Make predictions with global Ridge model"""
    feature_cols = [col for col in X_val.columns if col not in ['Store', 'Dept', 'IsHoliday']]
    X_scaled = ridge_scaler.transform(X_val[feature_cols])
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X_val.index)
    X_scaled_df['Store'] = store
    X_scaled_df['Dept'] = dept
    X_encoded = pd.get_dummies(X_scaled_df, columns=['Store', 'Dept'], drop_first=True)
    
    for col in ridge_feature_names:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    
    X_encoded = X_encoded[ridge_feature_names]
    return ridge_model.predict(X_encoded)

# Page config
st.set_page_config(page_title="Prophet vs ML Ensemble", layout="wide")
st.title("🏪 Retail Forecasting: Prophet vs ML Ensemble")
st.markdown("**Can Machine Learning beat statistical forecasting on holiday sales?**")

# Load data
df = load_sales_data()

# Sidebar
st.sidebar.header("Entity Selection")
available_entities = [
    "Store 1, Dept 1", "Store 1, Dept 2", "Store 1, Dept 3", "Store 1, Dept 4", "Store 1, Dept 5",
    "Store 2, Dept 1", "Store 2, Dept 2", "Store 2, Dept 3",
    "Store 3, Dept 1", "Store 3, Dept 2"
]
selected_entity = st.sidebar.selectbox("Select Store-Department", available_entities, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Data**: Walmart sales (2010-2012)")
st.sidebar.markdown("🎄 **10 holiday weeks** in dataset")
st.sidebar.markdown("🎯 **Goal**: Test ML on holidays")

# Parse entity
store = int(selected_entity.split(",")[0].split()[1])
dept = int(selected_entity.split(",")[1].split()[1])

# Filter data for selected entity
sub = df[(df["Store"] == store) & (df["Dept"] == dept)].copy()
ts = sub[["Date", "Weekly_Sales", "IsHoliday"]].rename(columns={"Date": "ds", "Weekly_Sales": "y"}).set_index("ds")

# Generate features
feat = make_features(ts.rename(columns={"y": "Weekly_Sales"}), target="Weekly_Sales").dropna()
feat['Store'] = store
feat['Dept'] = dept

X = feat.drop(columns=["Weekly_Sales"])
y_true = feat["Weekly_Sales"]

# Tabs
tabs = st.tabs(["📈 Prophet Baseline", "🤖 ML Ensemble", "⚖️ Head-to-Head Comparison"])

# TAB 1: Prophet Baseline
with tabs[0]:
    st.subheader("Prophet: Statistical Baseline")
    st.markdown("Time series decomposition (trend + seasonality + holidays)")
    
    prophet_model = load_prophet_model(store, dept)
    
    if prophet_model is None:
        st.error("⚠️ Prophet model not found. Train models first.")
    else:
        h = 12
        
        if len(feat) < h:
            st.warning("Not enough data for test set.")
        else:
            y_test = y_true.iloc[-h:]
            
            # Prophet predictions
            prophet_df = pd.DataFrame({'ds': y_test.index})
            prophet_forecast = prophet_model.predict(prophet_df)
            prophet_pred = prophet_forecast['yhat'].values
            
            # Metrics
            prophet_wape = (np.abs(y_test.values - prophet_pred).sum() / np.abs(y_test.values).sum()) * 100
            prophet_mae = np.mean(np.abs(y_test.values - prophet_pred))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Test WAPE", f"{prophet_wape:.2f}%")
            col2.metric("Test MAE", f"{prophet_mae:,.0f}")
            col3.metric("Test Period", "Last 12 weeks")
            
            # Plot
            comparison_df = pd.DataFrame({
                "Actual": y_test.values,
                "Prophet": prophet_pred
            }, index=y_test.index)
            
            st.line_chart(comparison_df)
            
            # Holiday analysis
            test_holidays = sub[sub.index.isin(y_test.index) & (sub["IsHoliday"] == True)]
            if not test_holidays.empty:
                st.info(f"🎄 Test set contains {len(test_holidays)} holiday weeks")
                
                holiday_indices = test_holidays.index
                holiday_actuals = y_test[holiday_indices]
                holiday_preds = pd.Series(prophet_pred, index=y_test.index)[holiday_indices]
                holiday_wape = (np.abs(holiday_actuals - holiday_preds).sum() / np.abs(holiday_actuals).sum()) * 100
                
                st.metric("Holiday WAPE", f"{holiday_wape:.2f}%",
                         delta=f"{holiday_wape - prophet_wape:+.2f}% vs overall",
                         delta_color="inverse")

# TAB 2: ML Ensemble
with tabs[1]:
    st.subheader("ML Ensemble: Ridge + LightGBM")
    st.markdown("Global models with adaptive weighting")
    
    global_lgbm = load_global_lgbm_model()
    ridge_model, ridge_scaler, ridge_feature_names = load_global_ridge_model()
    
    if global_lgbm is None or ridge_model is None:
        st.error("⚠️ ML models not found. Run training notebooks first.")
    else:
        h = 12
        
        if len(feat) < h:
            st.warning("Not enough data for test set.")
        else:
            X_test = X.iloc[-h:]
            y_test = y_true.iloc[-h:]
            y_train = y_true.iloc[:-h]
            
            # Ridge predictions
            ridge_pred = predict_ridge_global(X_test, store, dept, ridge_model, ridge_scaler, ridge_feature_names)
            
            # LightGBM predictions
            X_test_lgbm = X_test.copy()
            X_test_lgbm['Store'] = X_test_lgbm['Store'].astype('category')
            X_test_lgbm['Dept'] = X_test_lgbm['Dept'].astype('category')
            lgbm_pred = global_lgbm.predict(X_test_lgbm)
            
            # Ensemble
            ridge_w, lgbm_w, scenario, ensemble_pred = get_dynamic_ensemble_weights(
                y_test, y_train, ridge_pred, lgbm_pred
            )
            
            # Metrics
            ensemble_wape = (np.abs(y_test.values - ensemble_pred).sum() / np.abs(y_test.values).sum()) * 100
            ridge_wape = (np.abs(y_test.values - ridge_pred).sum() / np.abs(y_test.values).sum()) * 100
            lgbm_wape = (np.abs(y_test.values - lgbm_pred).sum() / np.abs(y_test.values).sum()) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Ensemble WAPE", f"{ensemble_wape:.2f}%", help=f"Ridge {ridge_w:.1%} + LightGBM {lgbm_w:.1%}")
            col2.metric("Ridge WAPE", f"{ridge_wape:.2f}%")
            col3.metric("LightGBM WAPE", f"{lgbm_wape:.2f}%")
            
            st.info(f"🎯 {scenario} • Weights: Ridge ({ridge_w:.1%}) + LightGBM ({lgbm_w:.1%})")
            
            # Plot
            comparison_df = pd.DataFrame({
                "Actual": y_test.values,
                "Ensemble": ensemble_pred,
                "Ridge": ridge_pred,
                "LightGBM": lgbm_pred
            }, index=y_test.index)
            
            st.line_chart(comparison_df)
            
            # Holiday analysis
            test_holidays = sub[sub.index.isin(y_test.index) & (sub["IsHoliday"] == True)]
            if not test_holidays.empty:
                st.info(f"🎄 Test set contains {len(test_holidays)} holiday weeks")
                
                holiday_indices = test_holidays.index
                holiday_actuals = y_test[holiday_indices]
                holiday_ensemble = pd.Series(ensemble_pred, index=y_test.index)[holiday_indices]
                holiday_wape = (np.abs(holiday_actuals - holiday_ensemble).sum() / np.abs(holiday_actuals).sum()) * 100
                
                st.metric("Holiday WAPE", f"{holiday_wape:.2f}%",
                         delta=f"{holiday_wape - ensemble_wape:+.2f}% vs overall",
                         delta_color="inverse")

# TAB 3: Comparison
with tabs[2]:
    st.subheader("Prophet vs ML Ensemble: Head-to-Head")
    
    prophet_model = load_prophet_model(store, dept)
    global_lgbm = load_global_lgbm_model()
    ridge_model, ridge_scaler, ridge_feature_names = load_global_ridge_model()
    
    if prophet_model is None or global_lgbm is None or ridge_model is None:
        st.error("⚠️ Models not found. Train all models first.")
    else:
        h = 12
        
        if len(feat) < h:
            st.warning("Not enough data.")
        else:
            X_test = X.iloc[-h:]
            y_test = y_true.iloc[-h:]
            y_train = y_true.iloc[:-h]
            
            # Prophet
            prophet_df = pd.DataFrame({'ds': y_test.index})
            prophet_forecast = prophet_model.predict(prophet_df)
            prophet_pred = prophet_forecast['yhat'].values
            
            # ML Ensemble
            ridge_pred = predict_ridge_global(X_test, store, dept, ridge_model, ridge_scaler, ridge_feature_names)
            X_test_lgbm = X_test.copy()
            X_test_lgbm['Store'] = X_test_lgbm['Store'].astype('category')
            X_test_lgbm['Dept'] = X_test_lgbm['Dept'].astype('category')
            lgbm_pred = global_lgbm.predict(X_test_lgbm)
            ridge_w, lgbm_w, scenario, ensemble_pred = get_dynamic_ensemble_weights(
                y_test, y_train, ridge_pred, lgbm_pred
            )
            
            # Calculate WAPE
            prophet_wape = (np.abs(y_test.values - prophet_pred).sum() / np.abs(y_test.values).sum()) * 100
            ensemble_wape = (np.abs(y_test.values - ensemble_pred).sum() / np.abs(y_test.values).sum()) * 100
            
            # Determine winner
            if ensemble_wape < prophet_wape:
                improvement = ((prophet_wape - ensemble_wape) / prophet_wape) * 100
                st.success(f"🏆 **ML Ensemble wins!** {improvement:.1f}% better than Prophet")
            else:
                degradation = ((ensemble_wape - prophet_wape) / prophet_wape) * 100
                st.info(f"🏆 **Prophet wins.** ML is {degradation:.1f}% worse")
            
            # Metrics
            st.markdown("### Overall Test Performance")
            col1, col2 = st.columns(2)
            col1.metric("Prophet WAPE", f"{prophet_wape:.2f}%")
            col2.metric("ML Ensemble WAPE", f"{ensemble_wape:.2f}%",
                       delta=f"{ensemble_wape - prophet_wape:+.2f}%",
                       delta_color="inverse")
            
            # Plot
            comparison_df = pd.DataFrame({
                "Actual": y_test.values,
                "Prophet": prophet_pred,
                "ML Ensemble": ensemble_pred
            }, index=y_test.index)
            
            st.line_chart(comparison_df)
            
            # Holiday-specific
            test_holidays = sub[sub.index.isin(y_test.index) & (sub["IsHoliday"] == True)]
            if not test_holidays.empty:
                st.markdown("### 🎄 Holiday Performance")
                
                holiday_indices = test_holidays.index
                holiday_actuals = y_test[holiday_indices]
                holiday_prophet = pd.Series(prophet_pred, index=y_test.index)[holiday_indices]
                holiday_ensemble = pd.Series(ensemble_pred, index=y_test.index)[holiday_indices]
                
                holiday_prophet_wape = (np.abs(holiday_actuals - holiday_prophet).sum() / np.abs(holiday_actuals).sum()) * 100
                holiday_ensemble_wape = (np.abs(holiday_actuals - holiday_ensemble).sum() / np.abs(holiday_actuals).sum()) * 100
                
                hcol1, hcol2 = st.columns(2)
                hcol1.metric("Prophet Holiday WAPE", f"{holiday_prophet_wape:.2f}%")
                hcol2.metric("Ensemble Holiday WAPE", f"{holiday_ensemble_wape:.2f}%",
                            delta=f"{holiday_ensemble_wape - holiday_prophet_wape:+.2f}%",
                            delta_color="inverse")
                
                if holiday_ensemble_wape < holiday_prophet_wape:
                    improvement = ((holiday_prophet_wape - holiday_ensemble_wape)/holiday_prophet_wape)*100
                    st.success(f"✅ ML Ensemble is **{improvement:.1f}% better on holidays!**")
                else:
                    degradation = ((holiday_ensemble_wape - holiday_prophet_wape)/holiday_prophet_wape)*100
                    st.warning(f"⚠️ Prophet is **{degradation:.1f}% better on holidays**")
