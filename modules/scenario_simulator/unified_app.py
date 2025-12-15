from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib

# Make repo root importable
import sys
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from modules.feature_engineering.feature_utils import make_features
from modules.scenario_simulator.engine import simulate_scenario

ARTIFACTS = repo_root / "artifacts"

@st.cache_data
def load_sales_frame():
    train = pd.read_csv(repo_root / "modules/baseline_prophet_forecast/data/train.csv")
    features = pd.read_csv(repo_root / "modules/baseline_prophet_forecast/data/features.csv")
    stores = pd.read_csv(repo_root / "modules/baseline_prophet_forecast/data/stores.csv")

    df = (
        train.merge(features, on=["Store", "Date", "IsHoliday"])
             .merge(stores, on="Store")
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Store", "Dept", "Date"])

@st.cache_resource
def load_lgbm_model():
    model_path = ARTIFACTS / "lgbm_model_store1_dept1.pkl"
    return joblib.load(model_path)

@st.cache_resource
def load_global_lgbm_model():
    model_path = ARTIFACTS / "global_lgbm_model.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None

def load_elasticity(store=1, dept=1):
    p = ARTIFACTS / f"elasticity_store{store}_dept{dept}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"store": store, "dept": dept, "elasticity": -1.2, "promo_uplift": 0.25}

def wape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return float((np.abs(y_true.values - y_pred).sum() / denom) * 100) if denom != 0 else np.nan

def bias_pct(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return float(((y_pred - y_true.values).sum() / denom) * 100) if denom != 0 else np.nan


@st.cache_data
def load_sku_clusters():
    p = ARTIFACTS / "sku_clusters.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

st.set_page_config(page_title="Retail Decision Intelligence", layout="wide")
st.title("Retail Forecasting & Decision Intelligence")

df = load_sales_frame()

st.sidebar.header("Entity Selection")
store = st.sidebar.number_input("Store", min_value=1, value=1, step=1)
dept = st.sidebar.number_input("Department", min_value=1, value=1, step=1)

sub = df[(df["Store"] == store) & (df["Dept"] == dept)].copy()
sub = sub.sort_values("Date")

if sub.empty:
    st.error("No data found for this Store–Dept.")
    st.stop()

ts = sub[["Date", "Weekly_Sales", "IsHoliday"]].rename(columns={"Date": "ds", "Weekly_Sales": "y"})
#ts = ts.set_index("ds").asfreq("W")

# Build feature table for forecasting KPIs
model_df = ts.rename(columns={"y": "Weekly_Sales"}).copy()
feat = make_features(model_df, target="Weekly_Sales").dropna()

# Holdout window for monitoring metrics
H = 12
X = feat.drop(columns=["Weekly_Sales"])
y_true = feat["Weekly_Sales"]

have_enough = len(feat) >= (H + 20)  # basic guard

pred = None
out = None

if have_enough:
    model = load_lgbm_model()
    X_val = X.iloc[-H:]
    y_val = y_true.iloc[-H:]
    pred = model.predict(X_val)
    out = pd.DataFrame({"Actual": y_val.values, "Forecast": pred}, index=y_val.index)

tabs = st.tabs(["KPI Dashboard", "Forecast (LightGBM)", "Global Model 🌍", "Anomalies", "Segmentation", "Inventory Risk", "Pricing Scenario"])

with tabs[0]:
    st.subheader("KPI Dashboard (Monitoring)")

    # Headline metrics
    recent_avg = float(ts["y"].tail(12).mean())
    latest_week = ts.index.max()
    latest_sales = float(ts["y"].iloc[-1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Week", str(latest_week.date()))
    c2.metric("Latest Weekly Sales", f"{latest_sales:,.0f}")
    c3.metric("Recent Avg (12 wks)", f"{recent_avg:,.0f}")

    if out is None:
        c4.metric("WAPE (last 12 wks)", "N/A")
        st.warning("Not enough history after feature generation to compute forecast KPIs.")
    else:
        w = wape(out["Actual"], out["Forecast"].values)
        b = bias_pct(out["Actual"], out["Forecast"].values)
        c4.metric("WAPE (last 12 wks)", f"{w:.2f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Bias (sum error / sum actual)", f"{b:.2f}%")
        c6.metric("MAE (last 12 wks)", f"{np.mean(np.abs(out['Actual'] - out['Forecast'])):,.0f}")
        c7.metric("Over-forecast weeks", str(int((out["Forecast"] > out["Actual"]).sum())))
        c8.metric("Under-forecast weeks", str(int((out["Forecast"] < out["Actual"]).sum())))

        st.markdown("### Forecast vs Actual (Monitoring Window)")
        st.line_chart(out)

        st.markdown("### Error Trend (Forecast - Actual)")
        err = (out["Forecast"] - out["Actual"])
        st.line_chart(err.rename("Error"))

        st.markdown("### Worst Weeks by Absolute Error")
        worst = out.copy()
        worst["abs_error"] = np.abs(worst["Actual"] - worst["Forecast"])
        st.dataframe(worst.sort_values("abs_error", ascending=False).head(10), use_container_width=True)

    st.divider()

    # Anomaly KPIs (reuse anomaly logic quickly)
    st.markdown("### Anomaly Summary")
    window = 8
    threshold = 2.0
    tmp = ts.copy()
    tmp["rolling_mean"] = tmp["y"].rolling(window).mean()
    tmp["rolling_std"] = tmp["y"].rolling(window).std()
    tmp["z"] = (tmp["y"] - tmp["rolling_mean"]) / tmp["rolling_std"]
    tmp = tmp.dropna()
    tmp["anomaly"] = tmp["z"].abs() > threshold

    anomaly_count_12 = int(tmp["anomaly"].tail(12).sum())
    last_anom_date = tmp[tmp["anomaly"]].index.max() if tmp["anomaly"].any() else None

    a1, a2, a3 = st.columns(3)
    a1.metric("Anomalies (last 12 wks)", str(anomaly_count_12))
    a2.metric("Most recent anomaly", str(last_anom_date.date()) if last_anom_date is not None else "None")
    a3.metric("Default rule", f"window={window}, |z|>{threshold}")

    st.divider()

    # Inventory risk headline (quick compute)
    st.markdown("### Inventory Risk Snapshot")
    lead_time = 2
    service_level = 0.95

    lt = ts["y"].rolling(lead_time).sum().dropna()
    mu, sigma = float(lt.mean()), float(lt.std())

    if sigma == 0 or np.isnan(sigma):
        st.info("Insufficient variance to compute inventory risk metrics.")
    else:
        from scipy.stats import norm
        z = float(norm.ppf(service_level))
        ss = z * sigma
        rop = mu + ss

        i1, i2, i3 = st.columns(3)
        i1.metric("Lead time (weeks)", str(lead_time))
        i2.metric("Safety Stock (95% SL)", f"{ss:,.0f}")
        i3.metric("Reorder Point (μ+SS)", f"{rop:,.0f}")

    st.divider()

    # Segmentation (if available)
    st.markdown("### SKU Segment (if exported)")
    clusters = load_sku_clusters()
    sku = f"{store}_{dept}"
    if clusters is None:
        st.info("Segmentation not available yet (export artifacts/sku_clusters.csv to enable this KPI).")
    else:
        row = clusters[clusters["sku"] == sku]
        if row.empty:
            st.info("No cluster found for this Store–Dept in the saved segmentation file.")
        else:
            st.metric("Segment / Cluster", str(int(row["cluster"].iloc[0])))


with tabs[1]:
    st.subheader("LightGBM Forecast (feature-engineered)")

    model = load_lgbm_model()

    model_df = ts.rename(columns={"y": "Weekly_Sales"}).copy()
    feat = make_features(model_df, target="Weekly_Sales").dropna()

    h = 12
    X = feat.drop(columns=["Weekly_Sales"])
    y_true = feat["Weekly_Sales"]

    if len(feat) < h + 60:
        st.warning("Not much history for robust features. Results may be unstable.")

    X_val = X.iloc[-h:]
    y_val = y_true.iloc[-h:]
    pred = model.predict(X_val)

    out = pd.DataFrame({"Actual": y_val.values, "Forecast": pred}, index=y_val.index)
    st.line_chart(out)

    wape = (np.abs(out["Actual"] - out["Forecast"]).sum() / np.abs(out["Actual"]).sum()) * 100
    st.metric("WAPE (last 12 weeks)", f"{wape:.2f}%")

with tabs[2]:
    st.subheader("🌍 Global Model Forecast (Cross-Entity Learning)")
    
    global_model = load_global_lgbm_model()
    
    if global_model is None:
        st.warning("⚠️ Global model not found. Run the `global_lgbm_forecast.ipynb` notebook first to generate `artifacts/global_lgbm_model.pkl`.")
    else:
        st.info("**Global Model**: Trained across multiple Store-Dept combinations. Can predict for entities with limited or zero historical data.")
        
        # Generate features for current entity
        model_df = ts.rename(columns={"y": "Weekly_Sales"}).copy()
        feat = make_features(model_df, target="Weekly_Sales").dropna()
        
        # Add entity identifiers
        feat['Store'] = store
        feat['Dept'] = dept
        
        h = 12
        if len(feat) < h + 10:
            st.warning("⚠️ Insufficient history for this Store-Dept combination.")
        else:
            X = feat.drop(columns=["Weekly_Sales"])
            y_true = feat["Weekly_Sales"]
            
            # Convert Store and Dept to categorical
            X['Store'] = X['Store'].astype('category')
            X['Dept'] = X['Dept'].astype('category')
            
            X_val = X.iloc[-h:]
            y_val = y_true.iloc[-h:]
            
            try:
                pred = global_model.predict(X_val)
                
                out = pd.DataFrame({"Actual": y_val.values, "Global_Forecast": pred}, index=y_val.index)
                st.line_chart(out)
                
                mae = np.mean(np.abs(out["Actual"] - out["Global_Forecast"]))
                wape = (np.abs(out["Actual"] - out["Global_Forecast"]).sum() / np.abs(out["Actual"]).sum()) * 100
                
                col1, col2 = st.columns(2)
                col1.metric("MAE (last 12 weeks)", f"${mae:,.2f}")
                col2.metric("WAPE (last 12 weeks)", f"{wape:.2f}%")
                
                st.markdown("---")
                st.markdown("**✅ Advantages of Global Model:**")
                st.markdown("- Learns from patterns across multiple stores/departments")
                st.markdown("- Can forecast for new stores/products with minimal history (cold-start)")
                st.markdown("- Single model to maintain vs. thousands of per-entity models")
                
            except Exception as e:
                st.error(f"❌ Error generating global forecast: {str(e)}")
                st.info("The global model may need features that aren't present in the current entity's data. Ensure the model was trained with compatible features.")

with tabs[3]:
    st.subheader("Anomaly Detection")

    # Load pre-computed anomalies from artifact
    try:
        anomalies = pd.read_csv("../../artifacts/anomalies_store1_dept1.csv", index_col=0, parse_dates=True)
        
        st.write(f"**Detected {len(anomalies)} anomalies** using rolling z-score method (window=8, threshold=2.0)")
        
        # Show time series with anomalies highlighted
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts.index, ts["y"], label="Weekly Sales", linewidth=1.5)
        ax.scatter(anomalies.index, anomalies["Weekly_Sales"], 
                  color="red", label="Anomaly", s=60, zorder=3)
        ax.set_xlabel("Date")
        ax.set_ylabel("Weekly Sales")
        ax.set_title("Sales with Detected Anomalies")
        ax.legend()
        st.pyplot(fig)
        
        # Show anomaly table
        st.write("**Anomaly Details:**")
        display_cols = ["Weekly_Sales", "z_score", "IsHoliday"]
        st.dataframe(anomalies[display_cols].sort_values("z_score", key=lambda s: s.abs(), ascending=False))
        
    except FileNotFoundError:
        st.warning("⚠️ Anomaly artifacts not found. Run the anomaly_detection notebook first to generate artifacts.")

with tabs[4]:
    st.subheader("SKU Segmentation")

    clusters = load_sku_clusters()
    sku = f"{store}_{dept}"

    if clusters is None:
        st.warning("No saved clusters found. Run the segmentation notebook and export artifacts/sku_clusters.csv.")
    else:
        row = clusters[clusters["sku"] == sku]
        if row.empty:
            st.info("This Store–Dept is not present in the saved cluster file.")
        else:
            cluster_id = int(row["cluster"].iloc[0])
            st.metric("Segment / Cluster", str(cluster_id))
            st.write("Use this segment to vary policies (service levels, safety stock multipliers, model strategy).")

with tabs[5]:
    st.subheader("Inventory Risk (Safety Stock + Stock-out Probability)")

    lead_time = st.slider("Lead time (weeks)", 1, 8, 2)
    service_level = st.slider("Service level", 0.80, 0.99, 0.95)

    # lead-time demand
    lt = ts["y"].rolling(lead_time).sum().dropna()
    mu, sigma = lt.mean(), lt.std()

    if sigma == 0 or np.isnan(sigma):
        st.warning("Insufficient variance to compute safety stock reliably.")
    else:
        from scipy.stats import norm
        z = norm.ppf(service_level)
        ss = z * sigma
        st.metric("Mean lead-time demand", f"{mu:,.0f}")
        st.metric("Safety stock", f"{ss:,.0f}")
        st.metric("Reorder point (mu + safety stock)", f"{(mu+ss):,.0f}")

        inv = np.linspace(mu*0.5, mu*2.0, 60)
        stockout = 1 - norm.cdf(inv, loc=mu, scale=sigma)
        chart = pd.DataFrame({"Stockout_Probability": stockout}, index=inv)
        st.line_chart(chart)

with tabs[6]:
    st.subheader("Pricing & Promotion Scenario")

    e = load_elasticity(store, dept)
    elasticity = st.number_input("Elasticity", value=float(e["elasticity"]), step=0.1)
    promo_uplift = st.number_input("Promo uplift (fraction)", value=float(e["promo_uplift"]), step=0.05)

    base = float(ts["y"].tail(12).mean())
    st.write(f"Baseline demand used (last 12-week avg): {base:,.0f}")

    price_change = st.slider("Price change (%)", -40, 40, 0)
    promo = st.selectbox("Promotion active?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    result = simulate_scenario(
        base_demand=base,
        price_change_pct=price_change,
        elasticity=elasticity,
        promo_flag=promo,
        promo_uplift=promo_uplift,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline", f"{result['base_demand']:,.0f}")
    c2.metric("Scenario", f"{result['scenario_demand']:,.0f}")
    c3.metric("Delta (%)", f"{result['delta_pct']:.1f}%")

