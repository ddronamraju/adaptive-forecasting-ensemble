import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import sys
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parents[2]
sys.path.append(str(repo_root))

from modules.feature_engineering.feature_utils import make_features

# Page configuration
st.set_page_config(
    page_title="Retail Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ========== Data Loading ==========
@st.cache_data
def load_data():
    """Load original data"""
    data_path = Path(__file__).parents[2] / "data"
    
    train = pd.read_csv(data_path / "train.csv", parse_dates=["Date"])
    features = pd.read_csv(data_path / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(data_path / "stores.csv")
    
    train = train.drop(columns=['IsHoliday'])
    
    df = (
        train
        .merge(features, on=["Store", "Date"], how="left")
        .merge(stores, on="Store")
    )
    
    df = df.sort_values(["Store", "Dept", "Date"])
    return df

@st.cache_resource
def load_models():
    """Load trained models"""
    artifacts_path = Path(__file__).parents[2] / "artifacts"
    
    # Load LightGBM
    global_lgbm = joblib.load(artifacts_path / "global_lgbm_model.pkl")
    
    # Load Ridge
    ridge_global = joblib.load(artifacts_path / "ridge_global.pkl")
    ridge_scaler = joblib.load(artifacts_path / "ridge_global_scaler.pkl")
    
    # Load ensemble weights
    with open(artifacts_path / "ensemble_weights.json", 'r') as f:
        ensemble_weights = json.load(f)
    
    # Load Prophet models
    prophet_models = {}
    selected_combos = [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 1), (2, 2), (2, 3),
        (3, 1), (3, 2),
    ]
    for store, dept in selected_combos:
        model_path = artifacts_path / f"prophet_store{store}_dept{dept}.pkl"
        if model_path.exists():
            prophet_models[(store, dept)] = joblib.load(model_path)
    
    return global_lgbm, ridge_global, ridge_scaler, ensemble_weights, prophet_models

@st.cache_data
def compute_all_predictions(_global_lgbm, _ridge_global, _ridge_scaler, _prophet_models, ensemble_weights, df):
    """Compute predictions for all entities"""
    selected_combos = [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 1), (2, 2), (2, 3),
        (3, 1), (3, 2),
    ]
    
    all_results = []
    VAL_SIZE = 12
    TEST_SIZE = 12
    
    for store, dept in selected_combos:
        ts = (
            df[(df["Store"] == store) & (df["Dept"] == dept)]
            [["Date", "Weekly_Sales", "IsHoliday"]]
            .set_index("Date")
            .sort_index()
        )
        
        feat_df = make_features(ts, target="Weekly_Sales").dropna()
        
        # Test set
        X_test_base = feat_df.drop(columns=["Weekly_Sales", "IsHoliday"]).iloc[-TEST_SIZE:]
        y_test = feat_df["Weekly_Sales"].iloc[-TEST_SIZE:]
        
        # Ridge predictions
        X_test_scaled = _ridge_scaler.transform(X_test_base)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_base.columns, index=X_test_base.index)
        X_test_scaled_df['Store'] = store
        X_test_scaled_df['Dept'] = dept
        X_test_final = pd.get_dummies(X_test_scaled_df, columns=['Store', 'Dept'], drop_first=True)
        
        for col in _ridge_global.feature_names_in_:
            if col not in X_test_final.columns:
                X_test_final[col] = 0
        X_test_final = X_test_final[_ridge_global.feature_names_in_]
        
        ridge_preds = _ridge_global.predict(X_test_final)
        ridge_wape = (np.abs(y_test.values - ridge_preds).sum() / np.abs(y_test.values).sum()) * 100
        
        # LightGBM predictions
        X_test_lgbm = feat_df.drop(columns=["Weekly_Sales"]).iloc[-TEST_SIZE:].copy()
        X_test_lgbm['Store'] = store
        X_test_lgbm['Dept'] = dept
        X_test_lgbm['Store'] = X_test_lgbm['Store'].astype('category')
        X_test_lgbm['Dept'] = X_test_lgbm['Dept'].astype('category')
        lgbm_preds = _global_lgbm.predict(X_test_lgbm)
        lgbm_wape = (np.abs(y_test.values - lgbm_preds).sum() / np.abs(y_test.values).sum()) * 100
        
        # Prophet predictions
        prophet_wape = None
        prophet_preds = None
        if (store, dept) in _prophet_models:
            prophet_df = pd.DataFrame({'ds': y_test.index})
            prophet_forecast = _prophet_models[(store, dept)].predict(prophet_df)
            prophet_preds = prophet_forecast['yhat'].values
            prophet_wape = (np.abs(y_test.values - prophet_preds).sum() / np.abs(y_test.values).sum()) * 100
        
        # Ensemble predictions
        key = f"{store}_{dept}"
        weights = ensemble_weights[key] if key in ensemble_weights else {"ridge_weight": 0.5, "lgbm_weight": 0.5}
        ensemble_preds = weights['ridge_weight'] * ridge_preds + weights['lgbm_weight'] * lgbm_preds
        ensemble_wape = (np.abs(y_test.values - ensemble_preds).sum() / np.abs(y_test.values).sum()) * 100
        
        all_results.append({
            "Store": store,
            "Dept": dept,
            "Entity": f"Store {store}, Dept {dept}",
            "Prophet_WAPE": prophet_wape,
            "Ridge_WAPE": ridge_wape,
            "LGBM_WAPE": lgbm_wape,
            "Ensemble_WAPE": ensemble_wape,
            "Ridge_Weight": weights['ridge_weight'],
            "LGBM_Weight": weights['lgbm_weight'],
            "Actuals": y_test.values,
            "Dates": y_test.index,
            "Ridge_Preds": ridge_preds,
            "LGBM_Preds": lgbm_preds,
            "Prophet_Preds": prophet_preds,
            "Ensemble_Preds": ensemble_preds,
            "IsHoliday": X_test_lgbm['IsHoliday'].values
        })
    
    return all_results

# ========== Main App ==========
def main():
    # Title
    st.title("📊 Retail Sales Forecasting Dashboard")
    st.markdown("### Adaptive Ensemble Model Performance Analysis")
    st.markdown("---")
    
    # Load data and models
    with st.spinner("Loading data and models..."):
        df = load_data()
        global_lgbm, ridge_global, ridge_scaler, ensemble_weights, prophet_models = load_models()
        all_results = compute_all_predictions(global_lgbm, ridge_global, ridge_scaler, prophet_models, ensemble_weights, df)
    
    # Convert to DataFrame
    results_df = pd.DataFrame([{
        "Store": r["Store"],
        "Dept": r["Dept"],
        "Entity": r["Entity"],
        "Prophet": f"{r['Prophet_WAPE']:.2f}%" if r['Prophet_WAPE'] else "N/A",
        "Ridge": f"{r['Ridge_WAPE']:.2f}%",
        "LightGBM": f"{r['LGBM_WAPE']:.2f}%",
        "Ensemble": f"{r['Ensemble_WAPE']:.2f}%",
        "Ridge_Wt": f"{r['Ridge_Weight']:.0%}",
        "LGBM_Wt": f"{r['LGBM_Weight']:.0%}",
    } for r in all_results])
    
    # ========== Section 1: Overall Metrics ==========
    st.header("🎯 Overall Model Performance")
    st.markdown("Average WAPE across all 10 store-department combinations")
    
    # Calculate averages
    avg_prophet = np.mean([r['Prophet_WAPE'] for r in all_results if r['Prophet_WAPE'] is not None])
    avg_ridge = np.mean([r['Ridge_WAPE'] for r in all_results])
    avg_lgbm = np.mean([r['LGBM_WAPE'] for r in all_results])
    avg_ensemble = np.mean([r['Ensemble_WAPE'] for r in all_results])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Prophet (Baseline)",
            f"{avg_prophet:.2f}%",
            delta=None,
            help="Statistical baseline model - per entity"
        )
    
    with col2:
        improvement = ((avg_prophet - avg_ridge) / avg_prophet) * 100
        st.metric(
            "Ridge Regression",
            f"{avg_ridge:.2f}%",
            delta=f"-{improvement:.1f}%",
            delta_color="inverse",
            help="Global linear model - good for extrapolation"
        )
    
    with col3:
        improvement = ((avg_prophet - avg_lgbm) / avg_prophet) * 100
        st.metric(
            "LightGBM",
            f"{avg_lgbm:.2f}%",
            delta=f"-{improvement:.1f}%",
            delta_color="inverse",
            help="Global tree-based model - handles cold-start"
        )
    
    with col4:
        improvement = ((avg_prophet - avg_ensemble) / avg_prophet) * 100
        st.metric(
            "⭐ Ensemble",
            f"{avg_ensemble:.2f}%",
            delta=f"-{improvement:.1f}%",
            delta_color="inverse",
            help="Adaptive weighted combination of Ridge + LightGBM"
        )
    
    # ========== Section 2: Model Comparison ==========
    st.markdown("---")
    st.header("📈 Model Comparison")
    
    col_chart, col_insights = st.columns([2, 1])
    
    with col_chart:
        fig, ax = plt.subplots(figsize=(10, 6))
        models = ['Prophet', 'Ridge', 'LightGBM', 'Ensemble']
        wapes = [avg_prophet, avg_ridge, avg_lgbm, avg_ensemble]
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700']
        
        bars = ax.bar(models, wapes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, wape in zip(bars, wapes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{wape:.2f}%',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Average WAPE (%)', fontsize=13, fontweight='bold')
        ax.set_title('Average Model Performance Across All Entities', fontsize=15, fontweight='bold')
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, max(wapes) * 1.15)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_insights:
        st.markdown("### 🏆 Key Insights")
        st.success(f"✅ **Ensemble reduces error by {((avg_prophet - avg_ensemble) / avg_prophet) * 100:.1f}%** vs Prophet baseline")
        st.info(f"📊 Evaluated on **10 entities** (store-dept combinations)")
        st.info(f"📅 Test period: **12 weeks** (out-of-sample)")
        st.info(f"⚖️ Validation period: **12 weeks** (for weight tuning)")
        
        # Best model count
        best_counts = {"Ridge": 0, "LightGBM": 0, "Ensemble": 0, "Prophet": 0}
        for r in all_results:
            wapes = {
                "Ridge": r['Ridge_WAPE'],
                "LightGBM": r['LGBM_WAPE'],
                "Ensemble": r['Ensemble_WAPE']
            }
            if r['Prophet_WAPE']:
                wapes["Prophet"] = r['Prophet_WAPE']
            
            best_model = min(wapes, key=wapes.get)
            best_counts[best_model] += 1
        
        st.markdown(f"**Wins by Model:**")
        for model, count in best_counts.items():
            if count > 0:
                st.markdown(f"- {model}: {count} entities")
    
    # ========== Section 3: Interactive Entity Analysis ==========
    st.markdown("---")
    st.header("🔍 Entity-Level Time Series Analysis")
    
    col_store, col_dept = st.columns(2)
    with col_store:
        selected_store = st.selectbox("Select Store", [1, 2, 3], index=0)
    with col_dept:
        available_depts = sorted(set([r['Dept'] for r in all_results if r['Store'] == selected_store]))
        selected_dept = st.selectbox("Select Department", available_depts, index=0)
    
    # Get selected entity data
    selected_result = next(r for r in all_results if r['Store'] == selected_store and r['Dept'] == selected_dept)
    
    # Display weights
    st.markdown(f"**Adaptive Ensemble Weights for Store {selected_store}, Dept {selected_dept}:**")
    weight_col1, weight_col2 = st.columns(2)
    with weight_col1:
        st.metric("Ridge Weight", f"{selected_result['Ridge_Weight']:.0%}")
    with weight_col2:
        st.metric("LightGBM Weight", f"{selected_result['LGBM_Weight']:.0%}")
    
    # Time series plot
    fig, ax = plt.subplots(figsize=(14, 7))
    weeks = np.arange(len(selected_result['Actuals']))
    
    ax.plot(weeks, selected_result['Actuals'], 'o-', label='Actual', 
            linewidth=3, markersize=10, color='black', zorder=5)
    ax.plot(weeks, selected_result['Ensemble_Preds'], 's-', 
            label=f"Ensemble (WAPE: {selected_result['Ensemble_WAPE']:.1f}%)", 
            alpha=0.9, linewidth=2.5, markersize=8, color='#FFD700')
    ax.plot(weeks, selected_result['Ridge_Preds'], '^-', 
            label=f"Ridge (WAPE: {selected_result['Ridge_WAPE']:.1f}%)", 
            alpha=0.7, linewidth=2, markersize=7, color='#66B2FF')
    ax.plot(weeks, selected_result['LGBM_Preds'], 'v-', 
            label=f"LightGBM (WAPE: {selected_result['LGBM_WAPE']:.1f}%)", 
            alpha=0.7, linewidth=2, markersize=7, color='#99FF99')
    
    if selected_result['Prophet_Preds'] is not None:
        ax.plot(weeks, selected_result['Prophet_Preds'], 'd-', 
                label=f"Prophet (WAPE: {selected_result['Prophet_WAPE']:.1f}%)", 
                alpha=0.6, linewidth=1.5, markersize=6, color='#FF9999')
    
    # Mark holidays
    for i, is_holiday in enumerate(selected_result['IsHoliday']):
        if is_holiday:
            ax.axvline(x=i, color='red', linestyle='--', alpha=0.3, linewidth=2)
            ax.text(i, ax.get_ylim()[1] * 0.98, '🎄', ha='center', fontsize=14)
    
    ax.set_title(f'Store {selected_store}, Dept {selected_dept}: Test Set Predictions (12 Weeks)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Week Number', fontsize=13)
    ax.set_ylabel('Weekly Sales ($)', fontsize=13)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ========== Section 4: Full Results Table ==========
    st.markdown("---")
    st.header("📋 Complete Results Table")
    
    # Format for display
    display_df = results_df.copy()
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Entity": st.column_config.TextColumn("Store-Dept", width="medium"),
            "Prophet": st.column_config.TextColumn("Prophet WAPE", width="small"),
            "Ridge": st.column_config.TextColumn("Ridge WAPE", width="small"),
            "LightGBM": st.column_config.TextColumn("LightGBM WAPE", width="small"),
            "Ensemble": st.column_config.TextColumn("Ensemble WAPE", width="small"),
            "Ridge_Wt": st.column_config.TextColumn("Ridge Wt", width="small"),
            "LGBM_Wt": st.column_config.TextColumn("LGBM Wt", width="small"),
        }
    )
    
    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="forecast_results.csv",
        mime="text/csv",
        help="Download the complete results table"
    )
    
    # ========== Footer ==========
    st.markdown("---")
    st.markdown("""
    **About This Dashboard:**
    - **Ensemble Model**: Adaptive weighted combination of Ridge Regression + LightGBM
    - **Weights**: Optimized per entity using validation set performance (no data leakage)
    - **Metric**: WAPE (Weighted Absolute Percentage Error) - lower is better
    - **Test Period**: Last 12 weeks of available data (held out from training)
    - **Prophet**: Baseline statistical model for comparison
    """)

if __name__ == "__main__":
    main()
