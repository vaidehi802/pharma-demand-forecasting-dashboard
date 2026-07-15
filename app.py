"""
app.py
------
RxForecast: Pharma Sales Demand & Territory Optimization Dashboard

Streamlit app combining:
  1. ARIMA quarterly demand forecasts by drug-category/product line (real Kaggle data)
  2. ML-driven physician under-prescriber risk scoring (simulated HCP panel data)
  3. A simple territory-reallocation recommendation view that combines both

Run locally:   streamlit run dashboard/app.py
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="RxForecast Dashboard", layout="wide")

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

PRODUCT_LABELS = {
    "M01AB": "M01AB — NSAIDs (acetic acid derivatives)",
    "M01AE": "M01AE — NSAIDs (propionic acid derivatives)",
    "N02BA": "N02BA — Analgesics (salicylic acid)",
    "N02BE": "N02BE — Analgesics (pyrazolones/anilides)",
    "N05B": "N05B — Anxiolytics",
    "N05C": "N05C — Hypnotics/sedatives",
    "R03": "R03 — Obstructive airway disease (asthma/COPD)",
    "R06": "R06 — Systemic antihistamines",
}


@st.cache_data
def load_all():
    sales = pd.read_csv(f"{DATA_DIR}/salesmonthly.csv", parse_dates=["datum"])
    forecast_results = pd.read_csv(f"{OUT_DIR}/forecast_results.csv")
    forecast_detail = pd.read_csv(f"{OUT_DIR}/forecast_detail.csv", parse_dates=["date"])
    physician_preds = pd.read_csv(f"{OUT_DIR}/physician_predictions.csv")
    with open(f"{OUT_DIR}/physician_model_metrics.json") as f:
        physician_metrics = json.load(f)
    feature_importance = pd.read_csv(f"{OUT_DIR}/physician_feature_importance.csv")
    return sales, forecast_results, forecast_detail, physician_preds, physician_metrics, feature_importance


sales, forecast_results, forecast_detail, physician_preds, physician_metrics, feature_importance = load_all()

st.title("💊 RxForecast")
st.caption("Pharma Sales Demand Forecasting & Territory Optimization — built on real Kaggle pharma sales data + simulated HCP commercial data")

tab1, tab2, tab3 = st.tabs(["📈 Demand Forecasting", "🎯 Physician Targeting", "🗺️ Territory Recommendations"])

# ---------------------------------------------------------------------------
# TAB 1 — Forecasting
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Quarterly Demand Forecast by Product Line")

    avg_mape = forecast_results["holdout_mape_pct"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg. Holdout MAPE", f"{avg_mape:.1f}%")
    c2.metric("Implied Forecast Accuracy", f"{100 - avg_mape:.1f}%")
    c3.metric("Product Lines Modeled", f"{len(forecast_results)}")

    product = st.selectbox(
        "Select product line",
        options=forecast_results["product"].tolist(),
        format_func=lambda p: PRODUCT_LABELS.get(p, p),
    )

    hist = sales[["datum", product]].rename(columns={product: "units"})
    detail = forecast_detail[forecast_detail["product"] == product]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["datum"], y=hist["units"], name="Historical Actuals", mode="lines"))
    holdout = detail[detail["type"] == "holdout_test"]
    fig.add_trace(go.Scatter(x=holdout["date"], y=holdout["forecast"], name="Holdout Forecast (test)", mode="lines+markers", line=dict(dash="dot")))
    future = detail[detail["type"] == "next_quarter_forecast"]
    fig.add_trace(go.Scatter(x=future["date"], y=future["forecast"], name="Next-Quarter Forecast", mode="lines+markers", line=dict(dash="dash", color="firebrick")))
    fig.update_layout(height=450, xaxis_title="Month", yaxis_title="Units Sold", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    row = forecast_results[forecast_results["product"] == product].iloc[0]
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Holdout MAPE", f"{row['holdout_mape_pct']:.1f}%")
    cc2.metric("ARIMA Order", row["order"])
    cc3.metric("Next Qtr Total (3 mo)", f"{row['next_quarter_total_units']:,.0f} units")

    with st.expander("All product lines — model comparison"):
        display_df = forecast_results.copy()
        display_df["product"] = display_df["product"].map(lambda p: PRODUCT_LABELS.get(p, p))
        st.dataframe(display_df, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — Physician ML model
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Under-Prescriber Risk Model (ML Classification)")
    st.info(
        "Physician-level prescribing data isn't publicly available (proprietary / "
        "HIPAA-adjacent). This model runs on a **simulated** HCP panel dataset built "
        "to mirror the feature relationships seen in real pharma commercial engagements.",
        icon="ℹ️",
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", physician_metrics["roc_auc"])
    m2.metric("Accuracy", f"{physician_metrics['accuracy']*100:.1f}%")
    m3.metric("Precision", physician_metrics["precision"])
    m4.metric("Base Rate (under-prescriber)", f"{physician_metrics['base_rate_under_prescriber']*100:.1f}%")

    fi_fig = px.bar(feature_importance, x="importance", y="feature", orientation="h",
                     title="Feature Importance — What Predicts Under-Prescribing")
    fi_fig.update_layout(yaxis=dict(categoryorder="total ascending"), height=350)
    st.plotly_chart(fi_fig, use_container_width=True)

    st.subheader("Highest-Risk Physicians (Most Recent Quarter)")
    region_filter = st.multiselect("Filter by region", options=sorted(physician_preds["region"].unique()))
    spec_filter = st.multiselect("Filter by specialty", options=sorted(physician_preds["specialty"].unique()))

    filtered = physician_preds.copy()
    if region_filter:
        filtered = filtered[filtered["region"].isin(region_filter)]
    if spec_filter:
        filtered = filtered[filtered["specialty"].isin(spec_filter)]

    st.dataframe(
        filtered.sort_values("under_prescriber_risk_score", ascending=False).head(50),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# TAB 3 — Territory recommendations
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Simulated Territory Reallocation Recommendations")
    st.caption(
        "Combines demand growth forecasts (Tab 1) with physician under-prescriber "
        "risk concentration (Tab 2) to flag regions where sales rep resourcing "
        "should shift."
    )

    region_risk = (
        physician_preds.groupby("region")
        .agg(
            physicians=("physician_id", "count"),
            avg_risk_score=("under_prescriber_risk_score", "mean"),
            high_risk_count=("under_prescriber_risk_score", lambda s: (s > 0.5).sum()),
        )
        .reset_index()
        .sort_values("avg_risk_score", ascending=False)
    )
    region_risk["high_risk_pct"] = (region_risk["high_risk_count"] / region_risk["physicians"] * 100).round(1)

    fig2 = px.bar(
        region_risk, x="region", y="avg_risk_score", color="high_risk_pct",
        title="Avg. Under-Prescriber Risk Score by Region",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Recommended Actions")
    top_region = region_risk.iloc[0]
    st.success(
        f"**{top_region['region']}** shows the highest concentration of under-prescribing "
        f"physicians ({top_region['high_risk_pct']}% flagged high-risk). Combined with the "
        f"demand forecast in Tab 1, this region is the top candidate for **additional rep "
        f"coverage or targeted digital/formulary-access outreach next quarter.**"
    )

    st.dataframe(region_risk, use_container_width=True)

st.divider()
st.caption(
    "Data sources: Kaggle 'Pharma Drug Sales' (real, 2014–2019 monthly) for demand forecasting · "
    "Simulated HCP panel dataset for physician-level ML classification, reflecting the standard "
    "constraint that prescriber-level data in this space is proprietary."
)
