import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.abspath("src"))

from analysis import compare_campaigns
from modeling import train_revenue_model
from agents import generate_campaign_recommendation

st.set_page_config(
    page_title="AI Experimentation & Revenue Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #555;
        margin-bottom: 20px;
    }

    .section-card {
        background-color: #f7f9fc;
        padding: 20px;
        border-radius: 14px;
        margin-bottom: 20px;
        border: 1px solid #e6eaf0;
    }

    .metric-label {
        font-size: 14px;
        color: #666;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">AI Experimentation & Revenue Optimization Platform</div>
    <div class="subtitle">
    Growth analytics command center for campaign performance, statistical testing,
    revenue prediction, and AI-powered optimization recommendations.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-card">
    <b>Platform Workflow:</b><br>
    Campaign Data → KPI Analytics → A/B Testing → ML Revenue Prediction → AI Growth Recommendations
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()



df = pd.read_csv("data/marketing_experiments.csv")

st.sidebar.header("Experiment Filters")
st.sidebar.markdown(
    """
    Filter campaigns and compare growth performance across acquisition,
    conversion, revenue, and retention metrics.
    """
)

selected_campaigns = st.sidebar.multiselect(
    "Select Campaigns",
    options=sorted(df["campaign"].unique()),
    default=sorted(df["campaign"].unique())
)

filtered_df = df[df["campaign"].isin(selected_campaigns)]

st.markdown("### Executive Performance Snapshot")

st.header("Executive KPI Overview")

avg_ctr = filtered_df["ctr"].mean()
avg_conversion_rate = filtered_df["conversion_rate"].mean()
avg_cac = filtered_df["cac"].mean()
avg_roas = filtered_df["roas"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg CTR", f"{avg_ctr:.2%}")
col2.metric("Avg Conversion Rate", f"{avg_conversion_rate:.2%}")
col3.metric("Avg CAC", f"${avg_cac:,.2f}")
col4.metric("Avg ROAS", f"{avg_roas:.2f}x")

st.divider()

st.markdown("### Campaign Performance Diagnostics")

st.header("Campaign Performance Comparison")

campaign_summary = (
    filtered_df
    .groupby("campaign")
    .agg({
        "impressions": "sum",
        "clicks": "sum",
        "conversions": "sum",
        "ad_spend": "sum",
        "revenue": "sum",
        "ctr": "mean",
        "conversion_rate": "mean",
        "cac": "mean",
        "roas": "mean",
        "customer_ltv": "mean"
    })
    .reset_index()
)
campaign_summary_text = campaign_summary.to_string()


fig_roas = px.bar(
    campaign_summary,
    x="campaign",
    y="roas",
    title="Average ROAS by Campaign",
    text_auto=True
)

st.plotly_chart(fig_roas, use_container_width=True)

fig_conversion = px.bar(
    campaign_summary,
    x="campaign",
    y="conversion_rate",
    title="Average Conversion Rate by Campaign",
    text_auto=".2%"
)

st.plotly_chart(fig_conversion, use_container_width=True)

st.divider()

st.header("Revenue vs Ad Spend")

fig_scatter = px.scatter(
    filtered_df,
    x="ad_spend",
    y="revenue",
    color="campaign",
    size="conversions",
    hover_data=["ctr", "conversion_rate", "cac", "roas"],
    title="Revenue Efficiency by Campaign"
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

st.header("Campaign Summary Table")

st.dataframe(
    campaign_summary,
    use_container_width=True
)

st.divider()

st.markdown("### Experiment Validity Check")

st.header("A/B Testing Statistical Analysis")
with st.expander("View Statistical Test Results", expanded=True):
    # put your A/B testing results here

    selected_metric = st.selectbox(
    "Select Metric for Statistical Comparison",
    [
        "conversion_rate",
        "ctr",
        "roas",
        "cac"
    ]
)

ab_test_results = compare_campaigns(
    filtered_df,
    metric=selected_metric
)

if ab_test_results:

    st.subheader("Experiment Results")

    st.write(
        f"""
        Comparing:
        {ab_test_results['campaign_1']} vs
        {ab_test_results['campaign_2']}
        """
    )

    st.write(f"T-Statistic: {ab_test_results['t_statistic']}")

    st.write(f"P-Value: {ab_test_results['p_value']}")

    if ab_test_results["significant"]:
        st.success(
            "Result is statistically significant (p < 0.05)"
        )
    else:
        st.warning(
            "Result is NOT statistically significant"
        )
st.divider()

st.markdown("### Predictive Revenue Intelligence")

st.header("Machine Learning Revenue Prediction")

with st.expander("View ML Model Diagnostics", expanded=True):
    # put model metrics and feature importance chart here

    model_results = train_revenue_model(filtered_df)

col1, col2 = st.columns(2)

col1.metric("Model MAE", f"${model_results['mae']:,.2f}")
col2.metric("Model R² Score", f"{model_results['r2']:.2f}")

st.subheader("Feature Importance")

importance_fig = px.bar(
    model_results["feature_importance"],
    x="importance",
    y="feature",
    orientation="h",
    title="Revenue Prediction Feature Importance"
)

st.plotly_chart(importance_fig, use_container_width=True)

st.divider()

st.markdown("### AI-Generated Growth Recommendations")

st.header("AI Campaign Optimization Recommendations")

with st.spinner("Generating AI recommendations..."):

    ai_recommendation = generate_campaign_recommendation(
        campaign_summary_text
    )

st.markdown("#### Consultant Recommendation Summary")
st.success(ai_recommendation)

st.divider()

st.caption(
    "Built with Python, Streamlit, Plotly, scikit-learn, scipy, Claude API, and experimentation analytics."
)

