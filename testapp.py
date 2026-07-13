import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.data_loader import DataLoader
from src.forecast import Forecaster
from src.evaluate import Evaluator

st.set_page_config(
    page_title="Airline Passenger Forecaster",
    page_icon="✈️",
    layout="wide"
)


def load_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #071120 0%, #0e1b2b 55%, #16253d 100%);
            color: #e9f2ff;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background:
                radial-gradient(circle at top left, rgba(255,255,255,0.2), transparent 24%),
                linear-gradient(135deg, #0f172a 0%, #1d4ed8 45%, #38bdf8 100%);
            padding: 2rem 2rem 1.7rem 2rem;
            border-radius: 24px;
            color: white;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.28);
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.14);
        }
        .hero-inner {
            display: grid;
            grid-template-columns: minmax(300px, 1.8fr) minmax(220px, 1fr);
            gap: 1.8rem;
            align-items: center;
        }
        .hero-copy {
            min-width: 0;
        }
        .hero-right {
            display: grid;
            gap: 1rem;
        }
        .hero-stat {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 18px;
            padding: 1.15rem 1rem;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.16);
        }
        .hero-stat h3 {
            margin: 0;
            color: #cfe6ff;
            font-size: 0.9rem;
            opacity: 0.85;
        }
        .hero-stat .value {
            margin: 0.5rem 0 0;
            font-size: 1.9rem;
            font-weight: 800;
            color: #ffffff;
        }
        .hero-callout {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 18px;
            padding: 1rem 1rem 0.95rem;
            color: #eaf4ff;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
        }
        .hero-callout p {
            margin: 0.4rem 0 0;
            opacity: 0.9;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        .hero-card h1 {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }
        .hero-card p {
            font-size: 1rem;
            opacity: 0.95;
            margin-bottom: 0;
            max-width: 700px;
        }
        .hero-pill {
            display: inline-block;
            background: rgba(255,255,255,0.16);
            color: white;
            padding: 0.38rem 0.75rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
            border: 1px solid rgba(255,255,255,0.22);
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #EEDC82;
            margin-bottom: 0.35rem;
        }
        .section-subtitle {
            color: #b8d7ff;
            margin-top: 0.15rem;
            opacity: 0.82;
            font-size: 0.95rem;
        }
        div[data-testid="stMetric"] {
            background: #708090;
            color: #10233f;
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 14px;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
            padding: 0.7rem 0.8rem;
        }
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: white;
            border: none;
            border-radius: 10px;
            width: 100%;
            padding: 0.55rem 0.8rem;
            font-weight: 600;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
        }
        .sidebar .block-container {
            padding-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            background: rgba(255,255,255,0.1);
            color: #dfeeff;
        }
        .stTabs [aria-selected="true"] {
            background: #2563eb;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_dataset_summary(df):
    return {
        "rows": len(df),
        "start": df.index.min().strftime("%b %Y"),
        "end": df.index.max().strftime("%b %Y"),
    }


load_custom_css()

with st.sidebar:
    st.image("assets/images.jpg", width=220, caption="Airline Passenger Forecasting")
    st.markdown("### Controls")
    future_months = st.slider("Forecast Horizon (Months)", 1, 24, 12)
    st.info("Adjust the slider to change how many months the model forecasts.")
    st.divider()
    st.caption("Built with Streamlit + TensorFlow")

loader = DataLoader("data/airline-passengers.csv")
df = loader.load_data()
summary = get_dataset_summary(df)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-inner">
            <div class="hero-copy">
                <div class="hero-pill">AI Forecasting Dashboard</div>
                <h1>✈️ Airline Passenger Analysis & Forecasting</h1>
                <p>Explore historical travel trends, understand seasonal movement, and generate future passenger forecasts with a sequence-based deep learning model.</p>
                <div class="hero-callout">
                    <strong>Fast insights, smoother planning.</strong>
                    <p>Use the interactive dashboard to inspect past passenger counts, track model accuracy, and forecast future demand.</p>
                </div>
            </div>
            <div class="hero-right">
                <div class="hero-stat">
                    <h3>Data points</h3>
                    <p class="value">144</p>
                </div>
                <div class="hero-stat">
                    <h3>History range</h3>
                    <p class="value">Jan 1949 – Dec 1960</p>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Built for time-series forecasting with sequence-based deep learning.")

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Dataset Rows", f"{summary['rows']}")
metric_col2.metric("Start Date", summary["start"])
metric_col3.metric("End Date", summary["end"])

tab1, tab2 = st.tabs(["🚀 Model Performance", "📈 Exploratory Data Analysis"])

with tab1:
    st.info("Temporarily disabled for debugging.")
    '''st.markdown('<div class="section-title">Model Accuracy Metrics</div>', unsafe_allow_html=True)
    try:
        Path("outputs").mkdir(exist_ok=True)
        mae, mse, rmse = Evaluator().evaluate()
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        m2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
        m3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
    except Exception as exc:
        st.warning(f"Model metrics are not available yet: {exc}")'''

with tab2:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown('<div class="section-title">Raw Data</div>', unsafe_allow_html=True)
        st.dataframe(df.reset_index(), height=350, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Historical Trend</div>', unsafe_allow_html=True)
        fig = px.line(
            df.reset_index(),
            x="month",
            y="passengers",
            markers=True,
            template="plotly_white",
            color_discrete_sequence=["#2563eb"],
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown('<div class="section-title">🔮 Generate Future Forecast</div>', unsafe_allow_html=True)

if st.button("Run RNN Model", use_container_width=True):
    with st.spinner("Analyzing temporal patterns..."):
        try:
            forecaster = Forecaster()
            future = forecaster.forecast(future_months)

            last_date = df.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=future_months,
                freq="MS",
            )

            forecast_df = pd.DataFrame({
                "Month": future_dates,
                "Predicted Passengers": future.flatten(),
            })
        except Exception as exc:
            st.error(f"Forecasting failed: {exc}")
            st.stop()

    st.success(f"Successfully generated forecast for {future_months} months.")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.subheader("Forecasted Values")
        st.dataframe(forecast_df, use_container_width=True)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv",
        )

    with res_col2:
        st.subheader("Combined Projection")
        fig_combined = go.Figure()
        fig_combined.add_trace(
            go.Scatter(
                x=df.index,
                y=df["passengers"],
                name="Historical",
                line=dict(color="#6c757d", width=2),
            )
        )
        fig_combined.add_trace(
            go.Scatter(
                x=forecast_df["Month"],
                y=forecast_df["Predicted Passengers"],
                name="Forecast",
                line=dict(color="#ff7f0e", width=3, dash="dot"),
            )
        )
        fig_combined.update_layout(
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=360,
        )
        st.plotly_chart(fig_combined, width="stretch")
