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
            background: #2F4F4F;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            background: #C3B091;
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            color: white;
            box-shadow: 0 8px 24px rgba(15, 91, 216, 0.18);
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #C3B091;
            margin-bottom: 0.3rem;
        }
        div[data-testid="stMetric"] {
            background: #2A3439;
            border: 1px solid #e5edf9;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
            padding: 0.75rem 0.8rem;
        }
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #0d6efd, #1d7ef5);
            color: white;
            border: none;
            border-radius: 8px;
            width: 100%;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #0b5ed7, #0d6efd);
        }
        .sidebar .block-container {
            padding-top: 1rem;
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
    st.title("Settings")
    future_months = st.slider("Forecast Horizon (Months)", 1, 24, 12)
    st.info("Adjust the slider to change how many months the model forecasts.")
    st.divider()

loader = DataLoader("data/airline-passengers.csv")
df = loader.load_data()
summary = get_dataset_summary(df)

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom:0.2rem;">✈️ Airline Passenger Analysis & Forecasting</h1>
        <p style="margin:0; opacity:0.95;">Explore historical travel trends and generate future passenger forecasts with an RNN-based model.</p>
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
    st.markdown('<div class="section-title">Model Accuracy Metrics</div>', unsafe_allow_html=True)
    try:
        Path("outputs").mkdir(exist_ok=True)
        mae, mse, rmse = Evaluator().evaluate()
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        m2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
        m3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
    except Exception as exc:
        st.warning(f"Model metrics are not available yet: {exc}")

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
        st.plotly_chart(fig_combined, use_container_width=True)
