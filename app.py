import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

from src.data_loader import download_solar_data
from src.data_processing import load_and_preprocess
from src.forecasting import feature_engineering, train_and_evaluate
from src.optimization import EVOptimizationParameters, optimize_charging

st.set_page_config(page_title="Carbon-Aware EV Charging", layout="wide")

st.title("Carbon-Aware EV Charging Dashboard")
st.markdown("""
This project optimizes Electric Vehicle (EV) charging schedules using **Big Data Analytics** (PySpark & Pandas)
and **Renewable Energy Forecasting** (Random Forest on Kaggle solar data).
The goal is to minimize grid carbon emissions by charging when local solar generation is highest.
""")

st.sidebar.header("Pipeline Control")

# --- Step 1: Data Management ---
if st.sidebar.button("1. Download & Process Data"):
    with st.spinner("Downloading data from Kaggle..."):
        download_solar_data()
    with st.spinner("Processing data (PySpark/Pandas)..."):
        load_and_preprocess()
    st.sidebar.success("Data downloaded and processed!")

# --- Step 2: Forecasting ---
if st.sidebar.button("2. Train Forecasting Model"):
    with st.spinner("Training Random Forest Model..."):
        df = pd.read_csv("data/processed_hourly.csv", parse_dates=['DATE_TIME'])
        df, features, target = feature_engineering(df)
        model, metrics, results_df = train_and_evaluate(df, features, target)

        st.session_state['metrics'] = metrics
        st.session_state['results_df'] = results_df

    st.sidebar.success("Model trained successfully!")

# Display Forecasting Metrics if available
if 'metrics' in st.session_state and 'results_df' in st.session_state:
    st.subheader("Renewable Energy Forecasting (Test Set)")

    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"{st.session_state['metrics']['MAE']:.2f} kW")
    col2.metric("Root Mean Squared Error (RMSE)", f"{st.session_state['metrics']['RMSE']:.2f} kW")

    results_df = st.session_state['results_df']
    # Show a slice of predictions vs actuals
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df['DATE_TIME'][:100], y=results_df['AC_POWER'][:100], name="Actual Generation (kW)"))
    fig.add_trace(go.Scatter(x=results_df['DATE_TIME'][:100], y=results_df['PREDICTED_AC_POWER'][:100], name="Predicted Generation (kW)"))
    fig.update_layout(title="Solar Forecast vs Actual (First 100 Test Hours)", xaxis_title="Time", yaxis_title="Power (kW)")
    st.plotly_chart(fig, use_container_width=True)


st.divider()

# --- Step 3: EV Charging Optimization ---
st.subheader("EV Charging Optimization")

st.markdown("""
**Assumptions:**
- **Grid Emissions:** Base factor is 0.82 kg CO2/kWh (Typical Indian Grid). This factor decreases linearly during hours of high local solar generation.
- **Cost Tariff:** Assumes a simple Time-of-Use tariff (Off-peak [10 PM - 6 AM]: 5 Rs/kWh | Peak [6 AM - 10 PM]: 8 Rs/kWh).
- **Efficiency:** Standard 90% charging efficiency from grid to battery.
""")

col1, col2, col3 = st.columns(3)
with col1:
    batt_cap = st.number_input("Battery Capacity (kWh)", min_value=10.0, max_value=100.0, value=40.0)
    charger_power = st.number_input("Charger Power (kW)", min_value=3.0, max_value=22.0, value=7.0)
with col2:
    curr_charge = st.number_input("Current Charge (kWh)", min_value=0.0, max_value=batt_cap, value=10.0)
    target_charge = st.number_input("Target Charge (kWh)", min_value=0.0, max_value=batt_cap, value=35.0)
with col3:
    window_hours = st.slider("Hours Available before Departure", min_value=4, max_value=24, value=12)
    start_hour_offset = st.slider("Start offset in test set (hours)", min_value=0, max_value=100, value=24)


if st.button("Run Optimization"):
    if not os.path.exists("src/models/test_predictions.csv"):
        st.error("Please run Step 1 & 2 first to generate forecast data.")
    else:
        forecast_df = pd.read_csv("src/models/test_predictions.csv", parse_dates=['DATE_TIME'])

        ev_params = EVOptimizationParameters(
            battery_capacity_kwh=batt_cap,
            current_charge_kwh=curr_charge,
            target_charge_kwh=target_charge,
            max_charger_power_kw=charger_power
        )

        results = optimize_charging(forecast_df, ev_params, start_hour_idx=start_hour_offset, hours_available=window_hours)

        if results['status'] == 'Infeasible':
            st.error(f"Infeasible: {results['message']}")
        else:
            st.success("Optimization Successful!")

            # Summary Metrics
            met = results['metrics']
            st.write("### Strategy Comparison")

            met_df = pd.DataFrame({
                "Strategy": ["Immediate", "Cheapest Time", "Carbon-Aware"],
                "Total Cost (Rs)": [met['Immediate']['cost_rs'], met['Cheapest']['cost_rs'], met['Carbon_Aware']['cost_rs']],
                "Total Emissions (kg CO2)": [met['Immediate']['emissions_kg'], met['Cheapest']['emissions_kg'], met['Carbon_Aware']['emissions_kg']]
            })
            st.table(met_df)

            # Charting Schedules
            sched = results['schedules']
            st.write("### Charging Schedules")

            sched_df = pd.DataFrame({
                "Time": results['timestamps'],
                "Immediate (kW)": sched['Immediate'],
                "Cheapest (kW)": sched['Cheapest'],
                "Carbon-Aware (kW)": sched['Carbon_Aware'],
                "Solar Forecast (Normalized)": [s / max(results['solar_forecast']) * charger_power if max(results['solar_forecast']) > 0 else 0 for s in results['solar_forecast']],
                "Effective Tariff (Rs)": results['tariffs']
            })

            fig2 = px.bar(sched_df, x="Time", y=["Immediate (kW)", "Cheapest (kW)", "Carbon-Aware (kW)"], barmode="group",
                         title="Charging Power over Time by Strategy")
            fig2.add_trace(go.Scatter(x=sched_df['Time'], y=sched_df['Solar Forecast (Normalized)'], name="Solar Forecast Profile", mode='lines+markers', line=dict(color='yellow')))
            st.plotly_chart(fig2, use_container_width=True)
