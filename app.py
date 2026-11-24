import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Import the LangGraph app we built earlier
from agent_graph import app as agent_app

load_dotenv()

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AgentWeather", page_icon="⛅", layout="wide")

st.title("⛅ AgentWeather Dashboard")
st.markdown("**Agentic Multi-Metric Weather & AQI Forecasting System**")
st.markdown("---")

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.header("Simulate Live Data")
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 65)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 12)
precipitation = st.sidebar.slider("Precipitation (mm)", 0.0, 10.0, 0.0)

if st.sidebar.button("Run Agent Pipeline"):
    with st.spinner("Agents are thinking..."):
        # Run the LangGraph agents
        initial_state = {
            "humidity": float(humidity),
            "wind_speed": float(wind_speed),
            "precipitation_mm": float(precipitation),
            "forecast_temp": 0.0,
            "alert_message": "",
            "drift_message": "",
            "drift_detected": False,
            "news_report": ""
        }
        
        final_state = agent_app.invoke(initial_state)
        
        # Display results in the main area
        st.success("Agent Pipeline Completed!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Forecasted Temperature", f"{final_state['forecast_temp']:.2f} °C")
        col2.metric("Alert Status", "Hazard!" if "HAZARD" in final_state['alert_message'] else "Normal")
        col3.metric("Drift Status", "Drift Detected" if final_state['drift_detected'] else "Stable")
        
        st.markdown("###  Report")
        st.info(final_state['news_report'])
        
        st.markdown("###  Agent Activity Panel")
        st.write(f"**Alert Agent:** {final_state['alert_message']}")
        st.write(f"**Drift Agent:** {final_state['drift_message']}")

st.markdown("---")

# --- 3. HISTORICAL DATA CHART ---
st.subheader("Historical Temperature (Last 7 Days)")
if os.path.exists("historical_data.csv"):
    df = pd.read_csv("historical_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get the last 7 days of data for a clean chart
    last_7_days = df.tail(24 * 7)
    
    st.line_chart(last_7_days.set_index('timestamp')['temperature_c'])
else:
    st.warning("No historical data found. Run backfill.py first.")