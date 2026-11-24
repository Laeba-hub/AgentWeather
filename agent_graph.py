import joblib
import pandas as pd
from typing import TypedDict
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# --- 1. DEFINE THE SHARED STATE ---
# This is the "clipboard" passed between all 4 agents
class AgentState(TypedDict):
    humidity: float
    wind_speed: float
    precipitation_mm: float
    forecast_temp: float
    alert_message: str
    drift_message: str
    drift_detected: bool
    news_report: str

# --- 2. FORECAST AGENT ---
def forecast_agent(state: AgentState):
    print("\nForecast Agent: Loading model and predicting...")
    model = joblib.load("models/best_temperature_model.pkl")
    
    # Prepare input data with the features we engineered
    input_data = pd.DataFrame({
        'humidity': [state['humidity']],
        'wind_speed': [state['wind_speed']],
        'precipitation_mm': [state['precipitation_mm']],
        'temp_lag_24h': [28.0], 
        'temp_lag_48h': [27.5],
        'temp_lag_72h': [27.0],
        'temp_rolling_24h': [27.8],
        'temp_rolling_72h': [27.6],
        'hour': [12],
        'month': [11]
    })
    
    prediction = model.predict(input_data)[0]
    print(f"Forecast Agent: Predicted Temperature = {prediction:.2f}°C")
    return {"forecast_temp": float(prediction)}

# --- 3. ALERT AGENT ---
def alert_agent(state: AgentState):
    print("Alert Agent: Checking for hazards...")
    temp = state['forecast_temp']
    alert = "No hazards detected. Weather is normal."
    
    if temp > 40.0:
        alert = "HAZARD: Extreme Heat Warning! Stay indoors."
    elif temp < 10.0:
        alert = "HAZARD: Extreme Cold Warning! Dress warmly."
        
    print(f"Alert Agent: {alert}")
    return {"alert_message": alert}

# --- 4. DRIFT AGENT ---
def drift_agent(state: AgentState):
    print("Drift Agent: Checking for model drift...")
    
    # Simulating the actual temperature that occurred (In production, you'd fetch this from the DB)
    actual_temp = 30.0 
    predicted_temp = state['forecast_temp']
    
    # Calculate the error
    error = abs(actual_temp - predicted_temp)
    drift_threshold = 5.0
    
    if error > drift_threshold:
        drift_msg = f"DRIFT DETECTED! Error is {error:.2f}°C (Threshold: {drift_threshold}). Triggering retraining."
        drift_status = True
    else:
        drift_msg = f"No drift. Error is {error:.2f}°C (Threshold: {drift_threshold})."
        drift_status = False
        
    print(drift_msg)
    return {"drift_message": drift_msg, "drift_detected": drift_status}

# --- 5. WEATHER REPORT AGENT (LLM) ---
def report_agent(state: AgentState):
    print("Weather Report Agent: Asking Groq LLM to write the news report...")
    
    # Initialize the Groq LLM
    llm = ChatGroq(
        temperature=0.7, 
        model_name="openai/gpt-oss-120b",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Create the prompt
    prompt = f"""
    You are a professional weather news anchor in Karachi.
    Write a short, 3-sentence news report based on this data:
    - Forecasted Temperature for tomorrow: {state['forecast_temp']:.1f}°C
    - Current Humidity: {state['humidity']}%
    - Alert Status: {state['alert_message']}
    - System Status: {state['drift_message']}
    
    Make it sound natural, informative, and professional.
    """
    
    try:
        response = llm.invoke(prompt)
        report = response.content
        print(f"Report Generated:\n{report}")
        
    except Exception as e:
        # FALLBACK: If the LLM fails, we don't crash the agent.
        print(f"LLM failed: {e}. Using fallback template.")
        report = f"Tomorrow's forecast is {state['forecast_temp']:.1f}°C with {state['humidity']}% humidity. {state['alert_message']}"
        
    return {"news_report": report}

# --- 6. BUILD THE LANGGRAPH ---
print("--- Building LangGraph ---")
workflow = StateGraph(AgentState)

# Add the 4 nodes
workflow.add_node("forecast", forecast_agent)
workflow.add_node("alert", alert_agent)
workflow.add_node("drift", drift_agent)
workflow.add_node("report", report_agent)

# Connect the nodes in the specific order required: Forecast -> Alert -> Drift -> Report
workflow.set_entry_point("forecast")
workflow.add_edge("forecast", "alert")
workflow.add_edge("alert", "drift")
workflow.add_edge("drift", "report")
workflow.add_edge("report", END)

# Compile the graph
app = workflow.compile()

# --- 7. RUN THE GRAPH ---
if __name__ == "__main__":
    print("\n--- Running the Agent Pipeline ---")

    # Initial state (Live weather data)
    initial_state = {
        "humidity": 61.0,
        "wind_speed": 10.3,
        "precipitation_mm": 0.0,
        "forecast_temp": 0.0,
        "alert_message": "",
        "drift_message": "",
        "drift_detected": False,
        "news_report": ""
    }

    # Execute the graph
    final_state = app.invoke(initial_state)

    print("\n--- Final Agent Output ---")
    print(f"Forecast: {final_state['forecast_temp']:.2f}°C")
    print(f"Alert: {final_state['alert_message']}")
    print(f"Drift Status: {final_state['drift_message']}")
    print(f"\nREPORT:\n{final_state['news_report']}")