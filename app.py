import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('traffic_predictor_model.joblib')
    scaler = joblib.load('traffic_scaler.joblib')
    return model, scaler

def prepare_features(hour, day, temperature, precipitation, 
                    special_event, road_work, vehicle_count):
    hour_sin = np.sin(2 * np.pi * hour/24)
    hour_cos = np.cos(2 * np.pi * hour/24)
    day_sin = np.sin(2 * np.pi * day/7)
    day_cos = np.cos(2 * np.pi * day/7)
    
    features = pd.DataFrame({
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos],
        'day_sin': [day_sin],
        'day_cos': [day_cos],
        'temperature': [temperature],
        'precipitation': [precipitation],
        'special_event': [special_event],
        'road_work': [road_work],
        'vehicle_count': [vehicle_count]
    })
    return features

def get_congestion_color(level):
    if level < 3:
        return 'green'
    elif level < 6:
        return 'yellow'
    elif level < 8:
        return 'orange'
    else:
        return 'red'

def main():
    st.title('AI-Powered Traffic Congestion Predictor')
    st.write('Predict traffic congestion levels based on various factors')

    # Sidebar inputs
    st.sidebar.header('Traffic Conditions')
    
    # Time selection
    current_time = datetime.now()
    selected_date = st.sidebar.date_input('Select Date', current_time)
    selected_time = st.sidebar.time_input('Select Time', current_time)
    
    # Environmental conditions
    temperature = st.sidebar.slider('Temperature (°C)', -10, 40, 20)
    precipitation = st.sidebar.slider('Precipitation (mm)', 0, 50, 0)
    
    # Traffic conditions
    vehicle_count = st.sidebar.number_input('Estimated Vehicle Count', 0, 5000, 1000)
    special_event = st.sidebar.checkbox('Special Event Nearby')
    road_work = st.sidebar.checkbox('Road Work Present')

    # Prepare features for prediction
    hour = selected_time.hour
    day = selected_date.weekday()
    
    # Load model and make prediction
    model, scaler = load_model()
    features = prepare_features(hour, day, temperature, precipitation,
                              special_event, road_work, vehicle_count)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    # Display current prediction
    st.header('Current Traffic Prediction')
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Congestion Level"},
        gauge = {
            'axis': {'range': [0, 10]},
            'bar': {'color': get_congestion_color(prediction)},
            'steps': [
                {'range': [0, 3], 'color': "lightgreen"},
                {'range': [3, 6], 'color': "lightyellow"},
                {'range': [6, 8], 'color': "orange"},
                {'range': [8, 10], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    
    st.plotly_chart(fig)

    # Display congestion level interpretation
    if prediction < 3:
        st.success('Low Traffic - Good time to travel!')
    elif prediction < 6:
        st.warning('Moderate Traffic - Some delays expected')
    elif prediction < 8:
        st.error('Heavy Traffic - Consider alternative routes')
    else:
        st.error('Severe Congestion - Avoid if possible!')

    # Show hourly forecast
    st.header('24-Hour Forecast')
    
    # Generate predictions for next 24 hours
    hours = list(range(24))
    forecasts = []
    
    for h in hours:
        features = prepare_features(h, day, temperature, precipitation,
                                  special_event, road_work, vehicle_count)
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        forecasts.append(pred)
    
    # Create forecast chart
    forecast_df = pd.DataFrame({
        'Hour': hours,
        'Congestion Level': forecasts
    })
    
    fig = px.line(forecast_df, x='Hour', y='Congestion Level',
                  title='24-Hour Congestion Forecast')
    fig.add_hline(y=3, line_dash="dash", line_color="green",
                  annotation_text="Low Traffic")
    fig.add_hline(y=6, line_dash="dash", line_color="orange",
                  annotation_text="Heavy Traffic")
    fig.add_hline(y=8, line_dash="dash", line_color="red",
                  annotation_text="Severe Congestion")
    
    st.plotly_chart(fig)

    # Traffic reduction recommendations
    st.header('Traffic Reduction Recommendations')
    
    if prediction >= 6:
        st.subheader('Alternative Routes')
        st.write('• Consider using side streets or alternate main roads')
        st.write('• Check navigation apps for real-time alternatives')
        
        st.subheader('Travel Time Adjustments')
        
        # Find best travel times
        best_times = [hours[i] for i in range(len(hours)) 
                     if forecasts[i] < 5]
        if best_times:
            st.write('Recommended travel times:')
            for time in best_times[:3]:
                st.write(f"• {time:02d}:00 (Predicted congestion: {forecasts[time]:.1f})")
        
        st.subheader('Additional Recommendations')
        st.write('• Consider public transportation options')
        st.write('• Work remotely if possible')
        st.write('• Carpool to reduce vehicle count')
    else:
        st.success('Current traffic conditions are favorable!')

if __name__ == '__main__':
    main()