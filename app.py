from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import sqlite3
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

def get_model():
    """Load the trained model with custom objects to handle missing metrics"""
    # Define custom objects to handle 'mse' function
    custom_objects = {
        'mse': tf.keras.losses.MeanSquaredError(),
        'mean_squared_error': tf.keras.losses.MeanSquaredError(),
        'mae': tf.keras.losses.MeanAbsoluteError(),
        'mean_absolute_error': tf.keras.losses.MeanAbsoluteError()
    }
    
    try:
        # Try to load with custom objects
        model = load_model('traffic_lstm_model.h5', custom_objects=custom_objects)
        return model
    except Exception as e:
        print(f"Error loading model with custom objects: {str(e)}")
        
        # Alternative approach: load model structure and weights separately
        try:
            # Load model architecture from JSON
            with open('model_architecture.json', 'r') as json_file:
                model_json = json_file.read()
            
            # Create model from JSON
            from tensorflow.keras.models import model_from_json
            model = model_from_json(model_json)
            
            # Load weights
            model.load_weights('model_weights.h5')
            
            # Compile model with appropriate loss and metrics
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return model
        except Exception as e:
            raise Exception(f"Failed to load model using alternative approach: {str(e)}")

def get_scalers():
    """Load the saved scalers"""
    with open('x_scaler.pkl', 'rb') as file:
        x_scaler = pickle.load(file)
    
    with open('y_scaler.pkl', 'rb') as file:
        y_scaler = pickle.load(file)
    
    return x_scaler, y_scaler

def get_db_connection():
    """Connect to the SQLite database"""
    conn = sqlite3.connect('traffic_data.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make traffic prediction based on input features"""
    # Parse input data from form
    date_str = request.form.get('date', '')
    time_str = request.form.get('time', '')
    
    # Extract hour (0-23)
    hour = int(time_str.split(':')[0]) if time_str else 0
    
    # Extract weekday (1-7, Monday is 1)
    weekday = int(request.form.get('day', 0)) + 1  # Adjust for 0-indexing
    
    # Extract month and day from date
    if date_str:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        month = date_obj.month
        month_day = date_obj.day
        year = date_obj.year
    else:
        month = 1
        month_day = 1
        year = 2020
    
    # Get other features
    temperature = float(request.form.get('temperature', 0))
    is_holiday = 1 if request.form.get('isholiday', 'no') == 'yes' else 0
    weather_type = request.form.get('x0', 'Clear')
    weather_desc = request.form.get('x1', 'sky is clear')
    
    # Get historical traffic data from database
    conn = get_db_connection()
    similar_conditions = conn.execute('''
        SELECT humidity, wind_speed, traffic_volume 
        FROM traffic_data 
        WHERE hour = ? AND weekday = ? AND month = ?
        LIMIT 10
    ''', (hour, weekday, month)).fetchall()
    
    # Default values if no similar conditions found
    humidity = 70
    wind_speed = 5
    
    # Get average values from similar conditions
    if len(similar_conditions) > 0:
        humidity = sum(row['humidity'] for row in similar_conditions) / len(similar_conditions)
        wind_speed = sum(row['wind_speed'] for row in similar_conditions) / len(similar_conditions)
    
    # Weather type and description mappings
    weather_type_mapping = {
        'Rain': 1, 'Clouds': 2, 'Clear': 3, 'Snow': 4, 'Mist': 5,
        'Drizzle': 6, 'Haze': 7, 'Thunderstorm': 8, 'Fog': 9, 'Smoke': 10, 'Squall': 11
    }
    
    weather_desc_mapping = {
        'SQUALLS': 1, 'Sky is Clear': 2, 'broken clouds': 3, 'drizzle': 4, 
        'few clouds': 5, 'fog': 6, 'freezing rain': 7, 'haze': 8, 
        'heavy intensity drizzle': 9, 'heavy intensity rain': 10, 'heavy snow': 11,
        'light intensity drizzle': 12, 'light intensity shower rain': 13, 'light rain': 14,
        'light rain and snow': 15, 'light shower snow': 16, 'light snow': 17,
        'mist': 18, 'moderate rain': 19, 'overcast clouds': 20,
        'proximity shower rain': 21, 'proximity thunderstorm': 22,
        'proximity thunderstorm with drizzle': 23, 'proximity thunderstorm with rain': 24,
        'scattered clouds': 25, 'shower snow': 26, 'sky is clear': 27,
        'sleet': 28, 'smoke': 29, 'snow': 30, 'thunderstorm': 31,
        'thunderstorm with drizzle': 32, 'thunderstorm with heavy rain': 33,
        'thunderstorm with light drizzle': 34, 'thunderstorm with light rain': 35,
        'thunderstorm with rain': 36, 'very heavy rain': 37
    }
    
    # Get previous traffic values for the lookback sequence
    # Using similar day patterns if available, otherwise using defaults
    traffic_history = conn.execute('''
        SELECT traffic_volume FROM traffic_data 
        WHERE hour = ? AND weekday = ? 
        ORDER BY date_time DESC LIMIT 6
    ''', (hour, weekday)).fetchall()
    
    # Prepare historical data for LSTM lookback
    historical_data = []
    for i in range(6):  # For 6 hours lookback
        if i < len(traffic_history):
            historical_data.append(traffic_history[i]['traffic_volume'])
        else:
            # Use average if not enough history
            historical_data.append(3000)  # Default value
    
    # Create feature sequence for LSTM
    features = []
    
    # Get typical traffic patterns
    hourly_traffic = conn.execute('''
        SELECT hour, AVG(traffic_volume) as avg_traffic 
        FROM traffic_data 
        GROUP BY hour
        ORDER BY hour
    ''').fetchall()
    
    weekday_traffic = conn.execute('''
        SELECT weekday, AVG(traffic_volume) as avg_traffic 
        FROM traffic_data 
        GROUP BY weekday
        ORDER BY weekday
    ''').fetchall()
    
    conn.close()
    
    # Get model and scalers
    try:
        model = get_model()
        x_scaler, y_scaler = get_scalers()
    except Exception as e:
        return render_template('prediction.html', 
                              error=f"Error loading model: {str(e)}",
                              prediction=None,
                              hourly_chart=None,
                              weekday_chart=None)
    
    # Create input sequence for LSTM (6 timesteps)
    for _ in range(6):
        # Features for each timestep in the order:
        # is_holiday, humidity, wind_speed, temperature, weekday, hour, month_day, year, month
        features.append([
            is_holiday, humidity, wind_speed, temperature,
            weekday, hour, month_day, year, month
        ])
    
    # Scale features
    features_array = np.array(features)
    scaled_features = x_scaler.transform(features_array)
    
    # Reshape for LSTM [samples, timesteps, features]
    lstm_input = scaled_features.reshape(1, 6, 9)
    
    # Make prediction
    try:
        scaled_prediction = model.predict(lstm_input)
        prediction = y_scaler.inverse_transform(scaled_prediction)[0][0]
    except Exception as e:
        return render_template('prediction.html', 
                              error=f"Error making prediction: {str(e)}",
                              prediction=None,
                              hourly_chart=None,
                              weekday_chart=None)
    
    # Generate hourly traffic chart
    hourly_chart = create_hourly_chart(hourly_traffic, hour)
    
    # Generate weekday traffic chart
    weekday_chart = create_weekday_chart(weekday_traffic, weekday)
    
    # Get traffic condition based on prediction
    traffic_condition = get_traffic_condition(prediction)
    
    return render_template('prediction.html', 
                          prediction=round(prediction),
                          traffic_condition=traffic_condition,
                          hourly_chart=hourly_chart,
                          weekday_chart=weekday_chart,
                          selected_hour=hour,
                          selected_weekday=weekday,
                          selected_weather=weather_type)

def create_hourly_chart(hourly_data, selected_hour):
    """Create hourly traffic chart with the selected hour highlighted"""
    plt.figure(figsize=(10, 5))
    
    hours = [row['hour'] for row in hourly_data]
    traffic = [row['avg_traffic'] for row in hourly_data]
    
    # Create bar chart
    bars = plt.bar(hours, traffic, color='skyblue')
    
    # Highlight selected hour
    if selected_hour in hours:
        idx = hours.index(selected_hour)
        bars[idx].set_color('red')
    
    plt.title('Average Traffic Volume by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Average Traffic Volume')
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    chart = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    return f"data:image/png;base64,{chart}"

def create_weekday_chart(weekday_data, selected_weekday):
    """Create weekday traffic chart with the selected weekday highlighted"""
    plt.figure(figsize=(10, 5))
    
    weekdays = [row['weekday'] for row in weekday_data]
    traffic = [row['avg_traffic'] for row in weekday_data]
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_map = {i+1: name for i, name in enumerate(weekday_names)}
    
    # Create bar chart
    bars = plt.bar([weekday_map.get(day, day) for day in weekdays], traffic, color='lightgreen')
    
    # Highlight selected weekday
    if selected_weekday in weekdays:
        idx = weekdays.index(selected_weekday)
        bars[idx].set_color('orange')
    
    plt.title('Average Traffic Volume by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Traffic Volume')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    chart = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    return f"data:image/png;base64,{chart}"

def get_traffic_condition(traffic_volume):
    """Return traffic condition based on volume"""
    if traffic_volume < 1000:
        return "Very Light"
    elif traffic_volume < 2500:
        return "Light"
    elif traffic_volume < 4000:
        return "Moderate"
    elif traffic_volume < 5500:
        return "Heavy"
    else:
        return "Very Heavy"

if __name__ == '__main__':
    app.run(debug=True)