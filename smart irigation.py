import requests

# Your ThingSpeak Write API Key
THINGSPEAK_WRITE_API_KEY = 'PXPUMQ4F30N0R48H'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import requests
import time

# --- ThingSpeak API Key ---
THINGSPEAK_WRITE_API_KEY = 'PXPUMQ4F30N0R48H'
THINGSPEAK_CHANNEL_ID = '2932567' # Optional, but good practice

# Load data
df = pd.read_csv("C:\Users\hp\Desktop\filtered_daily_weather_data.csv")

# Initial data overview
print("Initial Data Overview:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())

# Clean and preprocess
df.drop_duplicates(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Rain'] != 0]
imputer = SimpleImputer(strategy='mean')
df[['Rain', 'Temp Max', 'Temp Min']] = imputer.fit_transform(df[['Rain', 'Temp Max', 'Temp Min']])

# Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfYear'] = df['Date'].dt.dayofyear
df['Season'] = df['Month'] % 12 // 3 + 1

# Lag Features
df['Rain_Lag1'] = df['Rain'].shift(1)
df['Temp_Max_Lag1'] = df['Temp Max'].shift(1)
df['Temp_Min_Lag1'] = df['Temp Min'].shift(1)
df.dropna(inplace=True)

# Feature Scaling
features = ['Temp Max', 'Temp Min', 'Year', 'Month', 'Day', 'DayOfYear', 'Season',
            'Rain_Lag1', 'Temp_Max_Lag1', 'Temp_Min_Lag1']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Model training
X = df[features]
y = df['Rain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42),
                            param_grid, cv=3,
                            scoring='neg_mean_squared_error',
                            n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Rainfall prediction function
def predict_rainfall_with_context(temp_max, temp_min):
    # Rule-based condition
    if temp_max >= 33 and temp_min >= 22:
        return 0.00

    latest_row = df.sort_values('Date').iloc[-1].copy()

    input_dict = {
        'Temp Max': temp_max,
        'Temp Min': temp_min,
        'Year': latest_row['Year'],
        'Month': latest_row['Month'],
        'Day': latest_row['Day'],
        'DayOfYear': latest_row['DayOfYear'],
        'Season': latest_row['Season'],
        'Rain_Lag1': latest_row['Rain_Lag1'],
        'Temp_Max_Lag1': latest_row['Temp_Max_Lag1'],
        'Temp_Min_Lag1': latest_row['Temp_Min_Lag1'],
    }
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    prediction = best_model.predict(input_scaled)
    return prediction[0]

# Function to send data to ThingSpeak
def send_to_thingspeak(temperature, humidity, predicted_rain):
    url = f'http://api.thingspeak.com/update?api_key={THINGSPEAK_WRITE_API_KEY}&field1={temperature}&field2={humidity}&field3={predicted_rain}'
    try:
        response = requests.post(url)
        if response.status_code == 200:
            print(f'Data sent to ThingSpeak successfully! Response: {response.text}')
        else:
            print(f'Failed to send data to ThingSpeak. Status code: {response.status_code}')
    except requests.exceptions.RequestException as e:
        print(f'Error sending request to ThingSpeak: {e}')

if __name__ == "__main__":
    # Simulate getting current temperature and humidity (replace with your actual sensor readings)
    current_temperature = float(input("Enter the current temperature: "))
    current_humidity = float(input("Enter the current humidity: "))
    temp_max = float(input("Enter the maximum temperature for prediction: "))
    temp_min = float(input("Enter the minimum temperature for prediction: "))

    predicted_rainfall = predict_rainfall_with_context(temp_max, temp_min)
    print(f"\nPredicted rainfall considering temperature and recent weather data: {predicted_rainfall:.2f} mm")

    # Send the current temperature, humidity, and predicted rainfall to ThingSpeak
    send_to_thingspeak(current_temperature, current_humidity, predicted_rainfall)
    THINGSPEAK_CHANNEL_ID = '2932567' # Optional, but good practice

# The prediction from your ML model
prediction_value = 0.85  # Example prediction

# Construct the ThingSpeak Write API URL
write_url = f'http://api.thingspeak.com/update?api_key={THINGSPEAK_WRITE_API_KEY}&field1={prediction_value}'

try:
    response = requests.post(write_url)
    if response.status_code == 200:
        print(f'Prediction sent to ThingSpeak successfully! Response: {response.text}')
    else:
        print(f'Failed to send prediction to ThingSpeak. Status code: {response.status_code}')
except requests.exceptions.RequestException as e:
    print(f'Error sending request to ThingSpeak: {e}')
