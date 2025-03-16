import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import_train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Vehicle Fuel Efficiency Predictor",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("Vehicle Fuel Efficiency Prediction")
st.write("Enter vehicle specifications to predict fuel efficiency (km/l)")

# Function to load and prepare data
@st.cache_data
def load_data():
    # You should replace this with your actual dataset
    data = pd.DataFrame({
        'vehicle_type': ['Sedan', 'SUV', 'Hatchback', 'Sedan', 'SUV'] * 20,
        'fuel_type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol'] * 20,
        'kerb_weight': np.random.uniform(800, 2500, 100),
        'speed': np.random.uniform(60, 200, 100),
        'horse_power': np.random.uniform(50, 400, 100),
        'torque': np.random.uniform(90, 500, 100),
        'displacement': np.random.uniform(800, 4000, 100),
        'fuel_efficiency': np.random.uniform(8, 25, 100)
    })
    return data

# Load data and train model
data = load_data()

# Prepare features
le_vehicle = LabelEncoder()
le_fuel = LabelEncoder()

X = data.copy()
X['vehicle_type'] = le_vehicle.fit_transform(X['vehicle_type'])
X['fuel_type'] = le_fuel.fit_transform(X['fuel_type'])

y = X.pop('fuel_efficiency')

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Create input form
st.sidebar.header("Vehicle Specifications")

vehicle_type = st.sidebar.selectbox(
    "Vehicle Type",
    options=['Sedan', 'SUV', 'Hatchback']
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    options=['Petrol', 'Diesel']
)

kerb_weight = st.sidebar.slider(
    "Kerb Weight (kg)",
    min_value=800,
    max_value=2500,
    value=1200
)

speed = st.sidebar.slider(
    "Maximum Speed (km/h)",
    min_value=60,
    max_value=200,
    value=120
)

horse_power = st.sidebar.slider(
    "Horse Power (HP)",
    min_value=50,
    max_value=400,
    value=100
)

torque = st.sidebar.slider(
    "Torque (Nm)",
    min_value=90,
    max_value=500,
    value=200
)

displacement = st.sidebar.slider(
    "Engine Displacement (cc)",
    min_value=800,
    max_value=4000,
    value=1500
)

# Make prediction when user clicks the button
if st.sidebar.button("Predict Fuel Efficiency"):
    # Prepare input data
    input_data = pd.DataFrame({
        'vehicle_type': [vehicle_type],
        'fuel_type': [fuel_type],
        'kerb_weight': [kerb_weight],
        'speed': [speed],
        'horse_power': [horse_power],
        'torque': [torque],
        'displacement': [displacement]
    })
    
    # Transform categorical variables
    input_data['vehicle_type'] = le_vehicle.transform(input_data['vehicle_type'])
    input_data['fuel_type'] = le_fuel.transform(input_data['fuel_type'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display prediction
    st.header("Prediction Results")
    st.write(f"Estimated Fuel Efficiency: {prediction:.2f} km/l")
    
    # Display feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.subheader("Feature Importance")
    st.bar_chart(feature_importance.set_index('Feature'))

# Add some additional information
st.markdown("""
### About this predictor
This application uses a Random Forest model to predict vehicle fuel efficiency based on various specifications.
The model takes into account:
- Vehicle type (Sedan, SUV, Hatchback)
- Fuel type (Petrol, Diesel)
- Kerb weight
- Maximum speed
- Horse power
- Torque
- Engine displacement

Please note that this is a demonstration model and predictions may not be entirely accurate.
""")
