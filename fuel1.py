import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Vehicle Fuel Efficiency Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom Multiple Linear Regression Model
class CustomLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calculate coefficients using normal equation
        # β = (X^T X)^(-1) X^T y
        XTX = np.dot(X.T, X)
        XTX_inv = np.linalg.inv(XTX)
        XTy = np.dot(X.T, y)
        coefficients = np.dot(XTX_inv, XTy)
        
        self.intercept = coefficients[0]
        self.coefficients = coefficients[1:]
    
    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept

# Function to encode categorical variables
def encode_categorical(value, categories):
    return [1 if value == category else 0 for category in categories]

# Function to load and prepare data
@st.cache_data
def load_data():
    # Sample data - replace with your actual dataset
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'vehicle_type': np.random.choice(['Sedan', 'SUV', 'Hatchback'], n_samples),
        'fuel_type': np.random.choice(['Petrol', 'Diesel'], n_samples),
        'kerb_weight': np.random.uniform(800, 2500, n_samples),
        'speed': np.random.uniform(60, 200, n_samples),
        'horse_power': np.random.uniform(50, 400, n_samples),
        'torque': np.random.uniform(90, 500, n_samples),
        'displacement': np.random.uniform(800, 4000, n_samples)
    })
    
    # Generate target variable with some relationship to features
    data['fuel_efficiency'] = (
        15 - 0.002 * data['kerb_weight'] 
        - 0.02 * data['horse_power'] 
        + 0.01 * data['torque'] 
        - 0.001 * data['displacement']
        + np.random.normal(0, 1, n_samples)
    )
    
    return data

# Title and description
st.title("Vehicle Fuel Efficiency Prediction")
st.write("Enter vehicle specifications to predict fuel efficiency (km/l)")

# Load data
data = load_data()

# Prepare features for training
vehicle_types = ['Sedan', 'SUV', 'Hatchback']
fuel_types = ['Petrol', 'Diesel']

# Prepare training data
X_train = []
for _, row in data.iterrows():
    # Encode categorical variables
    vehicle_type_encoded = encode_categorical(row['vehicle_type'], vehicle_types)
    fuel_type_encoded = encode_categorical(row['fuel_type'], fuel_types)
    
    # Combine all features
    features = vehicle_type_encoded + fuel_type_encoded + [
        row['kerb_weight'],
        row['speed'],
        row['horse_power'],
        row['torque'],
        row['displacement']
    ]
    X_train.append(features)

X_train = np.array(X_train)
y_train = data['fuel_efficiency'].values

# Train model
model = CustomLinearRegression()
model.fit(X_train, y_train)

# Create input form
st.sidebar.header("Vehicle Specifications")

vehicle_type = st.sidebar.selectbox(
    "Vehicle Type",
    options=vehicle_types
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    options=fuel_types
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
    vehicle_type_encoded = encode_categorical(vehicle_type, vehicle_types)
    fuel_type_encoded = encode_categorical(fuel_type, fuel_types)
    
    input_features = np.array(
        vehicle_type_encoded + 
        fuel_type_encoded + 
        [kerb_weight, speed, horse_power, torque, displacement]
    ).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    
    # Display prediction
    st.header("Prediction Results")
    st.write(f"Estimated Fuel Efficiency: {prediction:.2f} km/l")
    
    # Display feature importance (using absolute values of coefficients)
    feature_names = (
        [f"vehicle_type_{vt}" for vt in vehicle_types] +
        [f"fuel_type_{ft}" for ft in fuel_types] +
        ['kerb_weight', 'speed', 'horse_power', 'torque', 'displacement']
    )
    
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': np.abs(model.coefficients)
    }).sort_values('Coefficient', ascending=False)
    
    st.subheader("Feature Coefficients (Absolute Values)")
    st.bar_chart(importance.set_index('Feature'))

# Add explanatory information
st.markdown("""
### About this predictor
This application uses a custom Multiple Linear Regression model to predict vehicle fuel efficiency based on various specifications.
The model takes into account:
- Vehicle type (Sedan, SUV, Hatchback)
- Fuel type (Petrol, Diesel)
- Kerb weight
- Maximum speed
- Horse power
- Torque
- Engine displacement

Note: This is a simplified model for demonstration purposes. The predictions are based on synthetic data and a basic linear regression implementation.

#### How it works:
1. Categorical variables (vehicle type and fuel type) are encoded using one-hot encoding
2. Numerical features are used directly in the model
3. The model uses the normal equation method to find the best-fit coefficients
4. Predictions are made using linear combinations of the input features
""")
