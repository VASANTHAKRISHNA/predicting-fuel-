import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import_train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset or create a placeholder
@st.cache_data
def load_data():
    data = pd.read_csv('vehicle_data.csv')
    return data

data = load_data()

# Preprocess data
label_encoder = LabelEncoder()
if 'Fuel Type' in data.columns:
    data['Fuel Type'] = label_encoder.fit_transform(data['Fuel Type'])

X = data[['Fuel Type', 'Kerb Weight', 'Engine Displacement', 'Torque', 'Horse Power', 'Vehicle Speed']]
y = data['Fuel Efficiency']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title('Vehicle Fuel Efficiency Prediction')
st.write(f'Model Performance: MSE = {mse:.2f}, R² = {r2:.2f}')

fuel_type = st.selectbox('Fuel Type', label_encoder.classes_)
kerb_weight = st.number_input('Kerb Weight (kg)', min_value=500, max_value=5000, value=1500)
engine_displacement = st.number_input('Engine Displacement (cc)', min_value=500, max_value=8000, value=2000)
torque = st.number_input('Torque (Nm)', min_value=50, max_value=1000, value=200)
horse_power = st.number_input('Horse Power (hp)', min_value=10, max_value=1000, value=150)
vehicle_speed = st.number_input('Vehicle Speed (km/h)', min_value=0, max_value=300, value=100)

if st.button('Predict Fuel Efficiency'):
    input_data = np.array([[label_encoder.transform([fuel_type])[0], kerb_weight, engine_displacement, torque, horse_power, vehicle_speed]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)[0]
    st.write(f'Predicted Fuel Efficiency: {prediction:.2f} km/l')
