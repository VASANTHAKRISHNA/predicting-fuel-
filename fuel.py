
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model selection import train test split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'your_dataset.csv' with your actual file)
# Ensure your dataset has columns for fuel type, kerb weight, engine displacement,
# torque, horsepower, vehicle speed, and fuel efficiency (target variable)
try:
    data = pd.read_csv('your_dataset.csv')
except FileNotFoundError:
    st.error("Dataset file not found. Please upload your dataset (your_dataset.csv) to the app's directory.")
    st.stop()


# Preprocessing (example, adapt to your specific data)
# Handle missing values, convert categorical variables to numerical
data = data.dropna()  # Remove rows with missing values (or use imputation)

# Convert fuel type to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['Fuel_Type'], drop_first=True) #Example, adapt to your column name


# Feature and Target variables
X = data.drop('Fuel_Efficiency', axis=1) #Replace with your target column
y = data['Fuel_Efficiency'] #Replace with your target column


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor (or another suitable model)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Streamlit app
st.title("Fuel Efficiency Prediction")

# Input features for prediction (User interface)
fuel_type = st.selectbox('Fuel Type', data['Fuel_Type'].unique()) #Adapt to your fuel type colum
kerb_weight = st.number_input("Kerb Weight (kg)")
engine_displacement = st.number_input("Engine Displacement (cc)")
torque = st.number_input("Torque (Nm)")
horsepower = st.number_input("Horsepower (hp)")
vehicle_speed = st.number_input("Vehicle Speed (km/h)")


# Create a dataframe for the input
input_data = pd.DataFrame({
    'Kerb_Weight': [kerb_weight],
    'Engine_Displacement': [engine_displacement],
    'Torque': [torque],
    'Horsepower': [horsepower],
    'Vehicle_Speed': [vehicle_speed]
})

#One-hot encode fuel type
input_data = pd.get_dummies(input_data)
missing_cols = set( X.columns ) - set( input_data.columns )
for c in missing_cols:
    input_data[c] = 0
input_data = input_data[X.columns]




# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Fuel Efficiency: {prediction[0]}")

# Display model performance metrics
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"R-squared (R2): {r2}")
