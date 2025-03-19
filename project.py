# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===========================
# Step 1: Create a Dummy Dataset
# ===========================

# Set random seed for reproducibility
np.random.seed(42)

# Generate date-time range for 3 years (hourly data)
date_rng = pd.date_range(start='2020-01-01', end='2023-01-01', freq='H')

# Number of records
num_records = len(date_rng)

# Generate synthetic air pollution and weather data
data = {
    'timestamp': date_rng,
    'PM2.5': np.random.uniform(10, 300, num_records),  # µg/m³
    'PM10': np.random.uniform(20, 400, num_records),   # µg/m³
    'NO2': np.random.uniform(5, 200, num_records),     # ppb
    'CO': np.random.uniform(0.1, 5.0, num_records),    # ppm
    'temperature': np.random.uniform(5, 40, num_records),  # °C
    'humidity': np.random.uniform(10, 100, num_records),   # %
    'wind_speed': np.random.uniform(0, 15, num_records),   # m/s
    'traffic_density': np.random.uniform(50, 1000, num_records)  # Vehicles per hour
}

# Create DataFrame
df = pd.DataFrame(data)

# Compute AQI (Simplified formula for learning purposes)
df['AQI'] = (df['PM2.5'] * 0.5) + (df['PM10'] * 0.3) + (df['NO2'] * 0.1) + (df['CO'] * 10)

# Save to CSV
df.to_csv('air_quality_data.csv', index=False)

print("✅ Dataset created successfully and saved as 'air_quality_data.csv'!")

# ===========================
# Step 2: Load and Explore the Dataset
# ===========================

# Load dataset
df = pd.read_csv('air_quality_data.csv')

# Show the first 5 rows
print("\n📌 First 5 rows of the dataset:")
print(df.head())

# Check dataset summary
print("\n📌 Dataset Info:")
print(df.info())

# Basic statistics
print("\n📌 Dataset Statistics:")
print(df.describe())

# ===========================
# Step 3: Data Preprocessing
# ===========================

# Check for missing values
print("\n📌 Missing Values in Dataset:")
print(df.isnull().sum())

# Define features and target variable
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'temperature', 'humidity', 'wind_speed', 'traffic_density']
X = df[features]  # Independent variables
y = df['AQI']  # Target variable

# ===========================
# Step 4: Split Data into Training and Testing Sets
# ===========================

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"\n✅ Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

# ===========================
# Step 5: Train the Machine Learning Model
# ===========================

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict AQI on test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Model Performance:")
print(f"✅ Mean Absolute Error (MAE): {mae}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse}")

# ===========================
# Step 6: Visualize Predictions
# ===========================

# Plot actual vs predicted AQI
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual AQI", color='blue')
plt.plot(y_pred[:100], label="Predicted AQI", color='red', linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("AQI")
plt.title("Actual vs Predicted AQI")
plt.legend()
plt.show()

# ===========================
# Step 7: Feature Importance
# ===========================

# Get feature importance from the trained model
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 4))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance")
plt.show()

print("\n✅ Project Completed Successfully! 🎉")

import pickle

# Save the trained model
with open("air_quality_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as 'air_quality_model.pkl'")
