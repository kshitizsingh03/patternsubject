import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a small dataset
np.random.seed(42)
data_size = 20

# Generating 6 random feature columns
X = np.random.rand(data_size, 6) * 10  # Random values between 0 and 10
y = 3*X[:,0] + 2*X[:,1] + 1.5*X[:,2] + np.random.randn(data_size) * 2  # Linear combination with some noise

# Convert to DataFrame
column_names = [f'Feature{i+1}' for i in range(6)]
df = pd.DataFrame(X, columns=column_names)
df['Target_Column'] = y

# Display first few rows
print("Dataset Head:")
print(df.head())

# Splitting into train-test sets
X_train, X_test, y_train, y_test = train_test_split(df[column_names], df['Target_Column'], test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Custom prediction
custom_input = np.array([[2, 4, 6, 8, 10, 12]])  # Example input
custom_prediction = model.predict(custom_input)
print("Prediction for custom input:", custom_prediction)