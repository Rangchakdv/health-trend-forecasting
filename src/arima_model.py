import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("dataset/health_data.csv")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Select parameter
data = df['SugarLevel']

# -------------------------------
# Train-Test Split
# -------------------------------
train = data[:50]
test = data[50:]

# -------------------------------
# Build ARIMA Model
# -------------------------------
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()

# -------------------------------
# Predict on test data
# -------------------------------
predictions = model_fit.forecast(steps=len(test))

# -------------------------------
# Plot comparison
# -------------------------------
plt.figure(figsize=(10,5))

plt.plot(train, label="Training Data")
plt.plot(test, label="Actual Data")
plt.plot(test.index, predictions, label="Predicted Data", linestyle='--')

plt.title("ARIMA Prediction vs Actual")
plt.xlabel("Date")
plt.ylabel("Sugar Level")
plt.legend()
plt.grid()

plt.show()
