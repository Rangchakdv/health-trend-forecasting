
import pandas as pd
import matplotlib.pyplot as plt

# Load clean dataset
df = pd.read_csv("dataset/health_data.csv")

# Convert Date column again (important)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# -------------------------------
# 1. Plot all parameters together
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df['SugarLevel'], label='Sugar Level')
plt.plot(df['BloodPressure'], label='Blood Pressure')
plt.plot(df['Weight'], label='Weight')

plt.title("Health Parameters Over Time")
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# 2. Moving Average (Trend)
# -------------------------------
df['Sugar_MA'] = df['SugarLevel'].rolling(window=5).mean()

plt.figure(figsize=(10, 5))
plt.plot(df['SugarLevel'], label='Original Sugar')
plt.plot(df['Sugar_MA'], label='5-Day Moving Average', linestyle='--')

plt.title("Sugar Level Trend (Moving Average)")
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# 3. Individual clean plots
# -------------------------------
df['BloodPressure'].plot(title="Blood Pressure Trend", figsize=(8,4))
plt.grid()
plt.show()

df['Weight'].plot(title="Weight Trend", figsize=(8,4))
plt.grid()
plt.show()
