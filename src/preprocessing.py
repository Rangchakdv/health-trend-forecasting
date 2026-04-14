import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv("dataset/health_data.csv")

print("Initial Data:")
print(df.head())

# Step 2: Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Step 3: Set Date as index
df.set_index('Date', inplace=True)

# Step 4: Sort data
df.sort_index(inplace=True)

# Step 5: Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Step 6: Fill missing values (if any)
df.ffill(inplace=True)

# Step 7: Final cleaned data
print("\nCleaned Data:")
print(df.head())

# Step 8: Plot graphs
df['SugarLevel'].plot(title="Sugar Level Over Time")
plt.show()

df['BloodPressure'].plot(title="Blood Pressure Over Time")
plt.show()

df['Weight'].plot(title="Weight Over Time")
plt.show()

# Step 9: Save cleaned dataset
df.to_csv("dataset/clean_health_data.csv")

print("\nClean dataset saved successfully!")
