import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("🩺 Health Trend Forecasting System")

# Upload file
uploaded_file = st.file_uploader("Upload Health Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Select parameter
    parameter = st.selectbox("Select Health Parameter", df.columns)

    data = df[parameter]

    # Plot original data
    st.subheader("Original Data")
    st.line_chart(data)

    # ARIMA Model
    model = ARIMA(data, order=(2,1,2))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=10)

    # Show prediction
    st.subheader("Forecast (Next 10 Days)")
    st.write(forecast)

    # Plot forecast
    fig, ax = plt.subplots()
    ax.plot(data, label="Actual")
    ax.plot(forecast, label="Forecast", linestyle='--')
    ax.legend()

    st.pyplot(fig)
