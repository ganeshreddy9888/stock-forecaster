from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from fredapi import Fred

# Set the Matplotlib backend to Agg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, span):
    ema = data['Close'].ewm(span=span, adjust=False).mean()
    return ema

def calculate_macd(data, short_span=12, long_span=26, signal_span=9):
    short_ema = calculate_ema(data, short_span)
    long_ema = calculate_ema(data, long_span)
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def fetch_fred_data(series_id, start, end):
    fred = Fred(api_key='c4bff72331d2a9be3266c2c05bd9d60d')
    fred_data = fred.get_series(series_id, start, end)
    fred_data = fred_data.reset_index()
    fred_data.columns = ['Date', series_id]
    return fred_data

def prepare_data(stock, start, end):
    data = yf.download(stock, start, end)
    data.reset_index(inplace=True)

    # Fetch FRED economic data
    gdp_series_id = 'GDP'
    inflation_series_id = 'CPIAUCSL'
    gdp_data = fetch_fred_data(gdp_series_id, start, end)
    inflation_data = fetch_fred_data(inflation_series_id, start, end)

    # Merge stock data with FRED data
    data = pd.merge(data, gdp_data, on='Date', how='left')
    data = pd.merge(data, inflation_data, on='Date', how='left')
    data.ffill(inplace=True)

    data['RSI'] = calculate_rsi(data)
    data['EMA_50'] = calculate_ema(data, 50)
    data['EMA_200'] = calculate_ema(data, 200)
    data['MACD'], data['MACD_Signal'], data['MACD_Hist'] = calculate_macd(data)

    features = ['RSI', 'EMA_50', 'EMA_200', 'MACD', 'MACD_Signal', 'MACD_Hist', gdp_series_id, inflation_series_id]
    target = 'Close'

    data = data.dropna()
    X = data[features].values
    y = data[target].values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))

    return X_scaled, y_scaled, scaler, data

def create_sequences(X, y, time_step=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_step):
        X_seq.append(X[i:(i + time_step)])
        y_seq.append(y[i + time_step])
    return np.array(X_seq), np.array(y_seq)

def train_lstm_model(X_train, y_train, X_val, y_val):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_val, y_val))
    return model

def calculate_accuracy(data):
    # Calculate the direction of the actual price movement
    data['Price_Change'] = data['Close'].diff().shift(-1)
    data['Actual_Direction'] = np.where(data['Price_Change'] > 0, 1, 0)

    # EMA200 accuracy
    data['EMA200_Signal'] = np.where(data['Close'] > data['EMA_200'], 1, 0)
    ema200_accuracy = (data['EMA200_Signal'] == data['Actual_Direction']).mean()

    # RSI accuracy
    data['RSI_Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, 0, np.nan))
    rsi_accuracy = (data['RSI_Signal'] == data['Actual_Direction']).mean()

    # MACD accuracy
    data['MACD_Signal_Line'] = data['MACD'] - data['MACD_Signal']
    data['MACD_Signal'] = np.where(data['MACD_Signal_Line'] > 0, 1, 0)
    macd_accuracy = (data['MACD_Signal'] == data['Actual_Direction']).mean()

    return ema200_accuracy, rsi_accuracy, macd_accuracy

def plot_forecast(data, predictions, future_dates, future_predictions, stock_symbol):
    fig = make_subplots(rows=1, cols=1)
    
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(x=data['Date'], y=predictions, mode='lines', name='Predicted Prices'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Forecasted Prices', line=dict(dash='dot', color='red')))
    
    fig.update_layout(
        title=f'Stock Price Forecast for {stock_symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        autosize=True,
        margin=dict(l=50, r=20, t=50, b=50),
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        modebar=dict(
            remove=['toImage', 'autoScale2d', 'editInChartStudio', 'hoverCompareCartesian', 'lasso2d', 'orbitRotation', 'pan2d', 'resetScale2d', 'select2d', 'sendDataToCloud', 'toggleSpikelines', 'zoom2d', 'zoomIn2d', 'zoomOut2d'],
            add=['zoomIn2d', 'zoomOut2d', 'toImage']
        ),
    )
    fig.update_xaxes(showspikes=True, spikecolor='black', spikesnap='cursor', showline=True)
    fig.update_yaxes(showspikes=True, spikecolor='black', spikesnap='cursor', showline=True)
    
    fig_html = fig.to_html(full_html=False, config={'displaylogo': False})
    return fig_html

def forecast_stock(stock, start, end):
    X_scaled, y_scaled, scaler, data = prepare_data(stock, start, end)
    time_step = 60
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_step)

    # Split data into train and test sets
    train_size = int(len(X_seq) * 0.8)
    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:], y_seq[train_size:]

    # Train LSTM model
    model = train_lstm_model(X_train, y_train, X_val, y_val)

    # Predicting on validation set
    predictions_scaled = model.predict(X_val)
    predictions = scaler.inverse_transform(predictions_scaled)

    # Inverse transform the actual prices
    actual_prices = scaler.inverse_transform(y_val)

    # Forecasting future prices
    last_sequence = X_scaled[-time_step:]
    future_predictions = []
    for _ in range(252):  # Predict for next year (252 trading days)
        last_sequence = np.expand_dims(last_sequence, axis=0)
        future_pred_scaled = model.predict(last_sequence)
        future_pred = scaler.inverse_transform(future_pred_scaled)
        future_predictions.append(future_pred[0][0])

        # Prepare new sequence with correct dimensions
        new_feature = np.array([[future_pred_scaled[0][0]]] * last_sequence.shape[2]).T
        new_sequence = np.append(last_sequence[0, 1:], new_feature, axis=0)
        last_sequence = new_sequence

    future_dates = pd.date_range(start=pd.to_datetime(end), periods=253)[1:]

    # Calculate accuracy
    ema200_accuracy, rsi_accuracy, macd_accuracy = calculate_accuracy(data)

    # Plot actual vs predicted prices using Plotly
    plot_html = plot_forecast(data, predictions, future_dates, future_predictions, stock)

    best_buying_price = find_best_buying_price(data)
    return plot_html, best_buying_price

def find_best_buying_price(data):
    # Calculate Fibonacci retracement levels
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    if max_price == min_price:
        return None

    diff = max_price - min_price
    retracement_levels = {
        'level_0': max_price,
        'level_236': max_price - 0.236 * diff,
        'level_382': max_price - 0.382 * diff,
        'level_50': max_price - 0.50 * diff,
        'level_618': max_price - 0.618 * diff,
        'level_100': min_price,
    }
    
    # Use the average of these levels as the best buying price
    avg_buying_price = np.mean(list(retracement_levels.values()))
    return avg_buying_price

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    stock_symbol = request.form['stock_symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    if stock_symbol and start_date and end_date:
        plot_html, best_buying_price = forecast_stock(stock_symbol, start_date, end_date)
        return render_template('results.html', plot=plot_html, best_buying_price=best_buying_price)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
