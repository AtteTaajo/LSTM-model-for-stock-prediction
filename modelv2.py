import keras
import tensorflow as tf
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ticker = "NOKIA.HE"  # Nokia Corporation stock ticker on Helsinki Stock Exchange

# Download all available historical data
data = yf.download(ticker, start="1990-01-01")



close = data['Close'].values


#Time period for training
years_for_training = 5
train_end_year = 2014
train_start_year = train_end_year - years_for_training + 1

train_mask = (data.index.year >= train_start_year) & (data.index.year <= train_end_year) 
test_mask = data.index.year > train_end_year

# Split the data
train_data = close[train_mask]
test_data = close[test_mask]

# Scale only the training data, then transform both sets
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
test_data_scaled = scaler.transform(test_data.reshape(-1, 1))

# Augment the training data with noise
noise_level = 0.13
  # Adjust this value as needed (e.g., 0.01 for 1% noise)
augmented_train_data_scaled = train_data_scaled + np.random.normal(0, noise_level, train_data_scaled.shape)

# x is the last 600 days of training data, y is the next day
lookback = 645
x_train, y_train = [], []
for i in range(lookback, len(augmented_train_data_scaled)):
    x_train.append(augmented_train_data_scaled[i-lookback:i, 0])
    y_train.append(augmented_train_data_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # shape: (samples, 90, 1)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(256, input_shape=(x_train.shape[1], 1), return_sequences=True))
model.add(keras.layers.LSTM(256, return_sequences=True))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=False)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(1))

model.summary()

#cp = ModelCheckpoint(filepath='modelv2.h5', save_best_only=True)

model.compile(optimizer=Adam(learning_rate=0.000025), loss=MeanSquaredError(), metrics=[keras.metrics.RootMeanSquaredError()])

model.fit(x_train, y_train, epochs=40, batch_size=6, validation_split=0.5)


# Prepare test data
x_test, y_test = [], []
for i in range(lookback, len(test_data_scaled)):
    x_test.append(test_data_scaled[i-lookback:i, 0])
    y_test.append(test_data_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Standard predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_orig, predictions))
mae = mean_absolute_error(y_test_orig, predictions)
r2 = r2_score(y_test_orig, predictions)

print(f"Standard predictions - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

'''

# Make predictions for the actual test period (2015–2025)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Rescale predictions back to original scale

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(data.index[train_mask], train_data, label='Training Data', color='blue')
plt.plot(data.index[test_mask], test_data, label='Test Data', color='orange')
plt.plot(data.index[test_mask][lookback:], predictions, label='Test Predictions', color='green')
plt.legend()
plt.show()
'''

'''
# Lisää recursive prediction******************
def predict_next_days(model, last_data, days=120):
    predictions = []
    current_data = last_data

    for _ in range(days):
        pred = model.predict(current_data)
        predictions.append(pred[0, 0])
        current_data = np.append(current_data[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    return np.array(predictions)

# Käytä mallia ennustamaan seuraavat 30 päivää
last_90_days = x_test[-1].reshape(1, lookback, 1)
future_predictions = predict_next_days(model, last_90_days)

# Muunna ennusteet alkuperäiseen mittakaavaan
future_predictions = scaler.inverse_transform(future_predictions.reshape(-1, 1))

# Lisää ennusteet plotiin
plt.figure(figsize=(14, 7))
plt.plot(data.index[train_mask], train_data, label='Training Data', color='blue')
plt.plot(data.index[test_mask], test_data, label='Test Data', color='orange')
plt.plot(data.index[test_mask][lookback:], predictions, label='Predictions', color='green')
plt.plot(pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=120, freq='B'), future_predictions, label='Future Predictions', color='red')
plt.legend()
plt.show()

'''

# Recursive prediction for the whole test period (2015–2025)
recursive_predictions = []
current_window = train_data_scaled[-lookback:].reshape(1, lookback, 1)

for _ in range(len(test_data_scaled)):
    pred = model.predict(current_window)
    recursive_predictions.append(pred[0, 0])
    current_window = np.append(current_window[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

# Inverse transform to get EUR values
recursive_predictions = scaler.inverse_transform(np.array(recursive_predictions).reshape(-1, 1))

# Plot
plt.figure(figsize=(14, 7))
plt.plot(data.index[train_mask], train_data, label='Training Data', color='blue')
plt.plot(data.index[test_mask], test_data, label='Test Data', color='orange')
plt.plot(data.index[test_mask], recursive_predictions, label='Recursive Test Predictions', color='purple')
plt.title("Nokia Stock Price Prediction with 5 Years of Training Data")
plt.xlabel("Date")
plt.ylabel("Price EUR")
plt.legend()
plt.grid(True)
plt.show()

# Calculate metrics (align lengths first)
rec_preds_aligned = recursive_predictions[-len(y_test_orig):]

rmse_rec = np.sqrt(mean_squared_error(y_test_orig, rec_preds_aligned))
mae_rec = mean_absolute_error(y_test_orig, rec_preds_aligned)
r2_rec = r2_score(y_test_orig, rec_preds_aligned)

print(f"Recursive predictions - RMSE: {rmse_rec:.4f}, MAE: {mae_rec:.4f}, R2: {r2_rec:.4f}")