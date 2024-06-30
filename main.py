# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import SimpleRNN, Dense

# # Load the data
# data = pd.read_csv('datas\AAPL.csv')  # Ensure your data has a 'Date' and 'Close' column
# data['Date'] = pd.to_datetime(data['Date'])
# data.set_index('Date', inplace=True)

# # Prepare the data for RNN
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# # Create sequences
# sequence_length = 60
# X = []
# y = []

# for i in range(sequence_length, len(scaled_data)):
#     X.append(scaled_data[i-sequence_length:i, 0])
#     y.append(scaled_data[i, 0])

# X = np.array(X)
# y = np.array(y)

# # Reshape the data
# X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# # Split into train and test sets
# split = int(0.8 * len(X))  # 80% training, 20% testing
# X_train, X_test = X[:split], X[split:]
# y_train, y_test = y[:split], y[split:]

# # Build the SimpleRNN model
# model = Sequential()
# model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
# model.add(SimpleRNN(units=50))
# model.add(Dense(units=1))

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

# # Make predictions
# train_predicted_stock_price = model.predict(X_train)
# test_predicted_stock_price = model.predict(X_test)

# # Inverse transform to get original scale
# train_predicted_stock_price = scaler.inverse_transform(train_predicted_stock_price)
# test_predicted_stock_price = scaler.inverse_transform(test_predicted_stock_price)

# # Plot the results
# plt.figure(figsize=(10, 6))

# # Plot actual data
# plt.plot(data.index[sequence_length:split+sequence_length], scaler.inverse_transform(y_train.reshape(-1, 1)), color='blue', label='Actual Train Stock Price')
# plt.plot(data.index[split+sequence_length:], scaler.inverse_transform(y_test.reshape(-1, 1)), color='green', label='Actual Test Stock Price')

# # Plot predicted data
# plt.plot(data.index[sequence_length:split+sequence_length], train_predicted_stock_price, color='red', label='Predicted Train Stock Price')
# plt.plot(data.index[split+sequence_length:], test_predicted_stock_price, color='orange', label='Predicted Test Stock Price')

# plt.title('Stock Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df = pd.read_csv('datas\AAPL.csv')  # Replace 'your_dataset.csv' with your actual dataset filename

print(df)
# Use 'Close' prices for simplicity, you can choose other features as needed
data = df['Close'].values.reshape(-1, 1)

print(data[0])
print(data[1])
print(data[2])
print(data)

# # Normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data)

# # Split data into training and testing sets
# train_size = int(len(scaled_data) * 0.8)
# train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# # Function to create sequences for RNN
# def create_sequences(data, seq_length):
#     xs, ys = [], []
#     for i in range(len(data) - seq_length - 1):
#         x = data[i:(i + seq_length)]
#         y = data[i + seq_length]
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)

# # Set sequence length
# seq_length = 10

# # Create sequences for training and testing
# X_train, y_train = create_sequences(train_data, seq_length)
# X_test, y_test = create_sequences(test_data, seq_length)

# # Reshape input to be [samples, time steps, features] expected by RNN
# X_train = np.reshape(X_train, (X_train.shape[0], seq_length, 1))
# X_test = np.reshape(X_test, (X_test.shape[0], seq_length, 1))

# # Build the RNN model
# model = Sequential()
# model.add(SimpleRNN(units=50, activation='relu', input_shape=(seq_length, 1)))
# model.add(Dense(units=1))

# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=32)

# # Evaluate the model
# train_loss = model.evaluate(X_train, y_train)
# test_loss = model.evaluate(X_test, y_test)

# print(f"Train Loss: {train_loss}")
# print(f"Test Loss: {test_loss}")

# # Predictions
# predicted = model.predict(X_test)
# predicted_prices = scaler.inverse_transform(predicted)

# # Plotting
# plt.figure(figsize=(14, 7))
# plt.plot(df.index[-len(predicted_prices):], df['Close'].values[-len(predicted_prices):], label='Actual Prices')
# plt.plot(df.index[-len(predicted_prices):], predicted_prices, label='Predicted Prices')
# plt.title('Stock Price Prediction using Simple RNN')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True)
# plt.show()
