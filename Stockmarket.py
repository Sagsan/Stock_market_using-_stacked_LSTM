import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
df = pd.read_csv(r'D:\raw_data.csv')

# Select the 'Close' price column for training
feature_column = 'Close'

# Check if the feature column exists in the dataset
if feature_column not in df.columns:
    raise ValueError(f"'{feature_column}' column not found in the dataset.")

# Select the feature column for training
data = df[[feature_column]]

# Convert the dataframe to a numpy array
dataset = data.values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60  # Number of time steps to look back
x_train, y_train = create_dataset(scaled_data, time_step)

# Reshape the data (LSTM expects 3D input)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=32)

# Predict future stock prices
future_data = scaled_data[-time_step:]  # Use the last 'time_step' data points for prediction
future_predictions = []
for i in range(30):  # Predict the next 30 days
    x_input = np.reshape(future_data, (1, time_step, 1))
    prediction = model.predict(x_input, verbose=0)
    future_predictions.append(prediction[0, 0])
    future_data = np.append(future_data[1:], prediction, axis=0)

# Inverse scale the predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Print the future predictions
print("Future Stock Price Predictions:")
print(future_predictions)
