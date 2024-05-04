import pandas as pd import numpy as np import matplotlib.pyplot as plt from tensorflow.keras.models import Sequential from tensorflow.keras.layers import LSTM, Dense from sklearn.preprocessing import MinMaxScaler from sklearn.metrics import mean_squared_error 

from sklearn.metrics import accuracy_score 
 
# Load data 
df = pd.read_csv('/content/drive/MyDrive/Uniqlo.csv', index_col=
0, parse_dates=True) df.dropna(inplace=True) 
 
# Scale data 
scaler = MinMaxScaler(feature_range=(0, 1)) scaled_data = scaler.fit_transform(df['Close'].values.reshape(-
1, 1)) 
 
# Define training and testing data train_size = int(len(scaled_data) * 0.8) train_data = scaled_data[:train_size] test_data = scaled_data[train_size:] 
 
# Define helper function to create data sequences for LSTM def create_sequences(data, sequence_length): 
    x = []     y = []     for i in range(len(data) - sequence_length - 1):         sequence = data[i:(i + sequence_length), 0]         target = data[i + sequence_length, 0] 
        x.append(sequence) 
        y.append(target)     return np.array(x), np.array(y) 
 
# Create sequences for LSTM sequence_length = 30 x_train, y_train = create_sequences(train_data, sequence_length) x_test, y_test = create_sequences(test_data, sequence_length)  
# Reshape data for LSTM input 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1
], 1)) 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1
)) 
 
# Define LSTM model architecture model = Sequential() model.add(LSTM(20, input_shape=(sequence_length, 1))) model.add(Dense(1)) model.compile(optimizer='adam', loss='mean_squared_error')  
from sklearn.metrics import accuracy_score 
 
# Load data 
df = pd.read_csv('/content/drive/MyDrive/Uniqlo.csv', index_col=
0, parse_dates=True) df.dropna(inplace=True) 
 
# Scale data 
scaler = MinMaxScaler(feature_range=(0, 1)) scaled_data = scaler.fit_transform(df['Close'].values.reshape(-
1, 1)) 
 
# Define training and testing data train_size = int(len(scaled_data) * 0.8) train_data = scaled_data[:train_size] test_data = scaled_data[train_size:] 
 
# Define helper function to create data sequences for LSTM 
def create_sequences(data, sequence_length): 
    x = []     y = []     for i in range(len(data) - sequence_length - 1):         sequence = data[i:(i + sequence_length), 0]         target = data[i + sequence_length, 0] 
        x.append(sequence) 
        y.append(target)     return np.array(x), np.array(y) 
 
# Create sequences for LSTM sequence_length = 30 x_train, y_train = create_sequences(train_data, sequence_length) x_test, y_test = create_sequences(test_data, sequence_length)  
# Reshape data for LSTM input 
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1
], 1)) 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1
)) 
 
# Define LSTM model architecture model = Sequential() model.add(LSTM(20, input_shape=(sequence_length, 1))) model.add(Dense(1)) model.compile(optimizer='adam', loss='mean_squared_error')  
# Train model 
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test)) 
 
# Plot loss curves 
plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('Model loss') plt.ylabel('Loss') plt.xlabel('Epoch') plt.legend(['Train', 'Validation'], loc='upper right') plt.show() 
 
# Make predictions on test data y_pred = model.predict(x_test) 
 
# Scale data back to original range y_pred = scaler.inverse_transform(y_pred) y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  
# Plot predicted vs actual values plt.plot(y_test, label='Actual') plt.plot(y_pred, label='Predicted') plt.title('Stock Price Prediction') plt.xlabel('Time') plt.ylabel('Price') plt.legend() plt.show() 
 
train_predict = model.predict(x_train) train_predict = scaler.inverse_transform(train_predict) y_train = scaler.inverse_transform([y_train]) train_score = np.sqrt(mean_squared_error(y_train[0], train_predi ct[:,0])) print('Train Score: %.2f RMSE' % (train_score))  
