from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import pandas as pd

def create_dataset(df, look_back=1):
    X, Y = [], []
    for i in range(len(df) - look_back):
        X.append(df[i:(i + look_back), 0])
        Y.append(df[i + look_back, 0])
    return np.array(X), np.array(Y)

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    df = pd.read_csv('data/forex_data.csv', index_col='date', parse_dates=True)
    df = df['return'].values.reshape(-1, 1)

    look_back = 1
    X, Y = create_dataset(df, look_back)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = create_model((X.shape[1], 1))
    model.fit(X, Y, epochs=20, batch_size=1, verbose=2)

    model.save('models/model.h5')
