import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

def plot_predictions(df, model, look_back=1):
    df_values = df['return'].values.reshape(-1, 1)
    X, _ = create_dataset(df_values, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    predictions = model.predict(X)

    plt.figure(figsize=(14, 7))
    plt.plot(df.index[look_back:], df['return'][look_back:], label='Real')
    plt.plot(df.index[look_back:], predictions, label='Previsto', color='red')
    plt.title('Previsão de Movimentação de Preços no Forex')
    plt.xlabel('Data')
    plt.ylabel('Retorno')
    plt.legend()
    plt.show()

def create_dataset(df, look_back=1):
    X, Y = [], []
    for i in range(len(df) - look_back):
        X.append(df[i:(i + look_back), 0])
        Y.append(df[i + look_back, 0])
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    df = pd.read_csv('data/forex_data.csv', index_col='date', parse_dates=True)
    model = load_model('models/model.h5')
    plot_predictions(df, model)
