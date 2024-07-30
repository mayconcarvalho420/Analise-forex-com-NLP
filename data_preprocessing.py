import pandas as pd
from alpha_vantage.foreignexchange import ForeignExchange
import numpy as np

API_KEY = 'your_alpha_vantage_api_key'

def get_forex_data(from_currency, to_currency, output_size='compact'):
    fx = ForeignExchange(key=API_KEY)
    data, _ = fx.get_currency_exchange_daily(from_symbol=from_currency, to_symbol=to_currency, outputsize=output_size)
    df = pd.DataFrame(data).T
    df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close'})
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    return df

def preprocess_data(df):
    df['return'] = df['close'].pct_change()
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = get_forex_data('EUR', 'USD')
    df = preprocess_data(df)
    df.to_csv('data/forex_data.csv')
