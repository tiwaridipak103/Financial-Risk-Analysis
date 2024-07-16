#%%
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np

def download_data(stock , start_date , end_date):
    data = {}
    ticker = yf.download(stock , start_date , end_date)
    data['Price'] = ticker['Adj Close']
    return pd.DataFrame(data)


if __name__ == '__main__':
    
    start  = '2015-01-01'
    end = '2020-01-01'

    stock_data = download_data('IBM' , start, end)

    stock_data['return'] = np.log(stock_data['Price'] / stock_data['Price'].shift(1))
    stock_data['move'] = stock_data['Price'] - stock_data['Price'].shift(1)

    # Average then the 0 values do not count
    stock_data['up'] = np.where(stock_data['move'] > 0 , stock_data['move'] , 0)
    stock_data['down'] = np.where(stock_data['move'] < 0 , stock_data['move'] , 0)

    # Relative Strength
    stock_data['average_gain'] = stock_data['up'].rolling(14).mean() 
    stock_data['average_loss'] = stock_data['down'].abs().rolling(14).mean() 

    RS  = stock_data['average_gain'] / stock_data['average_loss']

    stock_data['rsi'] = 100.0 - (100.0 /(1.0 + RS))

    stock_data = stock_data.dropna()


    print(stock_data)
    
    plt.plot(stock_data['rsi'])
    plt.show()
