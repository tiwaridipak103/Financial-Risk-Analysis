#%%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf




def download_data(stock , start_date , end_date):
    data = {}
    ticker = yf.download(stock , start_date , end_date)
    data['Price'] = ticker['Adj Close']
    #return pd.DataFrame(data)
    return data['Price'].values


if __name__ == '__main__':
    
    # start  = '2010-10-05'
    # end = '2014-01-05'

    # stock_data = download_data('IBM' , start, end)

    # plt.figure(figsize=(20,10))
    # plt.plot(stock_data)

    # plt.rc("figure", figsize=(20,10))
    # plot_acf(stock_data, lags=100)
    # plt.show()

    mu, sigma = 0, 0.1 # mean and standard deviation
    x = np.random.normal(mu, sigma, 1000)

    plt.figure(figsize=(20,10))
    plt.plot(x)

    plt.rc("figure", figsize=(20,10))
    plot_acf(x, lags=100)
    plt.show()

