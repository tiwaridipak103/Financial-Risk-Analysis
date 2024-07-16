import yfinance as yf
import pandas as pd


def download_data(stock , start_date , end_date):
    data = {}
    ticker = yf.download(stock , start_date , end_date)
    data['Price'] = ticker['Adj Close']
    return pd.DataFrame(data)


if __name__ == '__main__':
    
    start  = '2010-10-05'
    end = '2014-01-05'

    stock_data = download_data('IBM' , start, end)
    print(stock_data)