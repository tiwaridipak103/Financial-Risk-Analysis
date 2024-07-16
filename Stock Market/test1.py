# importing os module 
import os
from datetime import datetime
import backtrader as bt
from backtrader import plot
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress

if __name__ == '__main__':
    stocks = []
    cerebro = bt.Cerebro()

    with open(r'C:\Users\DELL\Desktop\Stock Market\archive\few_stocks.txt') as file_in:
        for line in file_in:
            stocks.append(line.strip('\n'))
            stock = line.strip('\n')

            # Path
            path = r'C:\Users\DELL\Desktop\Stock Market\archive\stocks\\' + stock + '.csv'

            try:
                
                df = pd.read_csv( path , parse_dates= True , index_col = 0)
                if len(df) > 100:
                    cerebro.adddata(bt.feeds.PandasData(dataname = df , plot = False, name=stock))

            except FileNotFoundError:
                print("The specified file was not found.")

    # Check if a specific data feed is in Cerebro
    symbol_to_check = 'SP'
    for data_feed in cerebro.datas:
        if data_feed._name == symbol_to_check:
            print(f"{symbol_to_check} data is in Cerebro.")
            print(data_feed.close)
            break
        else:
            print(f"{symbol_to_check} data is not in Cerebro.")

    # # Run the backtest
    # cerebro.run()

    # # Extract the data from the Cerebro instance
    # data_list = [data for data in cerebro.datas]

    # # Convert the data to a pandas DataFrame
    # df = pd.concat([data.close for data in data_list], axis=1, keys=[data._name for data in data_list])

    # # Print the first few rows of the DataFrame
    # print(df.head())
