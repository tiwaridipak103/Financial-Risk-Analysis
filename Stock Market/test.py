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

class MovingAverageStrategy(bt.Strategy):

    params = (('period_fast' , 30) , ('period_slow' , 200) ,)

    def __init__(self):
        self.close_data = self.data.close
        print(self.data[0])

        #usually this is where we create the indicators
        self.fast_sma = bt.indicators.MovingAverageSimple(self.close_data ,period = self.params.period_fast)

        self.slow_sma = bt.indicators.MovingAverageSimple(self.close_data ,period = self.params.period_slow)

    def next(self):

        # we have to check whether we have already opened a long position
        if not self.position:
            # we can open a long position if needed
            if self.fast_sma[0] > self.slow_sma[0] and self.fast_sma[-1] < self.slow_sma[-1]:
                #print('Buy')
                self.buy()

        else:
            # check whether to close the long position
            if self.fast_sma[0] < self.slow_sma[0] and self.fast_sma[-1] > self.slow_sma[-1]:
                #print('close')
                self.close()

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
                    cerebro.adddata(bt.feeds.PandasData(dataname = df , plot = False))

            except FileNotFoundError:
                print("The specified file was not found.")

            break


    cerebro.addobserver(bt.observers.Value)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate = 0.0)
    cerebro.addanalyzer(bt.analyzers.Returns)
    cerebro.addanalyzer(bt.analyzers.DrawDown)
    print('Done!!!')

    # # this is how we attach the strategy we have implemented
    cerebro.addstrategy(MovingAverageStrategy)

    cerebro.broker.set_cash(100000)
        
    #Commision fee is 1%    
    cerebro.broker.setcommission(0.01)
        
    print('Initial capital: $%.f' % cerebro.broker.getvalue())
    results = cerebro.run()

    # evalute the results
    print('Sharpe ratio : %.2f' %  results[0].analyzers.sharperatio.get_analysis()['sharperatio'])

    print('Return : %.2f%%' %  results[0].analyzers.returns.get_analysis()['rnorm100'])

    print('drawdown : %.2f%%' %  results[0].analyzers.drawdown.get_analysis()['drawdown'])

    print('Capital: %.2f' % cerebro.broker.getvalue())

            

    
