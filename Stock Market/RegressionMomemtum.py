from datetime import datetime
import backtrader as bt
from backtrader import plot
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress

def download_data(stocks , start_date , end_date):
    data = {}
    
    for stock in stocks:
        ticker = yf.download(stock , start , end)
        data[stock] = ticker['Adj Close']

    return pd.DataFrame(data)

def calculate_momentum(data):
    log_data = np.log(data)
    x_data = np.arange(len(log_data))
    beta , _ , r , _ , _ = linregress(x_data ,log_data )

    # we have to annualize the slope
    # there are 252 trading days in a year

    return (1+beta)**252 * (r**2)

class Momentum(bt.Indicator):
    # Every trading day has a momentum parameter
    # Except for the first 90 days
    lines = ('momentum_trends',)
    params = (('period' , 90),)

    def __init__(self):
        self.addminperiod = self.params.period

    def next(self):
        returns =  np.log(self.data.get(size = self.params.period))
        x = np.arange(len(returns))
        beta, _ , r , _ , _ = linregress(x,returns )
        annualized = (1 + beta) ** 252
        self.lines.momentum_trends[0] = annualized * r ** 2

class MomentumStrategy(bt.Strategy):

    def __init__(self):
        self.counter = 0
        self.indicators = {}
        self.sorted_data = []
        # We store the SP500 ( this is the index) data as the first item of the dataset
        self.spy = self.data[0]
        #all the other stocks (present in SP500) 
        self.stocks = self.data[1:]

        for stock in self.stocks:
            self.indicators[stock] = {}
            self.indicators[stock]['momentum'] = Momentum(stock.close , period = 90)
            self.indicators[stock]['sma100'] = bt.indicators.\
                MovingAverageSimple(stock.close , period = 100)
            self.indicators[stock]['atr20'] = bt.indicators.ATR(stock , period = 20)

            # SMA for SP500 index - because we open long positions when the SP500 index
            # is above it's SMA(200) BULLISH MARKET

            self.sma200 = bt.indicators.MovingAverageSimple(self.spy.close , period = 200)

    def prenext(self):
        # count the number of the days elapsed
        self.next()

    def next(self):
        # a week has passed so we have to make trades
        if self.counter % 5 == 0:
            self.update_portfolio()

        if self.counter % 10 == 0 :
            # 2 weeks have passed
            self.update_portfolio()

        self.counter += 1


    def rebalance_portfolio(self):

        self.sorted_data = list(filter(lambda stock_data : len(stock_data) > 100 , self.stocks))
        # sort the SP500 stocks based on Momentum
        self.sorted_data.sort(key = lambda stock: self.indicators[stock]['momentum'][0] )
        num_stocks = len(self.sorted_data)

        # sell stocks (close the long positions) - top 20%
        for index , single_stock in enumerate(self.sorted_data):
            # We can check whether are there open positions
             if self.getposition(self.data).size:
                 # if the stock is not in the top 20% then close the long position
                 # sell the stock if it's price falls below its 100 days MA

                 if index > 0.2 * num_stocks or single_stock < self.indicators[single_stock]['sma100']:
                     self.close(single_stock)

        # we open long positions when the SMA is below the SP500 index
        if self.spy < self.sma200:
            return
        
        # Consider the top 20% of the stocks and buy accordingly
        for index, single_stock in enumerate(self.sorted_data[:int(0.2 * num_stocks)]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash > 0 and not self.getposition(self.data).size:
                size = value * 0.001 / self.indicators[single_stock]["atr20"]
                self.buy(single_stock , size = size)  


    def update_positions(self):
        num_stocks = len(self.sorted_data)    

        # top 20 % momentum range
        for index , single_stock in enumerate(self.sorted_data[:int(0.2 * num_stocks)]):
            cash = self.broker.get_cash()
            value = self.broker.get_value()
            if cash > 0 :
                size = value * 0.001 / self.indicators[single_stock]["atr20"]
                self.order_target_size(single_stock, size)




if __name__ == '__main__':
    
    start  = '2011-04-01'
    end = '2013-01-01'

    stocks = ['IBM' , 'TSLA' , 'MSFT' , 'AAPL' , 'XOM' , 'CVX' , 'INTC']

    stocks_data = download_data(stocks , start, end)

    momentums = pd.DataFrame(columns = stocks)

    for stock in stocks:
        momentums[stock] = stocks_data[stock].rolling(90).\
                           apply(calculate_momentum , raw = False)
        
    plt.figure(figsize = (12,9))
    plt.xlabel('Days')
    plt.ylabel('Stock Price')

    bests = momentums.max().sort_values(ascending=False).index[:5]
    print(momentums.max().sort_values(ascending=False)[:5])

    for best in bests:
        end = momentums[best].index.get_loc(momentums[best].idxmax())
        returns = np.log(stocks_data[best].iloc[end-90 : end])
        x = np.arange(len(returns))
        slope , intercept , r_value , p_value , std_err = linregress(x,returns )
        plt.plot(np.arange(len(stocks_data[best][end - 90 : end+90])), stocks_data[best][end - 90 : end+90])
        plt.plot(x, np.e ** (intercept + slope * x))

    plt.show()