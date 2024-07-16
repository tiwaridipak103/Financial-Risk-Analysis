#%%
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np  


class MovingAverageCrossover:

    def __init__(self,capital , stock , start , end  , short_period , long_period):
        self.data = None
        self.is_long = False
        self.short_period = short_period
        self.long_period = long_period
        self.capital = capital
        self.equity  = [capital]
        self.stock = stock
        self.start = start
        self.end = end

    def download_data(self):
        stock_data = {}
        ticker = yf.download(self.stock , self.start , self.end)
        stock_data['Price'] = ticker['Adj Close']
        self.data =  pd.DataFrame(stock_data)

    def construct_signals(self):
  
        self.data['Short SMA'] = self.data['Price'].ewm(span=self.short_period , adjust = False).mean()
        self.data['Long SMA'] = self.data['Price'].ewm(span=self.long_period, adjust = False).mean()

        self.data['move'] = self.data['Price'] - self.data['Price'].shift(1)

        # Average then the 0 values do not count
        self.data['up'] = np.where(self.data['move'] > 0 , self.data['move'] , 0)
        self.data['down'] = np.where(self.data['move'] < 0 , self.data['move'] , 0)

        # Relative Strength
        self.data['average_gain'] = self.data['up'].rolling(14).mean() 
        self.data['average_loss'] = self.data['down'].abs().rolling(14).mean() 

        RS  = self.data['average_gain'] / self.data['average_loss']

        self.data['rsi'] = 100.0 - (100.0 /(1.0 + RS))

        self.data = self.data.dropna()

    def plot_signals(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.data['Price'] , label = 'Short Price' )
        plt.plot(self.data['Short SMA'] , label = 'Short SMA' , color = 'red')
        plt.plot(self.data['Long SMA'] , label = 'Long SMA' , color = 'blue')
        plt.title('Moving Average (MA) Crossover Trading Strategy with RSI Indicator')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.show()


    def simulate(self):
        #we consider all the trading days and decide whether to open a
        price_when_buy = 0

        for index , row  in self.data.iterrows() :
            #close the long position we have opened
            if row['Short SMA'] < row['Long SMA'] and self.is_long:
                self.equity.append(self.capital * row['Price'] / price_when_buy)
                self.is_long = False 
                

            elif  row['Short SMA'] > row['Long SMA'] and not self.is_long and row['rsi'] < 30:
                # Open a long position
                price_when_buy = row['Price']
                self.is_long = True 

    def plot_equity(self):
        print("Profit of the trading strategy: %.2f%%" % (
                      (float(self.equity[-1]) - float(self.equity[0])) /
                       float(self.equity[0]) * 100))
        
        print("Actual capital: $%0.2f" % self.equity[-1])
        plt.figure(figsize=(12,6))
        plt.title('Equity Curve')
        plt.plot(self.equity , label = 'Stock Price' , color = 'green')
        plt.xlabel('Date')
        plt.ylabel('Actual Capital ($)')
        plt.show()

    def plot_equity_without_profit_graph(self):
        plt.figure(figsize=(12,6))
        plt.title('Equity Curve')
        plt.plot(self.equity , label = 'Stock Price' , color = 'green')
        plt.xlabel('Date')
        plt.ylabel('Actual Capital ($)')
        plt.show()

    def show_stats(self):
        print("Profit of the trading strategy: %.2f%%" % (
                      (float(self.equity[-1]) - float(self.equity[0])) /
                       float(self.equity[0]) * 100))
        
        print("Actual capital: $%0.2f" % self.equity[-1])
        returns = (self.data['Price'] - self.data['Price'].shift(1)) / self.data['Price'].shift(1)
        ratio  = returns.mean()  / returns.std() * np.sqrt(252) # Anuallized the sharpe ratio with 252 numbers of days 
        print('Sharpe ratio: %.2f' % ratio)

if __name__ == '__main__':
    
    start  = datetime.datetime(2015,1,1)
    end = datetime.datetime(2020,1,1)

    
    strategy = MovingAverageCrossover(100 ,'IBM' , start, end , 30 , 100 )
    strategy.download_data()
    strategy.construct_signals()
    strategy.plot_signals()
    strategy.simulate()
    strategy.plot_equity_without_profit_graph()
    strategy.show_stats()