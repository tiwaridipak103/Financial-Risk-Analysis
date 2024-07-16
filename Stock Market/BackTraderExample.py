from datetime import datetime
import backtrader as bt
from backtrader import plot
import yfinance as yf
# Create a subclass of Strategy to define the indicators and logic

class MyStrategy(bt.Strategy):
    def __init__(self):
        #get the data we have provided
        self.close_data = self.data.close
        print(self.data[0])
        
    
    def next(self):
        print("%s - %s" % (self.data.datetime.date(0) ,   self.close_data[0]))

class MovingAverageStrategy(bt.Strategy):

    params = (('period_fast' , 30) , ('period_slow' , 200) ,)

    def __init__(self):
        self.close_data = self.data.close
        print(self.data[5])

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



 


cerebro = bt.Cerebro()  # create a "Cerebro" engine instance
#cerebro.broker.setcash(100000)
# Create a data feed
data = bt.feeds.PandasData(dataname=yf.download('MSFT', '2010-01-01', '2021-01-01'))

cerebro.adddata(data)  # Add the data feed

cerebro.addstrategy(MovingAverageStrategy)  # Add the trading strategy

cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.SharpeRatio , riskfreerate = 0)
cerebro.addanalyzer(bt.analyzers.Returns)
cerebro.addanalyzer(bt.analyzers.DrawDown)

cerebro.broker.set_cash(3000)
print('Initial capital: %.2f' % cerebro.broker.getvalue())

# commission fees - set 0.1%
cerebro.broker.setcommission(0.01)

# run the strategy
results = cerebro.run()  # run it all

# evalute the results
print('Sharpe ratio : %.2f' %  results[0].analyzers.sharperatio.get_analysis()['sharperatio'])

print('Return : %.2f%%' %  results[0].analyzers.returns.get_analysis()['rnorm100'])

print('drawdown : %.2f%%' %  results[0].analyzers.drawdown.get_analysis()['drawdown'])

print('Capital: %.2f' % cerebro.broker.getvalue())



