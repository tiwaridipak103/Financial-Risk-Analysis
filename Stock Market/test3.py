import backtrader as bt
import pandas as pd

class SimpleStrategy(bt.Strategy):
    def next(self):
        # Get the position for the current data feed
        position = self.getposition(self.data)
        print(f'the given position is : {position}')

        if self.data.close[0] > self.data.open[0]:
            # Buy on the first day if the closing price is higher than the opening price
            if not position.size:  # Check if there is no existing position
                print('hi')
                self.buy()

        elif self.data.close[0] < self.data.open[0]:
            # Sell on the second day if the closing price is lower than the opening price
            if position.size:  # Check if there is an existing position
                print('bye')
                self.sell()

# Sample data
data = pd.DataFrame({
    'date': ['2022-01-01', '2022-01-02', '2022-01-03','2022-01-04','2022-01-05' ],
    'open': [100, 110, 113, 109, 112],
    'high': [105, 115, 117, 115, 118],
    'low': [95, 105, 110, 109, 115],
    'close': [105, 100, 95, 85, 109],
    'volume': [100000, 120000, 130000, 109000, 140000],
}, index=pd.to_datetime(['2022-01-01', '2022-01-02','2022-01-03','2022-01-04','2022-01-05' ]))

# Create a Backtrader data feed
data_feed = bt.feeds.PandasData(dataname=data)

# Create a Backtrader Cerebro engine
cerebro = bt.Cerebro()
cerebro.adddata(data_feed)

# Add the strategy to the engine
cerebro.addstrategy(SimpleStrategy)

# Set the initial cash amount for the backtest
cerebro.broker.set_cash(10000)

# Print the starting cash amount
print(f'Starting Portfolio Value: {cerebro.broker.getvalue()}')

# Run the backtest
cerebro.run()

# Print the final cash amount
print(f'Ending Portfolio Value: {cerebro.broker.getvalue()}')