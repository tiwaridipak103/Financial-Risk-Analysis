import numpy as np
import pandas as pd
from pandas_datareader import data as wb

import yfinance as yf
PG = yf.download('PG', start = '2012-01-01', end='2017-01-01')

print(PG.head())

# Simple rate of return 