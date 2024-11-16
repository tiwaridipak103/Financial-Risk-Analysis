set.seed(1)

# Generate white noise terms
w <- rnorm(1000, mean = 0 , sd = 1)

# Defines the x time series
# x(t = 0) = x(t = 1) = 0
x <- rep(0, 1000)

# random walk x(t) = x(t-1) + w(t)
for(t in 3:1000) x[t] = w[t] + 0.4*w[t-1] +  0.9*w[t-2]

plot(x , type = 'l')
acf(x)

ma <- arima(x, order = c(0, 0 , 1))
ma

residue = x[t] - 0.4*w[t-1] 
acf(residue )

# Example with stocks 

library(plyr)
library(quantmod)
stocks <- c("AAPL")
data.env <- new.env()

### here we use l_ply so that we don't double save the data
### getSymbols() does this already so we just want to be memory efficient
### go through every stock and try to use getSymbols()
l_ply(stocks, function(sym) try(getSymbols(sym,env=data.env),silent=T))

### now we only want the stocks that got stored from getSymbols()
### basically we drop all "bad" tickers
stocks <- stocks[stocks %in% ls(data.env)]

### now we just loop through and merge our good stocks
### if you prefer to use an lapply version here, that is also fine
### since now we are just collecting all the good stock xts() objects
data <- xts()
for(i in seq_along(stocks)) {
    symbol <- stocks[i]
    data <- merge(data, Ad(get(symbol,envir=data.env)))
  }

plot(data)

# Log daily returns
returns = diff(log(data))

plot(returns)

acf(returns, na.action = na.omit)

ma <- arima(returns , order = c(0, 0 , 20))
ma

acf(ma$res[-1])

