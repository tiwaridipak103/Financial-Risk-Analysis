# Package needed for GARCH model
library(tseries)

# These are the coefficient for GARCH(1,1) model
alpha0 <- 0.1
alpha1 <- 0.4
beta1  <- 0.2

# White noise term values
w <- rnorm(2000)

# The actual x(t) time series 
x <- rep(0, 2000)

# Volatility squared values
sigma2 <- rep(0, 2000)

# GARCH(1, 1) model simulation
for(t in 2:2000){
  sigma2[t] <-  alpha0 + alpha1 * x[t-1]^2 + beta1 * sigma2[t-1]
  x[t] <- w[t] * sqrt(sigma2[t])
}

plot(x, type = 'l')

acf(x)  # this is when we are only ploting for residue 

acf(x*x) # this is when we are  ploting for square of residue , it is found to correlated

# Use the GARCH function
x.garch <- garch(x, trace = FALSE)
x.garch


# Applying in Real data 

library(plyr)
library(quantmod)
library(forecast)
stocks <- c("IBM")
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

# Calcumate the optimal q and p value with AIC
solution.aic <- Inf
solution.order <- c(0,0,0)

for(p in 1:4) for(d in 1:4) for(q in 1:4) {

   actual.aic <- AIC(arima(returns , order = c(p, d, q), optim.control = list(maxit = 1000)))
   
   # Lower the AIC better the model
   if (actual.aic < solution.aic) {
   solution.aic <- actual.aic
   solution.order <- c(p, d, q)
   solution.arima <- arima(returns , order = solution.order , optim.control = list(maxit = 1000))

}

}

solution.aic 
solution.order 
solution.arima

# No of serial corelation in the residual y(t) series
acf(resid(solution.arima), na.action = na.omit)

acf(resid(solution.arima)^2, na.action = na.omit) # it is serious correlation


# Use the GARCH function
result.garch <- garch(x, trace = FALSE)
result.residual = result.garch$res[-1]

# No of serial corelation in the residual y(t) series
acf(result.residual, na.action = na.omit)

acf(result.residual^2, na.action = na.omit) # it has no autocorrelation with square of residue

