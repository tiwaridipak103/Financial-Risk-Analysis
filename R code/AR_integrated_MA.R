# We can construct an Arima(p,d,q) simulation with the AR , I and MA components
x = arima.sim(n =2000, model = list(c(1, 1 ,1) , ar = 0.4 , ma = 0.8))
plot(x)

# We would like to approximate the process ARIMA
model = arima(x, order= c(1, 1 ,1))
model

acf(resid(model))

# Let's apply Ljung-Box test
# if the p > 005 , it means the residue are independent
# at the 95 % level so the ARMA is good fit

Box.test(resid(model) , lag = 20 , type = "Ljung-Box")


#######################################################################

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

# Let's apply Ljung-Box test
# if the p > 005 , it means the residue are independent
# at the 95 % level so the ARMA is good fit

Box.test(resid(solution.arima) , lag = 25 , type = "Ljung-Box")

#lets forcast the log daily returns in the coming 50 days!!!
plot(forecast(solution.arima, h = 50))
