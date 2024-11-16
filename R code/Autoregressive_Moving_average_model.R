
library(lmtest)

# We can construct an Arima(p,q) simulation with the AR and MA components
x = arima.sim(n =1000, model = list(ar = 0.4 , ma = -0.2))

plot(x)

# fit an ARIMA model with AR(1) and MA(1)
model = arima(x, order= c(1, 0 ,1))

model

coeftest(model)

#####################################################################

# We can construct an Arima(p,q) simulation with the AR and MA components
x = arima.sim(n =1000, model = list(ar = 0.5 , ma = -0.2))

plot(x)

# fit an ARIMA model with AR(1) and MA(1)
model = arima(x, order= c(1, 0 ,1))

model

coeftest(model)

###################################################################

# We can construct an Arima(p,q) simulation with the AR and MA components
x = arima.sim(n =1000, model = list(ar = c(0.4, -0.2, 0.6) , ma = c(0.6, -0.4)))

plot(x)

# Calcumate the optimal q and p value with AIC
solution.aic <- Inf
solution.order <- c(0,0,0)

for(i in 1:4) for(j in 1:4) {

   actual.aic <- AIC(arima(x, order = c(i, 0, j), optim.control = list(maxit = 1000)))
   
   # Lower the AIC better the model
   if (actual.aic < solution.aic) {
   solution.aic <- actual.aic
   solution.order <- c(i, 0, j)
   solution.arma <- arima(x, order = solution.order , optim.control = list(maxit = 1000))

}

}

solution.aic 
solution.order 
solution.arma

# No of serial corelation in the residual y(t) series
acf(resid(solution.arma))


# Let's apply Ljung-Box test
# if the p > 005 , it means the residue are independent
# at the 95 % level so the ARMA is good fit

Box.test(resid(solution.arma) , lag = 20 , type = "Ljung-Box")

#######################################################################

# Applying in Real data 

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

# Calcumate the optimal q and p value with AIC
solution.aic <- Inf
solution.order <- c(0,0,0)

for(i in 1:4) for(j in 1:4) {

   actual.aic <- AIC(arima(returns , order = c(i, 0, j), optim.control = list(maxit = 1000)))
   
   # Lower the AIC better the model
   if (actual.aic < solution.aic) {
   solution.aic <- actual.aic
   solution.order <- c(i, 0, j)
   solution.arma <- arima(returns , order = solution.order , optim.control = list(maxit = 1000))

}

}

solution.aic 
solution.order 
solution.arma

# No of serial corelation in the residual y(t) series
acf(resid(solution.arma), na.action = na.omit)

# Let's apply Ljung-Box test
# if the p > 005 , it means the residue are independent
# at the 95 % level so the ARMA is good fit

Box.test(resid(solution.arma) , lag = 20 , type = "Ljung-Box")

