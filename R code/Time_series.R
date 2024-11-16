set.seed(1)

# Generate white noise terms
w <- rnorm(2000, mean = 0 , sd = 1)

# Defines the x time series
x <- rep(0, 2000)

# random walk x(t) = x(t-1) + w(t)
for(t in 2:2000) x[t] = x[t-1] + w[t] 

plot(x , type = 'l')

# We can use the differentiating operator to check the correlation
acf(diff(x) , na.action = na.omit)
