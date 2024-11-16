set.seed(1)

# Generate white noise terms
w <- rnorm(1000, mean = 0 , sd = 1)

# Defines the x time series
# x(t = 0) = x(t = 1) = 0
x <- rep(0, 1000)

# random walk x(t) = x(t-1) + w(t)
for(t in 3:1000) x[t] = 0.6 * x[t-1]- 0.4 * x[t-2] + w[t]  # we can w as w[t] or w[t-1] , it does not matter since w is normally distributed.

plot(x , type = 'l')
acf(x)

# We can use the differentiating operator to check the correlation
acf(diff(x) , na.action = na.omit)

x.ar = ar(x, method = 'mle')

x.ar$order

x.ar$ar