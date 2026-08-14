# Prompt log 02 - the risk-parity solver bug I caught

## What I wanted
A true equal-risk-contribution risk-parity fund, distinct from equal weight.

## Prompt(s)
"Implement risk parity as equal risk contribution using the covariance matrix."

## What the assistant produced
An SLSQP solver minimising the squared differences of risk contributions.

## What was wrong or risky
On the 50-stock equity book it returned weights IDENTICAL to equal weight to
four decimals. The brief warns that optimisers on tiny daily-return covariances
can silently stall, and that is what happened: daily variances are about 1e-4 so
the objective sat near 1e-12 and the solver stopped at its equal-weight start.

## What I changed and why
I replaced the solver with the cyclical coordinate descent algorithm (Griveau
Billion, Richard and Roncalli, 2013), which is scale-invariant. I verified the
fix by checking the risk contributions were equal and that the equity risk-parity
volatility (14.65%) now sat between equal weight and minimum variance.