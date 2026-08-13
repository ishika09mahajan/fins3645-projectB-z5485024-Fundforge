# Prompt log 01 - the funds and the walk-forward backtest

## What I wanted
An out-of-sample portfolio engine for four methods (equal weight, minimum
variance, maximum Sharpe, risk parity) across equity, crypto and combined,
with no look-ahead.

## Prompt(s)
"Build a walk-forward out-of-sample backtest: 252-day trailing window, monthly
rebalance on the first trading day, long-only fully-invested weights, rf=0,
annualise equity/combined with 252 and crypto with 365, and charge a turnover
cost. Weights must come only from past data."

## What the assistant produced
A portfolios.py with the four optimisers and an oos_backtest that rebalances
monthly and returns net returns, weights, turnover and metrics.

## What was wrong or risky
The first max-Sharpe version was unconstrained and blew up out of sample
(around a -152% drawdown) because it took huge bets on noisy mean estimates.

## What I changed and why
I made the optimisers long-only with a budget constraint, re-ran the backtest,
and checked the weights summed to one and the drawdowns were sane before I
trusted any number.