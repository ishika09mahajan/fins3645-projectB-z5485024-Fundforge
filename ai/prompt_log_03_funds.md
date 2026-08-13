# Prompt log 03 - the sentiment model and the fusion

## What I wanted
A finance-tuned VADER sentiment index by sector, and a look-ahead-safe fusion
into the equity funds.

## Prompt(s)
"Score the headlines with VADER, extend its lexicon with my Part A finance words,
build an equal-weight sector Fear & Greed index, standardise it without
look-ahead, lag it, then tilt the fund weights by it."

## What the assistant produced
sentiment.py (finance-tuned VADER and the sector index) and fusion.py (the
exp-tilt overlay).

## What was wrong or risky
The first standardisation used the full-sample mean and standard deviation, which
is look-ahead: a live signal cannot know the full-sample mean.

## What I changed and why
I switched to an expanding-window z-score so each day uses only its own history,
and confirmed the tilt only uses sentiment dated before each rebalance. I kept
the honest result: the overlay helps the max-Sharpe funds and cuts drawdown but
slightly hurts the risk-minimising funds, and I did not tune the strength to hide
that.