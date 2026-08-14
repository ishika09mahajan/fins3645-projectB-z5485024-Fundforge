# AGENTS.md - Agent guide for the FundForge Part B project

## What this is
A multi-asset systematic fund platform (FINS3645 Part B). It builds four portfolio
types (equal weight, minimum variance, maximum Sharpe, risk parity) across three
universes (equity, crypto, combined), scores news headlines into a sector Fear &
Greed index, fuses that sentiment into the funds as a look-ahead-safe tilt, and
serves it all through a Streamlit dashboard that reads precomputed results.
Data source and full requirements: see PROJECT_BRIEF.md and context/.

## Live links
- Deployed app: https://ishika09mahajan-fins3645-projectb-z5485024-streamlit-app-ng0buy.streamlit.app
- Public repository: https://github.com/ishika09mahajan/fins3645-projectB-z5485024-Fundforge

## Layout
- src/etl.py         Station 1 - load and clean equities, crypto, news
- src/features.py    Station 2 - return panels and headline assembly
- src/portfolios.py  Station 3a - optimisers + walk-forward out-of-sample backtest
- src/sentiment.py   Station 3b - finance-tuned VADER -> sector sentiment index
- src/fusion.py      Station 3c - sentiment tilt overlay
- scripts/run_part_b.py    master script; writes every CSV under results/
- scripts/make_figures.py  figures from the saved CSVs
- streamlit_app.py   Station 4 - dashboard (reads results/ only)

## Reproduce
1. create .venv, install requirements.txt and requirements-dev.txt
2. python scripts/run_part_b.py     # results/data + results/tables
3. python scripts/make_figures.py   # results/figures
4. streamlit run streamlit_app.py

## Rules the assistant must follow
- No look-ahead: weights use only the trailing estimation window; the sentiment
  tilt uses only signal dated strictly before each rebalance, standardised on an
  expanding window.
- Annualise equity/combined with 252, crypto with 365. Risk-free rate = 0.
- The deployed app must never run VADER or a backtest at request time; it only
  reads the precomputed CSVs in results/.
- Do not commit raw data; the only committed data is the derived output in results/.
- Keep the four innovations intact: Ledoit-Wolf covariance shrinkage, a
  turnover/transaction-cost model, equal-risk-contribution risk parity (cyclical
  coordinate descent), and the finance-tuned sentiment overlay.

## How I check and correct the assistant's output
- I run every script myself and read the printed tables before trusting a result.
- I re-derive the key numbers. For example, I noticed the risk-parity fund was
  wrongly identical to equal weight on the 50-stock panel, and had the solver
  replaced with a proper equal-risk-contribution method before accepting it.
- I test for look-ahead by confirming both the weights and the sentiment tilt use
  only data dated before each rebalance.
- I keep prompt logs in ai/, and I only commit code I have read and understood.