# AI notes - how I used AI on Part B

I used an AI coding assistant throughout the build, and I directed and audited it
rather than pasting its output blindly. My rule was simple: I run every script
myself and read the printed tables before I trust a number.

Where AI helped: scaffolding the backtest engine, the Ledoit-Wolf shrinkage, the
Streamlit layout, and drafting boilerplate quickly.

Where AI was wrong, and what I did: (1) it first built an unconstrained
max-Sharpe that blew up out of sample, so I constrained it long-only; (2) its
risk-parity solver silently returned equal weight on the equity book, so I
diagnosed the scale-stall and replaced the solver with cyclical coordinate
descent; (3) its sentiment standardisation leaked look-ahead via the full-sample
mean, so I moved to an expanding window.

What is my own: all of the economic interpretation and written analysis in the
report is my own reasoning - the three recommendations, the reading of why each
fund behaves as it does, and the honest framing of the sentiment result. The
prompt logs in this folder record the specific prompts, the AI output, what was
wrong, and my fix with the reason.