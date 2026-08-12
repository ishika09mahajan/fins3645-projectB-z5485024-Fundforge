"""Station 3 - funds: optimal portfolios + walk-forward out-of-sample backtest.

Design (documented in the report):
  * 252-day rolling estimation window; rebalance the first trading day of each month.
  * Weights use returns strictly BEFORE the rebalance date (no look-ahead), held to
    the next rebalance.
  * Equity/combined annualise with 252; crypto-only with 365.
  * Long-only, fully invested (weights >= 0, sum to 1). rf = 0, stated.

Four methods: equal_weight, min_variance, max_sharpe (tangency), risk_parity.
Innovations: Ledoit-Wolf covariance shrinkage; a turnover/transaction-cost model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_EQUITY = 252
TRADING_DAYS_CRYPTO = 365
METHODS = ("equal_weight", "min_variance", "max_sharpe", "risk_parity")


# ---------------- covariance ----------------
def _sample_cov(window: pd.DataFrame) -> np.ndarray:
    return window.cov().to_numpy()


def _ledoit_wolf_cov(window: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a constant-correlation target (self-contained)."""
    X = window.to_numpy()
    t, n = X.shape
    X = X - X.mean(axis=0)
    sample = (X.T @ X) / t
    var = np.diag(sample).reshape(-1, 1)
    std = np.sqrt(var)
    denom = std @ std.T
    denom[denom == 0] = 1e-12
    corr = sample / denom
    r_bar = (corr.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    target = r_bar * denom
    np.fill_diagonal(target, var.ravel())
    y = X ** 2
    phi = ((y.T @ y) / t - sample ** 2).sum()
    gamma = np.linalg.norm(sample - target, "fro") ** 2
    kappa = phi / gamma if gamma > 0 else 0.0
    delta = float(np.clip(kappa / t, 0.0, 1.0))
    return delta * target + (1 - delta) * sample


def _cov(window: pd.DataFrame, method: str) -> np.ndarray:
    S = _ledoit_wolf_cov(window) if method == "lw" else _sample_cov(window)
    return S + np.eye(S.shape[0]) * 1e-8


# ---------------- optimisers ----------------
def _w_equal(n: int) -> np.ndarray:
    return np.repeat(1.0 / n, n)


def _w_min_variance(S: np.ndarray, long_only: bool) -> np.ndarray:
    n = S.shape[0]
    if not long_only:
        w = np.linalg.pinv(S) @ np.ones(n)
        return w / w.sum()
    w0 = _w_equal(n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(lambda w: w @ S @ w, w0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x if res.success else w0


def _w_max_sharpe(mu: np.ndarray, S: np.ndarray, rf: float, long_only: bool) -> np.ndarray:
    n = S.shape[0]
    excess = mu - rf
    if not long_only:
        w = np.linalg.pinv(S) @ excess
        s = w.sum()
        return w / s if abs(s) > 1e-12 else _w_equal(n)

    def neg_sharpe(w):
        vol = np.sqrt(w @ S @ w)
        return -(w @ excess) / vol if vol > 1e-12 else 0.0

    w0 = _w_equal(n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n
    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x if res.success else w0


def _w_risk_parity(S: np.ndarray, tol: float = 1e-12, max_iter: int = 20000) -> np.ndarray:
    n = S.shape[0]
    b = np.ones(n) / n
    vol = np.sqrt(np.diag(S))
    y = 1.0 / np.where(vol > 0, vol, 1.0)
    for _ in range(max_iter):
        y_prev = y.copy()
        for i in range(n):
            c = S[i] @ y - S[i, i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4.0 * S[i, i] * b[i])) / (2.0 * S[i, i])
        if np.linalg.norm(y - y_prev, 1) < tol:
            break
    return y / y.sum()

def compute_weights(window: pd.DataFrame, method: str, rf_daily: float = 0.0,
                    long_only: bool = True, cov_method: str = "sample") -> pd.Series:
    cols = window.columns
    n = len(cols)
    if method == "equal_weight":
        w = _w_equal(n)
    else:
        S = _cov(window, cov_method)
        if method == "min_variance":
            w = _w_min_variance(S, long_only)
        elif method == "max_sharpe":
            w = _w_max_sharpe(window.mean().to_numpy(), S, rf_daily, long_only)
        elif method == "risk_parity":
            w = _w_risk_parity(S)
        else:
            raise ValueError(f"unknown method {method!r}")
    return pd.Series(w, index=cols)


# ---------------- walk-forward backtest ----------------
def _rebalance_dates(index, start_i):
    idx = index[start_i:]
    s = pd.Series(idx, index=idx)
    return list(s.groupby([idx.year, idx.month]).first().values)


def oos_backtest(returns: pd.DataFrame, method: str = "min_variance",
                 window: int = 252, rf_daily: float = 0.0, long_only: bool = True,
                 cov_method: str = "sample", cost_bps: float = 10.0,
                 periods_per_year: int = TRADING_DAYS_EQUITY,
                 tilt=None, tilt_strength: float = 0.0):
    """Walk-forward OOS backtest. tilt = optional lagged signal (date x asset)."""
    returns = returns.dropna(how="all").copy()
    dates = returns.index
    if len(dates) <= window + 1:
        raise ValueError("not enough history for the estimation window")
    reb_dates = _rebalance_dates(dates, window)
    date_pos = {d: i for i, d in enumerate(dates)}

    weights_history, turnover_history, prev_w = {}, {}, None
    for reb in reb_dates:
        i = date_pos[reb]
        est = returns.iloc[i - window:i].dropna(axis=1, how="any")
        if est.shape[1] < 2:
            continue
        w = compute_weights(est, method, rf_daily, long_only, cov_method)
        if tilt is not None and tilt_strength != 0.0:
            past = tilt.loc[tilt.index < reb]
            if len(past):
                sig = past.iloc[-1].reindex(w.index).fillna(0.0)
                w = w * np.exp(tilt_strength * sig)
                if w.sum() > 0:
                    w = w / w.sum()
        weights_history[reb] = w
        if prev_w is not None:
            turnover_history[reb] = float(np.abs(w - prev_w.reindex(w.index).fillna(0.0)).sum())
        else:
            turnover_history[reb] = 1.0
        prev_w = w

    if not weights_history:
        raise ValueError("no rebalances produced weights - check the window/data")
    weights_df = pd.DataFrame(weights_history).T.sort_index().fillna(0.0)

    gross, net, ret_dates = [], [], []
    cost_rate = cost_bps / 10_000.0
    w_current, reb_set = None, set(weights_df.index)
    first_reb = weights_df.index.min()
    for d in dates:
        if d < first_reb:
            continue
        if d in reb_set:
            w_current = weights_df.loc[d]
            cost = cost_rate * turnover_history[d]
        else:
            cost = 0.0
        row = returns.loc[d].reindex(w_current.index).fillna(0.0)
        g = float((w_current * row).sum())
        gross.append(g); net.append(g - cost); ret_dates.append(d)

    net_s = pd.Series(net, index=ret_dates, name=method)
    turn_s = pd.Series(turnover_history).sort_index()
    metrics = performance_metrics(net_s, periods_per_year)
    metrics["avg_turnover"] = round(float(turn_s.mean()), 4)
    metrics["n_rebalances"] = int(len(weights_df))
    metrics["first_oos_date"] = str(net_s.index.min().date())
    return {"method": method, "daily_net": net_s,
            "daily_gross": pd.Series(gross, index=ret_dates, name="gross"),
            "weights": weights_df, "growth": (1 + net_s).cumprod(),
            "turnover": turn_s, "metrics": metrics}


def performance_metrics(daily_returns: pd.Series,
                        periods_per_year: int = TRADING_DAYS_EQUITY) -> dict:
    """Annualised return, annualised volatility, Sharpe, and max drawdown."""
    r = daily_returns.dropna()
    if len(r) == 0:
        return {k: float("nan") for k in ("ann_return", "ann_vol", "sharpe", "max_drawdown")}
    mean_d, std_d = r.mean(), r.std()
    ann_ret = (1 + mean_d) ** periods_per_year - 1
    ann_vol = std_d * np.sqrt(periods_per_year)
    sharpe = (mean_d / std_d) * np.sqrt(periods_per_year) if std_d > 0 else float("nan")
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return {"ann_return": round(float(ann_ret), 4), "ann_vol": round(float(ann_vol), 4),
            "sharpe": round(float(sharpe), 4), "max_drawdown": round(float(dd.min()), 4)}