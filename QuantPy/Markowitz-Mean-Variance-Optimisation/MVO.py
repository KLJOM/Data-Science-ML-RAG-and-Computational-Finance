"""
Markowitz Mean-Variance Optimisation (MVO) + Treynor-Black active portfolio.

Section 1 : Markowitz MVO - maximum Sharpe ratio, long-only, fully invested.
Section 2 : Treynor-Black - market-model alphas sized against residual variance,
            reported in both the classic TB form and the CFA Fundamental Law form
            dw_i = (alpha_i / sigma_e,i^2) * (sigma_A / IR).

Requires: pip install yfinance PyPortfolioOpt pandas numpy
"""

import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns

# ============================================================================
# CONFIGURATION  - edit these, not the code below
# ============================================================================

TICKERS = ["META", "AMD", "CAT", "MSTR", "MA", "V", "MC.PA",
           "ASML", "GOOGL", "CMCSA", "UNH", "PYPL"]

BENCHMARK = "^GSPC"            # market proxy. ACWI if you want a global benchmark.
START = "2021-01-01"
END = "2026-09-17"

RISK_FREE_RATE = 0.042         # annual. Set to the current short rate.
                            # COME BACK AND ACTUALLY ESTIMATE THE RISK FREE RATE BASED ON FRED DATABASE SINCE THIS IS US EQUITIES
                            # ECONOMETRICALLY 
                            # ALSO CREATE A PROGRAMME FOR SA EQUITIES - STATS SA, SARB

# Weekly returns, not daily: MC.PA closes hours before the US market, and daily
# data would bias its beta downward through non-synchronous trading.
RESAMPLE_RULE = "W-FRI"
PERIODS_PER_YEAR = 52

# Tickers quoted in a currency other than USD, mapped to their FX pair.
NON_USD_TICKERS = {"MC.PA": "EURUSD=X"}

USE_SHRINKAGE = False          # True -> Ledoit-Wolf covariance in the MVO section

# --- Treynor-Black inputs ---------------------------------------------------
# Alphas are FORECASTS. Regression alphas are used only as a placeholder.
# Supply your own to make this meaningful, e.g.
#   ALPHA_VIEWS = {"META": 0.03, "UNH": -0.02}   # annual, decimal
ALPHA_VIEWS = None

# Shrinkage applied to alpha forecasts. Theory says multiply raw forecasts by the
# R^2 of realised-on-forecast alpha, which is roughly IC^2. 1.0 = no shrinkage.
ALPHA_SHRINKAGE = 1.0

TB_LONG_ONLY = False           # True -> drop negative-alpha names before sizing

# --- CFA Fundamental Law inputs ---------------------------------------------
TARGET_ACTIVE_RISK = 0.05      # sigma_A, annual tracking error target
IC = 0.05                      # information coefficient (forecasting skill)
REBALANCES_PER_YEAR = 12
BREADTH = REBALANCES_PER_YEAR * len(TICKERS)   # BR, independent decisions/year

RULE = "=" * 78
THIN = "-" * 78


# ============================================================================
# DATA
# ============================================================================

def download_close(tickers, start, end):
    """Adjusted close prices as a DataFrame, one column per ticker."""
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return close


def build_price_panel():
    """Prices for all assets + benchmark, FX-converted to USD, common dates only."""
    prices = download_close(TICKERS, START, END)

    fx_pairs = sorted(set(NON_USD_TICKERS.values()))
    if fx_pairs:
        fx = download_close(fx_pairs, START, END)
        for ticker, pair in NON_USD_TICKERS.items():
            if ticker in prices.columns and pair in fx.columns:
                prices[ticker] = prices[ticker] * fx[pair].reindex(prices.index).ffill()

    bench = download_close([BENCHMARK], START, END).iloc[:, 0].rename(BENCHMARK)

    panel = prices.join(bench, how="inner").dropna(how="any")
    if panel.empty:
        raise RuntimeError("No overlapping price history. Check tickers and dates.")
    return panel[TICKERS], panel[BENCHMARK]


# ============================================================================
# SECTION 1 - MARKOWITZ MVO
# ============================================================================

def run_mvo(prices):
    # compounding=False gives arithmetic means. E(R_p) = w'mu only holds for
    # arithmetic returns; geometric means do not aggregate linearly.
    mu = expected_returns.mean_historical_return(prices, compounding=False)

    if USE_SHRINKAGE:
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    else:
        S = risk_models.sample_cov(prices)

    ef = EfficientFrontier(mu, S)
    ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
    weights = ef.clean_weights()

    print(RULE)
    print(" SECTION 1  |  MARKOWITZ MEAN-VARIANCE OPTIMISATION (MVO)")
    print(RULE)
    print(f" Objective      : maximum Sharpe ratio (tangency portfolio)")
    print(f" Constraints    : long-only, fully invested (sum w = 1)")
    print(f" Covariance     : {'Ledoit-Wolf shrinkage' if USE_SHRINKAGE else 'sample'}")
    print(f" Risk-free rate : {RISK_FREE_RATE:.2%}")
    print(THIN)
    print(f" {'Ticker':<10}{'Weight':>12}")
    print(THIN)
    for ticker, w in weights.items():
        print(f" {ticker:<10}{w * 100:>11.2f}%")
    print(THIN)
    print(f" {'TOTAL':<10}{sum(weights.values()) * 100:>11.2f}%")
    print(THIN)

    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
    print(f" Expected return {ret:>10.2%}")
    print(f" Volatility      {vol:>10.2%}")
    print(f" Sharpe ratio    {sharpe:>10.2f}")
    print()
    return weights


# ============================================================================
# SECTION 2 - TREYNOR-BLACK
# ============================================================================

def market_model(asset_excess, mkt_excess, ppy):
    """OLS of each asset's excess return on the market's. Returns annualised
    alpha, beta, residual variance, plus the t-stat on alpha and R^2."""
    x = mkt_excess.to_numpy(dtype=float)
    T = len(x)
    xbar = x.mean()
    sxx = ((x - xbar) ** 2).sum()

    rows = {}
    for col in asset_excess.columns:
        y = asset_excess[col].to_numpy(dtype=float)
        beta = ((x - xbar) * (y - y.mean())).sum() / sxx
        alpha = y.mean() - beta * xbar
        resid = y - alpha - beta * x
        rss = (resid ** 2).sum()
        s2e = rss / (T - 2)
        se_alpha = np.sqrt(s2e * (1.0 / T + xbar ** 2 / sxx))
        tss = ((y - y.mean()) ** 2).sum()
        rows[col] = {
            "alpha_ann": alpha * ppy,
            "beta": beta,
            "resid_var_ann": s2e * ppy,
            "resid_sd_ann": np.sqrt(s2e * ppy),
            "t_alpha": alpha / se_alpha,
            "r2": 1.0 - rss / tss,
        }
    return pd.DataFrame(rows).T


def treynor_black(alpha, resid_var, beta, mkt_prem, mkt_var):
    """Classic TB: build the active portfolio, then mix it with the market."""
    ratio = alpha / resid_var
    D = float(ratio.sum())
    if abs(D) < 1e-12:
        raise RuntimeError("sum(alpha/resid_var) is ~0; active portfolio undefined.")
    if D < 0:
        warnings.warn(
            "sum(alpha/resid_var) < 0: the active portfolio has negative alpha and "
            "the TB normalisation flips sign. Read Section 2c instead."
        )

    w_act = ratio / D
    alpha_A = float((w_act * alpha).sum())
    resid_var_A = float(((w_act ** 2) * resid_var).sum())
    beta_A = float((w_act * beta).sum())

    w0 = (alpha_A / resid_var_A) / (mkt_prem / mkt_var)
    denom = 1.0 + (1.0 - beta_A) * w0
    if abs(denom) < 1e-8:
        raise RuntimeError("Beta adjustment is singular; w* cannot be computed.")
    w_star = w0 / denom

    # w* = w0 / (1 + (1 - beta_A) w0) has a pole at w0 = 1/(beta_A - 1) when
    # beta_A > 1. Past it the expression still returns a root of the first-order
    # condition, but it is the MINIMUM-Sharpe branch, not the maximum.
    pole = 1.0 / (beta_A - 1.0) if beta_A > 1.0 else np.inf
    beyond_pole = bool(beta_A > 1.0 and w0 > pole)

    return {
        "w_active": w_act,
        "alpha_A": alpha_A,
        "resid_var_A": resid_var_A,
        "resid_sd_A": np.sqrt(resid_var_A),
        "beta_A": beta_A,
        "IR_A": alpha_A / np.sqrt(resid_var_A),
        "w0": w0,
        "w_star": w_star,
        "w_market": 1.0 - w_star,
        "final_active": w_star * w_act,
        "pole": pole,
        "beyond_pole": beyond_pole,
    }


def optimal_information_ratio(alpha, resid_var):
    """IR* = sqrt(sum alpha_i^2 / sigma_e,i^2). The best IR these alphas support."""
    return float(np.sqrt(((alpha ** 2) / resid_var).sum()))


def cfa_active_weights(alpha, resid_var, sigma_A, IR):
    """dw_i = (alpha_i / sigma_e,i^2) * (sigma_A / IR)."""
    return (alpha / resid_var) * (sigma_A / IR)


def run_treynor_black(prices, bench):
    rf_period = RISK_FREE_RATE / PERIODS_PER_YEAR

    px = prices.resample(RESAMPLE_RULE).last()
    bx = bench.resample(RESAMPLE_RULE).last()
    asset_ex = px.pct_change().dropna(how="any") - rf_period
    mkt_ex = bx.pct_change().dropna() - rf_period
    idx = asset_ex.index.intersection(mkt_ex.index)
    asset_ex, mkt_ex = asset_ex.loc[idx], mkt_ex.loc[idx]

    est = market_model(asset_ex, mkt_ex, PERIODS_PER_YEAR)
    mkt_prem = float(mkt_ex.mean() * PERIODS_PER_YEAR)
    mkt_var = float(mkt_ex.var(ddof=1) * PERIODS_PER_YEAR)

    print(RULE)
    print(" SECTION 2  |  TREYNOR-BLACK MODEL")
    print(RULE)
    print(f" Benchmark      : {BENCHMARK}")
    print(f" Frequency      : {RESAMPLE_RULE}   observations: {len(idx)}")
    print(f" Market premium : {mkt_prem:.2%}    Market vol: {np.sqrt(mkt_var):.2%}")
    print(f" Market Sharpe  : {mkt_prem / np.sqrt(mkt_var):.3f}")
    print()

    # --- 2a. market model estimates -----------------------------------------
    print(" 2a. MARKET MODEL ESTIMATES   R_i - rf = alpha + beta(R_M - rf) + e")
    print(THIN)
    print(f" {'Ticker':<9}{'alpha':>9}{'t(alpha)':>10}{'beta':>8}"
          f"{'sd(e)':>9}{'R^2':>8}{'a/var(e)':>11}")
    print(THIN)
    for t in est.index:
        r = est.loc[t]
        print(f" {t:<9}{r['alpha_ann']:>8.2%}{r['t_alpha']:>10.2f}{r['beta']:>8.2f}"
              f"{r['resid_sd_ann']:>8.2%}{r['r2']:>8.2f}"
              f"{r['alpha_ann'] / r['resid_var_ann']:>11.3f}")
    print(THIN)
    sig = (est["t_alpha"].abs() > 1.96).sum()
    print(f" Alphas significant at 5% ({'|t| > 1.96'}): {sig} of {len(est)}")
    print()

    # --- alpha forecasts -----------------------------------------------------
    alpha = est["alpha_ann"].copy()
    if ALPHA_VIEWS:
        source = "user-supplied views"
        alpha = pd.Series({t: ALPHA_VIEWS.get(t, 0.0) for t in alpha.index})
    else:
        source = "historical regression alphas (placeholder)"
    alpha = alpha * ALPHA_SHRINKAGE

    resid_var = est["resid_var_ann"]
    beta = est["beta"]

    if TB_LONG_ONLY:
        keep = alpha > 0
        alpha, resid_var, beta = alpha[keep], resid_var[keep], beta[keep]

    print(f" Alpha source   : {source}")
    print(f" Alpha shrinkage: {ALPHA_SHRINKAGE:.2f}"
          f"{'   (long-only: negative alphas dropped)' if TB_LONG_ONLY else ''}")
    print()

    # --- 2b. classic TB ------------------------------------------------------
    tb = treynor_black(alpha, resid_var, beta, mkt_prem, mkt_var)

    print(" 2b. ACTIVE PORTFOLIO AND MARKET MIX")
    print(THIN)
    print(f" {'Ticker':<9}{'w in A':>12}{'final w':>12}")
    print(THIN)
    for t in tb["w_active"].index:
        print(f" {t:<9}{tb['w_active'][t] * 100:>11.2f}%"
              f"{tb['final_active'][t] * 100:>11.2f}%")
    print(THIN)
    print(f" {'MARKET':<9}{'':>12}{tb['w_market'] * 100:>11.2f}%")
    print(THIN)
    print(f" alpha_A (active portfolio alpha) : {tb['alpha_A']:>9.2%}")
    print(f" sd(e_A) (active residual risk)   : {tb['resid_sd_A']:>9.2%}")
    print(f" beta_A                           : {tb['beta_A']:>9.3f}")
    print(f" IR_A = alpha_A / sd(e_A)         : {tb['IR_A']:>9.3f}")
    print(f" w0  (unadjusted active position) : {tb['w0']:>9.2%}")
    print(f" w*  (beta-adjusted)              : {tb['w_star']:>9.2%}")

    s_m = mkt_prem / np.sqrt(mkt_var)
    s_p = np.sqrt(s_m ** 2 + tb["IR_A"] ** 2)

    if tb["beyond_pole"]:
        safe = 0.5 * tb["pole"] / tb["w0"]
        print()
        print(" *** WARNING: beta adjustment has passed its pole. ***")
        print(f" w* = w0 / (1 + (1 - beta_A) w0) is singular at w0 = 1/(beta_A - 1)")
        print(f" = {tb['pole']:.2f}, and w0 = {tb['w0']:.2f} is beyond it. The value")
        print(" above is still a root of the first-order condition, but it is the")
        print(" MINIMUM-Sharpe branch. Do not trade it.")
        print(f" Cause: the alphas are implausibly large (IR_A = {tb['IR_A']:.2f}).")
        print(f" w0 scales linearly with alpha, so set ALPHA_SHRINKAGE below"
              f" {safe:.3f}")
        print(" or supply smaller ALPHA_VIEWS to bring w0 back under the pole.")
    else:
        print(f" Sharpe: market {s_m:.3f}  ->  TB portfolio {s_p:.3f}"
              f"   (S_P^2 = S_M^2 + IR_A^2)")
    print()

    # --- 2c. CFA Fundamental Law sizing --------------------------------------
    ir_star = optimal_information_ratio(alpha, resid_var)
    ir_law = IC * np.sqrt(BREADTH)

    print(" 2c. OPTIMAL ACTIVE WEIGHTS - CFA FUNDAMENTAL LAW FORM")
    print("     dw_i = (alpha_i / sigma_e,i^2) * (sigma_A / IR)")
    print(THIN)
    print(f" Target active risk sigma_A       : {TARGET_ACTIVE_RISK:>9.2%}")
    print(f" IR* implied by these alphas      : {ir_star:>9.3f}"
          f"   = sqrt(sum alpha_i^2 / sigma_e,i^2)")
    print(f" IR from Fundamental Law          : {ir_law:>9.3f}"
          f"   = IC {IC:.3f} x sqrt(BR {BREADTH})")
    print(THIN)

    dw_star = cfa_active_weights(alpha, resid_var, TARGET_ACTIVE_RISK, ir_star)
    dw_law = cfa_active_weights(alpha, resid_var, TARGET_ACTIVE_RISK, ir_law)

    print(f" {'Ticker':<9}{'dw using IR*':>16}{'dw using IC*sqrt(BR)':>24}")
    print(THIN)
    for t in dw_star.index:
        print(f" {t:<9}{dw_star[t] * 100:>15.2f}%{dw_law[t] * 100:>23.2f}%")
    print(THIN)
    print(f" {'SUM':<9}{dw_star.sum() * 100:>15.2f}%{dw_law.sum() * 100:>23.2f}%")
    print(THIN)

    risk_star = float(np.sqrt(((dw_star ** 2) * resid_var).sum()))
    risk_law = float(np.sqrt(((dw_law ** 2) * resid_var).sum()))
    print(f" Realised active risk, IR*        : {risk_star:>9.2%}  (hits target)")
    print(f" Realised active risk, IC*sqrt(BR): {risk_law:>9.2%}")
    print(f" E(R_A) = IR* x sigma_A           : {ir_star * TARGET_ACTIVE_RISK:>9.2%}")
    print()

    if not np.isclose(ir_law, ir_star, rtol=0.15):
        direction = "overstates" if ir_law > ir_star else "understates"
        print(f" NOTE: IC x sqrt(BR) = {ir_law:.3f} {direction} the skill actually")
        print(f"       present in these alphas (IR* = {ir_star:.3f}). The two inputs")
        print(f"       are inconsistent, so the IC x sqrt(BR) column misses the")
        print(f"       {TARGET_ACTIVE_RISK:.1%} tracking-error target by a factor of"
              f" {ir_star / ir_law:.2f}.")
        print(f"       Reconcile IC/BR with your alphas before trading this column.")
        print()

    if not tb["beyond_pole"]:
        print(" Note: the 'final w' column in 2b and the 'dw using IR*' column here are")
        print(" the same portfolio, scaled to different active-risk levels. TB fixes")
        print(" the scale via the market Sharpe ratio; the CFA form fixes it via")
        print(" sigma_A. Section 2c is immune to the pole above, so when 2b warns,")
        print(" use 2c.")
    print()
    return tb


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    prices, bench = build_price_panel()
    print()
    print(f" Price history: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}"
          f"   ({len(prices)} trading days, {len(prices.columns)} assets)")
    print()
    run_mvo(prices)
    run_treynor_black(prices, bench)
    print(RULE)
    print(" Alphas estimated from past returns are forecasts of nothing. Replace")
    print(" ALPHA_VIEWS with your own before treating Section 2 as an allocation.")
    print(RULE)