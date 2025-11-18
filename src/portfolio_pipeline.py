import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import yfinance as yf


def analyze_tickers(tickers_list, start, end):
    """
    Download price data and compute returns/visualizations.
    Returns monthly returns DataFrame.
    """
    prices = {}

    for t in tickers_list:
        df = yf.download(
            t, start=start, end=end,
            interval="1d", auto_adjust=False, progress=False
        )
        if not df.empty:
            prices[t] = df

    first = tickers_list[0]
    prep_data = pd.DataFrame(
        prices[first]['Adj Close']).rename(columns={'Adj Close': first})

    for t in tickers_list[1:]:
        prep_data[t] = prices[t]['Adj Close']

    return_data = prep_data.pct_change().dropna()

    cumulative_returns = (1 + return_data).cumprod() - 1
    cumulative_returns.plot(figsize=(12, 6))
    plt.title("Cumulative Returns")
    plt.grid(True)
    plt.show()

    monthly_returns = prep_data.resample("ME").ffill().pct_change().dropna()

    plt.figure(figsize=(10, 8))
    sns.heatmap(monthly_returns.cov(), annot=True, cmap="coolwarm")
    plt.title("Covariance Matrix (Monthly Returns)")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(monthly_returns.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix (Monthly Returns)")
    plt.show()

    return monthly_returns


def run_portfolio_womack(monthly_returns, return_floor=0.015):

    df = monthly_returns.copy()
    df_return = df.mean()
    df_cov = df.cov()

    def port_return(w):
        return np.dot(w, df_return.values)

    def port_risk(w):
        return np.dot(w, df_cov.values @ w)

    def optimize_for_risk_limit(r):
        n = len(df.columns)
        x0 = np.ones(n) / n
        bounds = [(0, 1)] * n

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: r - port_risk(w)},
            {"type": "ineq", "fun": lambda w: port_return(w) - return_floor},
        ]

        res = minimize(
            fun=lambda w: -port_return(w),
            x0=x0,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )
        w = res.x
        return w, port_return(w), port_risk(w)

    risk_levels = np.arange(0.001, 0.01, 0.0005)

    allocations = []
    rewards = []
    risks = []

    for r in risk_levels:
        w, ret, risk_val = optimize_for_risk_limit(r)
        allocations.append(w)
        rewards.append(ret)
        risks.append(risk_val)

    alloc_df = pd.DataFrame(allocations, index=risk_levels, columns=df.columns)

    reward_df = pd.DataFrame(
        {
            "risk_limit": risk_levels,
            "risk": risks,
            "return": rewards,
        }
    )

    return alloc_df, reward_df


def full_portfolio_pipeline(tickers, start, end):
    """
    Glue function:
    1. Runs analyze_tickers()
    2. Feeds returns into run_portfolio_womack()
    """

    monthly_returns = analyze_tickers(tickers, start, end)
    alloc_df, reward_df = run_portfolio_womack(monthly_returns)

    return {
        "monthly_returns": monthly_returns,
        "allocations": alloc_df,
        "reward_table": reward_df,
    }
