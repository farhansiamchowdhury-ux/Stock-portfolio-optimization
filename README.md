This project builds a full pipeline for analyzing stock performance and optimizing a portfolio using real market data.

It combines:

1. Market Data Collection (Yahoo Finance)

Downloads historical prices for any list of tickers

Computes daily and monthly returns

Plots cumulative performance

Displays covariance & correlation heatmaps

2. Portfolio Optimization (Mean–Variance Model)

Using monthly returns, we:

Compute expected returns

Compute covariance matrix

Run an optimization that finds the best portfolio weights

Sweep across different risk levels

Plot the efficient frontier

Plot how optimal weights change as risk tolerance changes
