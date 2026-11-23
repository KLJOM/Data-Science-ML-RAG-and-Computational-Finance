# Efficient Frontier: Cryptocurrency Portfolio Analysis

**London School of Economics — Department of Finance**  
*Instructor: [Your Name]*

This guide offers an educational walkthrough of `effApp.py`, a Python script for evaluating the risk and return of a cryptocurrency portfolio using real market data. The script is designed to be transparent, robust, and an excellent learning resource for portfolio construction using practical Python data tools.

---

## Table of Contents
- [Background: Efficient Frontier](#background-efficient-frontier)
- [What Does `effApp.py` Do?](#what-does-effappy-do)
- [Key Sections in the Code](#key-sections-in-the-code)
- [Step-by-Step Workflow](#step-by-step-workflow)
- [Customizing Your Portfolio](#customizing-your-portfolio)
- [Understanding and Interpreting Results](#understanding-and-interpreting-results)
- [Prerequisites & Setup](#prerequisites--setup)
- [Robust Handling of Missing Data](#robust-handling-of-missing-data)

---

## Background: Efficient Frontier
The Efficient Frontier is a foundational concept in Modern Portfolio Theory. It represents the set of optimal portfolios that offer the highest expected return for a given level of risk. Constructing and analyzing such portfolios helps finance professionals and students understand risk/return trade-offs and diversification benefits.

## What Does `effApp.py` Do?
- Downloads daily price data for selected cryptocurrencies.
- Computes mean returns and risk (standard deviation of returns) using historical data.
- Allows the user to specify portfolio weights (allocations) for each asset.
- Automatically handles missing or failed downloads, rebalancing weights if some data cannot be fetched.
- Prints out the annualized expected return and risk of the current portfolio allocation.

## Key Sections in the Code
1. **Imports and Dependencies:** Uses `numpy`, `pandas`, `datetime`, and [`yfinance`](https://github.com/ranaroussi/yfinance) for data and calculations.
2. **getData Function:** Downloads prices, calculates returns, validates data, and returns all necessary statistics for the assets.
3. **Weights Vector:** Manually set at the top of the script; should match the number of chosen cryptocurrencies.
4. **Portfolio Performance Function:** Calculates expected portfolio return and risk.
5. **Robust Error Handling:** If one or more cryptocurrencies fail to download, the code removes them and recalculates weights.
6. **Execution Block:** Downloads fresh data, computes statistics, applies robustness logic, and prints results.

## Step-by-Step Workflow
1. **Set Ticker List:**
   - Modify the `cryptolist` in the code to choose which assets to analyze. Example: `'BTC-USD', 'ETH-USD', 'ADA-USD', 'XRP-USD'`.
2. **Choose Portfolio Weights:**
   - Set the `weights` array to your desired allocations. All weights must sum to 1.0 (the code will normalize them if not).
3. **Historical Data Range:**
   - The last 365 days of price data are fetched by default.
4. **Run the Code:**
   - The script fetches data, handles errors, calculates annualized mean return and standard deviation (risk), and prints these values for your portfolio.
5. **Interpret Results:**
   - Printout displays annualized portfolio return (%) and risk (%) to two decimal places.

## Customizing Your Portfolio
- **Add/Remove Assets:** Edit the `cryptolist` variable.
- **Adjust Weights:** Change values in the `weights` numpy array. If data for a crypto cannot be downloaded, its weight is removed and the remaining weights are automatically scaled.

## Understanding and Interpreting Results
- **Returns:** Reflect the average annual growth rate you would expect with this portfolio allocation, based on past 1-year data. Remember, past performance is not indicative of future results!
- **Risk (Std Dev):** Indicates typical volatility to expect. Higher numbers mean greater swings in portfolio value.
- Optimal portfolios (on the efficient frontier) typically seek the highest return for a given risk, or the lowest risk for a given return.

## Prerequisites & Setup
- Install required Python libraries:
  ```bash
  pip install numpy pandas yfinance
  ```
- Adjust the script as needed, save changes, and run using e.g.:
  ```bash
  python effApp.py
  ```

## Robust Handling of Missing Data
- If Yahoo Finance cannot provide data for one or more tickers (e.g. due to delisting, network error, or typo), the script:
  - Warns you in the terminal.
  - Removes these assets from the analysis.
  - Re-calculates and normalizes the weights for remaining assets.
  - Ensures results are always coherent and never fail due to missing data.

---

*For further exploration, try testing different date ranges, assets, or custom weightings to observe the risk/return trade-offs on your own Efficient Frontier!*
