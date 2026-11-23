import numpy as np 
import datetime as dt
import yfinance as yf  # pyright: ignore[reportMissingImports]
import pandas as pd  # Ensure this is at the top if not already

# 01 - import data
"""
This funtion imports and extrapolates the data in the ways specified from yahoo finance
"""
def getData(crypto, start, end):
    stockData = yf.download(crypto, start=start, end=end, auto_adjust=False)
    # Handles both single and multiple tickers
    if isinstance(stockData, pd.Series) or (isinstance(stockData, pd.DataFrame) and 'Close' not in stockData):
        # Single asset fallback (older yfinance versions, or single crypto)
        close_data = stockData if isinstance(stockData, pd.Series) else stockData.iloc[:,0]
        stockData = pd.DataFrame(close_data)
        stockData.columns = [crypto[0]]
    else:
        stockData = stockData['Close']
    # Find valid cryptos (i.e., columns not all NaN)
    available_cryptos = list(stockData.columns[~stockData.isna().all(axis=0)])
    missing_cryptos = [c for c in crypto if c not in available_cryptos]
    if missing_cryptos:
        print(f"Warning: Missing data for: {missing_cryptos}. These will be skipped and weights rebalanced.")
    stockData = stockData[available_cryptos]
    returns = stockData.pct_change(fill_method=None)
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix, available_cryptos

# 02 - Portfolio Performance with weights mean returns and correlation show by the covariance matrix
# defining the weights matrix for the portfoloio performance calculations (list)

weights = np.array([0.4,0.15, 0.3, 0.15])

def portfolioPerformance(weights, meanReturns, covMatrix):
    # we want to get the standard deviation and the returns so we use the numpy module 
    returns = np.sum(meanReturns * weights)*252 # we have all our data in daily format so we scale by the number of trading days in a year
    # for the Std deviations we use the formula Portfolio Variance = w^T ∑w (reads weights transpose matrix with covariance matrix with weights)
    # simply the volatility multiplied by the time rule to reflect for the correct time scale
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252) # the weights multiplied by - the covariance multiplied by the weights --> give portfolio varience using dot product which is the volatility. Then it is multiplied by the trading days squared which is reffered to as the time rule

    return returns, std

cryptolist = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']

cryptocurencies = [crypto for crypto in cryptolist]

# defining the end date for the data being fetched
endDate = dt.datetime.now()
# defining the start date for the data being fecthed
startDate = endDate - dt.timedelta(days = 365)
print(getData(crypto = cryptocurencies, start=startDate, end=endDate))
# this will call the functions and fetch all the operations specified above in the functions
meanReturns, covMatrix, final_cryptos = getData(crypto = cryptocurencies, start=startDate, end=endDate)
if len(final_cryptos) != len(weights):
    # Map: for all cryptos in original list, keep only if present
    keep_idx = [i for i, c in enumerate(cryptocurencies) if c in final_cryptos]
    weights = weights[keep_idx]
    # Renormalize weights to sum to one (unless all zero from error)
    total = np.sum(weights)
    if total > 0:
        weights = weights / total
    print(f"Weights rebalanced for available cryptos: {final_cryptos}\nNew weights: {weights}")
returns, std = portfolioPerformance(weights, meanReturns, covMatrix) 

print("Return: " +str(round(returns*100,2)),"\nStandard Deviation: " +str(round(std*100,2))) # the numbers come out pretty ugly so we x 100 and round to 2 decimal places