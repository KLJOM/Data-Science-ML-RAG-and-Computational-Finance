import numpy as np 
import datetime as dt
import yfinance as yf  # pyright: ignore[reportMissingImports]

# 01 - import data
"""
This funtion imports and extrapolates the data in the ways specified from yahoo finance
"""
def getData(stocks, start, end):
    # Fetch data for multiple stock tickers using yfinance
    stockData = yf.download(stocks, start=start, end=end)
    print(stockData)
    # Selects only the close price to return 
    stockData = stockData["Close"]
    print(stockData)
    # Getting the returns a % changes 
    returns = stockData.pct_change()
    # From the percent changes we can now get the mean returns 
    meanReturns = returns.mean()
    # to understand how the assets are correlated with each other we want the covariance matrix
    covMatrix = returns.cov()
    print(covMatrix)
    return meanReturns, covMatrix

# 02 - Portfolio Performance with weights mean returns and correlation show by the covariance matrix
def portfolioPerformance(weights, meanReturns, covMatrix):
    # we want to get the standard deviation and the returns so we use the numpy module 
    returns = np.sum(meanReturns*weights)*252 # we have all our data in daily format so we scale by the number of trading days in a year
    # for the Std deviations we use the formula Portfolio Variance = w^T ∑w (reads weights transpose matrix with covariance matrix with weights)
    # simply the volatility multiplied by the time rule to reflect for the correct time scale
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252) # the weights multiplied by - the covariance multiplied by the weights --> give portfolio varience using dot product which is the volatility. Then it is multiplied by the trading days squared which is reffered to as the time rule

    return returns, std

stocklist = ['NVDA', 'PLTR', 'AMD', 'ORCL']
# need to add a .ax (austrailian stock market) to the end of all the stocks so the yahoo data is referenced properly - list comprehension is the solution
stocks = [stock for stock in stocklist]

# defining the end date for the data being fetched
endDate = dt.datetime.now()
# defining the start date for the data being fecthed
startDate = endDate - dt.timedelta(days = 365)

# defining the weights matrix for the portfoloio performance calculations (list) - %allocation of stocks

weights = np.array([0.5, 0.10, 0.2, 0.2])

# this will call the functions and fetch all the operations specified above in the functions
meanReturns, covMatrix = getData(stocks = stocks, start=startDate, end=endDate)
returns, std = portfolioPerformance(weights, meanReturns, covMatrix)

print("Return: " +str(round(returns*100,2)),"\nStandard Deviation: " +str(round(std*100,2))) # the numbers come out pretty ugly so we x 100 and round to 2 decimal places
