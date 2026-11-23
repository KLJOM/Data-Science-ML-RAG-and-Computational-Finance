import numpy as np 
import datetime as dt
import pandas as pd 
import yfinance as yf  # pyright: ignore[reportMissingImports]

# 01 - import data
"""
This funtion imports and extrapolates the data in the ways specified from yahoo finance
"""
# need to add a .ax (austrailian stock market) to the end of all the stocks so the yahoo data is referenced properly - list comprehension is the solution

stocklist = ['DSY.JO']
stocks = [stock for stock in stocklist]


# defining the end date for the data being fetched
endDate = dt.datetime(2026, 9, 12)

# defining the start date for the data being fecthed (2021/09/12)
startDate = dt.datetime(2021, 9, 12)

import os

def getData(stocks, start, end, price_filename="price_5yrs.csv", close_filename="close_price_5yrs.csv", label=""):
    # Fetch data using yfinance
    stockData = yf.download(stocks, start=start, end=end, multi_level_index=False)
    
    # Flatten MultiIndex columns if needed
    if isinstance(stockData.columns, pd.MultiIndex):
        if len(stockData.columns.levels[1]) == 1:
            stockData.columns = stockData.columns.get_level_values(0)

    # Reset index so 'Date' is preserved as a clean column
    stock_df = stockData.reset_index()
    if 'Date' in stock_df.columns:
        stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.strftime('%Y-%m-%d')

    # Reorder columns to standard OHLCV format
    desired_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    ordered_cols = [col for col in desired_cols if col in stock_df.columns]
    ordered_cols += [col for col in stock_df.columns if col not in ordered_cols]
    stock_df = stock_df[ordered_cols]

    # Save to the script's local folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    price_csv_path = os.path.join(script_dir, price_filename)
    close_price_csv_path = os.path.join(script_dir, close_filename)

    # 1. Save full OHLCV price data with Date
    stock_df.to_csv(price_csv_path, index=False)
    print(f"\n[{label or stocks}] Full price data saved to: {price_csv_path}")

    # 2. Select and save only Date and Close price
    close_cols = [c for c in ['Date', 'Close'] if c in stock_df.columns]
    stockclose_df = stock_df[close_cols]
    stockclose_df.to_csv(close_price_csv_path, index=False)
    print(f"[{label or stocks}] Close price data saved to: {close_price_csv_path}")

    print(f"Sample data for {label or stocks}:")
    print(stockclose_df.head(3))
    return stock_df, stockclose_df


# 1. Discovery Limited (DSY.JO)
getData(stocks=stocks, start=startDate, end=endDate, 
        price_filename="price_5yrs.csv", 
        close_filename="close_price_5yrs.csv", 
        label="Discovery (DSY.JO)")

# 2. FTSE/JSE Top 40 Index (^J200.JO)
getData(stocks='^J200.JO', start=startDate, end=endDate, 
        price_filename="top40_price_data.csv", 
        close_filename="top40_close_price.csv", 
        label="FTSE/JSE Top 40 Index")

# 3. FTSE/JSE All Share Index ALSI (^J203.JO)
getData(stocks='^J203.JO', start=startDate, end=endDate, 
        price_filename="alsi_price_data.csv", 
        close_filename="alsi_close_price.csv", 
        label="FTSE/JSE ALSI")