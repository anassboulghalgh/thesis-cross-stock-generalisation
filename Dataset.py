"""Downloads the daily stock data and builds Dataset_Thesis.csv.

Prices come from Yahoo Finance via yfinance and are adjusted for splits and dividends
(the default auto_adjust setting), so the closing price equals the adjusted closing price.
Running this again later can give slightly different prices, because Yahoo Finance
recalculates adjusted prices when new dividends are paid. The repository therefore
includes the exact Dataset_Thesis.csv used in the thesis.
"""
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "AMZN": "Technology",
    "GOOGL": "Technology", "TSLA": "Technology",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "PEP": "Food & Beverage", "KO": "Food & Beverage", "MDLZ": "Food & Beverage",
}
TICKERS = list(SECTORS)
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
OUTPUT_FILE = "Dataset_Thesis.csv"


def download_prices(tickers, start, end):
    """Download daily prices for all tickers and stack them into one long table."""
    raw_data = yf.download(tickers, start=start, end=end, group_by="ticker")
    all_stocks = []
    for ticker in tickers:
        df = raw_data[ticker].copy()
        df["Ticker"] = ticker
        df["Sector"] = SECTORS[ticker]
        all_stocks.append(df.reset_index())
    data = pd.concat(all_stocks, ignore_index=True)
    data["Adj Close"] = data["Close"]
    columns = ["Date", "Ticker", "Sector", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return data[columns].sort_values(["Ticker", "Date"])


def add_target(data):
    """Target: 1 if the next day's adjusted close is higher than today's, else 0."""
    data["Next_Day_Adj_Close"] = data.groupby("Ticker")["Adj Close"].shift(-1)
    data["Target"] = (data["Next_Day_Adj_Close"] > data["Adj Close"]).astype(int)
    return data


def add_indicators(data, tickers):
    """Returns and technical indicators, computed separately for each stock."""
    all_with_indicators = []
    for ticker in tickers:
        df = data[data["Ticker"] == ticker].sort_values("Date").copy()
        close = df["Adj Close"]

        df["Return"] = close.pct_change()
        for lag in range(1, 6):
            df[f"Return_lag{lag}"] = df["Return"].shift(lag - 1)

        df["SMA_10"] = SMAIndicator(close, window=10).sma_indicator()
        df["SMA_50"] = SMAIndicator(close, window=50).sma_indicator()
        df["RSI_14"] = RSIIndicator(close, window=14).rsi()

        macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()

        all_with_indicators.append(df)
    return pd.concat(all_with_indicators, ignore_index=True)


def build_dataset(tickers=TICKERS, start=START_DATE, end=END_DATE):
    """Download, add the target and the indicators, and drop rows with missing values."""
    data = download_prices(tickers, start, end)
    data = add_target(data)
    data = add_indicators(data, tickers)
    return data.dropna()


def main():
    data = build_dataset()
    print(data.shape)
    print(data["Ticker"].value_counts())
    data.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
