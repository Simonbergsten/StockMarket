import yfinance as yf


data = yf.download("SPY", start="1999-05-01", end="2019-10-04")
data.to_csv("spy.csv")

class Data:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = yf.download('SPY', start='2018-01-01', end='2021-06-30').to_csv()
