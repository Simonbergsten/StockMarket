import sqlalchemy
import pandas as pd
import yfinance as yf

wiki = 'https://en.wikipedia.org/wiki/'

tickersSensex = pd.read_html(wiki + 'BSE_SENSEX')[1].Symbol.to_list()
tickersDOW = pd.read_html(wiki + 'Dow_Jones_Industrial_Average')[1].Symbol.to_list()



def getData(tickers):
    data = []
    for ticker in tickers:
        data.append(yf.download(ticker).reset_index())

    return data

india, US = getData(tickersSensex), getData(tickersDOW)


def create_engine(name):
    engine = sqlalchemy.create_engine('sqlite:///'+name)
    return engine


indiaEngine, USEngine = create_engine('India'), create_engine('USA')

def toSQL(frames, symbols, engine):
    try:
        for frame, symbol in zip(frames, symbols):
            frame.to_sql(symbol, engine, index=False)
        print("Inserted successfully")
    except Exception as e:
        print("Insertion failed", e)


toSQL(US, tickersDOW, USEngine)

pd.read_sql('AAPL', USEngine)
pd.read_sql('SELECT * FROM AAPL WHERE Close > Open', USEngine)

