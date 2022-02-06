import yfinance as yf


def main():
    data = yf.download('APPS', start='2021-01-01')
    print(data['Adj Close'])


if __name__ == '__main__':
    main()
