# From: https://www.youtube.com/watch?v=iTDFfNvCtGU
imort talib
from zipline.api import order_target, record, symbol, order_percent_target

def initialize(context):
    stockList = ['FB', 'AMZN', 'NFLX', 'GOOGL', 'AAPL']

    context.stocks = [symbol(s) for s in stockList]

    context.target_pct_per_stock = 1.0 / len(context.stocks)

    context.LOW_RSI = 30
    context.HIGH_RSI = 70

def handle_data(context, data):

    # historical data
    prices = data.history(context.stocks, 'price', bar_count=20, frequency='id')

    rsis = {}

    # Loop through the stocks
    for stock in context.stocks:
        rsi = talib.RSI(prices[stock], timeperiod=14)[-1] # Look at the past 14 days RSI in order to be able to enter
        # a position the next day.
        rsis[stock] = rsi

        current_position = context.portfolio.positions[stock].amount

        # If the RSI is over 70 and we own share, time to sell.
        if rsi > context.HIGH_RSI and current_position > 0 and data.can_trade(stock):
            order_target(stock, 0)

        elif rsi < context.LOW_RSI and current_position == 0 and data.can_trade(stock):
            order_percent_target(stock, context.target_pct_per_stock)

    record(fb_rsi = symbol[('FB')],
           amzn_rsi = symbol[('AMZN')],
           aapl_rsi = symbol[('AAPL')],
           nflx_rsi = symbol[('NFLX')],
           googl_rsi = symbol[('GOOGL')])





