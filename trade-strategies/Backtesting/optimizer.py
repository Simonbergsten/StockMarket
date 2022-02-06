from empyrial import *

portfolio = ['AAPL', 'TSLA', 'UBER', 'MSFT', 'AMZN', 'V', 'FB', 'GME', 'SQ']

p = Engine(
    start_date = "2019-01-01",
    portfolio = portfolio,
    weight = (1.0 / len(portfolio) ),
    benchmark = ['SPY']
)

empyrial(p)

oracle(p)

optimizer(p, 'EF') # Remove benchmark and weight if running this.

