import backtrader as bt
import backtrader.analyzers as btanalyzers
import matplotlib
from datetime import datetime


class MaCrossStrategy(bt.Strategy):

    def __init__(self):
        ma_fast = bt.ind.SMA(period = 10)
        ma_slow = bt.ind.SMA(period = 50)

        self.crossover = bt.ind.CrossOver(ma_fast, ma_slow)


    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
            elif self.crossover < 0:
                self.close()

cerebro = bt.Cerebro()
data = bt.feeds.YahooFinanceData(dataname ='AAPL', fromdate = datetime(2010,1,1),
                                 todate = datetime(2020,1,1))
cerebro.adddata(data)

cerebro.broker.setcash(10000000.0)
cerebro.addstrategy(MaCrossStrategy)
cerebro.addsizer(bt.sizers.PercentSizer, percents = 10)


cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name = 'sharpe')
cerebro.addanalyzer(bt.analyzers.Transactions, _name = 'trans')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name = 'trades')


back = cerebro.run()

cerebro.broker.getvalue()

back[0].analyzers.sharpe.get_analysis()

cerebro.plot()


