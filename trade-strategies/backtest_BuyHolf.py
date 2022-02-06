from pyalgotrade import strategy
from pyalgotrade.barfeed import yahoofeed
from import_data import Data
import math

class BuyAndHoldStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument):
        super(BuyAndHoldStrategy, self).__init__(feed)
        self.instrument = instrument
        self.setUseAdjustedValues(True)
        self.position = None


    def onEnterOk(self, position):
        self.info(f"{position.getEntryOrder().getExecutionInfo}")


    def onBars(self, bars):
        bar = bars(self.instrument)
        self.info(bar.getClose())

        if self.position is None:
            close = bar.getAdjClose()
            broker = self.getBroker()
            cash = broker.getCash()

            quantity = math.floor(cash / close)
            self.position = self.enterLong(self.instrument, quantity)

instrument = 'SPY'
priceData = Data(instrument)
feed = yahoofeed.Feed()
feed.addBarsFromCSV(instrument, priceData.data)

# Instantiate strategy
run_strategy = BuyAndHoldStrategy(feed, "SPY")
run_strategy.run()
portfolio_value = run_strategy.getBroker().getEquity() + run_strategy.getBroker().getCash()
print(portfolio_value)