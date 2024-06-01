from abc import abstractmethod, ABC

from strategy_analyzer.stock import Stock


class Strategy(ABC):
    name = 'BaseStrategy'
    @abstractmethod
    def __init__(self, params:{}=None):
        self.params = params

    def calc_buy_sell(self, stock:Stock):
        pass

    def visualize(self, stock:Stock):
        pass