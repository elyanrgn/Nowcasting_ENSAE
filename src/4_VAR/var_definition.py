import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR


def stationarity_test(series, significance_level=0.05):
    result = sm.tsa.stattools.adfuller(series)
    p_value = result[1]
    is_stationary = p_value < significance_level
    return is_stationary, p_value


def make_stationary(series):
    is_stat, _ = stationarity_test(series)
    while not is_stat:
        series = series.diff().dropna()
        is_stat, _ = stationarity_test(series)

    return series


class model_VAR:
    def __init__(self, max_lag, data):
        self.data = data
        self.max_lag = max_lag

    def fit(self, criterion="aic"):
        for series in self.data.columns:
            self.data.loc[:, series] = make_stationary(self.data[series])
        self.model = VAR(self.data)
        self.results = self.model.fit(maxlags=self.max_lag, ic=criterion)

        return self.results

    def predict(self, steps):
        return self.results.forecast(self.data.values[-self.results.k_ar :], steps)

    def irf(self, steps):
        irf = self.results.irf(steps)
        irf.plot(orth=True)
        plt.show()


if __name__ == "__main__":
    # Example usage
    data = sm.datasets.macrodata.load_pandas().data
    dates = data[["year", "quarter"]].astype(int).astype(str)
    quarterly = dates["year"] + "Q" + dates["quarter"]

    from statsmodels.tsa.base.datetools import dates_from_str

    quarterly = dates_from_str(quarterly)

    mdata = data[["realgdp", "realcons", "realinv"]]

    mdata.index = pd.DatetimeIndex(quarterly)

    data = np.log(mdata).diff().dropna()
    data = data[["realgdp", "realcons", "realinv"]]
    var_model = model_VAR(max_lag=4, data=data)
    results = var_model.fit()
    print(results.summary())
    predictions = var_model.predict(steps=5)
    print(predictions)
    var_model.irf(steps=10)
