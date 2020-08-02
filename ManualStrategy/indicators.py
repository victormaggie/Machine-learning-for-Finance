
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os
#import seaborn as sns
from util import *
from matplotlib import dates

"Author: rren34"
"GTID: 903474021"

def author():
    return 'rren34'

class Indicators():

    def __init__(self, price):
        # here we add the normalization
        self.price = price

        # -----> price history of the stock
        # -----> helper data
        # -----> indicator itself
        # -----> LIKE : Price, SMA, Price/SMA

    # PART I:  price / sma ratio
    def simple_moving_average(self, windows=14):
        sma = self.price.rolling(window=windows).mean()
        sma.fillna(method='bfill', inplace=True)
        # normalization for the simple moving average
        # sma /= sma.iloc[0]
        return sma

    def sma_ratio(self, windows=14):
        """
        :param windows:
        :return:  sma_ratio > 1.05 SELL, sma_ratio < 0.95 indicator
        """
        sma = self.simple_moving_average(windows)
        sma_ratio = self.price / sma
        return sma_ratio

    # PART II:  Bollinger bond percent
    def bbr(self, windows=14):
        """
        Definition: a technical analysis tool defined by a set of lines plotted two
        standard deviations(positively and negatively) away from a simple moving average
        :param windows:
        :return:
        # reference: I watched the video of time series, I followed the video
        # copyright: reserved by Gatech.
        """
        sma = self.simple_moving_average(windows)
        # calculate the rolling standard deviation for the questions
        rolling_std = self.price.rolling(window=14, min_periods=14).std()
        rolling_std = rolling_std.fillna(method="bfill")
        top_band = sma + (2 * rolling_std)
        low_band = sma - (2 * rolling_std)
        # the bollinger band percent
        # ??? why if we use billinger percent cannot give us good result>>>??
        #bbr = (self.price - sma) / (top_band - low_band)
        bbr = (self.price - sma) / 2. / rolling_std
        # we return top_band, low_band as well for the report figures
        return bbr, top_band, low_band

    # PART III:  MACD   Moving average convergence divergence
    def macd(self):
        """
        :param data: data
        :return: momentum
        reference: Youtube MACD
        1. macd line ( 12 day EMA - 26 day EMA)
        2. Signal Line ( 9 day EMA of MACD line)
        3. MACD Histogram (MACD line - Signal line)
        """

        macd_12 = self.price.ewm(span=12).mean()
        macd_26 = self.price.ewm(span=26).mean()
        macd_line = macd_12 - macd_26
        signal_line = macd_line.ewm(span=9).mean()
        macd_hist = macd_line - signal_line
        # we can treat macd_hist > or < as a buy or sell signal
        return macd_line, signal_line, macd_hist

    # The other indicator I came up with as following:
    # Indicat\r 2: linear regression
    # Indicator 3: Kalman calculation  --> design Kalman filtering for the calculation
    # Fast_Fourier_Transformation

    def calculate_momentum(self, window = 7):
        return self.price / self.price.shift(window-1) -1


"""
Statistics : helper function definition
I use this function from my previous project 2
reference: RUI REN project 2
"""

def avg_daily_return(port_vals):
    daily_ret = (port_vals / port_vals.shift(1)) - 1
    return daily_ret

def std_daily_return(port_vals):
    daily_ret = (port_vals / port_vals.shift(1)) - 1
    return daily_ret

def sharpe_ratio(port_vals, k=np.sqrt(252)):
    return k * avg_daily_return(port_vals) / std_daily_return(port_vals)

def cummu_return(port_vals):
    cummu_ret = (port_vals / port_vals.iloc[0,0]) - 1
    return cummu_ret

def indicator_frame(prices):
    all_indicator = Indicators(prices)
    sma = all_indicator.simple_moving_average()
    sma_ratio = all_indicator.sma_ratio()
    bbr, top_band, low_band = all_indicator.bbr()
    momentum = all_indicator.calculate_momentum()
    macd, macd_signal, macd_divergence = all_indicator.macd()
    daily_ret = avg_daily_return(prices)
    std_daily_ret = std_daily_return(prices)

    df = pd.concat(
        [prices, daily_ret, std_daily_ret, momentum,
        sma, sma_ratio, bbr, top_band, low_band, macd, macd_signal, macd_divergence]
    , axis=1)

    df.columns = prices.columns.tolist() + [
        "daily_returns", "std_ev", "momentum", "SMA", "SMA_ratio",
        "bbr", "top_band", "low_band",  "macd", "macd_signal", "macd_divergence"
    ]
    df.fillna(method="bfill", inplace=True)
    return df

def test_code():

    # we only test the JPM data
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    index = pd.date_range(sd, ed)
    in_sample_data = pd.DataFrame(index)
    df_data = get_data(["JPM"], index)

    # Normalization for the data
    df_data = df_data / df_data.iloc[0,:]

    # We only analyze JPM, select the JPM data file here
    # print(df_JPM, '..')
    JPM_prices = df_data[["JPM"]]
    df = indicator_frame(df_data[["JPM"]])


    # MACD chart 2
    fig, ax = plt.subplots(3, 1, sharex=True)
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2008, 12, 31)
    index1 = pd.date_range(sd, ed)
    start = index1[0]
    end = index1[-1]

    df.loc[start:end][["JPM"]].plot(color="green", linewidth=0.9, ax=ax[0])
    ax[0].set_ylabel("Normalized value")
    hour_fmt = dates.DateFormatter('%m')
    ax[0].xaxis.set_tick_params(hour_fmt)
    df.loc[start:end][["macd", "macd_signal"]].plot(ax=ax[1])
    df.loc[start:end][["macd_divergence"]].plot(ax=ax[2])
    ax[2].axhline(y=0, color='r')
    ax[2].xaxis.set_tick_params(rotation=20)

    for i in range(len(ax)):
        ax[i].xaxis.grid(True, which='Major', linestyle='--')
        ax[i].yaxis.grid(True, which='Major', linestyle='--')
    plt.xlabel("Dates")
    plt.tight_layout()
    # plt.savefig("MACD2.png")

    # Bollinger Band ratio
    fig, ax = plt.subplots(2, 1, sharex=True)
    start = index[0]
    end = index[-1]

    df.loc[start:end][["JPM"]].plot(color="green", linewidth=0.9, ax=ax[0])
    df.loc[start:end][["SMA"]].plot(color="red", linewidth=0.9, ax=ax[0])
    df.loc[start:end][["top_band"]].plot(color="blue", linewidth=0.3, ax=ax[0])
    df.loc[start:end][["low_band"]].plot(color="blue", linewidth=0.3, ax=ax[0])
    ax[0].set_ylabel("Normalized value")
    num = len(df.loc[start:end][["low_band"]])

    ax[0].fill_between(
        df.loc[start:end]["SMA"].index, df.loc[start:end][["top_band"]].values.reshape(1, num)[0],
        df.loc[start:end][["low_band"]].values.reshape(1, num)[0],
        color='r',
        alpha=0.1)

    # plot bollinger ratio
    df.loc[start:end][["bbr"]].plot(legend=None, ax=ax[1])
    ax[1].axhline(y=1, color='red')
    ax[1].axhline(y=-1, color='red')

    for i in range(len(ax)):
        ax[i].xaxis.grid(True, which='Major', linestyle='dotted')
        ax[i].yaxis.grid(True, which='Major', linestyle='dotted')
    ax[1].xaxis.set_tick_params(rotation=20)
    plt.xlabel("Dates")

    plt.tight_layout()
    #plt.savefig("bbr1.png")
    plt.show()

    # 1 . MACD analysis
    fig, ax = plt.subplots(3, 1, sharex=True)
    start = JPM_prices.index.min()
    end = JPM_prices.index.max()

    df.loc[start:end][["JPM"]].plot(color='green', linewidth=0.9, ax=ax[0])
    ax[0].set_ylabel("Normalized value")
    df.loc[start:end][["macd", "macd_signal"]].plot(ax=ax[2])
    df.loc[start:end][["macd_divergence"]].plot(ax=ax[1])
    ax[1].axhline(y=0, color='r')
    hour_fmt = dates.DateFormatter('%m')
    ax[2].xaxis.set_tick_params(hour_fmt)
    ax[2].xaxis.set_tick_params(rotation=15)

    for i in range(len(ax)):
        ax[i].xaxis.grid(True, which='Major', linestyle='--')
        ax[i].yaxis.grid(True, which='Major', linestyle='--')
    plt.xlabel("Dates")
    plt.tight_layout()
    # plt.savefig("MACD_1.png")
    plt.show()

    # simple moving average
    fig, ax = plt.subplots(2,1, sharex=True)
    start = index[0]
    end = index[-1]
    df.loc[start:end][["JPM"]].plot(color="green", linewidth=0.9, ax=ax[0])
    df.loc[start:end][["SMA"]].plot(color='red', linewidth=0.9, ax=ax[0])

    df.loc[start:end][["SMA_ratio"]].plot(color='blue', linewidth=0.8, ax=ax[1])
    ax[1].axhline(y=1, color='r')
    for i in range(len(ax)):
        ax[i].xaxis.grid(True, which='Major', linestyle='dotted')
        ax[i].yaxis.grid(True, which='Major', linestyle='dotted')
    ax[1].xaxis.set_tick_params(rotation=20)
    ax[0].set_ylabel("Normalized value")
    plt.xlabel("Dates")
    plt.tight_layout()
    #plt.savefig("SMA_1.png")
    plt.show()

    fig, ax = plt.subplots(2, 1, sharex=True)
    start = "2008-1"
    end = "2008-12"
    df.loc[start:end][["JPM"]].plot(color="green", linewidth=0.9, ax=ax[0])
    df.loc[start:end][["SMA"]].plot(color="red", linewidth=0.9, ax=ax[0])
    # plot sharp
    df.loc[start:end]["SMA_ratio"].plot(color='b', linewidth=0.9, ax=ax[1])
    ax[1].axhline(y=1, color="r")
    for i in range(len(ax)):
        ax[i].xaxis.grid(True, which='Major', linestyle='dotted')
        ax[i].yaxis.grid(True, which='Major', linestyle='dotted')
    ax[1].xaxis.set_tick_params(rotation=20)
    hour_fmt = dates.DateFormatter('%m')
    ax[1].xaxis.set_tick_params(hour_fmt)
    ax[0].set_ylabel("Normalized value")
    plt.xlabel("Dates")
    plt.tight_layout()
    #plt.savefig("SMAR_2.png")
    plt.show()


if __name__ == "__main__":
    test_code()
"""
generate the charts illustrate in the reports
"""

