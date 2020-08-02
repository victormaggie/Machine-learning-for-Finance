import numpy as np
import pandas as pd
import datetime as dt
from indicators import *
from util import *
from TheoreticallyOptimalStrategy import TheoreticallyOptimalStrategy
from marketsimcode import compute_portvals
import time
import matplotlib.pyplot as plt

class ManualStrategy(TheoreticallyOptimalStrategy):

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.date(2010, 12, 31), sv=100000,
                    SMA_ratio_up=1.02, SMA_ratio_low=0.92, bbr_top=1.0, bbr_bottom=-1.0):

        # here we import data
        index = pd.date_range(sd, ed)
        stock_price, idx_price, trading_date = self.get_historical_price(symbol, index)

        # Indicator calculation
        indicator = indicator_frame(stock_price.to_frame(symbol))
        macd_divergence = indicator["macd_divergence"]
        bbr_ratio = indicator["bbr"]
        SMA_ratio = indicator["SMA_ratio"]
        # Trading positions (strategy)
        a = np.zeros((len(trading_date), 1))
        action = pd.Series(index=trading_date)
        i = 0
        for date in action.index.to_list():
            prev_date = action.index.get_loc(date) - 1
            if prev_date < 0:
                action.loc[date] = 0
                continue

            elif prev_date >= 0:
                prev_date = action.index[prev_date]
                if  (bbr_ratio.loc[date] < bbr_bottom and SMA_ratio.loc[date] < SMA_ratio_low) or (macd_divergence.loc[prev_date] < 0 and macd_divergence.loc[date] > 0):
                    # Buy the stock
                    action.loc[date] = 1
                elif (bbr_ratio.loc[date] > bbr_top and SMA_ratio.loc[date] > SMA_ratio_up) or (macd_divergence.loc[prev_date] > 0 and macd_divergence.loc[date] < 0 ):
                    # Short the stock
                    action.loc[date] = -1
                else:
                    action.loc[date] = 0

        df_trades = self.actionsToOrders(action)
        return df_trades.to_frame(symbol)

def test_code():
    ms = ManualStrategy()
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    interval = pd.date_range(sd, ed)
    trading_days = get_data(["SPY"], dates=interval).index
    sv = 100000
    start_share = 1000

    # Benchmark performance for the calculation
    benchmark_val = pd.DataFrame([start_share] + [0] * (len(trading_days)-1), columns=[symbol], index=trading_days)
    benchmark_val = compute_portvals(benchmark_val, start_val=sv, commission=9.95, impact=0.005)
    benchmark_val /= benchmark_val.iloc[0, :]

    # out-sample cases

    # Test: case
    df_trades_hard = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv, bbr_top=0.91,
                                   bbr_bottom=-0.91, SMA_ratio_up=1.02, SMA_ratio_low=0.99)
    port_vals_hard = compute_portvals(df_trades_hard, start_val=sv, commission=9.95, impact=0.005)

    port_vals_hard /= port_vals_hard.iloc[0, :]

    fig, ax = plt.subplots()
    benchmark_val.plot(ax=ax, color='green')
    port_vals_hard.plot(ax=ax, color='r')
    plt.legend(["Benchmark", "Manual Strategy"])
    plt.title("ManualStrategy for in sample data")
    plt.xlabel("Dates")
    plt.ylabel("Normalizaed value")

    for date, action in df_trades_hard[df_trades_hard[symbol] != 0].iterrows():
        if action[symbol] < 0:
            ax.axvline(date, color='green', alpha=0.5, linestyle='--', linewidth=1.1)
        elif action[symbol] > 0:
            ax.axvline(date, color='blue', alpha=0.5, linestyle='--', linewidth=1.1)
    plt.tight_layout()
    plt.show()



    # out of sample data calculation::

    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    interval = pd.date_range(sd, ed)
    trading_days = get_data(["SPY"], dates=interval).index
    # Benchmark performance for the calculation
    benchmark_val = pd.DataFrame([start_share] + [0] * (len(trading_days)-1), columns=[symbol], index=trading_days)
    benchmark_val = compute_portvals(benchmark_val, start_val=sv, commission=9.95, impact=0.005)
    benchmark_val /= benchmark_val.iloc[0, :]

    df_trades_naive = ms.testPolicy(symbol=symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=sv, bbr_top=0.91,
                                   bbr_bottom=-0.91, SMA_ratio_up=1.02, SMA_ratio_low=0.99)

    port_vals_naive = compute_portvals(df_trades_naive, start_val=sv, commission=9.95, impact=0.005)
    port_vals_naive /= port_vals_naive.iloc[0, :]
    # Figures
    fig, ax = plt.subplots()
    benchmark_val.plot(ax=ax, color="green")
    port_vals_naive.plot(ax=ax, color="red")
    ax = plt.gca()
    ax.xaxis.grid(True, which='Major', linestyle='--')
    ax.yaxis.grid(True, which='Major', linestyle='--')
    hour_fmt = dates.DateFormatter('%m')
    ax.xaxis.set_tick_params(hour_fmt)
    ax.xaxis.set_tick_params(rotation=20)
    plt.legend(["Benchmark", "Manual Strategy"])
    plt.title("Manual Strategy for out-sample data")
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    plt.show()

    for date, action in df_trades_naive[df_trades_naive[symbol] != 0].iterrows():
        if action[symbol] < 0:
            ax.axvline(date, color='k', alpha=0.5, linewidth=1.1)
        elif action[symbol] > 0:
            ax.axvline(date, color='blue', alpha=0.5, linewidth=1.1)

    plt.tight_layout()
    plt.show()

    """
    Statistic information of the calculations
    """

    # Benchmark
    cumm_returns_benchmark, avg_daily_returns_benchmark, std_daily_returns_benchmark, sharpe_ratio_benchmark = [
        cummu_return(benchmark_val),
        avg_daily_return(benchmark_val),
        std_daily_return(benchmark_val),
        sharpe_ratio(benchmark_val)
    ]

    # cumm_returns, avg_daily_returns, std_daily_returns, sharpe_ratio_ = [
    #     cummu_return(port_vals_hard),
    #     avg_daily_return(port_vals_hard),
    #     std_daily_return(port_vals_hard),
    #     sharpe_ratio(port_vals_hard)
    # ]

    #The out sample data
    cumm_returns, avg_daily_returns, std_daily_returns, sharpe_ratio_ = [
        cummu_return(port_vals_naive),
        avg_daily_return(port_vals_naive),
        std_daily_return(port_vals_naive),
        sharpe_ratio(port_vals_naive)
    ]
    """
    Print all the statistic information
    Reference: I modified the below code from marketsim project 5
    Copyright: Gatech
    """

    print(f"Date Range: {sd} to {ed}")
    print()
    print("Sharpe Ratio of Fund: {}".format(sharpe_ratio_.iloc[-1, 0]))
    print("Sharpe Ratio of Benchmark : {}".format(sharpe_ratio_benchmark.iloc[-1, 0]))
    print()
    print("Cumulative Return of Fund: {}".format(cumm_returns.iloc[-1, 0]))
    print("Cumulative Return of Benchmark : {}".format(cumm_returns_benchmark.iloc[-1, 0]))
    print()
    print("Standard Deviation of Fund: {}".format(std_daily_returns.iloc[-1, 0]))
    print("Standard Deviation of Benchmark : {}".format(std_daily_returns_benchmark.iloc[-1, 0]))
    print()
    print("Average Daily Return of Fund: {}".format(avg_daily_returns.iloc[-1, 0]))
    print("Average Daily Return of Benchmark : {}".format(avg_daily_returns_benchmark.iloc[-1, 0]))
    print()
    # Here we need multiply by the start value.
    print(f"Final Portfolio Value: {port_vals_hard.iloc[-1,0] * sv}")


def author():
    return "rren34"


if __name__ == "__main__":
    test_code()
