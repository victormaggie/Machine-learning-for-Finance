import pandas as pd
import numpy as np
import os
import datetime as dt
from indicators import *
from marketsimcode import compute_portvals
from util import get_data
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib import dates
import matplotlib as mpl

def author():
    return "rren34"

class TheoreticallyOptimalStrategy(object):
    def __init__(self):
        pass

    def actionsToOrders(self, actions):

        df_trades = pd.Series(index=actions.index)
        stock_holding = 0
        for date in actions.index.to_list():
            action = actions.loc[date]

            if action == 1:  # Long
                df_trades.loc[date] = {
                    0: 1000,
                    -1000: 2000,
                    1000: 0,
                }.get(stock_holding)

            elif action == -1:   # short
                df_trades.loc[date] = {
                    0: -1000,
                    1000: -2000,
                    -1000: 0,
                }.get(stock_holding)

            else:
                df_trades.loc[date] = 0

            stock_holding += df_trades.loc[date]
        return df_trades

    # get the historical price data
    def get_historical_price(self, symbol, date):
        port_data = get_data([symbol], date)
        stock_price = port_data[symbol]
        # normalize the data
        stock_price /= stock_price.iloc[0]
        # get the SPY data
        benchmark_price = port_data["SPY"]
        benchmark_price /= benchmark_price[0]
        trading_days = benchmark_price.index
        return stock_price, benchmark_price, trading_days

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.date(2009, 12, 31), sv=100000):

        # we need first get the historical data
        index = pd.date_range(sd, ed)
        stock_prices, idx_pic, time_trade = self.get_historical_price(symbol, index)


        df_action = pd.Series(index=time_trade)
        daily_ret = avg_daily_return(stock_prices)
        prev_date = None
        for date in df_action.index:
            if prev_date is None:
                df_action.loc[date] = 0
                prev_date = date
                continue

            if daily_ret.loc[date] > 0:
                df_action.loc[prev_date] = 1
            elif daily_ret.loc[date] < 0:
                df_action.loc[prev_date] = -1
            else:
                df_action.loc[prev_date] = 0  # DO Nothing

            prev_date = date

        df_trades = self.actionsToOrders(df_action)
        return df_trades.to_frame(symbol)

def test_code():
    Theor_opt_stg = TheoreticallyOptimalStrategy()

    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    interval = pd.date_range(start_date, end_date)
    # get the trading_days for the calculation
    trading_days = get_data(["SPY"], dates=interval).index
    # The initial value is 100000
    sv = 100000
    # The initial stock holding is 1000
    start_hold = [1000]

    # TheoreticallyOptimalStrategy
    df_trades = Theor_opt_stg.testPolicy(symbol, sd=start_date, ed=end_date, sv=sv)
    port_vals = compute_portvals(df_trades, start_val=sv, commission=0, impact=0)
    port_vals /=port_vals.iloc[0,:]

    benchmark_val = pd.DataFrame((start_hold + [0] * (len(trading_days) - 1)), columns=["JPM"], index=trading_days)
    benchmark_val = compute_portvals(benchmark_val, start_val=sv, commission=0, impact=0)
    benchmark_val /= benchmark_val.iloc[0, :]

    # plot for the calculation
    hour_fmt = dates.DateFormatter('%m/%Y')
    fig, ax = plt.subplots()
    benchmark_val.plot(ax=ax, color="green")
    port_vals.plot(ax=ax, color="red")
    ax.xaxis.grid(True, which='Major', linestyle='--')
    ax.yaxis.grid(True, which='Major', linestyle='--')
    ax.xaxis.set_tick_params(rotation=0)
    ax.xaxis.set_tick_params(hour_fmt)
    plt.legend(["Benchmark", "Optimal Strategy"])
    plt.title("Theoretically Optimal Strategy vs. Benchmark")
    plt.xlabel("Dates")
    plt.ylabel("Normalized value")
    plt.show()

    """
    Statistic information of the calculations
    """
    cumm_returns, avg_daily_returns, std_daily_returns, sharpe_ratio_ = [
        cummu_return(port_vals),
        avg_daily_return(port_vals),
        std_daily_return(port_vals),
        sharpe_ratio(port_vals)
    ]

    # Benchmark
    cumm_returns_benchmark, avg_daily_returns_benchmark, std_daily_returns_benchmark, sharpe_ratio_benchmark = [
        cummu_return(benchmark_val),
        avg_daily_return(benchmark_val),
        std_daily_return(benchmark_val),
        sharpe_ratio(benchmark_val)
    ]

    """
    Print all the statistic information
    Reference: I modified the below code from marketsim project 5
    Copyright: Gatech
    """

    print(f"Date Range: {start_date} to {end_date}")
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
    print(f"Final Portfolio Value: {port_vals.iloc[-1,0] * sv}")





