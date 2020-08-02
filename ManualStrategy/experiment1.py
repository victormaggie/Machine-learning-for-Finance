import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from util import *
from indicators import *
from ManualStrategy import *
from marketsimcode import compute_portvals
import StrategyLearner as sl

def author():
    return 'rren34'

def test_code():
    # we only use the JPM data and the same manual strategy
    # we test the in-sample data for the calculation

    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    index = pd.date_range(sd, ed)
    in_sample_data = pd.DataFrame(index)
    df_data = get_data(['JPM'], index)
    trading_days = df_data.index
    sv = 100000
    start_share = 1000

    # normalization for the data
    df_data = df_data / df_data.iloc[0, :]

    # we only analyze JPM data, select the JPM data file here
    # print(df_JPM, '...')

    JPM_prices = df_data[['JPM']]
    df_trade = indicator_frame(df_data[['SPY']])

    # Benchmark performance for the calculation

    benchmark_val = pd.DataFrame([start_share] + [0] * (len(trading_days) - 1),
                                 columns=[symbol], index=trading_days)
    benchmark_val = compute_portvals(benchmark_val, start_val=sv, commission=0, impact=0.005)
    benchmark_val /= benchmark_val.iloc[0, :]

    # Test: case
    ms = ManualStrategy()
    df_trades_hard = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv, bbr_top=0.91,
                                   bbr_bottom=-0.91, SMA_ratio_up=1.02, SMA_ratio_low=0.99)

    port_vals_hard = compute_portvals(df_trades_hard, start_val=sv, commission=0, impact=0.005)
    port_vals_hard /= port_vals_hard.iloc[0, :]

    # Test: case for strategy_learn
    learner = sl.StrategyLearner(verbose=False)

    learner.addEvidence(symbol=symbol, sd=sd, ed=ed, sv=sv)

    df_trades_strategy = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    port_vals_strategy = compute_portvals(df_trades_strategy, start_val=sv, commission=0, impact=0.005)
    port_vals_strategy /= port_vals_strategy.iloc[0, :]

    # Plot the data
    fig, ax = plt.subplots()
    benchmark_val.plot(ax=ax, color='green')
    port_vals_hard.plot(ax=ax, color='red')
    port_vals_strategy.plot(ax=ax, color='black')
    plt.legend(["Benchmark", "Manual Strategy", "Strategy Learner"])
    plt.title("Comparison of different method for in sample data")
    plt.xlabel("Dates")
    plt.ylabel("Normalizaed value")
    ax = plt.gca()
    ax.xaxis.grid(True, which='Major', linestyle='--')
    ax.yaxis.grid(True, which='Major', linestyle='--')
    plt.show()


"""
    for date, action in df_trades_hard[df_trades_hard[symbol] != 0].iterrows():
        if action[symbol] < 0:
            ax.axvline(date, color='green', alpha=0.5, linestyle='--', linewidth=1.1)
        elif action[symbol] > 0:
            ax.axvline(date, color='blue', alpha=0.5, linestyle='--', linewidth=1.1)
    plt.tight_layout()
    plt.show()

"""


