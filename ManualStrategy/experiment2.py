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
    # In-sample
    symbol = "JPM"
    sv = 100000
    commission = 0.0
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)
    period = pd.date_range(sd_in, ed_in)
    trading_days = get_data(["SPY"], dates=period).index

    ############################
    #
    # Cumulative return
    #
    ############################

    df_cum_ret = pd.DataFrame(
        columns=["Benchmark", "Manual Strategy", "QLearning Strategy"],
        index=np.linspace(0.0, 0.01, num=10)
    )
    for impact, _ in df_cum_ret.iterrows():
        print("Compare cumulative return against impact={}".format(impact))

        # Benchmark
        benchmark_trade = pd.DataFrame([1000] + [0] * (len(trading_days) - 1), columns=[symbol], index=trading_days)
        benchmark = compute_portvals(benchmark_trade, start_val=sv, commission=commission, impact=impact)
        benchmark /= benchmark.iloc[0, :]
        df_cum_ret.loc[impact, "Benchmark"] = cummu_return(benchmark)

        # Manual Strategy
        ms = ManualStrategy()
        df_trades_manual = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv, bbr_top=0.91,
                                        bbr_bottom=-0.91, SMA_ratio_up=1.01, SMA_ratio_low=0.99)
        portvals_manual = compute_portvals(df_trades_manual, start_val=sv, commission=commission, impact=impact)
        portvals_manual /= portvals_manual.iloc[0, :]
        df_cum_ret.loc[impact, "Manual Strategy"] = cummu_return(portvals_manual)

        # QLearning Strategy
        learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
        learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        df_trades_qlearning = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        portvals_qlearning = compute_portvals(df_trades_qlearning, start_val=sv, commission=commission, impact=impact)
        portvals_qlearning /= portvals_qlearning.iloc[0, :]
        df_cum_ret.loc[impact, "QLearning Strategy"] = cummu_return(portvals_qlearning)

    fig, ax = plt.subplots()
    df_cum_ret[["Benchmark"]].plot(ax=ax, color="g", marker="o")
    df_cum_ret[["Manual Strategy"]].plot(ax=ax, color="r", marker="o")
    df_cum_ret[["QLearning Strategy"]].plot(ax=ax, color="b", marker="o")
    plt.title("Cumulative return against impact on {} stock over in-sample period".format(symbol))
    plt.xlabel("Impact")
    plt.ylabel("Cumulative return")
    plt.grid()
    plt.tight_layout()
    plt.show()
    # plt.savefig("figures/experiment2_{}_cr_in_sample.png".format(symbol))


    nb_orders = pd.DataFrame(
        columns=["Benchmark", "Manual Strategy", "QLearning Strategy"],
        index= [0.2, 0.35, 0.5, 0.65, 0.8]
    )
    for impact, _ in nb_orders.iterrows():
        print ("Compare number of orders against impact={}".format(impact))

        # Benchmark
        benchmark_trade = pd.DataFrame([1000] + [0] * (len(trading_days) - 1), columns=[symbol], index=trading_days)
        nb_orders.loc[impact, "Benchmark"] = (np.abs(benchmark_trade[symbol]) > 0).sum()

        # Manual Strategy
        ms = ManualStrategy()
        df_trades_manual = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv, bbr_top=0.91,
                                        bbr_bottom=-0.91, SMA_ratio_up=1.01, SMA_ratio_low=0.99)
        nb_orders.loc[impact, "Manual Strategy"] = (np.abs(df_trades_manual[symbol]) > 0).sum()

        # QLearning Strategy
        learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)
        learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        df_trades_qlearning = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        nb_orders.loc[impact, "QLearning Strategy"] = (np.abs(df_trades_qlearning[symbol]) > 0).sum()

    fig, ax = plt.subplots()
    nb_orders[["Benchmark"]].plot(ax=ax, color="g", marker="o")
    nb_orders[["Manual Strategy"]].plot(ax=ax, color="r", marker="o")
    nb_orders[["QLearning Strategy"]].plot(ax=ax, color="b", marker="o")
    plt.title("Number of orders against impact on {} stock over in-sample period".format(symbol))
    plt.xlabel("Impact")
    plt.ylabel("Number of orders")
    ax = plt.gca()
    ax.xaxis.grid(True, which='Major', linestyle='--')
    ax.yaxis.grid(True, which='Major', linestyle='--')
    plt.grid()
    plt.tight_layout()
    plt.show()
    #plt.savefig("figures/experiment2_{}_norders_in_sample.png".format(symbol))