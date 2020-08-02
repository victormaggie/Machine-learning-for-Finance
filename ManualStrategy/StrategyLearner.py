"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: rren34 (replace with your User ID)
GT ID: 903474021 (replace with your GT ID)
"""

import datetime as dt
import pandas as pd
from util import *
import numpy as np
import random
from indicators import *
import QLearner as ql
import matplotlib.pyplot as plt
from marketsimcode import *
import time

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0, min_iter=20, max_iter=100):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.n_bins = 10
        self.num_states = self.n_bins ** 3
        self.mean = None
        self.std = None
        self.divergence_bins = None
        self.bbr_bins = None
        self.SMAr_bins = None
        self.min_iter = min_iter
        self.max_iter = max_iter

    # this method should create a QLearner, and train it for trading
    # this is for training
    def normalize_indicators(self, df, mu, sigma):
        return (df - mu) / sigma

    def discretilization(self, data, bins, n_bins):
        # use the numpy digitalize function to discretilization
        integer = np.digitize(data, bins, right=True)-1
        integer = np.clip(integer, 0, n_bins-1)
        return integer

    def get_dataframe(self, symbol, dates):
        data = get_data([symbol], dates)
        prices = data[symbol]
        # print('...\n', prices)
        # print('...\n', prices.iloc[0])
        prices = prices / prices.iloc[0]
        benchmark = data['SPY']
        benchmark /= benchmark.iloc[0]
        trading_days = benchmark.index
        return prices, trading_days

    def get_state(self, symbol, sd, ed):
        # get the in-sample data
        prices, trading_days = self.get_dataframe(symbol, pd.date_range(sd, ed))
        daily_return = avg_daily_return(prices)

        # indicators
        indicators = indicator_frame(prices.to_frame(symbol))
        self.mean = indicators.mean()
        self.std = indicators.std()

        if (self.std == 0).any():
            self.std = 1

        std_indicators = self.normalize_indicators(indicators, self.mean, self.std)
        divergence = std_indicators["macd_divergence"]
        bbr = std_indicators["bbr"]
        SMA_ratio = std_indicators['SMA_ratio']

        # Discretize the into bins
        ## MACD
        _, self.divergence_bins = pd.qcut(divergence, self.n_bins, retbins=True, labels=False)
        divergence_ind = self.discretilization(divergence, self.divergence_bins, self.n_bins)
        divergence_ind = pd.Series(divergence_ind, index=indicators.index)

        ## Bollinger bands
        _, self.bbr_bins = pd.qcut(bbr, self.n_bins, retbins=True, labels=False)
        bbr_ind = self.discretilization(bbr, self.bbr_bins, self.n_bins)
        bbr_ind = pd.Series(bbr_ind, index=indicators.index)

        ## SMA
        _, self.SMA_ratio_bins = pd.qcut(SMA_ratio, self.n_bins, retbins=True, labels=False)
        SMA_ratio_ind = self.discretilization(SMA_ratio, self.SMA_ratio_bins, self.n_bins)
        SMA_ratio_ind = pd.Series(SMA_ratio_ind, index=indicators.index)

        # Compute state of in-sample data
        discretized_indicators = pd.DataFrame(index=indicators.index)
        discretized_indicators['macd_divergence'] = divergence_ind.values
        discretized_indicators['bbr'] = bbr_ind.values
        discretized_indicators['SMA_ratio'] = SMA_ratio_ind.values
        discretized_indicators['mapping'] = divergence_ind.astype(str) + \
                                            bbr_ind.astype(str) + SMA_ratio_ind.astype(str)

        discretized_indicators['state'] = discretized_indicators['mapping'].astype(np.int)
        states = discretized_indicators['state']
        return states, daily_return

    def addEvidence(self, symbol="JPM",
                    sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 1, 1),
                    sv=10000):

        # state and reward
        states, daily_returns = self.get_state(symbol, sd, ed)

        self.learner = ql.QLearner(
            num_states=self.num_states,
            num_actions=3,    # long, short, do nothing
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=self.verbose
        )

        # Training loop
        i = 0
        converged = False
        df_trades_previous = None
        while (i <= self.min_iter) or (i <= self.max_iter and not converged):
            # iteration number and convergence check
            action = self.learner.querysetstate(states.iloc[0])

            holding = 0
            df_trades = pd.Series(index=states.index)
            # iterate the state as Series
            for day, state in states.iteritems():
                reward = holding * daily_returns.loc[day]   # reward
                if action != 2:
                    reward *= (1 - self.impact)
                action = self.learner.query(state, reward)
                if action == 0:   # Short
                    df_trades.loc[day] = {
                        -1000: 0,
                        0: -1000,
                        1000: -2000,
                    }.get(holding)
                elif action == 1:  # Long
                    df_trades.loc[day] = {
                        -1000: 2000,
                        0: 1000,
                        1000: 0
                    }.get(holding)
                elif action == 2:    # do Nothing
                    df_trades.loc[day] = 0
                else:
                    raise Exception("Unknown trading action to make: {}".format(action))

                holding += df_trades.loc[day]

            # check for convergence
            # if the df_trades convergence for the calculation
            if (df_trades_previous is not None) and (df_trades.equals(df_trades_previous)):
                converged = True

            df_trades_previous = df_trades
            i += 1

    # this method should use the existing policy and test it against new data
    # test the out sample data

    def testPolicy(self, symbol="JPM",
        sd=dt.datetime(2010,1,1),
        ed=dt.datetime(2011,12,31),
        sv=100000):

        states, daily_returns = self.get_state(symbol, sd, ed)

        holding = 0
        df_trades = pd.Series(index=states.index)
        for day, state in states.iteritems():
            action = self.learner.querysetstate(state)
            if action == 0: # short
                df_trades.loc[day] = {
                    -1000: 0,
                    0: -1000,
                    1000: -2000,
                }.get(holding)

            elif action == 1:  # Long
                df_trades.loc[day] = {
                    -1000: 2000,
                    0: 1000,
                    1000: 0
                }.get(holding)

            elif action == 2: # Do nothing
                df_trades.loc[day] = 0

            else:
                raise Exception("Unknown trading action to make: {}".format(action))

            holding += df_trades.loc[day]
        return df_trades.to_frame(symbol)

    def author(self):
        return "rren34"

def test_code():
    sv = 100000
    commission = 0.0
    impact = 0.0
    sd_in_train = dt.datetime(2008, 1, 1)
    ed_in_train = dt.datetime(2009, 12, 31)

    sd_out_test = dt.datetime(2010, 1, 1)
    ed_out_test = dt.datetime(2011, 12, 31)

    # the last day --> short 1000 shares
    bench = lambda trading_days: [1000] + [0] * (len(trading_days) - 2) + [-1000]

    for symbol in ["JPM", "ML4T-220", "AAPL", "UNH", "SINE_FAST_NOISE"]:
        print('###############')
        print('{}'.format(symbol))
        print('###############')

        ############################

        # In sample
        period = pd.date_range(sd_in_train, ed_in_train)
        trading_days = get_data(['SPY'], dates=period).index

        # Benchmark in-sample
        benchmark_trade = pd.DataFrame(bench(trading_days), columns=[symbol], index=trading_days)
        benchmark_trade = compute_portvals(benchmark_trade, start_val=sv, commission=commission, impact=impact)
        benchmark = benchmark_trade / benchmark_trade.iloc[0, :]

        # Train
        print('Training....')
        learner = StrategyLearner(verbose=False, impact=impact, commission=commission)
        start = time.time()
        learner.addEvidence(symbol=symbol, sd=sd_in_train, ed=ed_in_train, sv=sv)
        print("addEvidence() on in-sample completes in {} sec".format(time.time() - start))

        # Test : in-sample
        print('Testing in-sample...')
        start = time.time()
        df_trades = learner.testPolicy(symbol=symbol, sd=sd_in_train, ed=ed_in_train, sv=sv)
        print('testPolicy() on in-sample completes in in {} sec'.format(time.time() - start))
        portvals_train = compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)
        portvals_train = portvals_train / portvals_train.iloc[0, :]

        if cummu_return(portvals_train) > 1.0:
            print("cummulative return in-sample greater than 1.0: {}".format(cummu_return(portvals_train)))
        else:
            print("ERROR cummulative return in-sample NOT greater than 1.0: {}".format(cummu_return(portvals_train)))
        if cummu_return(portvals_train) > cummu_return(benchmark):
            print("cummulative return in-sample greater than benchmark: {} vs {}".format(cummu_return(portvals_train), cummu_return((benchmark))))
        else:
            print("ERROR cummulative return in-sample NOT greater than 1.0: {}".format(cummu_return(portvals_train), cummu_return((benchmark))))

        fig, ax = plt.subplots()
        benchmark.plot(ax=ax, color='green')
        portvals_train.plot(ax=ax, color='red')
        plt.legend(['Benchmark', "QLearner Strategy"])
        plt.title('QLearner strategy on {} stock in-sample period'.format(symbol))
        plt.xlabel('Dates')
        plt.ylabel('Normalized value')
        for day, order in df_trades[df_trades[symbol] != 0].iterrows():
            if order[symbol] < 0:    # Short
                ax.axvline(day, color='k', alpha=0.5)
            elif order[symbol] > 0:  # Long
                ax.axvline(day, color='b', alpha=0.5)
        plt.tight_layout()
        plt.show()
        # plt.savefig("figures/QLearner_{}_in_sample.png".format(symbol))

        ###############################
        ###
        ###     Out of Sample
        ###

        # test : out-of-sample
        print("Testing out of sample ...")
        period = pd.date_range(sd_out_test, ed_out_test)
        trading_days = get_data(['SPY'], dates=period).index

        # Benchmark out of sample
        benchmark_trade = pd.DataFrame(bench(trading_days), columns=[symbol], index=trading_days)
        benchmark = compute_portvals(benchmark_trade, start_val=sv, commission=commission, impact=impact)
        benchmark /= benchmark.iloc[0, :]

        # testPolicy out of sample
        start = time.time()
        df_trades = learner.testPolicy(symbol=symbol, sd=sd_out_test, ed=ed_out_test, sv=sv)
        print("testPolicy() on out-of-sample completes in in {} sec".format(time.time() - start))
        portvals_test = compute_portvals(df_trades, start_val=sv, commission=commission, impact=impact)
        portvals_test = portvals_test / portvals_test.iloc[0, :]

        if cummu_return(portvals_test) > 1.0:
            print("cumulative return out-of-sample greater than 1.0: {}".format(cummu_return(portvals_test)))
            cummu_return(portvals_test)
        else:
            print("ERROR cumulative return out-of-sample NOT greater than 1.0: {}".format(cummu_return(portvals_test)))
            cummu_return(portvals_test)
        if cummu_return(portvals_test) > cummu_return(benchmark):
            print("cumulative return out-of-sample greater than benchmark: {} vs {}".format( cummu_return(portvals_test), cummu_return(benchmark)))
            cummu_return((portvals_test))
        else:
            print("ERROR cumulative return out-of-sample NOT greater than benchmark: {} vs {}".format(cummu_return(portvals_test), cummu_return(benchmark)))
            cummu_return(portvals_test)

        fig, ax = plt.subplots()
        benchmark.plot(ax=ax, color='green')
        portvals_test.plot(ax=ax, color='red')
        plt.legend(['Benchmark', 'QLearner Strategy'])
        plt.title("QLearner strategy on {} stock over out of sample period".format(symbol))
        plt.xlabel('Dates')
        plt.ylabel('Normalized value')
        for day, order in df_trades[df_trades[symbol] != 0].iterrows():
            if order[symbol] < 0:
                ax.axvline(day, color='k', alpha=0.5)
            elif order[symbol] > 0:
                ax.axvline(day, color='b', alpha=0.5)
        plt.tight_layout()
        plt.show()
        # plt.savefig('figures/Qlearner_{}_out_of_sample.png'.format(symbol))

def tests():
    symbol = "JPM"
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)
    sv = 100000

    # check the test policy() code
    learner = StrategyLearner(verbose=False)
    learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    df_trades_previous = None

    for i in range(20):
        df_trades = learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
        if (df_trades_previous is not None) and not (df_trades_previous.equals(df_trades)):
            raise Exception('testPolicy() does not always return the same result')
    print("OK: testPolicy() always returns the same result")

    # testPolicy() method should be much faster than your addEvidence() method
    learner = StrategyLearner(verbose=False)
    start = time.time()
    learner.addEvidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    time_addEvidence = time.time() - start

    start = time.time()
    learner.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    time_testPolicy = time.time() - start

    if time_testPolicy >= time_addEvidence:
        print("testPolicy() is not faster than addEvidence !!! {} VS {}".format(time_testPolicy, time_addEvidence))
    print("testPolicy() is not faster than addEvidence !!! {} VS {}".format(time_testPolicy, time_addEvidence))

if __name__=="__main__":
    print("One does not simply think up a strategy")
