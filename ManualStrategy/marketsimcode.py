"""
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

import pandas as pd
from datetime import datetime
import pandas as pd
from util import get_data, plot_data


def compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # STEP 1: df_price data frame
    sd = df_trades.index.min()
    ed = df_trades.index.max()
    index = pd.date_range(sd, ed)

    # List the trading days
    SPY_benchmark = get_data(['SPY'], dates=index)

    trading_days = []
    for day in index:
        if day in SPY_benchmark.index:
            trading_days.append(day)

    # Fetch stocks prices
    symbols = df_trades.columns.tolist()
    stock = {}
    for s in symbols:
        stock[s] = get_data([s], dates=index, colname='Adj Close')
        stock[s] = stock[s].resample("D").fillna(method="ffill")
        stock[s] = stock[s].fillna(method="bfill")

    port_vals = pd.DataFrame(index=trading_days, columns=["port_val"] + symbols)

    # Compute portfolio value for each trading day in the period
    curr_val = start_val
    pre_date = None
    for date in trading_days:

        if pre_date is not None:
            port_vals.loc[date, :] = port_vals.loc[pre_date, :]
            port_vals.loc[date, "port_val"] = 0
        else:
            port_vals.loc[date, :] = 0

        # Execute orders
        if date in df_trades.index:
            date_orders = df_trades.loc[[date]]
            for stk in date_orders.columns:
                order = date_orders.iloc[0].loc[stk]
                shares = abs(order)
                stock_price = stock[stk].loc[date, stk]

                if order < 0:     # Short the stock
                    stock_price *= (1 - impact)
                    curr_val = curr_val + stock_price * shares
                    curr_val = curr_val - commission
                    port_vals.loc[date, stk] = port_vals.loc[date, stk] - shares

                elif order > 0:   # long the stock
                    stock_price *= (1 + impact)
                    curr_val = curr_val - stock_price * shares
                    curr_val = curr_val - commission
                    port_vals.loc[date, stk] = port_vals.loc[date, stk] + shares

        # Update portfolio value
        for s in symbols:
            stock_price = stock[s].loc[date, s]
            port_vals.loc[date, "port_val"] += port_vals.loc[date, s] * stock_price
        port_vals.loc[date, "port_val"] += curr_val
        # update the current day for the calculation
        pre_date = date

    port_vals = port_vals.sort_index(ascending=True)
    return port_vals.iloc[:, 0].to_frame()


def author():
    return "rren34"
