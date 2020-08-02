"""
Template for implementing QLearner  (c) 2015 Tucker Balch

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

Student Name: RUI REN (replace with your name)
GT User ID: rren34 (replace with your User ID)
GT ID: 903474021 (replace with your GT ID)
"""

import numpy as np
import random as rand
import pandas as pd
import matplotlib.pyplot as plt
import random

"""
for action: up, down, left and right
for the beginning, we will use the epsilon greedy strategy
for the confidence of q learning, then we random decay rate
"""

class QLearner(object):

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):
        # dyna = 200

        # number of states
        self.s = num_states
        # number of actions
        self.a = num_actions
        # learning rate
        self.alpha = alpha
        # discount values
        self.gamma = gamma
        # random rate --> epsilon greedy method
        self.rar = rar
        # random decay rate --> epsilon decay rate
        self.radr = radr
        # dyna number
        self.dyna = dyna
        self.verbose = verbose

        # initialization the q_table
        self.q_table = np.zeros((self.s, self.a))
        # memorization the previous states and actions
        self.old_state = 0
        self.old_action = 0

        if dyna > 0:
            # T<s, a, s'>
            self.T = np.zeros((self.s, self.a, self.s))
            self.T_count = np.full((self.s, self.a, self.s), 0.00001)
            # R<s, a>
            self.Reward = np.zeros((self.s, self.a))

    def querysetstate(self, s):

        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        # random choice for the action --> exploration
        if random.uniform(0., 1.) <= self.rar:
            action = rand.randint(0, self.a - 1)
        # choice the action from q_table --> exploitation
        else:
            action = np.argmax(self.q_table[s, :])

        self.old_action = action
        self.old_state = s
        # epsilon decay

        if self.verbose: print(f"s = {s}, a = {action}")
        return action

    def query(self, s_prime, r):
        """
        @ <s, a, s_prime, r>
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The immediate reward
        @returns: The selected action
        """

        # We try 2000 episodes

        # Step 2: update the q_table according to the new action
        self.q_table[self.old_state, self.old_action] = self.q_table[self.old_state, self.old_action] + self.alpha * (
                r + self.gamma * (np.max(self.q_table[s_prime, :])) - self.q_table[self.old_state, self.old_action])

        if self.dyna > 0:
            # No.1 make the direction for the T and R model

            self.T_count[self.old_state, self.old_action, s_prime] = self.T_count[
                                                                         self.old_state, self.old_action, s_prime] + 1
            self.T[self.old_state, self.old_action, :] = self.T_count[self.old_state, self.old_action,
                                                         :] / self.T_count[self.old_state, self.old_action, :].sum()
            self.Reward[self.old_state, self.old_action] = (1 - self.alpha) * self.Reward[
                self.old_state, self.old_action] + self.alpha * r

            # calculate the pobability of choose s_prime
            # vectorization for the calculation
            # get rid of the for--loop
            state_hall = np.random.randint(0, self.s, self.dyna, int)
            action_hall = np.random.randint(0, self.a, self.dyna, int)
            dyna_r = self.Reward[state_hall, action_hall]
            dyna_s_prime = np.argmax(self.T[state_hall, action_hall, :], axis=1)
            dyna_action = np.argmax(self.q_table[dyna_s_prime, :], axis=1)

            self.q_table[state_hall, action_hall] = self.q_table[state_hall, action_hall] + self.alpha * (
                        dyna_r + self.gamma *
                        self.q_table[dyna_s_prime, dyna_action] - self.q_table[state_hall, action_hall])

        action = self.querysetstate(s_prime)
        # we use the exploration and exploitation method for random decay
        # we use the reduce epsilon
        self.rar = self.rar * self.radr

        if self.verbose: print(f"s = {s_prime}, a = {action}, r={r}")
        return action

    def author(self):
        """
        author : RRen34
        """
        return "rren34"


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
