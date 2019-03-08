#!/usr/bin/env python

"""
Simulates a K-arm bandit model selection experiment using conventional bandit methods
"""

import os
import argparse
from arm import *
from multiarm import *
from epsilon_greedy import *
from ucb import *


def bandit(Msim, bandit_algo, num_arms=10, budget=40):

    # 1. Setup K-arm bandit problem
    arms = [Msim.sample_arm() for ii in range(num_arms)]
    resource = 0

    # 2. Initialize by pulling each arm once
    print("arm: initial reward => final reward")
    print("-----------------------------------")
    for k in xrange(len(arms)):
        reward = arms[k].pull()
        bandit_algo.update(k, reward)
        print("%d: %.4f => %.4f" %(k,reward,arms[k].final_reward()))
        resource += 1

    # 3. Start bandit exploration-exploitation
    print("budget set to %d steps in aggregate." %budget)
    best_reward = 0.0
    while resource < budget:
        chosen_arm = bandit_algo.select_arm()
        reward = arms[chosen_arm].pull()
        regret = abs(Msim.oracle_reward - reward)
        print("step=%d, chosen=%d, reward=%.4f, regret=%.4f" %(resource, chosen_arm, 
                                                               reward, regret))
        print(bandit_algo)
        bandit_algo.update(chosen_arm, reward)
        resource += 1

    print("step=%d, chosen=%d, reward=%.4f, regret=%.4f" %(resource, chosen_arm,
                                                           reward, regret))
    print(bandit_algo)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Model selection simulation with conventional K-arm Bandit methods")
    p.add_argument('-d', '--datadir', required=True,
                   help="data directory w/ Sockeye metrics files, e.g. data/wnmt18-de-en/")
    p.add_argument('-k', '--num_arms', required=True, type=int,
                   help="Number of arms/models (k) to try in a K-arm bandit")
    p.add_argument('-b', '--budget', required=True, type=int,
                   help="Total budget for K-arm bandit")
    p.add_argument('-a', '--algo', default='epsilon-greedy',
                   help="Specifies which k-arm bandit algorithm to run {epsilon-greedy, ucb1}")
    p.add_argument('-p', '--plot', action='store_true', default=False,
                  help="Generate learning curve plot showing MultiArm experiment result")
    p.add_argument('-r', '--randomize', action='store_true', default=False,
                  help="Randomize the sampling of models from metrics files")

    args = p.parse_args()

    # 1. Setup MultiArm simulation experiment
    Msim = MultiArm(args.datadir, args.randomize)
    #print("---- Initial Status ----\n" + str(Msim))

    # 1. Choose algorithm
    if args.algo == "epsilon-greedy":
        bandit_algo = EpsilonGreedy(args.num_arms)
    elif args.algo == "ucb1":
        bandit_algo = UCB1(args.num_arms)
    else:
        print("Unknown bandit algorithm: %s. Exiting" %(args.algo))
        exit()

    # 2. Run bandit algorithm and show results
    bandit(Msim, bandit_algo, args.num_arms, args.budget)
    print("---- Simuluation results ----\n" + str(Msim))

    if args.plot:
        Msim.plot(args.budget)
