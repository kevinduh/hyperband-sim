#!/usr/bin/env python

"""
Simulates a K-arm bandit model selection experiment using the Hyperband method

Hyperband code is adapted from:
https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
"""

import os
import argparse
from arm import *
from multiarm import *
from math import log, ceil
import numpy as np


def hyperband(Msim, max_iter, eta=3):

    
    print("---- Hyperband starting ----")
    logeta = lambda x: log(x)/log(eta)

    # s_max+1 is the number of unique executions of Successive Halving inner loop
    s_max = int(logeta(max_iter))  

    # B is the estimated resource budget for an inner loop, but this is not exact
    B = (s_max+1)*max_iter

    # total_resource tracks the overall number of checkpoints/iterations run in Hyperband
    total_resource = 0

    # Outerloop for Hyperband. Run s_max+1 versions of Successive Halving.
    # Each s represents a different "bracket". It trades off #models considered (n) 
    # with minimum #iterations run before Successive Halving starts to prune (r)
    # e.g. (n=3, r=2) means 3 models are considered, and pruning starts after checkpoint 2
    # The amount of pruning is determined by eta (default 3, or 2 both reasonable)
    for s in reversed(range(s_max+1)): 

        # initial number of arms/models
        n = int(ceil(int(B/max_iter/(s+1))*eta**s)) 

        # initial number of iterations to run model
        # note original code assumed max_iter was divisible by eta:
        # i.e. r = max_iter*eta**(-s)
        r = int(max_iter*eta**(-s)) 

        msg = "Successive halving bracket s=%d: total arms n=%d, min step r=%d" %(s,n,r)
        print(msg)

        # Sample n arms/models
        T = [Msim.sample_arm() for ii in range(n)]

        # total steps/iterations/checkpoint so far in the Successive Halving inner loop
        resource_inner = 0 

        # keeps track of previous iteration
        prev_iter = 0

        # Successive Halving inner loop
        for i in range(s+1):

            # n_i models will be advanced to the checkpoint at r_i
            n_i = n*eta**(-i) 
            r_i = r*eta**(i)

            # num_pull is the number of iterations/checkpoints advanced
            num_pull = r_i - prev_iter
            prev_iter = r_i
            resource_inner += len(T)*num_pull
            total_resource += len(T)*num_pull

            # logging...
            msg = "  n_i=%d arms, evaluated %d steps (current_step r_i=%d), innerloop_resource=%d" %(n_i, num_pull,r_i,resource_inner)
            print(msg)

            # records reward after iterations are advanced
            rewards = [a.pull(num_pull) for a in T]

            # choose only the best ones and prune the remaining arms/models
            chosen_arms = np.argsort(rewards)[int( n_i/eta ):]
            T = [ T[j] for j in chosen_arms ]

        #print(Msim)

    print("Total resource usage: %d\n" %total_resource)



if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Model selection simulation with Hyperband")
    p.add_argument('-d', '--datadir', required=True,
                   help="data directory w/ Sockeye metrics files, e.g. data/wnmt18-de-en/")
    p.add_argument('-i', '--max_iter', required=True, type=int,
                   help="Hyperband max_iter parameter. Max number of checkpoints to run")
    p.add_argument('-e', '--eta', type=int, default=3,
                   help="1/eta models are kept in Hyperband's Successive Halving.")
    p.add_argument('-p', '--plot', action='store_true', default=False,
                  help="Generate learning curve plot showing MultiArm experiment result")
    p.add_argument('-r', '--randomize', action='store_true', default=False,
                  help="Randomize the sampling of models from metrics files")

    args = p.parse_args()
    
    # 1. Setup MultiArm simulation experiment
    Msim = MultiArm(args.datadir, args.randomize)
    #print("---- Initial Status ----\n" + str(Msim))

    # 2. Run Hyperband algorithm and show results
    hyperband(Msim, args.max_iter, args.eta)
    print("---- Simuluation results ----\n" + str(Msim))

    if args.plot:
        Msim.plot(args.max_iter)
