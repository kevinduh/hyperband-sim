#!/usr/bin/env python

"""
Model Selection as Multi-Arm Bandit Experiment: This class encapsulates 
a set of arms/models and provides convenience functions for summarizing 
the playback of the training experiments. Note that if the number of real
models prepared in the datadir is less than the number of arms requested, 
this class will re-sample and duplicate some arms. 
"""

from arm import *
import os
import random
from copy import copy
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


class MultiArm():

    def __init__(self, datadir, randomize=False, file_limit=10000):
        """Read raw metrics files and create 'arms' for simulation

        :param datadir: root directory to search for metrics files
        :param randomize: if true, shuffle order of metrics files for randomized experiments
        :param file_limit: limit on number of metrics files to read in simulation

        :return: list of Arm() representing the multi-arm bandit problem
        :return: oracle reward (float) from the best arm/model
        """

        # 1. Gather metrics files
        metrics_files = [os.path.join(dp,f) for dp,dn,fn in os.walk(datadir) for f in fn if f == 'metrics']
        self.num_arms_used = min(len(metrics_files), file_limit)

        msg = "Found %d metrics files in %s" % (len(metrics_files), datadir)
        #msg += "; Using %d models/arms in simulation:" % (self.num_arms_used)
        print(msg)

        # 2. Create Arm() object for each metrics file
        if randomize:
            random.shuffle(metrics_files)
        self.arms = [Arm(m) for m in metrics_files[:self.num_arms_used]]

        # 3. Print statistics
        self.oracle_reward = 0.0
        self.oracle_arm = 0
        #print("\n==== Details of each arm/model ====")
        for k in range(len(self.arms)):
            # rename arm with integer id for easy identification
            self.arms[k].name = str(k) + ":" + self.arms[k].name
            #print(self.arms[k])
            if self.oracle_reward < self.arms[k].final_reward():
                self.oracle_reward = self.arms[k].final_reward()
                self.oracle_arm = k
        print("oracle arm=%d, oracle reward=%f\n" %(self.oracle_arm,
                                                    self.oracle_reward))

        # 4. Create generator
        def arm_generator():
            # track duplicate resampling of same arm, d[arm name]->[list of id]
            self.duplicates = defaultdict(list)
            i = -1
            while True:
                i += 1
                if i >= self.num_arms_used:
                    a = copy(self.arms[i%self.num_arms_used])
                    a.current_step = 0
                    self.arms.append(a)
                    self.duplicates[a.name].append(i)
                else:
                    a = self.arms[i%self.num_arms_used]
                yield a
        self.arm_generator = arm_generator()

        # End __init__


    def sample_arm(self):
        """Returns an arm (simulates training a model)
        """
        return self.arm_generator.next()


    def __str__(self):
        """Prints snapshot of current Multi-Arm Bandit experiment status
        
        :return: string
        """
        s = "=== Multi-Arm experiment status ===\n"
        s += "id current_step max_step current_reward final_reward name\n"
        for ii, a in enumerate(self.arms):
            s += "%d %d %d %f %f %s\n" %(ii, a.current_step, a.max_step,
                                         a.current_reward(),
                                         a.final_reward(), a.name)

        d = self.stats()
        s += "chosen_arm=%d final_reward=%f " %(d['best_arm'], d['best_final_reward'])
        s += "oracle=%f regret=%f resource=%d " %(self.oracle_reward,
                                                 d['regret'],
                                                 d['resource'])
        s += "num_arms_examined=%d\n" % d['num_arms_examined']
        return s


    def stats(self):
        """Get current Multi-Arm Bandit experiment statistics

        :return: dictionary with best_arm, regret, resource, etc.
        """
        stats_dict = {}
        stats_dict['best_arm'] = 0
        stats_dict['best_current_reward'] = 0.0
        stats_dict['resource'] = 0
        stats_dict['num_arms_examined'] = 0

        for ii, a in enumerate(self.arms):
            stats_dict['resource'] += a.current_step
            if a.current_reward() > stats_dict['best_current_reward']:
                stats_dict['best_arm'] = ii
                stats_dict['best_current_reward'] = a.current_reward()
            if a.current_step > 0:
                stats_dict['num_arms_examined'] += 1

        stats_dict['best_final_reward'] = self.arms[stats_dict['best_arm']].final_reward()
        stats_dict['regret'] = self.oracle_reward - stats_dict['best_final_reward']

        return stats_dict


    def plot(self, max_step, filename="tmp.png"):
        """Plot the Multi-Arm Bandit simulation

        The full curve is plotted for each arm/model
        Note curves are smoothed monotonically-increasing version of the real BLEU curve
        Current step where the training stops is indicated by a dot

        :param max_step: max steps to draw in figure
        :param filename: output filename for plot
        """
        for a in self.arms:
            c = a.get_curve()
            m = min(len(c), max_step+1)
            plt.plot(range(m),c[:m])
            plt.scatter(a.current_step, a.current_reward())

        d = self.stats()
        title = "#arms=%d\n" %(len(self.arms))
        title += "chosen_arm=%d\n" %(d['best_arm'])
        title += "current_reward=%f\n" %(d['best_current_reward'])
        title += "final_reward=%f\n" %(d['best_final_reward'])
        title += "regret=%f resource=%d" %(d['regret'], d['resource'])
        plt.legend(loc='lower right', title=title)
        plt.ylabel('BLEU (validation set)')
        plt.xlabel('steps')
        #plt.show()
        plt.savefig(filename)


