#!/usr/bin/env python

"""
An arm is equivalent to a model's training run
Each time we pull this arm, we train by one more checkpoint/iteration/step
The arm is initilized by reading a metrics file
"""

from collections import defaultdict

class Arm():
    """The arm in a multi-arm bandit problem.
    Corresponds to a single model's learning curve extracted from Sockeye NMT metrics file

    :param metricsfile: metrics file generated by a Sockeye training process
    """

    def __init__(self, metricsfile):
        # 1. read metrics file
        self.name = metricsfile
        metrics = defaultdict(dict)
        with open(metricsfile, 'r') as fid:
            for line in fid:
                f = line.split()
                step = int(f.pop(0))
                metrics[step] = {k:float(v) for (k,v) in [x.split('=') for x in f]}

        self.curve = [0.0]
        self.current_step = 0 # actually step starts at 1

        # Create BLEU curve: curve[step] = max_{i=0,1,..,step}(bleu[i])
        # This defined to be monotonically-increasing, i.e. the best BLEU up to current step
        for step in range(1,len(metrics)+1):
            if 'bleu-val' in metrics[step]:
                self.curve.append(max(metrics[step]['bleu-val'], self.curve[step-1]))
                # find max step where bleu-val is seen (account for incompleteness in metricsfile)
                self.max_step = step
            else:
                self.curve.append(self.curve[step-1])


    def __str__(self):
        curve = self.get_curve()
        s1 = ", ".join(['%.4f']*len(curve)) % tuple(curve)
        s2 = "%s len=%d [ %s ]" % (self.name, len(curve), s1)
        return s2


    def pull(self, how_many=1):
        """Pull the arm for n times, i.e. train for n checkpoints
        
        :param how_many: how many times to pull the arm
        :return: reward after pulling (float)
        """
        try:
            self.current_step += how_many
            return self.current_reward()
        except:
            raise Exception("Invalid index: current=%s how_many=%s" %(self.current_step, how_many))


    def get_curve(self):
        """Get curve (e.g. monotonically-smoothed BLEU learning curve)
        
        :return: list of floats representing curve
        """
        return self.curve


    def final_reward(self):
        """Returns final reward (e.g. final BLEU) of a learning curve
        """
        return self.curve[self.max_step]


    def current_reward(self):
        """Returns reward at the current time step
        """
        s = min(self.current_step, self.max_step)
        return self.curve[s]

