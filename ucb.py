import math

class UCB1():
    """UCB1 algorithm for K-arm bandits
    """
    def __init__(self, n_arms):
        self.counts = [0 for x in range(n_arms)]
        self.values = [0.0 for x in range(n_arms)]


    def __str__(self):
        s = "UCB1 algorithm status:\n"
        s += "  pull count of each arm: %s\n" %(self.counts)
        s += "  estimated value of each arm: %s\n" %(self.values)
        return s


    def select_arm(self):
        n_arms = len(self.counts)
        for a in xrange(n_arms):
            if self.counts[a] == 0:
                return a

        total_counts = sum(self.counts)
        ucb_values = [0.0 for a in range(n_arms)]
        for a in xrange(n_arms):
            bound = math.sqrt((2*math.log(total_counts))/float(self.counts[a]))
            ucb_values[a] = self.values[a] + bound
            print a,bound," ",
        m = max(ucb_values)
        return ucb_values.index(m)


    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        #alpha = 1.0/float(self.counts[chosen_arm])
        alpha = 1.0
        value = self.values[chosen_arm]
        new_value = value + alpha * (reward - value)
        self.values[chosen_arm] = new_value

