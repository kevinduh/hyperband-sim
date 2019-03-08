import random

class EpsilonGreedy():
    """Epsilon-Greedy algorithm for K-arm bandits
    """
    def __init__(self, n_arms, epsilon=0.1):
        self.epsilon = epsilon
        self.counts = [0 for x in range(n_arms)]
        self.values = [0.0 for x in range(n_arms)]


    def __str__(self):
        s = "Espilon-Greedy algorithm status:\n"
        s += "  pull count of each arm: %s\n" %(self.counts)
        s += "  estimated value of each arm: %s\n" %(self.values)
        return s

        
    def select_arm(self):
        if random.random() > self.epsilon:
            m = max(self.values)
            return self.values.index(m)
        else:
            return random.randrange(len(self.values))


    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        #alpha = 1.0/float(self.counts[chosen_arm])
        alpha = 1.0
        value = self.values[chosen_arm]
        new_value = value + alpha * (reward - value)
        self.values[chosen_arm] = new_value

