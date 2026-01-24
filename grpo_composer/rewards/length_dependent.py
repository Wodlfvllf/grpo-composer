"""
GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach for
Concise Mathematical Reasoning in Language Models

Length-Dependent Accuracy Reward
The core idea is to reward correct completions not
uniformly but in proportion to their relative con-
ciseness. Given a question q and a set of model-
generated responses {oi}, we first isolate the subset
of correct responses and compute the mean µand
standard deviation σ of their token lengths.

z_i = (length(o_i) - µ) / (σ + epsilon)

Raccuracy(o|q) = exp(−αz), if o is correct,
0, if o is incorrect.

"""

class GRPOLEADReward:
    def __init__(self, responses, labels, alpha, epsilon):
        self.responses = responses
        self.labels = labels
        self.alpha = alpha
        self.epsilon = epsilon

    def compute_rewards(self):
        mean_length = np.mean([len(response) for response in self.responses])
        std_length = np.std([len(response) for response in self.responses])

        rewards_standardized = []
        for idx, response in enumerate(self.responses):
            if self.labels[idx] == 1:
                z = (len(response) - mean_length) / (std_length + self.epsilon)
                reward = np.exp(-self.alpha * z)
            else:
                reward = -1
            rewards_standardized.append(reward)

        return np.array(rewards_standardized)