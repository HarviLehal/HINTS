import numpy as np
import random
from Tree import Tree
from tqdm import tqdm


class HINTS():

    def __init__(self, x, levels, log_branch_factor, theta0, target, proposal, M, stepsize):
        self.tree = Tree(x, levels, log_branch_factor)   # Tree Object
        self.tree.build_tree()                           # Build Tree
        self.target = target                        # Target Distribution
        # self.tree_structure = tree.data_tree        # Data tree structure
        # self.leaves = tree.leaves                   # Data tree leaf nodes
        self.theta0 = theta0                        # Initial Parameter values
        self.logpdf = target.logpdf                 # Logpdf of Target
        self.proposal = proposal                    # Proposal Method
        self.M = M                                  # Number of Iterations
        self.plot = target.plot                     # Parameter Plotting
        self.levels = levels                        # Number of Levels
        self.log_branch_factor = log_branch_factor  # Log Branch Factor
        # self.design = tree.design                   # Design Matrix
        # self.scenarios = tree.scenarios             # Number of Scenarios
        self.stepsize = stepsize                    # Stepsize

    def prop(self, theta):                  # Theta previous parameter, Theta_n proposal parameter
        theta_n = self.proposal(theta, self.stepsize)
        return theta_n

    def ratio(self, x, theta, theta_n):
        a = self.logpdf(x, *theta_n)        # logpdf of proposal
        b = self.logpdf(x, *theta)          # logpdf of previous
        a = a-b                             # Acceptance Ratio
        a = np.exp(a)                       # Exponent
        u = np.random.uniform(0, 1, 1)
        if u <= a:
            # print("ACCEPTED")
            # print("theta_n: ", theta_n)
            return theta_n                  # Accept Proposal
        else:
            # print("REJECTED")
            # print("theta: ", theta)
            return theta                    # Reject Proposal

    def ratio_test(self, x, theta, theta_n):
        a = self.logpdf(x, *theta_n)        # logpdf of proposal
        b = self.logpdf(x, *theta)          # logpdf of previous
        a = a-b                             # Acceptance Ratio
        a = np.exp(a)                       # Exponent
        u = np.random.uniform(0, 1, 1)
        if u <= a:
            return 1                    # Accept Proposal
        else:
            # print("REJECTED")
            # print("theta: ", theta)
            return 0                    # Reject Proposal

    def rand_leaf_selection(self):              # Random Leaf Selection
        rand_leaf = random.choice(self.leaves)
        return rand_leaf

    def level_set(self, level):                 # Level Set
        level_set = []
        for x in self.tree_structure:
            if x.level == level:
                level_set.append(x)
        return level_set

    def common_parent(self, node):        # Common Parent Node Set
        common_parent_set = []
        level_set = self.level_set(node.level)  # Level Set
        for x in level_set:           # ADD SOMETHING TO FACTOR LEVELS IN
            if x.parent_id == node.parent_id:
                common_parent_set.append(x)
        return common_parent_set

    def init_leaf_set(self, node_ids):
        leaf_data = []
        for node_id in sorted(node_ids):
            node = self.tree_structure[node_id]
            if node.level == self.levels:
                leaf_data.append(node)
        return leaf_data

    def parent(self, node):
        level_set = self.level_set(node.level - 1)  # Level Set
        for x in level_set:           # ADD SOMETHING TO FACTOR LEVELS IN
            if x.node_id == node.parent_id:
                return x

    def sampler(self):
        thetas = []                                                         # Initialise theta parameter list
        theta_level = [[] for i in range(self.levels+1)]
        theta_level[self.levels].append(self.theta0)
        thetas.append(self.theta0)                                          # Append initial theta
        for iter in tqdm(range(self.M-1)):                                  # HINTS Iterations
            init_leaf = self.tree.rand_leaf_selection()                     # Random Leaf Selection
            parent = self.tree.parent(init_leaf)                            # Set Parent Node to initial leaf
            for level in reversed(range(0, self.levels+1)):                 # Iterate through levels
                if level == self.levels:                                    # If level is lowest
                    common_parent_set = self.tree.common_parent(init_leaf)  # Common Parent Set of initial leaf
                else:
                    common_parent_set = self.tree.common_parent(parent)     # Common Parent Set of next level

                for index, node in enumerate(common_parent_set):            # Iterate through common parent set
                    thetan = {}
                    for j in range(len(self.theta0)):                       # Iterate through parameters
                        thetan[j] = self.prop(thetas[-1][j])                # Propose new parameter
                    thetan = self.ratio(node.data, list(thetas[-1].values()), list(thetan.values()))  # Acceptance Ratio
                    p = {}
                    for j in range(len(self.theta0)):                       # Iterate through parameters
                        p[j] = thetan[j]                                    # Append new theta
                    thetas.append(p)                                        # Append new theta
                    theta_level[level].append(p)
                parent = self.tree.parent(common_parent_set[0])             # Parent Node
                if parent is not None:
                    thetan = self.ratio(parent.data, list(thetas[-1].values()), list(thetas[-len(common_parent_set)].values()))  # Acceptance Ratio
                else:
                    pass
        self.thetas = thetas
        self.theta_level = theta_level

