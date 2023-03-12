import numpy as np
from scipy.stats import multivariate_normal
import random
from Tree import Tree
from MCMC import MCMC
from Target import *
from Proposal import Proposal
from sklearn.datasets import make_spd_matrix
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

    def sampler(self):
        thetas = []                                                         # Initialise theta parameter list
        theta_level = [[] for i in range(self.levels+1)]
        theta_level[self.levels].append(self.theta0)
        thetas.append(self.theta0)                                          # Append initial theta
        for iter in tqdm(range(self.M-1)):                                  # HINTS Iterations
            init_leaf = self.tree.rand_leaf_selection()                     # Random Leaf Selection
            parent = self.tree.parent(init_leaf)
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
        self.thetas = thetas
        # self.plot(thetas)
        self.theta_level = theta_level
        # for i in range(len(theta_level)):
        #     self.plot(theta_level[i])
        self.plot(theta_level[0])

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
        parentnode = []
        level_set = self.level_set(node.level - 1)  # Level Set
        for x in level_set:           # ADD SOMETHING TO FACTOR LEVELS IN
            if x.node_id == node.parent_id:
                return x





# TEST DATA SET

mu = np.array([2, 4, 6])
sigma = np.eye(3)

x = multivariate_normal.rvs(mu, sigma, 20480)

mu0 = np.array([5, 7, 9])
sigma0 = np.eye(3)*2
theta0 = {0: mu0, 1: sigma0}

# WHACKY TEST DATA SET

# mu = np.array([3, 5, 7, 9])                                                                 # Sampling Mean
# sigma = make_spd_matrix(n_dim=4) * 10
# x = multivariate_normal.rvs(mu, sigma, 1024)
# mu0 = np.array([4, 8, 12, 16])
# sigma0 = make_spd_matrix(n_dim=4) * 20
# theta0 = {0: mu0, 1: sigma0}

z = HINTS(x, 3, 2, theta0, Gaussian, Proposal.rw3, 500, 0.01)
z.sampler()

z = HINTS(x, 4, 2, theta0, Gaussian, Proposal.rw3, 500, 0.01)
z.sampler()

z = HINTS(x, 5, 2, theta0, Gaussian, Proposal.rw3, 500, 0.01)
z.sampler()


z2 = MCMC(x, theta0, Gaussian, Proposal.rw3, 0.01)
z2.mcmc(2000)
