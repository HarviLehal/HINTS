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
        tree = Tree(x, levels, log_branch_factor)   # Tree Object
        tree.build_tree()                           # Build Tree
        self.target = target                        # Target Distribution
        # tree.leaf_nodes()                           # Leaf Nodes
        self.tree_structure = tree.data_tree             # Data tree structure
        self.leaves = tree.leaves                   # Data tree leaf nodes
        self.theta0 = theta0                        # Initial Parameter values
        self.logpdf = target.logpdf                 # Logpdf of Target
        self.proposal = proposal                    # Proposal Method
        self.M = M                                  # Number of Iterations
        self.plot = target.plot                     # Parameter Plotting
        self.levels = levels                        # Number of Levels
        self.log_branch_factor = log_branch_factor  # Log Branch Factor
        self.design = tree.design                   # Design Matrix
        self.scenarios = tree.scenarios             # Number of Scenarios
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

    # def common_parent_node_ids(self, common_parent_set):            # Common Parent Node Ids
    #     node_ids = [node.node_id for node in common_parent_set]
    #     return sorted(node_ids)

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

    def sampler(self):
        thetas = [[] for i in range(self.levels+1)]               # Initialise theta level list
        total = []
        propo = []
        total.append(self.theta0)
        thetas[self.levels].append(self.theta0)                           # Append initial theta
        for iter in tqdm(range(self.M-1)):                      # HINTS Iterations
            init_leaf = self.rand_leaf_selection()                # Random Leaf Selection
            parent = self.parent(init_leaf)
            for level in reversed(range(1, self.levels+1)):                    # Iterate through levels   (DONT FORGET TO DO self.levels - level so it starts at the bottom))
                if level == self.levels:                # If level is lowest
                    common_parent_set = self.common_parent(init_leaf)  # Common Parent Set
                else:
                    common_parent_set = self.common_parent(parent)  # Common Parent Set
                for index, node in enumerate(common_parent_set):                      # Iterate through common parent set
                    thetan = {}
                    for j in range(len(self.theta0)):               # Iterate through parameters
                        thetan[j] = self.prop(thetas[level][iter+index][j])  # Propose new parameter
                    p = {}
                    for j in range(len(self.theta0)):
                        p[j] = thetan[j]                # Append new theta
                    thetas[level].append(p)                         # Append new theta
                    propo.append(p)
                    thetan = self.ratio(node.data, list(thetas[level][iter+index].values()), list(thetan.values()))  # Acceptance Ratio
                    p = {}
                    for j in range(len(self.theta0)):
                        p[j] = thetan[j]                # Append new theta
                    thetas[level].append(p)                         # Append new theta
                    total.append(p)
                parent = self.parent(common_parent_set[0])                     # Parent Node
                thetan = self.ratio(parent.data, list(thetas[level][iter].values()), list(thetas[level][iter+len(common_parent_set)].values()))  # Acceptance Ratio
                p = {}
                for j in range(len(self.theta0)):
                    p[j] = thetan[j]
                thetas[level-1].append(p)
                total.append(p)
        self.thetas = thetas
        self.total = total
        self.plot(self.total)
        # self.propo = propo
        # self.plot(self.propo)
        # for i in range(len(self.thetas)):
        #     self.plot(self.thetas[i])


# TEST DATA SET

mu = np.array([2, 4, 6])
sigma = np.eye(3)

x = multivariate_normal.rvs(mu, sigma, 10240)

mu0 = np.array([3, 5, 7])
sigma0 = np.eye(3)*2
theta0 = {0: mu0, 1: sigma0}

# WHACKY TEST DATA SET

# mu = np.array([3, 5, 7, 9])                                                                 # Sampling Mean
# sigma = make_spd_matrix(n_dim=4) * 10
# x = multivariate_normal.rvs(mu, sigma, 10240)
# mu0 = np.array([4, 8, 12, 16])
# sigma0 = make_spd_matrix(n_dim=4) * 10
# theta0 = {0: mu0, 1: sigma0}


z = HINTS(x, 4, 2, theta0, Gaussian, Proposal.rw3, 10000, 0.1)

z.sampler()


# z2 = MCMC(x, theta0, Gaussian, Proposal.rw3, 0.1)
# z2.mcmc(500)
