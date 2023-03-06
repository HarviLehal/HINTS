import numpy as np
from scipy.stats import multivariate_normal
import random
from Tree import Tree
from MCMC import MCMC
from Target import *
from Proposal import Proposal
from sklearn.datasets import make_spd_matrix
from tqdm import tqdm

# TEST DATA SET

mu = np.array([2, 4, 6])
sigma = np.eye(3)*0.25

x = multivariate_normal.rvs(mu, sigma, 1024000)

mu0 = np.array([8, 16, 22])
sigma0 = np.eye(3)
theta0 = {0: mu0, 1: sigma0}

# WHACKY TEST DATA SET

# mu = np.array([3, 5, 7, 9])                                                                 # Sampling Mean
# sigma = make_spd_matrix(n_dim=4) * 10
# x = multivariate_normal.rvs(mu, sigma, 10240)
# mu0 = np.array([4, 8, 12, 16])
# sigma0 = make_spd_matrix(n_dim=4) * 10
# theta0 = {0: mu0, 1: sigma0}


class HINTS():

    def __init__(self, x, levels, log_branch_factor, theta0, target, proposal, M, stepsize):
        tree = Tree(x, levels, log_branch_factor)   # Tree Object
        tree.build_tree()                           # Build Tree
        self.target = target                        # Target Distribution
        tree.leaf_nodes()                           # Leaf Nodes
        self.tree_structure = tree.data             # Data tree structure
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
            return theta_n                  # Accept Proposal
        else:
            return theta                    # Reject Proposal

    def init_leaf_selection(self):
        init_leaf = random.choice(self.leaves)      # Randomly select initial leaf node
        init_leaf_set = []                          # Initialise set of leaf nodes
        for x in self.leaves:                       # Select all leaf nodes with same parent
            if x.parent_id == init_leaf.parent_id:
                init_leaf_set.append(x)
        return init_leaf_set

    def init_leaf_set_node_ids(self):               # Initialise leaf set node ids
        init_leaf_set = self.init_leaf_selection()
        node_ids = [node.node_id for node in init_leaf_set]
        return sorted(node_ids)

    def init_leaf_set(self, node_ids):              # Initialise leaf set
        leaf_data = []
        for node_id in sorted(node_ids):
            node = self.tree_structure[node_id]
            if node.level == self.levels:
                leaf_data.append(node)
        return leaf_data

    def sampler(self):
        thetas = []                                             # Initialise theta bottom level list
        thetas2 = []                                            # Initialise theta 2nd lowest level list
        thetas.append(self.theta0)                              # Append initial theta
        init_leaf_ID = self.init_leaf_set_node_ids()            # Initialise leaf set node ids
        init_leaf_sets = self.init_leaf_set(init_leaf_ID)       # Initialise leaf set
        for i in tqdm(range(self.M-1)):                         # HINTS Iterations
            for id in init_leaf_ID:                             # Iterate through leaf set
                thetan = {}
                for j in range(len(self.theta0)):               # Iterate through parameters
                    thetan[j] = self.prop(thetas[i][j])         # Propose new parameter
                thetan = self.ratio(self.tree_structure[id].data, list(thetas[i].values()), list(thetan.values()))  # Acceptance Ratio
                p = {}
                for j in range(len(self.theta0)):
                    p[j] = thetan[j]
                thetas.append(p)                                # Append new theta

            for x in self.tree_structure:
                if init_leaf_sets[0].parent_id == x.node_id:    # Select parent node of leaf set
                    parent = x
            thetan = {}
            for j in range(len(self.theta0)):                   # Iterate through parameters
                thetan[j] = thetas[i-len(init_leaf_ID)][j]      # Propose new parameter
            thetan = self.ratio(parent.data, list(thetas[i].values()), list(thetan.values()))   # Acceptance Ratio
            p = {}
            for j in range(len(self.theta0)):                   # Append new theta to theta2
                p[j] = thetan[j]
            thetas2.append(p)
        self.thetas = thetas
        self.thetas2 = thetas2
        self.plot(self.thetas)
        self.plot(self.thetas2)

    def sampler2(self):
        thetas = [[] for i in range(self.levels)]               # Initialise theta level list
        thetas[0].append(self.theta0)                           # Append initial theta
        for iter in tqdm(range(self.M-1)):                      # HINTS Iterations
            init_leaf_ID = self.init_leaf_set_node_ids()        # Initialise leaf set node ids
            for x in self.tree_structure:
                if init_leaf_sets[0].parent_id == x.node_id:    # Select parent node of leaf set
                    parent = x
            for levels in range(self.levels):

            init_leaf_sets = self.init_leaf_set(init_leaf_ID)   # Initialise leaf set
            for id in init_leaf_ID:                             # Iterate through leaf set
                thetan = {}
                for j in range(len(self.theta0)):               # Iterate through parameters
                    thetan[j] = self.prop(thetas[0][iter][j])   # Propose new parameter
                thetan = self.ratio(self.tree_structure[id].data, list(thetas[0][iter].values()), list(thetan.values()))  # Acceptance Ratio



z = HINTS(x, 4, 2, theta0, Gaussian, Proposal.rw3, 10000, 0.1)

z.sampler()

