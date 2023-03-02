import numpy as np
from scipy.stats import multivariate_normal
import random
from Tree import Tree
from MCMC import MCMC
from Target import *
from Proposal import Proposal
# TEST DATA SET

mu = np.array([3, 5, 7])
sigma = np.eye(3)

x = multivariate_normal.rvs(mu, sigma, 1024)

mu0 = np.array([2, 4, 6])
sigma0 = np.eye(3)*2
theta0 = {0: mu0, 1: sigma0}

# TEST TREE

# tree = Tree(x, 4, 2)
# tree.build_tree()
# tree.leaf_nodes()
# tree.leaves

# leaves = []
# for x in tree.data:
#     if x.level == tree.levels:
#         leaves.append(x)


# init_leaf = random.choice(tree.leaves)

# init_leaf_set = []
# for x in tree.leaves:
#     if x.parent_id == init_leaf.parent_id:
#         init_leaf_set.append(x)

# for x in init_leaf_set:
#     print(x.__dict__)


class HINTS():

    def __init__(self, x, levels, log_branch_factor, theta0, target, proposal, M):
        tree = Tree(x, levels, log_branch_factor)   # Tree Object
        tree.build_tree()                           # Build Tree
        self.target = target                        # Target Distribution
        tree.leaf_nodes()                           # Leaf Nodes
        self.data = tree.data                       # Data tree structure
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

    def init_leaf_selection(self):
        init_leaf = random.choice(self.leaves)
        init_leaf_set = []
        for x in self.leaves:
            if x.parent_id == init_leaf.parent_id:
                init_leaf_set.append(x)
        return init_leaf_set

    def init_leaf_set_node_ids(self):
        init_leaf_set = self.init_leaf_selection()
        node_ids = [node.node_id for node in init_leaf_set]
        return sorted(node_ids)

    def init_leaf_set_data(self):
        node_ids = self.init_leaf_set_node_ids()
        leaf_data = []
        for node_id in sorted(node_ids):
            node = self.data[node_id]
            if node.level == self.levels:
                leaf_data.append(node.data)
        return leaf_data


z = HINTS(x, 4, 2, theta0, Gaussian, Proposal.rw3, 1000)
a = z.init_leaf_selection()
b = z.init_leaf_set_node_ids()
c = z.init_leaf_set_data()

for i in a:
    print(i.__dict__)
print(b)
print(c)
