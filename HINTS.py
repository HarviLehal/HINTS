import numpy as np
from scipy.stats import multivariate_normal
import random
from Tree import Tree
from MCMC import MCMC

# TEST DATA SET

mu = np.array([3, 5, 7])
sigma = np.eye(3)

x = multivariate_normal.rvs(mu, sigma, 1024)

mu0 = np.array([2, 4, 6])
sigma0 = np.eye(3)*2
theta0 = {0: mu0, 1: sigma0}

# TEST TREE

tree = Tree(x, 4, 2)
tree.build_tree()
tree.leaf_nodes()
tree.leaves

# leaves = []
# for x in tree.data:
#     if x.level == tree.levels:
#         leaves.append(x)


init_leaf = random.choice(tree.leaves)

init_leaf_set = []
for x in tree.leaves:
    if x.parent_id == init_leaf.parent_id:
        init_leaf_set.append(x)

for x in init_leaf_set:
    print(x.__dict__)


class HINTS():

    def __init__(self, x, levels, log_branch_factor, theta0, target, proposal, M):
        tree = Tree(x, levels, log_branch_factor)
        tree.build_tree()
        tree.leaf_nodes()
        self.data = tree.data               # Data tree structure
        self.leaves = tree.leaves           # Data tree leaf nodes
        self.theta0 = theta0                # Initial Parameter values
        self.logpdf = target.logpdf         # Logpdf of Target
        self.proposal = proposal            # Proposal Method
        self.M = M                          # Number of Iterations
        self.plot = target.plot             # Parameter Plotting
