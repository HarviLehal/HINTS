import numpy as np
from scipy.stats import multivariate_normal
import random
from tree import Tree
from MCMC import MCMC


mu = np.array([3, 5, 7])
sigma = np.eye(3)

x = multivariate_normal.rvs(mu, sigma, 1024)

mu0 = np.array([2, 4, 6])
sigma0 = np.eye(3)*2
theta0 = {0: mu0, 1: sigma0}


tree = Tree(x, 4, 2)
tree.build_tree()

leaves = []
for x in tree.data:
    if x.level == tree.levels:
        leaves.append(x)


init_leaf = random.choice(leaves)

init_leaf_set = []
for x in leaves:
    if x.parent_id == init_leaf.parent_id:
        init_leaf_set.append(x)

for x in init_leaf_set:
    print(x.__dict__)
