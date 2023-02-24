import numpy as np
from scipy.stats import multivariate_normal
import random


class Node:
    def __init__(self, node_id, data, level=None):
        self.node_id = node_id
        self.data = data
        self.level = level


class Tree:
    def __init__(self, x, levels, log_branch_factor):
        self.levels = levels
        self.log_branch_factor = log_branch_factor
        self.design = np.array([1] + [2 ** self.log_branch_factor for _ in range(self.levels)])
        self.scenarios = 1 * 2 ** (self.levels * self.log_branch_factor)
        self.x = np.split(x, self.scenarios)

    def build_tree(self):
        D = []
        node = 0
        for i in range(self.levels):
            if i == 0:
                D.append([])
                for j in range(self.scenarios):
                    node_id = node + j
                    data = self.x[j]
                    D[i].append(Node(node_id, data, self.levels - i))
                z = int(self.scenarios/self.design[-1])
                for j in range(self.scenarios):
                    D[i][j].parent_id = self.scenarios + j % z
                    D[i][j].level = self.levels - i

            else:
                D.append([])
                for j in range(self.levels**(self.levels-i)):
                    node_id = node + j
                    data = self.x[j]
                    D[i].append(Node(node_id, data, self.levels - i))

                z = int(self.levels**(self.levels-i-1))
                for j in range(self.levels**(self.levels-i)):
                    D[i][j].parent_id = node + self.levels**(self.levels-i) + j % z
                    D[i][j].level = self.levels - i
            node += self.levels**(self.levels-i)
        D.append([])
        node_id = node
        data = self.x
        level = 0
        D[-1].append(Node(node_id, data, level))
        D = [elem for sublist in D for elem in sublist]
        self.data = D


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
