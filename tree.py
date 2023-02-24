import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
from MCMC import MCMC
import argparse

mu = np.array([3, 5, 7])
sigma = np.eye(3)

x = multivariate_normal.rvs(mu, sigma, 1024)

mu0 = np.array([2, 4, 6])
sigma0 = np.eye(3)*2
theta0 = {0: mu0, 1: sigma0}

parser = argparse.ArgumentParser()
args = parser.parse_known_args()[0]
log_branch_factor = 2
args.levels = 4

args.design = np.array([1] + [2 ** log_branch_factor for l in range(args.levels)])
args.scenarios = 1 * 2 ** (args.levels * log_branch_factor)

x = np.split(x, args.scenarios)               # split data into subsets for leaf nodes


D = []
node = 0
for i in range(args.levels):
    if i == 0:
        D.append([])
        for j in range(args.scenarios):
            d = {'node': node + j, 'data': x[j]}
            D[i].append(d)
        z = int(args.scenarios/args.design[-1])
        for j in range(args.scenarios):
            D[i][j]['parent'] = args.scenarios + j % z
            D[i][j]['level'] = args.levels - i

    else:
        D.append([])
        for j in range(args.levels**(args.levels-i)):
            d = {'node': node + j, 'data': x[j]}
            D[i].append(d)

        z = int(args.levels**(args.levels-i-1))
        for j in range(args.levels**(args.levels-i)):
            D[i][j]['parent'] = node + args.levels**(args.levels-i) + j % z
            D[i][j]['level'] = args.levels - i
    node += args.levels**(args.levels-i)

D.append([])
d = {'node': node, 'data': x, 'level': 0}
D[-1].append(d)
D = [elem for sublist in D for elem in sublist]


class Tree():

    def __init__(self, x, levels, log_branch_factor):
        self.levels = levels
        self.log_branch_factor = log_branch_factor
        self.design = np.array([1] + [2 ** self.log_branch_factor for l in range(self.levels)])
        self.scenarios = 1 * 2 ** (self.levels * self.log_branch_factor)
        self.x = np.split(x, self.scenarios)

    def tree_data(self):
        D = []
        node = 0
        for i in range(self.levels):
            if i == 0:
                D.append([])
                for j in range(self.scenarios):
                    d = {'node': node + j, 'data': x[j]}
                    D[i].append(d)
                z = int(self.scenarios/self.design[-1])
                for j in range(self.scenarios):
                    D[i][j]['parent'] = self.scenarios + j % z
                    D[i][j]['level'] = self.levels - i

            else:
                D.append([])
                for j in range(self.levels**(self.levels-i)):
                    d = {'node': node + j, 'data': x[j]}
                    D[i].append(d)

                z = int(self.levels**(self.levels-i-1))
                for j in range(self.levels**(self.levels-i)):
                    D[i][j]['parent'] = node + self.levels**(self.levels-i) + j % z
                    D[i][j]['level'] = self.levels - i
            node += self.levels**(self.levels-i)

        D.append([])
        d = {'node': node, 'data': x, 'level': 0}
        D[-1].append(d)
        D = [elem for sublist in D for elem in sublist]
        return D


# CHATGPT VERSION

class Node:
    def __init__(self, node, data, parent=None, level=None):
        self.node = node
        self.data = data
        self.parent = parent
        self.level = level

class Tree:
    def __init__(self, x, levels, log_branch_factor):
        self.levels = levels
        self.log_branch_factor = log_branch_factor
        self.design = np.array([1] + [2 ** self.log_branch_factor for l in range(self.levels)])
        self.scenarios = 1 * 2 ** (self.levels * self.log_branch_factor)
        self.x = np.split(x, self.scenarios)
        self.nodes = []

    def build_nodes(self):
        node = 0
        for i in range(self.levels):
            if i == 0:
                for j in range(self.scenarios):
                    node_value = node + j
                    data_value = self.x[j]
                    parent_value = self.scenarios + j % int(self.scenarios/self.design[-1])
                    level_value = self.levels - i
                    self.nodes.append(Node(node_value, data_value, parent_value, level_value))
            else:
                for j in range(self.levels**(self.levels-i)):
                    node_value = node + j
                    data_value = self.x[j]
                    parent_value = node + self.levels**(self.levels-i) + j % int(self.levels**(self.levels-i-1))
                    level_value = self.levels - i
                    self.nodes.append(Node(node_value, data_value, parent_value, level_value))
            node += self.levels**(self.levels-i)

        d = {'node': node, 'data': self.x, 'level': 0}
        self.nodes.append(Node(node, self.x, None, 0))

    def tree_data(self):
        self.build_nodes()
        return [node.__dict__ for node in self.nodes]
    










    import numpy as np

class Node:
    def __init__(self, node_id, data, parent_id=None, level=None):
        self.node_id = node_id
        self.data = data
        self.parent_id = parent_id
        self.level = level

class LeafNode(Node):
    def __init__(self, node_id, data, level):
        super().__init__(node_id, data, level=level)

class InnerNode(Node):
    def __init__(self, node_id, data, parent_id, level):
        super().__init__(node_id, data, parent_id=parent_id, level=level)

class Tree:
    def __init__(self, x, levels, log_branch_factor):
        self.levels = levels
        self.log_branch_factor = log_branch_factor
        self.design = np.array([1] + [2 ** self.log_branch_factor for l in range(self.levels)])
        self.scenarios = 1 * 2 ** (self.levels * self.log_branch_factor)
        self.x = np.split(x, self.scenarios)

    def tree_data(self):
        D = []
        node = 0
        for i in range(self.levels):
            if i == 0:
                D.append([])
                for j in range(self.scenarios):
                    node_id = node + j
                    data = self.x[j]
                    D[i].append(LeafNode(node_id, data, self.levels - i))
                z = int(self.scenarios/self.design[-1])
                for j in range(self.scenarios):
                    D[i][j].parent_id = self.scenarios + j % z
                    D[i][j].level = self.levels - i

            else:
                D.append([])
                for j in range(self.levels**(self.levels-i)):
                    node_id = node + j
                    data = self.x[j]
                    D[i].append(InnerNode(node_id, data, None, self.levels - i))

                z = int(self.levels**(self.levels-i-1))
                for j in range(self.levels**(self.levels-i)):
                    D[i][j].parent_id = node + self.levels**(self.levels-i) + j % z
                    D[i][j].level = self.levels - i
            node += self.levels**(self.levels-i)

        D.append([])
        node_id = node
        data = self.x
        level = 0
        D[-1].append(LeafNode(node_id, data, level))
        D = [elem for sublist in D for elem in sublist]
        return D