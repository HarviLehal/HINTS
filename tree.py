import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
from HINTS import HINTS
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
