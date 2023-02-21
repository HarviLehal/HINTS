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


# 4 levels branching by 4 each level provides 128 leaf nodes (from strens repo)

parser = argparse.ArgumentParser()
args = parser.parse_known_args()[0]
log_branch_factor = 2
args.levels = 4

args.design = np.array([1] + [2 ** log_branch_factor for l in range(args.levels)])
args.scenarios = 1 * 2 ** (args.levels * log_branch_factor)


# my way of forming the trees

x = np.split(x, args.scenarios)               # split data into subsets for leaf nodes

D = []

for i in tqdm(range(args.scenarios)):
    d = {'node': i, 'data': x[i]}
    D.append(d)

args.design[-1]     # terminal leafs per node

z = args.scenarios/args.design[-1]
parent = "parent"
for i in range(args.scenarios):
    D[i][parent+str(args.levels)] = args.scenarios + i % z     # every 64th subset has the same parent
    d = {'node': args.scenarios + (i % z), 'data': np.concatenate((D[int(i%z)]['data'], D[int(i%z)]['data']), axis=0)}
    D.append(d)


z1 = z/args.design[-2]

for i in range(args.scenarios):
    D[i][parent+str(args.levels-1)] = z+args.scenarios + i % z1     # every 64th subset has the same parent
