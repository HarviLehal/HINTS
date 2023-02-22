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

D1 = []

for i in range(args.scenarios):
    d = {'node': i, 'data': x[i]}
    D1.append(d)

args.design[-1]     # terminal leafs per node

z = int(args.scenarios/args.design[-1])
parent = "parent"
for i in range(args.scenarios):
    D1[i]['parent'] = args.scenarios + i % z     # every 64th subset has the same parent


D2 = []
for i in range(z):
    w = []
    for dico in D1:
        print(dico)
        if dico['parent'] == args.scenarios + i:
            w.append(dico['data'])
    w = np.vstack(w)
    d = {'node': args.scenarios + i, 'data': w}
    D2.append(d)

z1 = int(z/args.design[-2])
for i in range(z):
    D2[i]['parent'] = args.scenarios + z + i % z1     # every 64th subset has the same parent

D3 = []
for i in range(z1):
    w = []
    for dico in D2:
        print(dico)
        if dico['parent'] == args.scenarios + z + i:
            w.append(dico['data'])
    w = np.vstack(w)
    d = {'node': args.scenarios + z + i, 'data': w}
    D3.append(d)

z2 = z1/args.design[-3]
for i in range(z1):
    D3[i]['parent'] = args.scenarios + z1 + z + i % z2     # every 64th subset has the same parent



# ATTEMPT AT LOOPING
D=[]
for i in range(args.levels):
   D.append([])
   for j in range(args.levels**(args.levels-i)):
        d = {'node': j, 'data': x[j]}
        D[i].append(d)

    z = int(args.levels**(args.levels-i-1))
    for j in range(args.levels**(args.levels-i)):
        D[i][j]['parent'] = args.levels**(args.levels-i) + j % z
