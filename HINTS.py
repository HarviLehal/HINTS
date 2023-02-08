import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
from scipy.stats import chi2
import seaborn as sns
import target


class HINTS():

    def __init__(self, x, dist, proposal):
        self.target = dist                # target distribution
        self.proposal = proposal            # proposal style
        self.x = x
        self.dim = len(x[0])

    def mcmc_step(self, x, theta, theta_n=None):    # x is the data, theta is initial parameter
        if theta_n is not None:                     # proposal provided (for the union of sets stage of HINTS)
            pass
        else:
            theta_n = self.proposal(theta)          # proposal step
        a = self.target.logpdf(x, theta_n)          # logpdf of proposal
        b = self.target.logpdf(x, *theta)           # logpdf of previous
        a = a-b                                     # Acceptance Ratio
        a = np.exp(a)                               # Exponent
        u = np.random.uniform(0, 1, 1)
        if u <= a:
            return theta_n                          # Accept Proposal
        else:
            return theta                            # Reject Proposal

    def mcmc(self, M, x, theta):                            # Test mcmc sampler?
        self.M = M                                          # Number of iterations
        thetas = np.zeros((M))                              # Blank array for parameter values
        thetas[0] = theta                                   # Initial parameter value
        for i in range(M-1):
            thetas[i+1] = self.mcmc_step(x, thetas[i])      # mcmc loop
        self.thetas = thetas
        return self.thetas                                  # final array of parameters from mcmc

    def plot(self):
        plt.plot(self.thetas)
        plt.show()
        sns.kdeplot(self.thetas)
        plt.show()

    # def HINTS_node(self, level, parent, theta):
        # Aim to have a tree of the following form:
        # [node number, [DATA], parent node number, level]


theta0 = np.array([4])
x = multivariate_normal.rvs(0, 1, 1000)
# x = np.split(x,100)                     # split data into subsets for leaf nodes
# USE np.union1d(x[a], x[b], x[c],...) FOR THE HIGHER LEVELS OF THE TREE MAYBE?
z = HINTS(theta0)
# z.mcmc_step(x,z.mcmc_step(x, theta0))
z.mcmc(10000, theta0)
z.plot()
