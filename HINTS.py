import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
from scipy import stats


class HINTS():

    def __init__(self, sigma0):
        self.sigma0 = sigma0                # Initial Parameter value
        self.dim = len(sigma0)              # Dimension of Problem

    def logpdf(self, x, mean, var):
        # n = len(x)
        # prec = var**(-1)                                    # Precision
        # f = np.vstack(x-mean)                               # x - mu vectorised
        # a = -n*np.log(np.sqrt(var))*(-0.5 * prec*f.T@f)     # logpdf of Gaussian
        a = -0.5 * np.sum((x - mean) ** 2 / var + np.log(2 * np.pi * var))      # lodpdf of Gaussian as defined by ChatGPT which seems to work
        return a

    def mcmc_step(self, x, sigma, sigma_n=None):    # x is the data, sigma is initial parameter
        mu = np.mean(x)                             # Mean of Chosen Sample
        if sigma_n is not None:                     # proposal provided (for the union of sets stage of HINTS)
            pass
        else:
            sigma_n = sigma + np.eye(self.dim)*np.random.normal(size=1)     # a random walk?
        a = self.logpdf(x, mu, sigma_n)                                     # logpdf of proposal
        b = self.logpdf(x, mu, sigma)                                       # logpdf of previous
        a = a-b                                                             # Acceptance Ratio
        a = np.exp(a)                                                       # Exponent
        u = np.random.uniform(0, 1, 1)
        if u <= a:
            return sigma_n                                                  # Accept Proposal
        else:
            return sigma                                                    # Reject Proposal

    # def HINTS_node(self, level, parent, theta):
        # Aim to have a tree of the following form:
            # [node number, [DATA], parent node number, level]

    def mcmc(self, M, x, sigma):                            # Test mcmc sampler?
        self.M = M                                          # Number of iterations
        sigmas = np.zeros((M))                              # Blank array for parameter values
        sigmas[0] = sigma                                   # Initial parameter value
        for i in range(M-1):
            sigmas[i+1] = self.mcmc_step(x, sigmas[i])      # mcmc loop
        self.sigmas = sigmas
        return self.sigmas                                  # final array of parameters from mcmc

    def plot(self):
        plt.plot(self.sigmas)
        plt.show()


sigma0 = np.array([4])
x = multivariate_normal.rvs(0,1,10)
z = HINTS(sigma0)
# z.mcmc_step(x,z.mcmc_step(x, sigma0))
z.mcmc(10000, x, sigma0)
z.plot()
