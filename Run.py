import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
from scipy.stats import chi2
import seaborn as sns
import HINTS


class proposal():

    def propose(theta, dim):

        # theta_n = np.random.multivariate_normal(theta, self.q_cov)
        # theta_n = theta + np.sqrt(self.q_cov)*np.random.randn()
        theta_n = theta + np.eye(dim)*np.random.normal(size=1)
        return theta_n


class gaussian():

    def __init__(self, x):
        self.theta = [np.mean(x, axis=0), np.eye(len(x[0]))*np.var(x)]

    def logpdf(x, mu, sigma):
        # a = -0.5 * np.sum((x - mu) ** 2 / sigma + np.log(2 * np.pi * sigma))      # lodpdf of Gaussian as defined by ChatGPT which seems to work
        a = np.sum(multivariate_normal.logpdf(x, mu, sigma))
        return a                # logpdf


class poisson():

    def __init__(self, x):
        self.theta = [np.mean(x, axis=0)]

    def logpdf(self, data, lambda_):
        if lambda_ <= 0:
            return -np.inf
        else:
            logpdf = 0
            for i in data:
                if i < 0:
                    return -np.inf
                logpdf += i*np.log(lambda_) - lambda_ - np.log(np.math.factorial(i))
            return logpdf


# x = multivariate_normal.rvs([0, 1], np.eye(2), 1000)
# y = gaussian(x)
x = np.random.poisson(10, 1000)
y = poisson(x)
# x = np.split(x,100)                     # split data into subsets for leaf nodes
# USE np.union1d(x[a], x[b], x[c],...) FOR THE HIGHER LEVELS OF THE TREE MAYBE?
z = HINTS.HINTS(x, y.theta, y.logpdf, proposal.propose, 1000)
# z.mcmc_step(x,z.mcmc_step(x, theta0))
z.mcmc()
z.plot()
