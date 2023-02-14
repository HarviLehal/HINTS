import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
from scipy.stats import chi2
import seaborn as sns
import HINTS


class Proposal():

    def propose(theta, dim):
        if dim == 1:
            theta_n = theta + np.random.normal(size=1)
        else:
            theta_n = theta + np.eye(dim)*np.random.normal(size=1)
        return theta_n


class Gaussian():

    def __init__(self, x):
        mu = np.mean(x, axis=0)
        sigma = np.eye(len(x[0]))*np.var(x)
        self.theta = {0: mu, 1: sigma}      # [0:mean, 1:variance]

    def logpdf(data, mu, sigma):
        # a = -0.5 * np.sum((x - mu) ** 2 / sigma + np.log(2 * np.pi * sigma))  # lodpdf of Gaussian as defined by ChatGPT which seems to work
        a = np.sum(multivariate_normal.logpdf(data, mu, np.absolute(sigma)))
        return a                # logpdf


class Poisson():

    def __init__(self, x):
        mu = np.mean(x, axis=0)
        self.theta = {0: mu}      # [lambda]

    def logpdf(data, lambda_):
        if lambda_ <= 0:
            return -np.inf
        else:
            logpdf = 0
            for i in data:
                if i < 0:
                    return -np.inf
                logpdf += i*np.log(lambda_) - lambda_ - np.log(np.math.factorial(i))
            return logpdf


mu = np.array([0, 1])
sigma = np.eye(2)*(-1)


x = multivariate_normal.rvs(mu, sigma, 1000)
y = Gaussian(x)

mu0 = np.array([2, 4])
sigma0 = np.eye(2)*4
theta0 = {0: mu0, 1: sigma0}


# x = np.random.poisson(10, 1000)
# y = poisson(x)
# x = np.split(x,100)                     # split data into subsets for leaf nodes
# USE np.union1d(x[a], x[b], x[c],...) FOR THE HIGHER LEVELS OF THE TREE MAYBE?
z = HINTS.HINTS(x, theta0, Gaussian.logpdf, Proposal.propose, 1000)
# z.mcmc_step(x,z.mcmc_step(x, theta0))
z.mcmc()
z.plot()
