import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
from scipy.stats import chi2
import seaborn as sns
import HINTS

class proposal():

    def propose(self, theta, dim):

        # theta_n = np.random.multivariate_normal(theta, self.q_cov)
        # theta_n = theta + np.sqrt(self.q_cov)*np.random.randn()
        theta_n = theta + np.eye(dim)*np.random.normal(size=1)
        return theta_n


class gaussian():

    def __init__(self, x):
        self.x = x              # data
        self.mu = np.mean(x)    # mean
        self.sigma = np.var(x)  # variance

    def logpdf(x, mu, sigma):
        a = -0.5 * np.sum((x - mu) ** 2 / sigma + np.log(2 * np.pi * sigma))      # lodpdf of Gaussian as defined by ChatGPT which seems to work
        return a                # logpdf



x = multivariate_normal.rvs([0,1], np.eye(2), 1000)
logpdf = logpdf()
theta = np.var(x)
# x = np.split(x,100)                     # split data into subsets for leaf nodes
# USE np.union1d(x[a], x[b], x[c],...) FOR THE HIGHER LEVELS OF THE TREE MAYBE?
z = HINTS.HINTS(x, theta, gaussian.logpdf, proposal.propose)
# z.mcmc_step(x,z.mcmc_step(x, theta0))
z.mcmc(10000, x, theta)
z.plot()
