import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
import scipy.stats as ss
import seaborn as sns
import HINTS
import Proposal


class Gaussian():

    def __init__(self, x):
        mu = np.mean(x, axis=0)
        sigma = np.eye(len(x[0]))*np.var(x)
        self.theta = {0: mu, 1: sigma}      # [0:mean, 1:variance]

    def logpdf(data, mu, sigma):
        a = np.sum(multivariate_normal.logpdf(data, mu, np.absolute(sigma)))
        return a


mu = np.array([0, 1])
sigma = np.eye(2)*2

x = multivariate_normal.rvs(mu, sigma, 1000)
y = Gaussian(x)

mu0 = np.array([2, 4])
sigma0 = np.eye(2)*6
theta0 = {0: mu0, 1: sigma0}


# x = np.split(x,100)                     # split data into subsets for leaf nodes
# USE np.union1d(x[a], x[b], x[c],...) FOR THE HIGHER LEVELS OF THE TREE MAYBE?


z = HINTS.HINTS(x, theta0, Gaussian.logpdf, Proposal.Proposal.propose, 100000)
z.mcmc()

mean = []
for i in range(len(z.thetas)):
    mean.append(z.thetas[i][0])
plt.plot(mean)
plt.show()

variance = []
for i in range(len(z.thetas)):
    variance.append(z.thetas[i][1])
variance = np.sum(variance, axis=1)
plt.plot(np.absolute(variance))
plt.show()
