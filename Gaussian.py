import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from HINTS import HINTS
from Proposal import Proposal


class Gaussian():

    def __init__(self, x):
        mu = np.mean(x, axis=0)
        sigma = np.eye(len(x[0]))*np.var(x)
        self.theta = {0: mu, 1: sigma}      # [0:mean, 1:variance]

    def logpdf(data, mu, sigma):
        a = np.sum(multivariate_normal.logpdf(data, mu, np.absolute(sigma)))
        return a


mu = np.array([3, 5, 7])
sigma = np.eye(3)*5

x = multivariate_normal.rvs(mu, sigma, 1000)
y = Gaussian(x)

mu0 = np.array([2, 4, 6])
sigma0 = np.eye(3)*10
theta0 = {0: mu0, 1: sigma0}


# x = np.split(x,100)               # split data into subsets for leaf nodes
# USE np.union1d(x[a], x[b], x[c],...) FOR THE HIGHER LEVELS OF THE TREE MAYBE?


z = HINTS(x, theta0, Gaussian.logpdf, Proposal.propose, 100000)
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
