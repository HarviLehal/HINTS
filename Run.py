import numpy as np
import scipy.stats as ss
from MCMC import MCMC
from Proposal import Proposal
from Target import *


# Cauchy
loc = 0                                         # Sampling Location
scale = 1                                       # Sampling Scale
x = ss.cauchy.rvs(loc=loc, scale=scale, size=100)

loc0 = np.array([2])
scale0 = np.array([4])
theta0 = {0: loc0, 1: scale0}                   # Initial Parameters

cau = MCMC(x, theta0, Cauchy, Proposal.rw, 0.5)
cau.mcmc(10000)


# Exponential
scale = 1                                       # Sampling Scale
x = ss.expon.rvs(loc=0, scale=scale, size=100)

scale0 = np.array([3])
theta0 = {0: scale0}                            # Initial Parameter

exp = MCMC(x, theta0, Expon, Proposal.rw, 0.5)
exp.mcmc(10000)


# Gaussian
mu = np.array([3, 5, 7])                        # Sampling Mean
sigma = np.eye(3)                               # Sampling Covariance
sigma = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
x = ss.multivariate_normal.rvs(mu, sigma, 100)

mu0 = np.array([2, 4, 6])
sigma0 = np.eye(3)*2
sigma0 = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
theta0 = {0: mu0, 1: sigma0}                    # Initial Parameters

gau = MCMC(x, theta0, Gaussian, Proposal.rw, 0.5)
gau.mcmc(10000)


# Poisson
mu = 5                                          # Sampling mean
x = np.random.poisson(mu, 100)
mu0 = np.array([15])
theta0 = {0: mu0}                               # Initial Parameter

poi = MCMC(x, theta0, Poisson, Proposal.rw, 0.5)
poi.mcmc(10000)
