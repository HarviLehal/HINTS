import numpy as np
import scipy.stats as ss
from MCMC import MCMC
from Proposal import Proposal
from Target import *
from sklearn.datasets import make_spd_matrix
from HINTS import HINTS
from Tree import Tree

# Cauchy
loc = 0                                             # Sampling Location
scale = 1                                           # Sampling Scale
x = ss.cauchy.rvs(loc=loc, scale=scale, size=1024)  # Sample from Cauchy

loc0 = np.array([2])                                # Initial Location
scale0 = np.array([4])                              # Initial Scale
theta0 = {0: loc0, 1: scale0}                       # Initial Parameters

cau_MCMC = MCMC(x, theta0, Cauchy, Proposal.rw3, 0.01)
cau_MCMC.mcmc(1000)
cau_HINTS = HINTS(x, 5, 2, theta0, Cauchy, Proposal.rw3, 1000, 0.01)
cau_HINTS.sampler()


# Exponential
scale = 1                                           # Sampling Scale
x = ss.expon.rvs(loc=0, scale=scale, size=1024)     # Sample from Exponential

scale0 = np.array([3])                              # Initial Scale
theta0 = {0: scale0}                                # Initial Parameter

exp_MCMC = MCMC(x, theta0, Expon, Proposal.rw3, 0.01)
exp_MCMC.mcmc(1000)
exp_HINTS = HINTS(x, 5, 2, theta0, Expon, Proposal.rw3, 1000, 0.01)
exp_HINTS.sampler()

# 3 value Gaussian
mu = np.array([3, 5, 7])                                            # Sampling Mean
sigma = np.array([[1, 0.75, 0.9], [0.75, 2, 0.5], [0.9, 0.5, 3]])   # Sampling Covariance
sigma = Gaussian.get_near_spd(sigma)                                # Make Covariance Positive Definite
x = ss.multivariate_normal.rvs(mu, sigma, 1024)                     # Sample from Gaussian

mu0 = np.array([2, 4, 6])                                           # Initial Mean
sigma0 = np.array([[2, 1.75, 1.9], [1.75, 3, 1.5], [1.9, 1.5, 4]])  # Initial Covariance
sigma0 = Gaussian.get_near_spd(sigma0)                              # Make Covariance Positive Definite
theta0 = {0: mu0, 1: sigma0}                                        # Initial Parameters

gau_MCMC = MCMC(x, theta0, Gaussian, Proposal.rw3, 0.01)
gau_MCMC.mcmc(1000)
gau_HINTS = HINTS(x, 5, 2, theta0, Gaussian, Proposal.rw3, 1000, 0.01)
gau_HINTS.sampler()


# 4 value Gaussian

mu = np.array([3, 5, 7, 9])                                         # Sampling Mean
sigma = make_spd_matrix(n_dim=4) * 2                                # Sampling Covariance
x = ss.multivariate_normal.rvs(mu, sigma, 1024)                     # Sample from Gaussian

mu0 = np.array([4, 8, 12, 16])                                      # Initial Mean
sigma0 = make_spd_matrix(n_dim=4) * 2                               # Initial Covariance
theta0 = {0: mu0, 1: sigma0}                                        # Initial Parameters

gau2_MCMC = MCMC(x, theta0, Gaussian, Proposal.rw3, 0.01)
gau2_MCMC
gau2_HINTS = HINTS(x, 5, 2, theta0, Gaussian, Proposal.rw3, 1000, 0.01)
gau2_HINTS.sampler()

# Poisson

mu = 5                                                              # Sampling Mean
x = np.random.poisson(mu, 1024)                                     # Sample from Poisson

mu0 = np.array([8])                                                # Initial Mean
theta0 = {0: mu0}                                                   # Initial Parameter

poi_MCMC = MCMC(x, theta0, Poisson, Proposal.rw3, 0.01)
poi_MCMC.mcmc(1000)
poi_HINTS = HINTS(x, 5, 2, theta0, Poisson, Proposal.rw3, 10000, 0.01)
poi_HINTS.sampler()
