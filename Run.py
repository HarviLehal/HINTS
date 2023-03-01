import numpy as np
import scipy.stats as ss
from MCMC import MCMC
from Proposal import Proposal
from Target import *
from sklearn.datasets import make_spd_matrix

# Cauchy
loc = 0                                         # Sampling Location
scale = 1                                       # Sampling Scale
x = ss.cauchy.rvs(loc=loc, scale=scale, size=100)

loc0 = np.array([2])
scale0 = np.array([4])
theta0 = {0: loc0, 1: scale0}                   # Initial Parameters

cau = MCMC(x, theta0, Cauchy, Proposal.rw3, 0.1)
cau.mcmc(10000)


# Exponential
scale = 1                                       # Sampling Scale
x = ss.expon.rvs(loc=0, scale=scale, size=100)

scale0 = np.array([3])
theta0 = {0: scale0}                            # Initial Parameter

exp = MCMC(x, theta0, Expon, Proposal.rw3, 0.1)
exp.mcmc(10000)

# TO MAKE SURE GAUSSIAN WORKS, THE FOLLOWING CODE MAKES SURE THE COV MATRIX IS SPD
def get_near_spd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


# 3 value Gaussian
mu = np.array([3, 5, 7])                                            # Sampling Mean
sigma = np.array([[1, 0.75, 0.9], [0.75, 2, 0.5], [0.9, 0.5, 3]])   # Sampling Covariance DOESNT WORK WITHOUT SPD FUNCTION
sigma = get_near_spd(sigma)
x = ss.multivariate_normal.rvs(mu, sigma, 1000)

mu0 = np.array([2, 4, 6])                                           # Initial Mean
sigma0 = np.array([[2, 1.75, 1.9], [1.75, 3, 1.5], [1.9, 1.5, 4]])  # Initial Covariance DOESNT WORK WITHOUT SPD FUNCTION
sigma0 = get_near_spd(sigma0)
theta0 = {0: mu0, 1: sigma0}                                        # Initial Parameters

gau2 = MCMC(x, theta0, Gaussian, Proposal.rw3, 0.01)
gau2.mcmc(100000)


# 4 value Gaussian

mu = np.array([3, 5, 7, 9])                                                                 # Sampling Mean
sigma = np.array([[0.5, 2.5, 3, 3.5], [2.5, 1, 4, 4.5], [3, 4, 1.5, 5], [3.5, 4.5, 5, 2]])  # Sampling Covariance DOESNT WORK WITHOUT SPD FUNCTION
sigma = get_near_spd(sigma)
sigma = make_spd_matrix(n_dim=4)
x = ss.multivariate_normal.rvs(mu, sigma, 1000)
mu0 = np.array([4, 8, 12, 16])
sigma0 = np.array([[1.5, 3.5, 4, 4.5], [3.5, 2, 5, 5.5], [4, 5, 2.5, 6], [4.5, 5.5, 6, 3]]) # Initial Covariance DOESNT WORK WITHOUT SPD FUNCTION
sigma0 = get_near_spd(sigma0)
sigma0 = make_spd_matrix(n_dim=4)
theta0 = {0: mu0, 1: sigma0}                                                                # Initial Parameters

gau2 = MCMC(x, theta0, Gaussian, Proposal.rw3, 0.01)
gau2.mcmc(50000)


mu = np.array([3, 5, 7, 9])                                                                 # Sampling Mean
sigma = make_spd_matrix(n_dim=4)
x = ss.multivariate_normal.rvs(mu, sigma, 1000)
mu0 = np.array([4, 8, 12, 16])
sigma0 = make_spd_matrix(n_dim=4)
theta0 = {0: mu0, 1: sigma0}                                                                # Initial Parameters

gau2 = MCMC(x, theta0, Gaussian, Proposal.rw3, 0.01)
gau2.mcmc(50000)

# Poisson
mu = 5                                          # Sampling mean
x = np.random.poisson(mu, 100)
mu0 = np.array([15])
theta0 = {0: mu0}                               # Initial Parameter

poi = MCMC(x, theta0, Poisson, Proposal.rw3, 0.1)
poi.mcmc(50000)
