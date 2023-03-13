import numpy as np
import scipy.stats as ss
from Proposal import Proposal
from Target import *
from HINTS import HINTS
from timeit import timeit
from MCMC import MCMC

mu = np.array([2, 4, 6])                                         # Sampling Mean
sigma = np.eye(3) * 15                              # Sampling Covariance
x = ss.multivariate_normal.rvs(mu, sigma, 4096-)                     # Sample from Gaussian

mu0 = np.array([10, 20, 30])                                      # Initial Mean
sigma0 = np.eye(3) * 20                               # Sampling Covariance
theta0 = {0: mu0, 1: sigma0}                                        # Initial Parameters

# gau_MCMC = MCMC(x, theta0, Gaussian, Proposal.rw3, 10000, 0.01)
gau_HINTS1 = HINTS(x, 4, 3, theta0, Gaussian, Proposal.rw3, 1000, 0.01)
gau_HINTS2 = HINTS(x, 5, 2, theta0, Gaussian, Proposal.rw3, 1000, 0.01)
gau_HINTS3 = HINTS(x, 6, 1, theta0, Gaussian, Proposal.rw3, 1000, 0.01)

a = timeit(lambda: gau_HINTS1.sampler(), number=5)
b = timeit(lambda: gau_HINTS2.sampler(), number=5)
c = timeit(lambda: gau_HINTS3.sampler(), number=5)

print("HINTS1 is " + str(a/b) + " times faster than HINTS2")
print("HINTS1 is " + str(a/c) + " times faster than HINTS3")
print("HINTS2 is " + str(b/c) + " times faster than HINTS3")
print("HINTS2 is " + str(b/a) + " times faster than HINTS1")
print("HINTS3 is " + str(c/a) + " times faster than HINTS1")
print("HINTS3 is " + str(c/b) + " times faster than HINTS2")

gau_HINTS1.sampler()
gau_HINTS2.sampler()
gau_HINTS3.sampler()

gau_HINTS1.plot(gau_HINTS1.theta_level[0])
gau_HINTS2.plot(gau_HINTS2.theta_level[0])
gau_HINTS3.plot(gau_HINTS3.theta_level[0])
