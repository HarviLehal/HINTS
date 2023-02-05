import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm


class HINTS():

    def __init__(self, sigma0, step):
        self.sigma0 = sigma0                # Initial Parameter value
        self.dim = len(sigma0)              # Dimension of Problem
        self.step = step                    # Random walk step size

    def mcmc_step(self, x, sigma, sigma_n=None):  # x is the data, sigma is initial parameter
        mu = np.mean(x)         # Mean of Chosen Sample
        if sigma_n is not None: # proposal provided
            pass
        else:
            sigma_n = multivariate_normal.rvs(sigma, np.eye(self.dim)*self.step**2)   # replace with a random walk which adds a value onto the covariance diagonal  (value + random normal/sqrt(sigma^2))
            # sigma_n = sigma + np.eye(self.dim)*np.random.normal(size=1)*self.step        # would this do the above?
        prec_n = sigma_n**(-1)                          # Precision of proposal
        prec = sigma*(-1)                               # Precision of prior
        f = np.vstack(x-mu)                             # x - mu vectorised
        a = (0.5 * prec_n*f.T@f)-(0.5 * prec*f.T@f)     # Log Acceptance Ratio
        a = np.exp(a)                                   # Exponent
        u = np.random.uniform(0, 1, 1)
        if u <= a:
            # print("Accept Proposal: "+ str(sigma_n)+"\n Reject: "+ str(sigma))
            return sigma_n
        else:
            # print("Reject Proposal: "+ str(sigma_n)+"\n Accept: "+ str(sigma))
            return sigma

    # def HINTS_node(self, level, parent, theta):
        # Aim to have a tree of the following form:
            # [node number, [DATA], parent node number, level]


    def mcmc(self, M, x, sigma):                                # Test mcmc sampler?
        sigmas = np.zeros(M)
        sigmas[0] = sigma
        for i in range(M-1):
            sigmas[i+1] = self.mcmc_step(x, sigmas[i])
        self.sigmas = sigmas
        return self.sigmas
    
    def plot(self):
        plt.plot(self.sigmas)
        plt.show()



sigma0 = np.array([4])
x = multivariate_normal.rvs(0,1,1000)
print(x)
# x = np.array([1,2,3,4,5,6])
mu = np.array([1])
z = HINTS(sigma0, 1)
# z.mcmc_step(x,z.mcmc_step(x, sigma0))
z.mcmc(1000, x, sigma0)
z.plot()