import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm


class HINTS():

    def __init__(self, mu, sigma0, step):
        self.mu = mu
        self.sigma0 = sigma0
        self.dim = len(sigma0)
        self.step = step

    def mcmc_step(self, x, sigma, sigma_n=None):  # q is proposal dist and f is posterior dist (How to incorporate this into sampler, or just leave it as gaussian for now?)
        mu = np.mean(x)
        if sigma_n is not None:
            pass
        else:
            sigma_n = multivariate_normal.rvs(sigma, np.eye(self.dim)*self.step**2)

        prec_n = sigma_n**(-1)                          # Precision of proposal
        prec = sigma*(-1)                               # Precision of prior
        f = np.vstack(x-mu)
        a = (0.5 * prec*f.T@f)-(0.5 * prec_n*f.T@f)     # Acceptance Ratio
        u = np.random.uniform(0, 1, 1)
        if u <= a:
            print("Accept: "+ str(sigma_n))
            return sigma_n
        else:
            print("Reject: "+ str(sigma_n))
            return sigma

    def HINTS_node(self, level, parent, theta):
        # Aim to have a tree of the following form:
            # [node number, [DATA], parent node number, level]
            




sigma0 = np.array([0.5])
x = np.array([1, 2, 3, 4, 5, 6])
mu = np.array([4])
z = HINTS(mu, sigma0, 2)
z.mcmc(x, sigma0)
