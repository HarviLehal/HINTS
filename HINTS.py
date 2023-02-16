import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
from scipy.stats import chi2
import seaborn as sns


class HINTS():

    def __init__(self, x, theta0, logpdf, proposal, M):
        self.x = x                          # Data
        self.theta0 = theta0                  # Initial Parameter values
        self.logpdf = logpdf                # Logpdf of Target
        self.proposal = proposal            # Proposal Method
        self.M = M                          # Number of Iterations

    def prop(self, theta):    # Theta previous parameter, Theta_n proposal parameter
        # if theta_n is not None:                     # Proposal provided (for the Union of sets stage of HINTS)
        #     pass
        # else:
        dim = 1
        try:
            dim = np.shape(theta)[1]
        except IndexError as e:
            pass
        theta_n = self.proposal(theta, dim)    # proposal step
        return theta_n

    def ratio(self, x, theta, theta_n):

        a = self.logpdf(x, *theta_n)                     # logpdf of proposal
        b = self.logpdf(x, *theta)                       # logpdf of previous
        a = a-b                                         # Acceptance Ratio
        a = np.exp(a)                                   # Exponent
        u = np.random.uniform(0, 1, 1)
        if u <= a:
            return theta_n                          # Accept Proposal
        else:
            return theta                            # Reject Proposal

    def mcmc(self):                            # Test mcmc sampler
        thetas = []         # blank list to save parameters
        thetas.append(self.theta0)  # add initial parameter values
        for i in tqdm(range(self.M-1)):   # for each iteration:
            thetan = {}     # blank dictionary for the proposal
            for j in range(len(self.theta0)):       # for each parameter:
                thetan[j] = self.prop(thetas[i][j])     # new proposal for parameter j using latest parameter j value
            thetan = self.ratio(self.x, list(thetas[i].values()), list(thetan.values()))    # M-H accept-reject step of new values against old values
            p = {}
            for j in range(len(self.theta0)):       # for each parameter:
                p[j] = thetan[j]
            thetas.append(p)   # append parameter values onto the list
        self.thetas = thetas        # save list

    def plot(self):
        plt.plot(self.thetas)
        plt.show()
        sns.kdeplot(self.thetas)
        plt.show()
