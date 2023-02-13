import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
from scipy.stats import chi2
import seaborn as sns


class HINTS():

    def __init__(self, x, theta, logpdf, proposal, M):
        self.x = x                          # Data
        self.theta = theta                  # Parameters of Target
        self.dim = np.size(self.theta[0])   # Dimension of Problem
        self.logpdf = logpdf                # Logpdf of Target
        self.proposal = proposal            # Proposal Method
        self.M = M                          # Number of Iterations

    def mcmc_step(self, x, theta, theta_n=None):    # Theta previous parameter, Theta_n proposal parameter
        if theta_n is not None:                     # Proposal provided (for the Union of sets stage of HINTS)
            pass
        else:
            theta_n = self.proposal(theta, np.size(theta))    # proposal step
        print("********INITIAL DATA**********")
        print(x)
        print("******************************")
        print("******INITIAL PARAMETER******")
        print(theta)
        print("******************************")
        print("******PROPOSAL PARAMETER******")
        print(theta_n)
        print("******************************")
        a = self.logpdf(x, theta_n)                     # logpdf of proposal
        b = self.logpdf(x, theta)                       # logpdf of previous
        a = a-b                                         # Acceptance Ratio
        a = np.exp(a)                                   # Exponent
        u = np.random.uniform(0, 1, 1)
        if u <= a:
            return theta_n                          # Accept Proposal
        else:
            return theta                            # Reject Proposal

    def mcmc(self):                            # Test mcmc sampler
        thetas = {}
        for i in range(len(self.theta)):
            thetas[i] = np.zeros((self.M, np.size(self.theta[i])))  # Blank array for parameter values
            thetas[i][0] = self.theta[i]                            # Initial parameter values
        # thetas = np.zeros((self.M, len(self.theta)))
        for j in range(self.M-1):
            for i in range(len(self.theta)):
                thetas[i][j+1] = self.mcmc_step(x=self.x, theta=thetas[i][j])      # mcmc loop
        self.thetas = thetas                                # save vector of parameters
        return self.thetas                                  # final array of parameters from mcmc

    def plot(self):
        plt.plot(self.thetas)
        plt.show()
        sns.kdeplot(self.thetas)
        plt.show()
