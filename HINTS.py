import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
import scipy.stats

class HINTS():

    def __init__(self,mu,sigma):
        self.mu=mu
        self.sigma=sigma


    def mcmc(self, x):  # q is proposal dist and f is posterior dist (How to incorporate this into sampler, or just leave it as gaussian for now?)
        dim=len(x)      # dimension of sample
        xn = np.zeros((dim))                           # set up vector for new sample
        xn = multivariate_normal.rvs(x, np.eye(dim))   # take new sample

        a = multivariate_normal.logpdf(xn, self.mu, self.sigma)     # define mu and sigma in init since they wont change for the entire thing? 
        b = multivariate_normal.logpdf(x, self.mu, self.sigma)      
        alpha = a-b
        u = np.random.uniform(0,1,1)
        if u <= alpha:
            return xn
        else:
            return x    # do I want to return a new value or new parameter estimates? or am I confusing theta for a parameter when it is infact the data?


z=HINTS([1.2345],[0.5])

z.mcmc([1.2345])