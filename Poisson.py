import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
import scipy.stats as ss
import seaborn as sns
import HINTS

class Proposal():

    def propose(theta, dim):
        print(dim)
        if dim == 1:
            theta_n = theta + np.random.normal(size=1)*10
        else:
            theta_n = theta + np.eye(dim)*np.random.normal(size=1)
        return theta_n


class Poisson():

    def __init__(self, x):
        mu = np.mean(x, axis=0)
        self.theta = {0: mu}      # [lambda]

    def logpdf(data, lambda_):
        if lambda_ <= 0:
            return -np.inf
        else:
            logpdf = 0
            for i in data:
                if i < 0:
                    return -np.inf
                logpdf += i*np.log(lambda_) - lambda_ - np.log(np.math.factorial(i))
            return logpdf


x = np.random.poisson(10, 1000)
y = Poisson(x)
theta0 = {0: 15}


z = HINTS.HINTS(x, theta0, Poisson.logpdf, Proposal.propose, 100000)
z.mcmc()
