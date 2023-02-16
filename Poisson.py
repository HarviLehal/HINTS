import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
import scipy.stats as ss
import seaborn as sns
import HINTS
import Proposal


class Poisson():

    def __init__(self, x):
        mu = np.mean(x, axis=0)
        self.theta = {0: mu}      # [lambda]

    def logpdf(data, mean):
        if mean <= 0:
            return -np.inf
        else:
            logpdf = 0
            for i in data:
                if i < 0:
                    return -np.inf
                logpdf += i*np.log(mean) - mean - np.log(np.math.factorial(i))
            return logpdf


x = np.random.poisson(10, 1000)
y = Poisson(x)
theta0 = {0: 15}


z = HINTS.HINTS(x, theta0, Poisson.logpdf, Proposal.Proposal.propose, 100000)
z.mcmc()


mean = []
for i in range(len(z.thetas)):
    mean.append(z.thetas[i][0])
plt.plot(mean)
plt.show()
