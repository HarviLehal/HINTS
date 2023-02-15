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
        if dim == 1:
            theta_n = theta + np.random.normal(size=1)
        else:
            theta_n = theta + np.eye(dim)*np.random.normal(size=1)
        return theta_n


class Expon():

    def logpdf(dataset, scale):
        if scale <= 0:
            return -np.inf
        else:
            logpdf = 0
            for x in dataset:
                if x < 0:
                    return -np.inf
                logpdf += -np.log(scale) - x/scale
            return logpdf


x = ss.expon.rvs(loc=0, scale=1, size=1000)
theta0 = {0: 3}


z = HINTS.HINTS(x, theta0, Expon.logpdf, Proposal.propose, 10000)
z.mcmc()


mean = []
for i in range(len(z.thetas)):
    mean.append(z.thetas[i][0])
plt.plot(mean)
plt.show()
