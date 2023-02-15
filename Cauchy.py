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


class Cauchy():

    def logpdf(dataset, loc, scale):
        if scale <= 0:
            return -np.inf
        else:
            logpdf = 0
            for x in dataset:
                logpdf += -np.log(np.pi) - np.log(scale) - np.log(1 + ((x - loc)/scale)**2)
            return logpdf


loc = 0
scale = 1
x = ss.cauchy.rvs(loc=loc, scale=scale, size=100)
theta0 = {0: 2, 1: 4}


z = HINTS.HINTS(x, theta0, Cauchy.logpdf, Proposal.propose, 100000)
z.mcmc()

loc = []
for i in range(len(z.thetas)):
    loc.append(z.thetas[i][0])
plt.plot(loc)
plt.show()

scale = []
for i in range(len(z.thetas)):
    scale.append(z.thetas[i][1])
plt.plot(scale)
plt.show()