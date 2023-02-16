import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm
import scipy.stats as ss
import seaborn as sns
import HINTS
import Proposal


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


scale = 1
x = ss.expon.rvs(loc=0, scale=scale, size=1000)
scale0 = np.array([3])
theta0 = {0: scale0}


z = HINTS.HINTS(x, theta0, Expon.logpdf, Proposal.Proposal.propose, 10000)
z.mcmc()


mean = []
for i in range(len(z.thetas)):
    mean.append(z.thetas[i][0])
plt.plot(mean)
plt.show()
