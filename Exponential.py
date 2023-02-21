import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from HINTS import HINTS
from Proposal import Proposal


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


z = HINTS(x, theta0, Expon, Proposal.rw, 1000)
z.mcmc()


mean = []
for i in range(len(z.thetas)):
    mean.append(z.thetas[i][0])
plt.plot(mean)
plt.show()
