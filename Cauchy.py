import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import HINTS
import Proposal


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
loc0 = np.array([2])
scale0 = np.array([4])
theta0 = {0: loc0, 1: scale0}


z = HINTS.HINTS(x, theta0, Cauchy.logpdf, Proposal.Proposal.propose, 100000)
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
