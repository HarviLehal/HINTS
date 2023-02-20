import matplotlib.pyplot as plt
import numpy as np
from HINTS import HINTS
from Proposal import Proposal


class Poisson():

    def __init__(self, x):
        mu = np.mean(x, axis=0)
        self.theta = {0: mu}      # [lambda]

    def logpdf(data, mean):
        if mean <= 0:
            return -np.inf
        else:
            mean = int(mean)     # integer to nearest whole number
            logpdf = 0
            for i in data:
                if i < 0:
                    return -np.inf
                logpdf += i*np.log(mean) - mean - np.log(np.math.factorial(i))
            return logpdf


mu = 5
x = np.random.poisson(mu, 1000)
y = Poisson(x)
mu0 = np.array([15])
theta0 = {0: mu0}


z = HINTS(x, theta0, Poisson.logpdf, Proposal.propose, 1000)
z.mcmc()


mean = []
for i in range(len(z.thetas)):
    mean.append(z.thetas[i][0])
plt.plot(mean)
plt.show()
