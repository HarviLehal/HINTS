import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


class Cauchy():

    def logpdf(dataset, loc, scale):
        if scale <= 0:
            return -np.inf
        else:
            logpdf = 0
            for x in dataset:
                logpdf += -np.log(np.pi) - np.log(scale) - np.log(1 + ((x - loc)/scale)**2)
            return logpdf

    def plot(parameters):
        loc = []
        scale = []
        for i in range(len(parameters)):
            loc.append(parameters[i][0])
            scale.append(parameters[i][1])
        plt.plot(loc)
        plt.show()
        plt.plot(scale)
        plt.show()


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

    def plot(parameters):
        mean = []
        for i in range(len(parameters)):
            mean.append(parameters[i][0])
        plt.plot(mean)
        plt.show()


class Gaussian():

    def logpdf(data, mu, sigma):
        logpdf = np.sum(multivariate_normal.logpdf(data, mu, np.absolute(sigma)))
        return logpdf

    def plot(parameters):
        mean = []
        variance = []
        for i in range(len(parameters)):
            mean.append(parameters[i][0])
            variance.append(parameters[i][1])
        variance = np.sum(variance, axis=1)
        plt.plot(mean)
        plt.show()
        plt.plot(np.absolute(variance))
        plt.show()


class Poisson():

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

    def plot(parameters):
        mean = []
        for i in range(len(parameters)):
            mean.append(parameters[i][0])
        plt.plot(mean)
        plt.show()
