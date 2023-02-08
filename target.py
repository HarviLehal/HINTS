import numpy as np


class gaussian():

    def __init__(self, x):
        self.x = x
        self.mu = np.mean(x)
        self.sigma = np.var(x)

    def logpdf(self):
        a = -0.5 * np.sum((self.x - self.mu) ** 2 / self.sigma + np.log(2 * np.pi * self.sigma))      # lodpdf of Gaussian as defined by ChatGPT which seems to work
        return a
