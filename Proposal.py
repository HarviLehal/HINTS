import numpy as np


class Proposal():

    def rw(theta, stepsize):
        dim = 1
        try:
            dim = np.shape(theta)[1]
        except IndexError:  # Added as 1D arrays have no dim in np.shape()
            pass
        if dim == 1:
            theta_n = theta + np.random.normal(scale=stepsize, size=1)
        else:
            theta_n = theta + np.eye(dim)*np.random.normal(scale=stepsize, size=1)
        return theta_n
