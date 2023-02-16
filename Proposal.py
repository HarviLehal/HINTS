import numpy as np


class Proposal():

    def propose(theta, dim):
        if dim == 1:
            theta_n = theta + np.random.normal(size=1)
        else:
            theta_n = theta + np.eye(dim)*np.random.normal(size=1)
        return theta_n
