import numpy as np


class Proposal():

    def propose(theta):
        dim = 1
        try:
            dim = np.shape(theta)[1]
        except IndexError:              # Added due to 1D arrays having no dimensions in the shape function
            pass
        if dim == 1:
            theta_n = theta + np.random.normal(size=1)
        else:
            theta_n = theta + np.eye(dim)*np.random.normal(size=1)
        return theta_n
