import numpy as np


def gaussian(x, mean, var):
    mu = np.mean(x)
    # n = len(x)                                                            # No.of samples
    # prec = var**(-1)                                                      # Precision
    # f = np.vstack(x-mean)                                                 # x - mu vectorised
    # a = -n*np.log(np.sqrt(var))*(-0.5 * prec*f.T@f)                       # logpdf of Gaussian
    a = -0.5 * np.sum((x - mean) ** 2 / var + np.log(2 * np.pi * var))      # lodpdf of Gaussian as defined by ChatGPT which seems to work
    return a
