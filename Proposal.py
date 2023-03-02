import numpy as np
from numpy.linalg import eigvals


class Proposal():

    def rw1(theta, stepsize):
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

# ALTERNATIVE CHATGPT AIDED VERSION????

    def rw2(theta, stepsize):
        dim = 1
        try:
            dim = np.shape(theta)[1]
        except IndexError:
            pass
        if dim == 1:
            theta_n = theta + np.random.normal(scale=stepsize, size=1)
        else:
            # Ensure the matrix is symmetric
            theta = (theta + theta.T) / 2
            # Compute eigenvalues and eigenvectors of the matrix
            w, v = np.linalg.eigh(theta)
            # Apply random perturbation to the eigenvalues
            w_proposed = w + stepsize * np.random.normal(size=w.shape)
            # Ensure eigenvalues are positive
            w_proposed[w_proposed < 0] = 0
            # Reconstruct the proposed covariance matrix
            theta_n = v @ np.diag(w_proposed) @ v.T
            # Ensure proposed matrix is positive semi-definite
            if np.any(eigvals(theta_n) < 0):
                # If proposed matrix is not positive semi-definite,
                # project it onto the nearest positive semi-definite matrix
                eigvals_proposed, eigvecs_proposed = np.linalg.eigh(theta_n)
                eigvals_proposed[eigvals_proposed < 0] = 0
                theta_n = eigvecs_proposed @ np.diag(eigvals_proposed) @ eigvecs_proposed.T
            # Ensure proposed matrix is symmetric
            theta_n = (theta_n + theta_n.T) / 2
        return theta_n

# ALTERNATIVE CHATGPT AIDED VERSION WITH 1D CORRECTION?

    def rw3(theta, stepsize):
        dim = 1
        try:
            dim = np.shape(theta)[1]
        except IndexError:
            pass
        if dim == 1:
            theta_n = []
            for i in theta:
                theta_n.append(i+np.random.normal(scale=stepsize, size=1))
            theta_n = np.ndarray.flatten(np.array(theta_n))
        else:
            # Ensure the matrix is symmetric
            theta = (theta + theta.T) / 2
            # Compute eigenvalues and eigenvectors of the matrix
            w, v = np.linalg.eigh(theta)
            # Apply random perturbation to the eigenvalues
            w_proposed = w + 0.1 * np.random.normal(size=w.shape)
            # Ensure eigenvalues are positive
            w_proposed[w_proposed < 0] = 0
            # Reconstruct the proposed covariance matrix
            theta_n = v @ np.diag(w_proposed) @ v.T
            # Ensure proposed matrix is positive semi-definite
            if np.any(eigvals(theta_n) < 0):
                # If proposed matrix is not positive semi-definite,
                # project it onto the nearest positive semi-definite matrix
                eigvals_proposed, eigvecs_proposed = np.linalg.eigh(theta_n)
                eigvals_proposed[eigvals_proposed < 0] = 0
                theta_n = eigvecs_proposed @ np.diag(eigvals_proposed) @ eigvecs_proposed.T
            # Ensure proposed matrix is symmetric
            theta_n = (theta_n + theta_n.T) / 2
        return theta_n
