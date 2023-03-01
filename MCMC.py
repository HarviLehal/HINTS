import numpy as np
from tqdm import tqdm


class MCMC():

    def __init__(self, x, theta0, target, proposal, stepsize):
        self.x = x                          # Data
        self.theta0 = theta0                # Initial Parameter values
        self.logpdf = target.logpdf         # Logpdf of Target
        self.proposal = proposal            # Proposal Method
        self.stepsize = stepsize            # Stepsize
        self.plot = target.plot             # Parameter Plotting

    def prop(self, theta):                  # Theta previous parameter, Theta_n proposal parameter
        theta_n = self.proposal(theta, self.stepsize)
        return theta_n

    def ratio(self, x, theta, theta_n):

        a = self.logpdf(x, *theta_n)        # logpdf of proposal
        b = self.logpdf(x, *theta)          # logpdf of previous
        a = a-b                             # Acceptance Ratio
        a = np.exp(a)                       # Exponent
        u = np.random.uniform(0, 1, 1)
        if u <= a:
            return theta_n                  # Accept Proposal
        else:
            return theta                    # Reject Proposal

    def mcmc(self, M):                                                                         # Test mcmc sampler
        thetas = []                                                                         # blank list to save param
        thetas.append(self.theta0)                                                          # add initial param values
        for i in tqdm(range(M-1)):                                                     # for each iteration:
            thetan = {}                                                                     # blank dict for the prop
            for j in range(len(self.theta0)):                                               # for each parameter:
                thetan[j] = self.prop(thetas[i][j])                                         # new proposal for param j
            thetan = self.ratio(self.x, list(thetas[i].values()), list(thetan.values()))    # M-H accept-reject step
            p = {}
            for j in range(len(self.theta0)):                                               # for each parameter:
                p[j] = thetan[j]
            thetas.append(p)                                                                # append param val to list
        self.thetas = thetas                                                                # save list
        self.plot(self.thetas)
