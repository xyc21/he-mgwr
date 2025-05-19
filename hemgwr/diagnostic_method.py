import numpy as np


def getAICc(gwr):
    n = gwr.N
    k = gwr.tr_S()
    aicc = -2.0 * gwr.llf() + 2.0 * n * (k + 1.0) / ( n - k - 2.0)
    return aicc


def getAIC(gwr):
    k = gwr.tr_S()
    aic = -2.0 * gwr.llf() + 2.0 * (k + 1)
    return aic





def getCV(gwr):
    aa = gwr.resid_response.reshape((-1, 1)) / (1.0 - gwr.influ)
    cv = np.sum(aa ** 2) / gwr.n
    return cv


def corr(cov):
    invsd = np.diag(1 / np.sqrt(np.diag(cov)))
    cors = np.dot(np.dot(invsd, cov), invsd)
    return cors