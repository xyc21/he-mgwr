from tqdm import tqdm
import numpy as np

from .diagnostic_method import getAICc, getAIC, getCV
from .kernel import Kernel


import time

from .tool import k_nearest_neighbors_all


class GWR():
    """
    Parameters
    ----------
    coords : array-like
        n*2 array of (x, y) coordinates for n observations; also used to calibrate positions when 'points' is set to None.

    y : array
        n*1 array of dependent variable values.

    X : array
        n*k array of independent variables, excluding the intercept.

    bw : scalar
        Bandwidth value, can be a distance or the number of nearest neighbors; user-specified or obtained via Sel_BW.

    kernel : str
        Type of kernel function used for weighting observations. Options:
        'gaussian'
        'bisquare'

    fixed : boolean
        If True, use a distance-based kernel; if False, use an adaptive (nearest-neighbor) kernel (default).

    constant : boolean
        If True, include an intercept in the model (default); if False, exclude the intercept.


    name_x : list of str
        Names of the independent variables, used for output.

    Attributes
    ----------
    coords : array-like
        n*2 array of (x, y) coordinates used for calibration.

    y : array
        n*1 array of dependent variable values.

    X : array
        n*k array of independent variables.

    bw : scalar
        Bandwidth value, either distance or number of neighbors.

    sigma2_v1 : boolean
        Denominator form for sigma-squared diagnostics.

    kernel : str
        Kernel function used for weighting.

    fixed : boolean
        Whether the kernel is fixed-distance or adaptive.

    constant : boolean
        Whether an intercept is included.


    n : int
        Number of observations.

    k : int
        Number of independent variables.

    Example
    -------
    # Basic model calibration
    """

    def __init__(self, coords, Y,X, bw, kernel_type='bisquare', fixed=False, constant=True,
                   name_x=None,gpu_num=-1 ):
        """
        Initialize class
        """
        self.constant = constant#是否添加截距项
        self.coords = coords#二维地理坐标
        self.Y=Y
        self.X=X
        self.bw = bw#带宽
        self.kernel_type = kernel_type#核函数类型
        self.fixed = fixed#是否为固定型
        self.name_x = name_x
        self.gpu_num=gpu_num

        self.N=X.shape[0]
        self.K=X.shape[1]

    def fit(self,lite=False,all_distances=None):
        """
        Initialize class

        Parameters
        ----------
        lite : bool, optional
            Whether to estimate a lightweight version of GWR.
            Computes only the minimum required diagnostics.
            Bandwidth selection can be faster.
            Default is False.

        Returns
        -------
        object
            If lite is False, returns a GWRResult instance;
            otherwise, returns a GWRResultLite instance.
        """


        if self.constant:

            ones_column = np.ones((self.N, 1))

            self.X = np.hstack((ones_column, self.X))
            self.K += 1
        else:
            self.X = self.X

        S, predy, beta =Kernel(self.coords,self.Y,self.X,self.N,self.K,self.bw,self.kernel_type,self.fixed,self.gpu_num).run(all_distances=all_distances)
        redis=self.Y.reshape((-1,1))-predy


        return GWRResult(self,beta,predy,S,redis)


class GWRResult():
    """
    Base class containing common attributes for all GWR regression models.

    Parameters
    ----------
    model : GWR object
        Pointer to the GWR object containing the estimation parameters.

    params : array
        n*k array of estimated coefficients.

    predy : array
        n*1 array of predicted y values.

    S : array
        n*n hat matrix.

    CCT : array
        n*k scaled variance-covariance matrix.

    name_x : list of str
        Names of the independent variables used for output.
    """


    def __init__(self,model, beta, predy, S, resid):

        self.predy = predy
        self.S = S
        self.Y=model.Y.reshape((-1,1))
        self.N=model.N

        self._cache = {}

        self.beta=beta
        self.resid = resid
        self.K=model.K
        self.model=model
        self.name_x=model.name_x
    def tr_S(self):
        return np.sum(self.S)

    def llf(self):
        nobs2 = self.Y.shape[0] / 2.0
        SSR = np.sum((self.Y - self.predy) ** 2, axis=0)
        llf = -np.log(SSR) * nobs2
        llf -= (1 + np.log(np.pi / nobs2)) * nobs2
        return llf



    def resid_ss(self):
        u = self.resid.flatten()
        return np.dot(u, u.T)
    # def df_model(self):
    #     return self.N - self.tr_S
    # def sigma2(self):
    #     return (self.resid_ss / (self.N - self.tr_S))
    def aicc(self):
        return getAICc(self)
    def aic(self):
        return getAIC(self)
    def cv(self):
        return getCV(self)
    def calculate_r2_adj_r2(self):

        ss_tot = np.sum((self.Y - np.mean(self.predy)) ** 2)


        ss_res = np.sum((self.Y - self.predy) ** 2)


        self.r2 = 1 - (ss_res / ss_tot)

        n = len(self.Y)

        self.adj_r2 = 1 - ((1 - self.r2) * (n - 1) / (n - self.tr_S() - 1))
    def evaluation(self):
        self.rss=self.resid_ss()
        self.traceS=self.tr_S()
        self.df_model=self.N - self.traceS
        self.sigma2=self.rss / (self.N - self.traceS)
        self._llf=self.llf()
        self._aicc=self.aicc()
        self._aic=self.aic()

        self.mae = np.mean(np.abs(self.Y - self.predy))


        self.rmse = np.sqrt(np.mean((self.Y - self.predy) ** 2))




    def summary(self):
        self.evaluation()
        self.calculate_r2_adj_r2()
        if self.name_x is not None:
            XNames = list(self.name_x)
            if len(XNames) < self.K:
                XNames = ["Intercept"] + XNames

        else:
            XNames = ["X" + str(i) for i in range(self.K)]
        summary = '=' * 75 + '\n'
        summary += "%37s\n" % ('summary')
        summary += '=' * 75 + '\n'
        summary += "%-60s %14d\n" % ('Number of observations:', self.N)
        summary += "%-60s %14d\n\n" % ('Number of covariates:', self.K)
        summary += "%s\n" % ('Geographically Weighted Regression (GWR) Results')
        summary += '-' * 75 + '\n'

        if self.model.fixed:
            summary += "%-50s %20s\n" % ('Spatial kernel:',
                                         'Fixed ' + self.model.kernel_type)
        else:
            summary += "%-54s %20s\n" % ('Spatial kernel:',
                                         'Adaptive ' + self.model.kernel_type)

        summary += "%-62s %12.3f\n" % ('Bandwidth used:', self.model.bw)

        summary += "\n%s\n" % ('Diagnostic information')
        summary += '-' * 75 + '\n'


        summary += "%-62s %12.3f\n" % ('Residual sum of squares:',
                                       self.rss)
        summary += "%-62s %12.3f\n" % (
            'Effective number of parameters (trace(S)):', self.traceS)
        summary += "%-62s %12.3f\n" % ('Degree of freedom (n - trace(S)):',
                                       self.df_model)
        summary += "%-62s %12.3f\n" % ('Sigma estimate:', np.sqrt(self.sigma2))
        summary += "%-62s %12.3f\n" % ('Log-likelihood:', self._llf)
        summary += "%-62s %12.3f\n" % ('AIC:', self._aic)
        summary += "%-62s %12.3f\n" % ('AICc:', self._aicc)
        summary += "%-62s %12.3f\n" % ('Root mean square error', self.rmse)
        summary += "%-62s %12.3f\n" % ('Mean absolute error', self.mae)
        summary += "%-62s %12.3f\n" % ('R2:', self.r2)
        summary += "%-62s %12.3f\n" % ('Adjusted R2:', self.adj_r2)



        summary += "\n%s\n" % ('Summary Statistics For GWR Parameter Estimates')
        summary += '-' * 75 + '\n'
        summary += "%-20s %10s %10s %10s %10s %10s\n" % ('Variable', 'Mean', 'STD',
                                                         'Min', 'Median', 'Max')
        summary += "%-20s %10s %10s %10s %10s %10s\n" % (
            '-' * 20, '-' * 10, '-' * 10, '-' * 10, '-' * 10, '-' * 10)
        for i in range(self.K):
            summary += "%-20s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (
                XNames[i][:20], np.mean(self.beta[:, i]), np.std(self.beta[:, i]),
                np.min(self.beta[:, i]), np.median(self.beta[:, i]),
                np.max(self.beta[:, i]))

        summary += '=' * 75 + '\n'
        print(summary)
        return summary


