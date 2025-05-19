from hemgwr.gwr import GWR
import numpy as np

class MGWR(GWR):
    """
    Parameters
    ----------
    coords : array-like
        n*2 array of (x, y) coordinates for n observations; also used for calibration when 'points' is set to None.

    y : array
        n*1 array of dependent variable values.

    X : array
        n*k array of independent variables, excluding the intercept.

    bw : scalar
        Bandwidth value, either a distance or a number of nearest neighbors; specified by the user or obtained via Sel_BW.

    kernel : str
        Type of kernel function used to weight observations. Options:
        'gaussian'
        'bisquare'

    fixed : bool
        If True, use a fixed (distance-based) kernel; if False, use an adaptive (nearest-neighbor) kernel (default).

    constant : bool
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
        Bandwidth value, either a distance or a number of nearest neighbors.

    kernel : str
        Kernel function used to weight observations.

    fixed : bool
        Whether the kernel is fixed or adaptive.

    constant : bool
        Whether an intercept is included in the model.

    n : int
        Number of observations.

    k : int
        Number of independent variables.

    Example
    -------
    # Basic model calibration
    """

    def __init__(self, coords, Y, X,  selector,
                 kernel_type='bisquare', fixed=False, constant=True, name_x=None,gpu_num=1):
        """
        Initialize class
        """
        self.selector = selector
        self.bws = self.selector.bw[0]  #final set of bandwidth
        self.bws_history = selector.bw[1]  #bws history in backfitting
        self.bw_init = self.selector.bw_init  #initialization bandiwdth

        GWR.__init__ (self, coords, Y, X, self.bw_init,  kernel_type=kernel_type, fixed=fixed, constant=constant,
                      name_x=None, gpu_num=gpu_num)
        self.selector = selector
        # self.P = None
        # self.exog_resid = None
        # self.exog_scale = None
        self.name_x = name_x
        self.gpu_num = gpu_num

    def fit(self):
        if self.constant:

            ones_column = np.ones((self.N, 1))

            self.X = np.hstack((ones_column, self.X))
            self.K += 1
        else:
            self.X = self.X
        params = self.selector.params
        predy = np.sum(self.X * params, axis=1).reshape(-1, 1)

        return MGWRResult(self,params,predy)



class MGWRResult():
    """
    Base class that includes common attributes for all GWR regression models.

    Parameters
    ----------
    model : GWR object
        Reference to a GWR object containing the estimated parameters.

    params : array
        n*k array of estimated coefficients.

    predy : array
        n*1 array of predicted y-values.

    S : array
        n*n hat matrix.

    CCT : array
        n*k scaled variance-covariance matrix.

    name_x : list of str
        Names of the independent variables, used for output.
    """


    def __init__(self,model, beta, predy):

        self.predy = predy
        self.Y=model.Y.reshape((-1,1))
        self.N=model.N
        self.beta=beta
        self.K=model.K
        self.model=model
        self.name_x=model.name_x

    def calculate_r2_adj_r2(self):
        """
        Compute R-squared (R²) and adjusted R-squared.

        Parameters
        ----------
        y_true : numpy array
            True (observed) values.

        y_pred : numpy array
            Predicted values.

        num_features : int
            Number of features (independent variables).

        Returns
        -------
        r2 : float
            R-squared value.

        adj_r2 : float
            Adjusted R-squared value.
        """


        ss_tot = np.sum((self.Y - np.mean(self.predy)) ** 2)


        ss_res = np.sum((self.Y - self.predy) ** 2)


        self.r2 = 1 - (ss_res / ss_tot)


        n = len(self.Y)


        self.adj_r2 = 1 - ((1 - self.r2) * (n - 1) / (n - self.K - 1))

    def residual_sum_of_squares(self):
        """
        Compute the Residual Sum of Squares (RSS).

        Parameters
        ----------
        y_true : numpy array or list
            Actual (true) values.

        y_pred : numpy array or list
            Predicted values.

        Returns
        -------
        float
            Residual Sum of Squares (RSS).
        """

        residuals = np.array(self.Y) - np.array(self.predy)
        self.rss = np.sum(residuals ** 2)

        self.mae = np.mean(np.abs(self.Y - self.predy))

        self.rmse = np.sqrt(np.mean((self.Y - self.predy) ** 2))

    def summary(self):
        self.calculate_r2_adj_r2()
        self.residual_sum_of_squares()
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

        summary += "%s\n" % ('Multi-Scale Geographically Weighted Regression (MGWR) Results')
        summary += '-' * 75 + '\n'
        # summary += "%-54s %20s\n" % ('Model type', self.model.kernel_type)
        if self.model.fixed:
            summary += "%-50s %20s\n" % ('Spatial kernel:',
                                         'Fixed ' + self.model.kernel_type)
        else:
            summary += "%-54s %20s\n" % ('Spatial kernel:',
                                         'Adaptive ' + self.model.kernel_type)

        summary += "%-54s %20s\n" % ('Criterion for optimal bandwidth:',
                                     self.model.selector.criterion)

        if self.model.selector.rss_score:
            summary += "%-54s %20s\n" % ('Score of Change (SOC) type:', 'RSS')
        else:
            summary += "%-54s %20s\n" % ('Score of Change (SOC) type:',
                                         'Smoothing f')
        summary += "%-54s %20s\n\n" % ('Termination criterion for MGWR:',
                                       self.model.selector.tol_multi)

        summary += "%s\n" % ('MGWR bandwidths')
        summary += '-' * 75 + '\n'
        summary += "%-15s %14s %16s\n" % (
            'Variable', 'Bandwidth',self.model.selector.criterion)
        for j in range(self.K):
            value = self.model.selector.bw[5][len(self.model.selector.bw[5]) - self.K + j][-1][1][0]
            result = round(value, 5)
            summary += "%-14s %15.3f %20s\n" % (
                XNames[j], self.model.bws[j],result
            )
        summary += "\n%s\n" % ('Diagnostic information')
        summary += '-' * 75 + '\n'
        summary += "%-62s %12.3f\n" % ('Residual sum of squares:', self.rss)
        summary += "%-62s %12.3f\n" % ('Root mean square error', self.rmse)
        summary += "%-62s %12.3f\n" % ('Mean absolute error', self.mae)


        summary += "%-62s %12.3f\n" % ('R2', self.r2)
        # summary += "%-62s %12.3f\n" % ('Adjusted R2', self.adj_r2)
        summary += "\n%s\n" % ('Summary Statistics For MGWR Parameter Estimates')
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
        summary += '-' * 75 + '\n'
        summary += "%-62s %12.3f\n" % ('iterations:', len(self.model.selector.bw[2]))
        summary += "%-62s %12.3f\n" % ('Average run time per iteration:', self.model.selector.MRunTime/len(self.model.selector.bw[2]))
        summary += "%-62s %12.3f\n" % ('Search bandwidth running time(s):', self.model.selector.MRunTime)


        summary += '=' * 75 + '\n'

        print(summary)
        return summary



