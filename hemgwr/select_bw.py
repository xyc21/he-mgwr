import sys
import time

import numpy as np
from .Search import golden_section, multi_bw_search
from .diagnostic_method import getAIC,getAICc,getCV
from .gwr import GWR
from .tool import find_min_max_distances, gpuInformation, get_gpuids
from .kernel import Kernel

Diag = {'AICc': getAICc, 'AIC': getAIC, 'CV': getCV}
class Select_bw():
    def __init__(self, coords,Y, X,fixed=False,kernel_type='bisquare', multi=False,
                 constant=True, gpu_num=-1):
        self.bw = None
        self.coords = np.array(coords)
        self.Y = Y
        self.X = X
        self.fixed = fixed
        self.kernel_type = kernel_type
        self.multi = multi
        self._functions = []
        self.constant = constant
        self.search_params = {}
        self.gpu_num=gpu_num
        self.N=X.shape[0]
        self.K=X.shape[1]
        self.RunTime=None

    def search(self,  criterion='AICc',
               bw_min=None, bw_max=None, tol=1.0e-6,
               max_iter=200,  tol_multi=1.0e-5,
               rss_score=False, max_iter_multi=200, multi_bw_min=[None],
               multi_bw_max=[None], bws_same_times=5, verbose=False,gpuinfor=True,all_distances=None):

        k = self.X.shape[1]
        if self.constant:  # k is the number of covariates
            k += 1
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.bws_same_times = bws_same_times
        self.verbose = verbose
        self.all_distances=all_distances
        if len(multi_bw_min) == k:
            self.multi_bw_min = multi_bw_min
        elif len(multi_bw_min) == 1:
            self.multi_bw_min = multi_bw_min * k
        else:
            print(
                "multi_bw_min must be a list containing a single item, or a list with k items corresponding to each covariate (including the intercept)")
            sys.exit()
        if len(multi_bw_max) == k:
            self.multi_bw_max = multi_bw_max
        elif len(multi_bw_max) == 1:
            self.multi_bw_max = multi_bw_max * k
        else:
            print("multi_bw_max must be a list containing a single item, or a list with k items corresponding to each covariate (including the intercept)")
            sys.exit()
        self.tol = tol
        self.max_iter = max_iter
        self.tol_multi = tol_multi
        self.rss_score = rss_score
        self.max_iter_multi = max_iter_multi
        self.search_params['criterion'] = criterion
        self.search_params['bw_min'] = bw_min
        self.search_params['bw_max'] = bw_max
        self.search_params['tol'] = tol
        self.search_params['max_iter'] = max_iter
        if gpuinfor:
            get_gpuids(self.gpu_num)

        if self.multi:
            M_start_time=time.time()
            self.mgwr_bw()
            M_end_time=time.time()
            self.MRunTime=M_end_time-M_start_time
            self.params = self.bw[3]  # params n by k
            self.sel_hist = self.bw[-2]  # bw searching history
            self.mgwr_sel_hist=self.bw[1]
            self.bw_init = self.bw[-1]  # scalar, optimal bw from initial gwr model
        else:
            start_time = time.time()
            self.gwr_bw()
            end_time=time.time()
            self.RunTime=end_time-start_time
            self.sel_hist = self.bw[-1]

        return self.bw[0]
    def gwr_bw(self):
        gwr_func = lambda bw: Diag[self.criterion](GWR(
            self.coords, self.Y, self.X, bw,  kernel_type=
            self.kernel_type, fixed=self.fixed, constant=self.constant,gpu_num=self.gpu_num).fit(lite=True,all_distances=self.all_distances))
        self._optimized_function = gwr_func
        a, c = self._init_section()
        delta = 0.38197  # 1 - (np.sqrt(5.0)-1.0)/2.0
        self.bw = golden_section(a, c, delta, gwr_func, self.tol,
                                 self.max_iter, self.bw_max, self.fixed,
                                 self.verbose)
    def mgwr_bw(self):
        y=self.Y
        if self.constant:
            ones_column = np.ones((self.N, 1))
            self.X = np.hstack((ones_column, self.X))
            self.K+=1
        else:
            self.X= self.X
        def gwr_function(y, x, bw):
            return GWR(self.coords, y, x,bw,  kernel_type=self.kernel_type,
                       fixed=self.fixed,  constant=False,gpu_num=self.gpu_num).fit(
                           lite=True,all_distances=self.all_distances)

        def bw_function(y, x):
            selector = Select_bw(self.coords, y, x,
                              kernel_type=self.kernel_type,fixed=self.fixed,
                              constant=False, gpu_num=self.gpu_num)
            return selector

        def sel_function(bw_func, bw_min=None, bw_max=None):
            return bw_func.search(criterion=self.criterion,
                bw_min=bw_min, bw_max=bw_max,  tol=self.tol,
                max_iter=self.max_iter, verbose=False,gpuinfor=False,all_distances=self.all_distances)

        self.bw = multi_bw_search(self.Y, self.X, self.N, self.K, self.tol_multi,
                           self.max_iter_multi, self.rss_score, gwr_function,
                           bw_function, sel_function, self.multi_bw_min, self.multi_bw_max,
                           self.bws_same_times, verbose=self.verbose)

    def _init_section(self):
        n = np.array(self.coords).shape[0]
        if self.constant:
            n_vars =  self.K + 1
        else:
            n_vars = self.K
        if not self.fixed:
            a = 40 + 2 *n_vars
            if n>20000:
                c = 20000
            else:
                c=n
        else:
            min_dist,max_dist=find_min_max_distances(self.coords)

            a = min_dist / 2.0
            c = max_dist * 2.0

        if self.bw_min is not None:
            a = self.bw_min
        if self.bw_max is not None and self.bw_max is not np.inf:
            c = self.bw_max

        return a, c