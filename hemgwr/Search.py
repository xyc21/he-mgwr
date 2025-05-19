import time
from copy import deepcopy
from tqdm.auto import tqdm
import numpy as np
def golden_section(a, c, delta, function, tol, max_iter, bw_max, fixed,
                   verbose=False):
    b = a + delta * np.abs(c - a)
    d = c - delta * np.abs(c - a)

    opt_score = np.inf
    diff = 1.0e9
    iters = 0
    output = []
    dict = {}
    while np.abs(diff) > tol and iters < max_iter and a != np.inf:
        iters += 1
        if not fixed:
            b = np.round(b).astype(int)
            d = np.round(d).astype(int)
        # Print b and d if not integers

        if b in dict:
            score_b = dict[b]
        else:
            score_b = function(b)
            dict[b] = score_b
            if verbose:
                print("Bandwidth: ", np.round(b, 2), ", score: ",
                      "{0:.6f}".format(score_b[0]))

        if d in dict:
            score_d = dict[d]
        else:
            score_d = function(d)
            dict[d] = score_d
            if verbose:
                print("Bandwidth: ", np.round(d, 2), ", score: ",
                      "{0:.6f}".format(score_d[0]))

        if score_b <= score_d:
            opt_val = b
            opt_score = score_b
            c = d
            d = b
            b = a + delta * np.abs(c - a)

        else:
            opt_val = d
            opt_score = score_d
            a = b
            b = d
            d = c - delta * np.abs(c - a)

        output.append((opt_val, opt_score))

        opt_val = np.round(opt_val, 2)
        if (opt_val, opt_score) not in output:
            output.append((opt_val, opt_score))

        diff = score_b - score_d
        score = opt_score

    if a == np.inf or bw_max == np.inf:
        score_ols = function(np.inf)
        output.append((np.inf, score_ols))

        if score_ols <= opt_score:
            opt_score = score_ols
            opt_val = np.inf

        if verbose:
            print("Bandwidth: ", np.inf, ", score: ",
                  "{0:.2f}".format(score_ols[0]))

    return opt_val, opt_score, output
def multi_bw_search(Y,X, N, K, tol_multi,
                           max_iter_multi, rss_score, gwr_function,
                           bw_function, sel_function, multi_bw_min, multi_bw_max,
                           bws_same_times, verbose):
    if all(item is not None for item in multi_bw_min) and all(item is not None for item in multi_bw_max):
        bw_min_mean=sum(multi_bw_min) / len(multi_bw_min)
        bw_max_mean=sum(multi_bw_max) / len(multi_bw_max)
    else:
        bw_min_mean=None
        bw_max_mean=None
    bw = sel_function(bw_function(Y, X), bw_min=bw_min_mean, bw_max=bw_max_mean)
    gwr_model = gwr_function(Y, X, bw)
    bw_gwr = bw
    _err=gwr_model.resid
    params=gwr_model.beta
    XB = np.multiply(params, X)
    if rss_score:
        rss = np.sum((_err)**2)

    iters = 0
    scores = []
    delta = 1e6
    BWs = []
    bw_stable_counter = 0
    bws = np.zeros(K)
    gwr_sel_hist = []
    # for iters in tqdm(range(1, max_iter + 1), desc='Backfitting'):
    for iters in range(1, max_iter_multi + 1):
        new_XB = np.zeros_like(X)
        params = np.zeros_like(X)
        st=time.time()

        for j in tqdm(range(K),desc='Backfitting'):
            temp_y = XB[:, j].reshape((-1, 1)).astype(np.float64)
            temp_y = (temp_y + _err).reshape(-1)
            temp_X = X[:, j].reshape((-1, 1)).astype(np.float64)
            bw_class = bw_function(temp_y, temp_X)
            if bw_stable_counter >= bws_same_times:
                bw = bws[j]
            else:
                bw = sel_function(bw_class,multi_bw_min[j], multi_bw_max[j])
                gwr_sel_hist.append(deepcopy(bw_class.sel_hist))

            gwr_model = gwr_function(temp_y, temp_X, bw)
            _err = gwr_model.resid.reshape((-1, 1))
            param = gwr_model.beta.reshape((-1,))
            new_XB[:, j] = gwr_model.predy.reshape(-1)
            params[:, j] = param
            bws[j] = bw
            if not bw_class.fixed:
                bws = bws.astype(int)  # 将数组转换为整型

        # If bws remain the same as from previous iteration
        if (iters > 1) and np.all(BWs[-1] == bws):
            bw_stable_counter += 1
        else:
            bw_stable_counter = 0

        num = np.sum((new_XB - XB) ** 2) / N
        den = np.sum(np.sum(new_XB, axis=1) ** 2)
        score = (num / den) ** 0.5
        XB = new_XB

        if rss_score:
            predy = np.sum(np.multiply(params, X), axis=1).reshape((-1, 1))
            new_rss = np.sum((Y.reshape((-1,1)) - predy) ** 2)
            score = np.abs((new_rss - rss) / new_rss)
            rss = new_rss
        scores.append(deepcopy(score))
        delta = score
        BWs.append(deepcopy(bws))

        if verbose:
            print("Current iteration:", iters, ",SOC:", np.round(score, 7))
            print("Bandwidths:", ', '.join([str(bw) for bw in bws]))
        et=time.time()
        if delta < tol_multi:
            break

    opt_bws = BWs[-1]
    return (opt_bws, np.array(BWs), np.array(scores), params, _err, gwr_sel_hist, bw_gwr)