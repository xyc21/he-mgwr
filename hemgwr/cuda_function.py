import numba
from numba import cuda
import math


@cuda.jit
def calculate_B_matrix(dev_coords, dev_X, dev_B, N, K, bandWidth, data_size_per_gpu, start_index, use_gussian):
    tx = cuda.threadIdx.x
    tid = tx + cuda.blockDim.x * cuda.blockIdx.x
    regression_point_id = tx + cuda.blockDim.x * cuda.blockIdx.x + start_index
    B_per_size = K * K
    if tid >= data_size_per_gpu:
        return

    shared_array_B = cuda.shared.array(shape=(0,), dtype=numba.float64)  # shape=0 表示动态分配
    for i in range(K):
        for j in range(K):
            shared_array_B[tx * B_per_size + i * K + j] = 0.0
            for xi in range(N):
                if use_gussian:
                    weight = _gussian(bandWidth[regression_point_id], dev_coords[regression_point_id, 0],
                                      dev_coords[regression_point_id, 1],
                                      dev_coords[xi, 0], dev_coords[xi, 1])
                else:
                    weight = _bi_square(bandWidth[regression_point_id], dev_coords[regression_point_id, 0],
                                        dev_coords[regression_point_id, 1],
                                        dev_coords[xi, 0], dev_coords[xi, 1])
                shared_array_B[tx * B_per_size + i * K + j] += dev_X[xi, i] * weight * dev_X[xi, j]

            dev_B[tid, i * K + j] = shared_array_B[tx * B_per_size + i * K + j]


@cuda.jit
def Calculate_S_PY(dev_coords, dev_X, dev_Y, dev_pinvB, dev_S, dev_predy, dev_Beta, N, K, bandWidth, data_size_per_gpu,
                   start_index, use_gussian):
    tx = cuda.threadIdx.x
    tid = tx + cuda.blockDim.x * cuda.blockIdx.x
    regression_point_id = tx + cuda.blockDim.x * cuda.blockIdx.x + start_index
    if tid >= data_size_per_gpu:
        return
    B_per_size = K * K
    share_s_start_id = tx * (2 + K)
    share_py_start_id = share_s_start_id + 1
    share_Beta_start_id = share_s_start_id + 2

    shared_array = cuda.shared.array(shape=(0,), dtype=numba.float64)  # shape=0 表示动态分配

    shared_array[share_s_start_id] = 0.0
    shared_array[share_py_start_id] = 0.0
    if use_gussian:

        W_ii = _gussian(bandWidth[regression_point_id], dev_coords[regression_point_id, 0], dev_coords[regression_point_id, 1],
                        dev_coords[regression_point_id, 0], dev_coords[regression_point_id, 1])
    else:
        W_ii = _bi_square(bandWidth[regression_point_id], dev_coords[regression_point_id, 0], dev_coords[regression_point_id, 1],
                          dev_coords[regression_point_id, 0], dev_coords[regression_point_id, 1])
    for i in range(K):
        shared_array[share_Beta_start_id + i] = 0.0
        xi_pinvB = 0.0
        for j in range(K):
            xi_pinvB += dev_X[regression_point_id, j] * dev_pinvB[regression_point_id, j * K + i]
        shared_array[share_s_start_id] += xi_pinvB * dev_X[regression_point_id, i] * W_ii
        xT_W_Y = 0.0
        for xi in range(N):
            if use_gussian:
                xT_W_Y += dev_X[xi, i] * _gussian(bandWidth[regression_point_id], dev_coords[regression_point_id, 0], dev_coords[regression_point_id, 1],
                                                  dev_coords[xi, 0], dev_coords[xi, 1]) * dev_Y[xi]
                Beta = 0.0
                for j in range(K):
                    Beta += dev_pinvB[regression_point_id, i * K + j] * dev_X[xi, j]
                shared_array[share_Beta_start_id + i] += Beta * _gussian(bandWidth[regression_point_id],dev_coords[regression_point_id,0],
                                                                         dev_coords[regression_point_id,1],
                                                                         dev_coords[xi,0], dev_coords[xi,1]) * dev_Y[xi]

            else:
                xT_W_Y += dev_X[xi, i] * _bi_square(bandWidth[regression_point_id], dev_coords[regression_point_id, 0], dev_coords[regression_point_id, 1],
                                                  dev_coords[xi, 0], dev_coords[xi, 1]) * dev_Y[xi]
                Beta = 0.0
                for j in range(K):
                    Beta += dev_pinvB[regression_point_id, i * K + j] * dev_X[xi, j]
                shared_array[share_Beta_start_id + i] += Beta * _bi_square(bandWidth[regression_point_id],dev_coords[regression_point_id,0],
                                                                         dev_coords[regression_point_id,1],
                                                                         dev_coords[xi,0], dev_coords[xi,1]) * dev_Y[xi]
        shared_array[share_py_start_id] += xi_pinvB * xT_W_Y
        dev_Beta[tid,i]=shared_array[share_Beta_start_id + i]
    dev_S[tid]=shared_array[share_s_start_id]
    dev_predy[tid]=shared_array[share_py_start_id]


@cuda.jit(device=True)
def _gussian(bandWidth, U1, V1, U2, V2):
    dist = ((U1 - U2) ** 2 + (V1 - V2) ** 2) ** 0.5
    return math.exp(-0.5 * (dist / bandWidth) ** 2)


@cuda.jit(device=True)
def _bi_square(band_width, u1, v1, u2, v2):
    dist = math.sqrt((u1 - u2) ** 2 + (v1 - v2) ** 2)
    Hs = dist / band_width
    if dist>=band_width:
        return 0
    else:
        return (1 - Hs ** 2) ** 2
