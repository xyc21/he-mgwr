from numba import njit,prange
import numpy as np
from numba import cuda

@njit(parallel=True)
def k_nearest_neighbors(coords, k):
    k=k-1
    eps = 1.0000001
    n = coords.shape[0]

    indices = np.empty(n, dtype=np.int64)
    distances = np.empty(n, dtype=np.float64)

    for i in prange(n):

        dists = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))


        kth_index = np.argpartition(dists, k)[k]
        indices[i] = kth_index
        distances[i] = dists[kth_index]* eps

    return indices, distances


@njit(parallel=True)
def k_nearest_neighbors_all(coords, k=20000):
    eps = 1.0000001
    n = coords.shape[0]



    distances = np.empty((n, k), dtype=np.float64)

    for i in prange(n):

        dists = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))


        k_indices = np.argpartition(dists, k)[:k]
        k_dists = dists[k_indices]

        sorted_indices = np.argsort(k_dists)
        k_indices = k_indices[sorted_indices]
        k_dists = k_dists[sorted_indices]


        distances[i, :] = k_dists * eps

    return distances
@njit(parallel=True)
def find_min_max_distances(coords):

    n = coords.shape[0]
    thread_min_dists = np.full(n, np.inf)
    thread_max_dists = np.full(n, -np.inf)

    for i in prange(n):

        other_coords = np.concatenate((coords[:i], coords[i + 1:]))


        dists = np.sqrt(np.sum((other_coords - coords[i]) ** 2, axis=1))


        thread_min_dists[i] = np.min(dists)
        thread_max_dists[i] = np.max(dists)


    min_dist = np.min(thread_min_dists)
    max_dist = np.max(thread_max_dists)

    return min_dist, max_dist
def gpuInformation():
    gpu_count = len(cuda.gpus)


    gpus = []
    for i in range(gpu_count):
        gpu = cuda.gpus[i]
        gpus.append((gpu.id, gpu.name))


    for gpu_id, gpu_name in gpus:
        print(f"GPU ID: {gpu_id}, Name: {gpu_name}")
    return gpus
def get_gpuids(gpu_num):
    gpu_ids = []
    if gpu_num == -1:
        gpu_count = len(cuda.gpus)

        for i in range(gpu_count):
            gpu = cuda.gpus[i]
            gpu_ids.append(gpu.id)
            print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
    else:
        gpu_count = len(cuda.gpus)
        if gpu_count >= gpu_num:
            for i in range(gpu_num):
                gpu = cuda.gpus[i]
                gpu_ids.append(gpu.id)
                print(f"GPU ID: {gpu.id}, Name: {gpu.name}")