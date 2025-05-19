import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import sys
from tqdm import tqdm
from .cuda_function import *
from .tool import k_nearest_neighbors, k_nearest_neighbors_all

all_distances=None

class Kernel:
    def __init__(self, coords, Y, X, N, K, bandWidth, kernel_type, fixed, gpu_num):
        self.X = X
        self.Y = Y
        self.coords = coords
        self.N = N
        self.K = K
        self.bandWidth = bandWidth
        self.kernel_type = kernel_type
        self.fixed = fixed
        self.gpu_num = gpu_num

    def get_gpuids(self,verbose):
        gpu_ids = []
        if self.gpu_num == -1:
            gpu_count = len(cuda.gpus)

            for i in range(gpu_count):
                gpu = cuda.gpus[i]
                gpu_ids.append(gpu.id)
                if verbose:
                    print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
            return gpu_ids
        else:
            gpu_count = len(cuda.gpus)
            if gpu_count >= self.gpu_num:
                for i in range(self.gpu_num):
                    gpu = cuda.gpus[i]
                    gpu_ids.append(gpu.id)
                    if verbose:
                        print(f"GPU ID: {gpu.id}, Name: {gpu.name}")
                return gpu_ids
            else:
                print("The number of available GPUs is less than the specified GPU count.")
                sys.exit()



    def run(self, all_distances=None):



        if self.fixed:
            start_time=time.time()
            S,predy,beta=fixedKernel(self.coords, self.Y, self.X, self.N, self.K, self.bandWidth, self.kernel_type,
                        self.gpu_num).run()
            end_time=time.time()

            return S, predy, beta
        else:




            start_time=time.time()


            S, predy, beta =adaptiveKernel(self.coords, self.Y, self.X, self.N, self.K,  self.bandWidth, self.kernel_type,
                           self.gpu_num).run(all_distances)
            end_time=time.time()
            # print("运行时间：",end_time-start_time)
            return S, predy, beta


    @staticmethod
    def distribute_data_evenly_with_indices(total_data, num_threads):

        base_data = total_data // num_threads


        remainder = total_data % num_threads


        data_size_per_gpu_list = [base_data + 1 if i < remainder else base_data for i in range(num_threads)]


        start_indices = [0] * num_threads
        for i in range(1, num_threads):
            start_indices[i] = start_indices[i - 1] + data_size_per_gpu_list[i - 1]

        return data_size_per_gpu_list, start_indices

    @staticmethod
    def compute_pseudo_inverses_cpu(matrices, M):
        """
        Compute the pseudoinverse of multiple matrices using SVD on the CPU and return them as row vectors.

        Parameters
        ----------
        matrices : np.ndarray
            A 2D array of shape (N, M^2), containing N flattened MxM matrices.

        M : int
            The dimension of each square matrix (size MxM).

        Returns
        -------
        np.ndarray
            An array of shape (N, M^2), where each row is the flattened pseudoinverse of the corresponding input matrix.
        """
        # Ensure the input is a NumPy array and convert to float64

        matrices = np.array(matrices, dtype=np.float64)
        output_matrices = []

        if matrices.shape[1] != M * M:
            raise ValueError(f"Each flattened matrix should have a length of {M * M}, but got {matrices.shape[1]}")

        regularization = 1000 * np.finfo(np.float64).eps

        for flat_matrix in matrices:

            matrix = flat_matrix.reshape(M, M)

            matrix += np.eye(M) * regularization


            u, s, vh = np.linalg.svd(matrix, full_matrices=False)

            s_inv = np.array([1 / val if val > 1e-10 else 0 for val in s])

            pseudo_inv_matrix = vh.T @ np.diag(s_inv) @ u.T


            output_matrices.append(pseudo_inv_matrix.flatten())

        return np.array(output_matrices)




class fixedKernel(Kernel):
    """
    Initialize the class.

    Parameters
    ----------
    dev_XOne : array
        Input data array.

    coords : array
        Coordinates for variable U.

    N : int
        Number of observations.

    K : int
        Number of independent variables.

    bandwidth : float
        Bandwidth parameter, e.g., for kernel processing.

    kernel_type : str
        Type of kernel function.
    """


    def __init__(self, coords, Y, X, N, K, bandWidth, kernel_type, gpu_num):
        self.X = X  # 输入数据
        self.Y = Y  # 输入数据
        self.coords = coords  # 变量 U
        self.N = N  # 数据的维度 N
        self.K = K  # 数据的维度 K
        self.bandWidth = bandWidth  # 带宽参数
        self.kernel_type = kernel_type  # 核函数的类型
        self.gpu_num = gpu_num
        self.B = None

    def _xwx(self, gpu_id, data_size_per_gpu, start_index, use_gussian,distances):
        host_b = np.zeros((data_size_per_gpu, self.K * self.K), dtype=np.float64)

        cuda.select_device(gpu_id)

        stream = cuda.stream()

        dev_X = cuda.to_device(self.X, stream=stream)
        dev_coords = cuda.to_device(self.coords, stream=stream)
        dev_B = cuda.device_array(host_b.shape, dtype=np.float64, stream=stream)
        dev_distances=cuda.to_device(distances, stream=stream)
        threads_per_block = 64
        if self.K > 13:
            threads_per_block = 16
        elif self.K > 8:
            threads_per_block = 32

        num_blocks = math.ceil(data_size_per_gpu / threads_per_block)

        shared_memory_size = threads_per_block * self.K * self.K * 8


        calculate_B_matrix[num_blocks, threads_per_block, stream, shared_memory_size](dev_coords, dev_X,
                                                                                      dev_B, self.N, self.K,
                                                                                      dev_distances,
                                                                                      data_size_per_gpu, start_index,
                                                                                      use_gussian)

        cuda.synchronize()

        dev_B.copy_to_host(host_b, stream=stream)
        return host_b

    def _s_predy_beta(self, gpu_id, data_size_per_gpu, start_index, use_gussian,distances):
        host_s = np.zeros((data_size_per_gpu), dtype=np.float64)
        host_predy = np.zeros((data_size_per_gpu), dtype=np.float64)
        host_beta = np.zeros((data_size_per_gpu, self.K), dtype=np.float64)

        cuda.select_device(gpu_id)

        stream = cuda.stream()

        dev_X = cuda.to_device(self.X, stream=stream)
        dev_coords = cuda.to_device(self.coords, stream=stream)
        dev_Y = cuda.to_device(self.Y, stream=stream)
        dev_pinvB = cuda.to_device(self.B, stream=stream)
        dev_distances=cuda.to_device(distances, stream=stream)
        #
        dev_S = cuda.device_array(host_s.shape, dtype=np.float64, stream=stream)
        dev_predy = cuda.device_array(host_predy.shape, dtype=np.float64, stream=stream)
        dev_Beta = cuda.device_array(host_beta.shape, dtype=np.float64, stream=stream)

        threads_per_block = 64
        if self.K > 13:
            threads_per_block = 16
        elif self.K > 8:
            threads_per_block = 32

        num_blocks = math.ceil(data_size_per_gpu / threads_per_block)

        shared_memory_size = threads_per_block * (self.K + 2) * 8

        Calculate_S_PY[num_blocks, threads_per_block, stream, shared_memory_size](dev_coords, dev_X, dev_Y, dev_pinvB,
                                                                                  dev_S, dev_predy, dev_Beta,
                                                                                  self.N, self.K, dev_distances,
                                                                                  data_size_per_gpu, start_index,
                                                                                  use_gussian)

        cuda.synchronize()

        dev_S.copy_to_host(host_s, stream=stream)
        dev_predy.copy_to_host(host_predy, stream=stream)
        dev_Beta.copy_to_host(host_beta, stream=stream)
        return host_s.reshape((-1,1)), host_predy.reshape((-1,1)), host_beta

    def run(self):
        kernel_type=None
        if self.kernel_type=="gaussian":
            kernel_type=True
        elif self.kernel_type=="bisquare":
            kernel_type=False
        # self.gpuInformation()
        gpu_ids = self.get_gpuids(False)
        data_size_per_gpu_list, start_indices = self.distribute_data_evenly_with_indices(self.N, len(gpu_ids))
        all_host_b = []
        all_host_s = []
        all_host_predy = []
        all_host_beta = []
        distances = np.full(self.N, self.bandWidth)

        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []
            for gpu_id, data_size_per_gpu, start_index in zip(gpu_ids, data_size_per_gpu_list, start_indices):

                futures.append(executor.submit(self._xwx, gpu_id, data_size_per_gpu, start_index, kernel_type,distances))


            for future in futures:
                host_b = future.result()
                all_host_b.append(host_b)
        self.B = np.concatenate(all_host_b, axis=0)
        self.B = Kernel.compute_pseudo_inverses_cpu(self.B, self.K)
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []
            for gpu_id, data_size_per_gpu, start_index in zip(gpu_ids, data_size_per_gpu_list, start_indices):

                futures.append(executor.submit(self._s_predy_beta, gpu_id, data_size_per_gpu, start_index, kernel_type,distances))


            for future in futures:
                host_s, host_predy, host_beta = future.result()
                all_host_s.append(host_s)
                all_host_predy.append(host_predy)
                all_host_beta.append(host_beta)
        S=np.concatenate(all_host_s, axis=0)
        predy=np.concatenate(all_host_predy, axis=0)
        beta=np.concatenate(all_host_beta, axis=0)

        return S,predy,beta





class adaptiveKernel(Kernel):
    """
    Initialize the class.

    Parameters
    ----------
    dev_XOne : array
        Input data array.

    coords : array
        Coordinates corresponding to variable U.

    N : int
        Number of observation points.

    K : int
        Number of independent variables.

    bandwidth : float
        Bandwidth parameter, e.g., used for kernel processing.

    kernel_type : str
        Type of kernel function.
    """


    def __init__(self, coords, Y, X, N, K, bandWidth, kernel_type, gpu_num):
        self.X = X
        self.Y = Y
        self.coords = coords
        self.N = N
        self.K = K
        self.bandWidth = bandWidth
        self.kernel_type = kernel_type
        self.gpu_num = gpu_num
    def _xwx(self, gpu_id, data_size_per_gpu, start_index, use_gussian,distances):
        host_b = np.zeros((data_size_per_gpu, self.K * self.K), dtype=np.float64)

        cuda.select_device(gpu_id)

        stream = cuda.stream()

        dev_X = cuda.to_device(self.X, stream=stream)
        dev_coords = cuda.to_device(self.coords, stream=stream)
        dev_B = cuda.device_array(host_b.shape, dtype=np.float64, stream=stream)
        dev_distances=cuda.to_device(distances, stream=stream)

        threads_per_block = 64
        if self.K > 13:
            threads_per_block = 16
        elif self.K > 8:
            threads_per_block = 32

        num_blocks = math.ceil(data_size_per_gpu / threads_per_block)

        shared_memory_size = threads_per_block * self.K * self.K * 8


        calculate_B_matrix[num_blocks, threads_per_block, stream, shared_memory_size](dev_coords, dev_X,
                                                                                      dev_B, self.N, self.K,
                                                                                      dev_distances,
                                                                                      data_size_per_gpu, start_index,
                                                                                      use_gussian)

        cuda.synchronize()

        dev_B.copy_to_host(host_b, stream=stream)
        return host_b
    def _s_predy_beta(self, gpu_id, data_size_per_gpu, start_index, use_gussian,distances):
        host_s = np.zeros((data_size_per_gpu), dtype=np.float64)
        host_predy = np.zeros((data_size_per_gpu), dtype=np.float64)
        host_beta = np.zeros((data_size_per_gpu, self.K), dtype=np.float64)

        cuda.select_device(gpu_id)

        stream = cuda.stream()

        dev_X = cuda.to_device(self.X, stream=stream)
        dev_coords = cuda.to_device(self.coords, stream=stream)
        dev_Y = cuda.to_device(self.Y, stream=stream)
        dev_pinvB = cuda.to_device(self.B, stream=stream)
        dev_distances=cuda.to_device(distances, stream=stream)

        dev_S = cuda.device_array(host_s.shape, dtype=np.float64, stream=stream)
        dev_predy = cuda.device_array(host_predy.shape, dtype=np.float64, stream=stream)
        dev_Beta = cuda.device_array(host_beta.shape, dtype=np.float64, stream=stream)
        threads_per_block = 64

        if self.K > 13:
            threads_per_block = 16
        elif self.K > 8:
            threads_per_block = 32

        num_blocks = math.ceil(data_size_per_gpu / threads_per_block)

        shared_memory_size = threads_per_block * (self.K + 2) * 8

        Calculate_S_PY[num_blocks, threads_per_block, stream, shared_memory_size](dev_coords, dev_X, dev_Y, dev_pinvB,
                                                                                  dev_S, dev_predy, dev_Beta,
                                                                                  self.N, self.K, dev_distances,
                                                                                  data_size_per_gpu, start_index,
                                                                                  use_gussian)

        cuda.synchronize()

        dev_S.copy_to_host(host_s, stream=stream)
        dev_predy.copy_to_host(host_predy, stream=stream)
        dev_Beta.copy_to_host(host_beta, stream=stream)

        return host_s.reshape((-1,1)), host_predy.reshape((-1,1)), host_beta
    def run(self,all_distances):
        kernel_type=None
        if self.kernel_type=="gaussian":
            kernel_type=True
        elif self.kernel_type=="bisquare":
            kernel_type=False
        st=time.time()
        if all_distances is None:
            indices, distances=k_nearest_neighbors(self.coords,self.bandWidth)

        else:
            distances = np.ascontiguousarray(all_distances[:, self.bandWidth-1].reshape(-1))
        et=time.time()

        gpu_ids = self.get_gpuids(False)
        data_size_per_gpu_list, start_indices = self.distribute_data_evenly_with_indices(self.N, len(gpu_ids))
        all_host_b = []
        all_host_s = []
        all_host_predy = []
        all_host_beta = []

        st=time.time()
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []
            for gpu_id, data_size_per_gpu, start_index in zip(gpu_ids, data_size_per_gpu_list, start_indices):

                futures.append(executor.submit(self._xwx, gpu_id, data_size_per_gpu, start_index, kernel_type,distances))

            for future in futures:
                host_b = future.result()
                all_host_b.append(host_b)
        self.B = np.concatenate(all_host_b, axis=0)
        self.B = Kernel.compute_pseudo_inverses_cpu(self.B, self.K)
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = []
            for gpu_id, data_size_per_gpu, start_index in zip(gpu_ids, data_size_per_gpu_list, start_indices):

                futures.append(executor.submit(self._s_predy_beta, gpu_id, data_size_per_gpu, start_index, kernel_type,distances))

            for future in futures:
                host_s, host_predy, host_beta = future.result()
                all_host_s.append(host_s)
                all_host_predy.append(host_predy)
                all_host_beta.append(host_beta)
        S=np.concatenate(all_host_s, axis=0)
        predy=np.concatenate(all_host_predy, axis=0)
        beta=np.concatenate(all_host_beta, axis=0)
        et=time.time()
        # print("GWRTime",et-st)
        return S,predy,beta
