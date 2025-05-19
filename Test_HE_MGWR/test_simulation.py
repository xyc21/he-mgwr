import pandas as pd
from hemgwr.gwr import GWR
from hemgwr.mgwr import MGWR
from hemgwr.read_data import DataReader
from hemgwr.select_bw import Select_bw
from hemgwr.tool import gpuInformation, k_nearest_neighbors_all
import time

def save_summary_to_txt(summary, file_path):
    # Generate the summary
    summary_content = summary

    # Save the summary to a text file
    with open(file_path, 'w') as file:
        file.write(summary_content)

    print(f"Summary saved to {file_path}")

def Record_the_running_time():
    num_independent_vars =[4,6,8,10,15]
    dataset_size =[2500,10000,50625,100489,250000,504100,1000000]
    for j in num_independent_vars:
        for i in dataset_size:
            print('=' * 75 + '\n')
            print(f"Start computing dataset k{j}, file name: simulate_data_{i}_{j}_parameter.csv")
            filename = f"../simulation/k{j}/simulate_data_{i}_{j}_parameter.csv"
            reader = DataReader(filename)
            data=reader.read_csv()
            coords=reader.get_coordinates()

            x,y=reader.standardize_data()
            num = x.shape[0]
            header=reader.get_xName()
            if x.shape[0]>=20000:
                all_distances = k_nearest_neighbors_all(coords)
            else:
                all_distances=None
            selector = Select_bw(coords, y, x,fixed=False,kernel_type="bisquare",multi=True, constant=True)
            selector.search(verbose=True,rss_score=True,all_distances=all_distances,multi_bw_min=[20])

            mgwr = MGWR(coords, y, x, selector, constant=True,name_x=header).fit()
            summary=mgwr.summary()

            get_time = time.time()
            # 转换为本地时间
            local_time = time.localtime(get_time)


            formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", local_time)
            pathName = f"MGWR_summary_{num}_{j}_{formatted_time}.txt"
            save_summary_to_txt(summary,pathName)
def Diff_GPU_running_time():
    gpu_counts = [1, 2, 4, 6, 8]
    num_independent_vars = [4, 6, 8, 10, 15]
    dataset_size = [2500, 10000, 50625, 100489, 250000, 504100, 1000000]
    for num_gpus in gpu_counts:
        for i in dataset_size:
            print('=' * 75 + '\n')
            print(f"Start computing dataset k10, file name: simulate_data_{i}_10_parameter.csv")
            filename = f"../simulation/k10/simulate_data_{i}_10_parameter.csv"
            reader = DataReader(filename)
            data = reader.read_csv()
            coords = reader.get_coordinates()

            x, y = reader.standardize_data()
            num = x.shape[0]
            header = reader.get_xName()
            if x.shape[0] >= 20000:
                all_distances = k_nearest_neighbors_all(coords)
            else:
                all_distances = None
            selector = Select_bw(coords, y, x, fixed=False, kernel_type="bisquare", multi=True, constant=True,gpu_num=num_gpus)
            selector.search(verbose=True, rss_score=True, all_distances=all_distances, multi_bw_min=[20])

            mgwr = MGWR(coords, y, x, selector, constant=True, name_x=header).fit()
            summary = mgwr.summary()

            get_time = time.time()

            local_time = time.localtime(get_time)


            formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", local_time)
            pathName = f"MGWR_summary_{num}_10_{formatted_time}_GPUnum_{num_gpus}.txt"
            save_summary_to_txt(summary, pathName)


if __name__ == "__main__":

    # Performance of HE-MGWR: running time under different numbers of independent variables (4, 6, 8, 10, and 15)
    # as the number of observations increases.
    Record_the_running_time()


    # Performance of HE-MGWR: running time under different numbers of GPUs (1, 2, 4, 6, and 8)
    # as the number of observations increases.
    Diff_GPU_running_time()