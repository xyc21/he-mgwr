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
    dataset_sizes =[1,2,3,4,5]
    for j in range(10):
        for datasize in dataset_sizes:
            print('=' * 75 + '\n')
            print(f"Start computing, file name: sample_{datasize}.csv")
            filename = f"../sampleData/sample_{datasize}.csv"
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

            local_time = time.localtime(get_time)


            formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", local_time)
            pathName = f"MGWR_summary_{num}_{j}_{formatted_time}.txt"
            save_summary_to_txt(summary,pathName)


if __name__ == "__main__":

    # Performance of HE-MGWR on the sample dataset: running time as the number of observations increases.
    Record_the_running_time()

