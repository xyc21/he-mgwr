## HE-MGWR:A Highly Scalable Multi-GPU Based Open Source Implementation Scheme for MGWR (Multiscale Geographically Weighted Regression

---

## Installation guide:

The HE-MGWR program relies on libraries such as numba for its functionality. The simplest way to install these dependencies is to use conda:
```bash

conda install numba tqdm numpy pandas
conda install cudatoolkit=11.8
```
### HE-MGWR Program Access Instructions

The Python package for the HE-MGWR program has been uploaded to  this repository. You can access and download it from there.

## Examples

Example call to the HE-MGWR to fit GWR model:

```bash
    coords=reader.get_coordinates()
    x,y=reader.standardize_data()
    selector = Select_bw(coords, y, x,fixed=False,kernel_type="bisquare",multi=True, constant=True)
    selector.search(verbose=False,rss_score=True,max_iter_multi=20)
    mgwr = MGWR(coords, y, x, selector, constant=True,name_x=header).fit()
```
#### where:
coords: array  
An array of spatial coordinates (longitude and latitude), 
where each row represents the location of a sample point.

y: array  
The dependent (response) variable values.

x: array  
The independent (explanatory) variable values.

fixed=False:bool   
Determines whether to use a fixed bandwidth (True) or adaptive bandwidth (False). 

kernel_type="bisquare": str   
Specifies the kernel function used to compute spatial weights. "bisquare" or "gaussian".

constant=True: bool  
Indicates whether to include an intercept (constant term) in the model.

#### CSV

| X-coord | y-coord  | Y  | X1  | X2  | X3  | ... | Xk  |
|---------|-----|-----|-----|-----|-----|-----|-----|
| ...     | ... | ... | ... | ... | ... | ... | ... |
| ...     | ... | ... | ... | ... | ... | ... | ... |

#### where:      
X-coord: X coordinate of the location point  
Y-coord: Y coordinate of the location point  
y: dependent variable  
X1...Xk: independent   


## Folder description:
- hemger: Python package  
- simulation: Contains simulated data and the code for generating it  
- sampleData: Dataset used to demonstrate code execution  
- test_HE_MGWR: Contains instruction files for reproducing experiments  
  - test_simulation: Instructions for using HE-MGWR on simulated data  
  - test_sampleData: Instructions for demonstrating HE-MGWR on the "AOD-PM2.5" dataset  


