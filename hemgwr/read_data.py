import numpy as np
import pandas as pd

class DataReader:
    """
    Parameters
    ----------
    path : str
        Path to the data file, which can be a CSV or XLSX file.

    spherical : bool, optional
        If True, indicates spherical coordinates (longitude and latitude);
        if False, assumes projected coordinates (default).

    Attributes
    ----------
    self.path : str
        File path.

    self.data : np.ndarray
        Array to store the loaded data; initialized as None.

    self.df : pd.DataFrame
        DataFrame to store the loaded data; initialized as None.

    self.spherical : bool
        Flag indicating whether spherical coordinates are used.
    """

    def __init__(self, path,spherical=False):
        """
        Initialize class with the file path.
        """
        self.path = path
        self.data = None
        self.df = None
        self.spherical = spherical

    def read_csv(self,spherical=False):
        self.spherical=spherical

        self.df = pd.read_csv(self.path, dtype=np.float64)
        self.data = self.df.to_numpy()
        return self.data

    def read_xlsx(self,spherical=False):
        self.spherical = spherical

        self.df = pd.read_excel(self.path, dtype=np.float64)
        self.data = self.df.to_numpy()
        return self.data

    def get_coordinates(self):

        if self.data is not None:
            if self.spherical:
                coordinates = self.data[:, :2]
            else:
                coordinates = self.data[:, :2]  # 提取前两列作为坐标
            # if coordinates.dtype != np.float32:
            #     coordinates = coordinates.astype(np.float32)  # 确保数据类型为float32
            return coordinates
        else:
            raise ValueError("Data has not been loaded. Please read the file first.")

    def get_dependent_variable(self):

        if self.data is not None:
            return self.data[:, 2]  # Assuming y is the third column
        else:
            raise ValueError("Data has not been loaded. Please read the file first.")

    def get_independent_variables(self):

        if self.data is not None:
            return self.data[:, 3:]
        else:
            raise ValueError("Data has not been loaded. Please read the file first.")

    def get_headers(self):

        if self.df is not None:
            return self.df.columns.tolist()
        else:
            raise ValueError("Data has not been loaded. Please read the file first.")

    def get_xName(self):

        if self.df is not None:
            return self.df.columns.tolist()[3:]
        else:
            raise ValueError("Data has not been loaded. Please read the file first.")

    def standardize_data(self):
        """
        Standardize the input feature matrix X and target variable y.

        Parameters
        ----------
        X : ndarray
            Feature matrix (2D array).

        y : ndarray
            Target variable (1D array or column vector).

        Returns
        -------
        X_standardized : ndarray
            Standardized feature matrix.

        y_standardized : ndarray
            Standardized target variable (as a column vector).
        """

        X_standardized = (self.get_independent_variables() - self.get_independent_variables().mean(axis=0)) / self.get_independent_variables().std(axis=0)


        y = self.get_dependent_variable().reshape((-1, 1))


        y_standardized = (y - y.mean(axis=0)) / y.std(axis=0)

        return X_standardized, y_standardized.reshape(-1)

