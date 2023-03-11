"""
A collection of Python functions and classes
"""


import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler


__author__ = "Erik Matovic"
__version__ = "1.0"
__email__ = "xmatovice@stuba.sk"
__status__ = "Development"


def check_null_values(df: pd.DataFrame) -> None:
    """
    Print NULL values.
    :param: df - pandas dataframe
    :returns: nothing
    """
    for col in df:
        print(col, df[col].isnull().values.any())


def print_sum_null(df: pd.DataFrame) -> None:
    """
    Print sum of NULL values.
    :param: df - pandas dataframe
    :returns: nothing
    """
    print(df.isnull().sum())


def rescale(df: pd.DataFrame, col: str) -> Tuple[MinMaxScaler, np.ndarray]:
    """
    Rescale values using scikit-learn's MinMaxScaler.
    :param: df - dataframe
    :returns: rescaled values
    """
    scaler = MinMaxScaler()

    # The scaler expects the data to be shaped as (x, y), 
    # so we add a dimension using reshape.
    values = df[col].values.reshape(-1, 1)
    values_scaled = scaler.fit_transform(values)

    return scaler, values_scaled


def to_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    """
    """
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def split_data(data_raw: np.ndarray, seq_len: int, train_split: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test
