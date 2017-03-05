from mvpa2.base.hdf5 import h5save, h5load
import numpy as np
import os


def load_data(data_path, runs=None):
    """

    Parameters
    ----------
    data_path: Path to data.
    runs: If not None, which chunks/runs to keep.

    Returns
    -------
    List of datasets.
    """
    if not os.path.exists(data_path):
        raise ValueError("Data path doesn't exist: {0}".format(data_path))

    dss = h5load(data_path)
    if not isinstance(dss, (list, tuple, np.ndarray)):
        raise TypeError("Input datasets should be a sequence "
                        "(of type list, tuple, or ndarray) of datasets.")

    if runs is None:
        return dss
    else:
        dss = [sd.select(sadict={'chunks': runs}) for sd in dss]
        return dss

def save_data(data, save_path):
    """

    Returns
    -------

    """
    if os.path.exists(save_path):
        raise Warning("File % already exists.\nOverwriting.".format(save_path))
    h5save(save_path, data)
