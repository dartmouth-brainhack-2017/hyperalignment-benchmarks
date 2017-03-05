from mvpa2.mappers.zscore import zscore
from mvpa2.mappers.base import Mapper
from hyperbench.psc import psc
import numpy as np


def normalize(dss, norm_type):
    if norm_type == 'zscore':
        if isinstance(dss, (list, tuple, np.ndarray)):
            _ = [zscore(sd, chunks_attr=None) for sd in dss]
        else:
            zscore(dss, chunks_attr=None)
    elif norm_type == 'percent_signal_change':
        if isinstance(dss, (list, tuple, np.ndarray)):
            _ = [psc(sd, chunks_attr=None) for sd in dss]
        else:
            psc(dss, chunks_attr=None)
    elif norm_type == 'demean':
        if isinstance(dss, (list, tuple, np.ndarray)):
            _ = [psc(sd, scale=False, chunks_attr=None) for sd in dss]
        else:
            psc(dss, scale=False, chunks_attr=None)

    return dss

