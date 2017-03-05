# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data normalization by Percent Signal Change adapted from ZscoreMapper."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base.dochelpers import _str, borrowkwargs, _repr_attrs
from mvpa2.mappers.zscore import ZScoreMapper
from mvpa2.datasets import Dataset

@borrowkwargs(ZScoreMapper, '__init__')
class PSCMapper(ZScoreMapper):
    """
    Percent signal change mapper

    """
    def __init__(self, scale=True, *args, **kwargs):

        """

        :param scale: Boolean variable to determine whether to scale be mean to
                      get percent signal change or just get de-mean signal.
        :param args:
        :param kwargs:
        """
        self.scale = scale
        super(PSCMapper, self).__init__(*args, **kwargs)

    def _zscore(self, samples, mean, std):
        scaling_factors = mean if self.scale else np.ones(shape=mean.shape)
        return super(PSCMapper, self)._zscore(samples, mean, scaling_factors)


@borrowkwargs(PSCMapper, '__init__')
def psc(ds, **kwargs):
    """In-place PSC of a `Dataset` or `ndarray`.

    This function behaves identical to `PSCMapper`. The only difference is
    that the actual psc is done in-place -- potentially causing a
    significant reduction of memory demands.

    Parameters
    ----------
    ds : Dataset or ndarray
      The data that will be Z-scored in-place.
    **kwargs
      For all other arguments, please see the documentation of `ZScoreMapper`.
    """
    pscm = PSCMapper(**kwargs)
    pscm._secret_inplace_zscore = True
    # train
    if isinstance(ds, Dataset):
        pscm.train(ds)
    else:
        pscm.train(Dataset(ds))
    # map
    mapped = pscm.forward(ds)
    # and append the mapper to the dataset
    if isinstance(mapped, Dataset):
        mapped._append_mapper(pscm)
