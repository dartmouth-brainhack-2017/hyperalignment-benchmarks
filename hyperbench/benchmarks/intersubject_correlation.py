from itertools import combinations
from mvpa2.measures.corrcoef import pearson_correlation
from mvpa2.datasets import Dataset


def intersubject_correlation(dss, reference_ds=0):
    """
    Computes voxelwise inter-subject time series correlation
    in a pairwise fashion for a list of Datasets. Datasets
    must all be the same shape. Resulting dataset of pairwise
    correlations will inherit Dataset attributes from
    reference data set [Default: first data set in list].
    """

    # Check if input list contains Datasets, ndarrays
    dss = [Dataset(ds) if not type(ds) == Dataset else ds for ds in dss]

    ds_shape = dss[reference_ds].shape
    n_features = ds_shape[1]

    for ds in dss: assert ds.shape == ds_shape

    # Compute time series correlation per voxel per subject pair
    correlations = []
    for pair in combinations(dss, 2):
        pair_map = []
        for feature in xrange(n_features):
            pair_map.append(pearson_correlation(pair[0].samples[:, feature],
                                                   pair[1].samples[:, feature]))
        correlations.append(pair_map)

    # Resulting correlation map inherits attributes of referece data set
    correlations_ds = Dataset(correlations,
                                 fa=dss[reference_ds].fa,
                                 a=dss[reference_ds].a)
    correlations_ds.sa['pairs'] = list(combinations(range(len(dss)), 2))

    assert correlations_ds.shape[0] == len(dss) * (len(dss) - 1) / 2
    assert correlations_ds.shape[1] == n_features

    return correlations_ds
