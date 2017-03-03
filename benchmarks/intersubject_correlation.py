def intersubject_correlation(dss, reference_ds=0):
    """
    Computes voxelwise inter-subject time series correlation
    in a pairwise fashion for a list of Datasets. Datasets
    must all be the same shape. Resulting dataset of pairwise
    correlations will inherit Dataset attributes from
    reference data set [Default: first data set in list].
    """
    from mvpa2.measures.corrcoef import pearson_correlation
    from itertools import combinations

    ds_shape = dss[reference_ds].shape
    n_voxels = ds_shape[1]

    for ds in dss: assert ds.shape == ds_shape

    correlations = []
    for pair in combinations(dss, 2):
        pair_map = []
        for voxel in xrange(n_voxels):
            pair_map.append(mv.pearson_correlation(pair[0].samples[:, voxel],
                                                   pair[1].samples[:, voxel]))
        correlations.append(pair_map)

    correlations_ds = mv.Dataset(correlations,
                                 fa=dss[reference_ds].fa,
                                 a=dss[reference_ds].a)
    correlations_ds.sa['pairs'] = list(combinations(range(len(dss)), 2))

    assert correlations_ds.shape[0] == len(dss) * (len(dss) - 1) / 2
    assert correlations_ds.shape[1] == n_voxels

    return correlations_ds
