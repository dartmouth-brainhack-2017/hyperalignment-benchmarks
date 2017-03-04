from __future__ import absolute_import
from hyperbench.dataio.hdf5_data import load_data, save_data
from hyperbench.extenals.srm import SRM


def srm_alignment(input_data, nfeatures, output_data, mask, output_suffix, training_runs,
                   testing_runs, **kwargs):
    """

    Parameters
    ----------
    input_data: Input data path.
    output_data: Output data path.
    nfeatures: Number of target features in the output data.
    mask: Path to mask file.
    output_suffix: Filename suffix for saving aligned data.
    training_runs: List of runs to be used for training.
    testing_runs: List of runs to be used for testing.
    kwargs: Passed onto SRM

    Returns
    -------
    Nothing
    """
    # XXX TODO Use mask to load from nifti file
    dss_train = load_data(input_data, training_runs) #, mask)
    dss_test = load_data(input_data, testing_runs) #, mask)
    # Prepare data for SRM
    # SRM expects numpy arrays
    # and data as featuresXsamples
    dss_train = [sd.samples.T for sd in dss_train]
    dss_test = [sd.samples.T for sd in dss_test]

    # Initialize SRM
    srm = SRM(features=nfeatures, **kwargs)
    # Run hyperalignment on training data
    srmfit = srm.fit(dss_train)
    # Align and save data
    dss_aligned = {}
    for split, dss in (('train', dss_train),
                       ('test', dss_test)):
        dss_hyper = srm.transform(dss)
        # To transpose data to samplesXfeatures
        dss_hyper = [sd.T for sd in dss_hyper]
        if output_data is not None:
            save_data(dss_hyper, output_suffix+split)
        dss_aligned[split] = dss_hyper

    return dss_aligned