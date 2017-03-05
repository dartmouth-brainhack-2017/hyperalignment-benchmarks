from mvpa2.algorithms.hyperalignment import Hyperalignment
from hyperbench.dataio.hdf5_data import load_data, save_data
from hyperbench.normalize import normalize


def hyperalignment(input_data, output_data, mask, output_suffix, training_runs,
                   testing_runs, normalization, **kwargs):
    """

    Parameters
    ----------
    input_data: Input data path.
    output_data: Output data path.
    mask: Path to mask file.
    output_suffix: Filename suffix for saving aligned data.
    training_runs: List of runs to be used for training.
    testing_runs: List of runs to be used for testing.
    kwargs: Passed onto Hyperalignment

    Returns
    -------
    Nothing
    """
    # XXX TODO Use mask to load from nifti file
    dss_train = load_data(input_data, training_runs) #, mask)
    dss_test = load_data(input_data, testing_runs) #, mask)
    # Normalize/pre-process data here
    dss_train = normalize(dss_train, norm_type=normalization)
    dss_test = normalize(dss_test, norm_type=normalization)

    # Initialize hyperalignment
    ha = Hyperalignment(**kwargs)
    # Run hyperalignment on training data
    hmappers = ha(dss_train)
    # Align and save data
    dss_aligned = {}
    for split, dss in (('train', dss_train),
                       ('test', dss_test)):
        dss_hyper = [hm.forward(sd) for hm, sd in zip(hmappers, dss)]
        if output_data is not None:
            save_data(dss_hyper, output_suffix+split)
        dss_aligned[split] = dss_hyper

    return dss_aligned
