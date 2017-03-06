import pytest
from hyperbench.alignments.hyperalignment import hyperalignment
from hyperbench.alignments.srm import srm_alignment
from hyperbench.benchmarks.intersubject_correlation import \
    intersubject_correlation
from hyperbench.benchmarks.timesegmentclassification import \
    timesegments_classification

@pytest.mark.parametrize("alignment", ['hyper', 'srm'])
@pytest.mark.parametrize("normalization", ['zscore', 'percent_signal_change', 'demean'])
@pytest.mark.parametrize("benchmark", ['isc', 'clf'])
def test_basics(alignment, normalization, benchmark):
    data_path = './data/hyperalignment_tutorial_data_2.4_slim.hdf5.gz'
    # Smoke tests for now
    if alignment == 'hyper':
        dss_aligned = hyperalignment(data_path, None, None, None, range(4),
                                     range(5, 8), normalization=normalization,
                                     output_dim=None)
    elif alignment == 'srm':
        dss_aligned = srm_alignment(data_path, 20, None, None, None, range(4),
                                     range(5, 8), normalization=normalization)
    else:
        pass

    if benchmark == 'isc':
        corrs_train = intersubject_correlation(dss_aligned['train'])
        corrs_test = intersubject_correlation(dss_aligned['test'])
        print("Training mean ({0}):{1}\n".format(alignment, corrs_train.samples.mean()))
        print("Test mean ({0}):{1}\n".format(alignment, corrs_test.samples.mean()))
    elif benchmark == 'clf':
        errors_train = timesegments_classification(dss_aligned['train'], do_zscore=True)
        errors_test = timesegments_classification(dss_aligned['test'], do_zscore=True)
        print("Training error ({0}:{1}\n".format(alignment, errors_train.mean()))
        print("Test error ({0}):{1}\n".format(alignment, errors_test.mean()))
