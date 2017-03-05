"""
Hyperalignment Benchmarks
=========================
"""
import argparse


def get_data():
    """
    Function to load a BIDS dataset into pymvpa.
    :return:
    """
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('infiles', nargs='+',
                        help='Input files to run the hyperalignment benchmark')
    parser.add_argument('outfiles', nargs='+',
                        help='Output prefix to save the hyperaligned data')
    parser.add_argument('--normalization', '-n',
                        choices=('zscore', 'percent_signal_change'),
                        default='zscore',
                        help='What kind of normalization step to apply')
    # TODO
    return parser.parse_args()


def main():
    """Run the beast"""
    args = parse_args()

    pass


if __name__ == '__main__':
    main()
    
def get_data_split(data):
    """
    Function to split data into half
    
    Parameters
    ----------
    data : typical hyperalignment input dataset 
    
    Returns
    -------
    ds_train : training data
    ds_test : test data
    """
    from __future__ import division

    nruns = set(data[0].sa.chunks)
    nruns = len(nruns)
    split_num = nruns/2

    for test_run in range(nruns):
        # split in training and testing set
        ds_train = [sd[sd.sa.chunks <= split_num, :] for sd in ds_all]
        ds_test = [sd[sd.sa.chunks > split_num, :] for sd in ds_all]
    
    return ds_train, ds_test