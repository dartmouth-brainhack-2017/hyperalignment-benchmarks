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
