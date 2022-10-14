import tifffile as tiff
import numpy as np
import os
import argparse


class BC:
    VERB = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main(parser):

    # collect arguments
    args = parser.parse_args()
    autof_tiff_path = args.input_autof_tiffpath[0]

    print()

    # define hardcoded number of levels of scattering, and with which values

    # load the autof tiff

    # extract the shape

    # automatically evaluate how split the matrix to generate incremental scattering areas and compact fibrosis

    # generate the actual fake scattering matrix

    # save the scattering as a separate channel in a tiff file












if __name__ == '__main__':

    my_parser = argparse.ArgumentParser(
        description='Generate a phantom from a autofluorescence tiff adding a fake striped scattering.')

    my_parser.add_argument('-i',
                           '--input-autof-tiffpath',
                           nargs='+',
                           help='Path of tiff file of the autofluorescence signal to use.',
                           required=True)

    main(my_parser)