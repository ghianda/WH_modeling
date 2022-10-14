import tifffile as tiff
import numpy as np
import os
import argparse


class BC:
    VERB = '\033[95m'
    B = '\033[94m'
    G = '\033[92m'
    Y = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# PHANTOM SHAPE:
# ------------------------------------------------------------------------------------------------------
# |       |         |            |         |            |         |     |            |         |       |
# |       |         |            |         |            |         |     |            |         |       |
# |  myo  | compact | diff lvl 1 | compact | diff lvl 2 | compact | ... | diff lvl n | compact |  myo  |
# |       |         |            |         |            |         |     |            |         |    000|
# |       |         |            |         |            |         |     |            |         |    000|  <- crop angle
# ------------------------------------------------------------------------------------------------------
# |<  dw >|<-  cw ->|<--  dw --->|<-  cw ->|<--  dw --->|
#                   |<----  "segment" ---->|
#                   |<-------  sw -------->|


#
def main(parser):

    # collect arguments
    args = parser.parse_args()
    autof_tiff_path = args.input_autof_tiffpath[0]

    # =================== USER INPUT HARDCODED ============================================
    # define hardcoded number of levels of scattering, and with which values
    scatt_lvls  = [4, 5, 6, 7, 8, 9]  # levels of scattering to be modeled as diffuse fibrosis
    lvls_ratio  = 2  # ratio =(width of diffuse fib area / width of compact fib area)
    # =======================================================================================

    print(BC.Y + "*** START GENERATING PHANTOM FROM: ***" + BC.ENDC)
    print(BC.B + autof_tiff_path + BC.ENDC + '\n')

    ''' ------------------------------------------- PREPARE DATA -------------------------------------------------'''
    tiff_name = os.path.basename(autof_tiff_path)
    dir_path  = os.path.dirname(autof_tiff_path)

    # load tiff
    tissue = tiff.imread(autof_tiff_path)
    tissue = np.moveaxis(tissue, 0, -1)  # (z, y, x) -> (r, c, z) = (YXZ)

    # shape of tiff file
    shape_yxz = tissue.shape
    print("Loaded {}".format(tiff_name))
    print("Shape (YXZ): ", shape_yxz)

    # how compose the phantom
    n_segments = len(scatt_lvls)  # see above
    dw = shape_yxz[1]  # width of the diffuse fibrosois area
    cw = int(dw / lvls_ratio)  # width of the compact fibrosis area
    sw = dw + cw  # segment_width

    # shape of the phantom
    depth = shape_yxz[2]
    width = shape_yxz[0]
    long  = dw + (n_segments * sw) + cw + dw  #  dw are the extremes

    shape_phntm_YXZ = [width, long, depth]
    print("Phantom shape (YXZ) will be: ", shape_phntm_YXZ, end=' - ')
    print("splitted in {} segments".format(n_segments))

    ''' ------------------------------generate the actual phantom channels ---------------------------------'''

    # fake channels
    autof, scatt = np.zeros(shape_phntm_YXZ), np.zeros(shape_phntm_YXZ)

    # first segment:
    # myo
    autof[:, 0:dw, :] = np.copy(tissue)
    scatt[:, 0:dw, :] = 0
    # compact
    autof[:, dw:sw, :] = 0
    scatt[:, dw:sw, :] = 255

    # following segments
    for (i, lvl) in enumerate(scatt_lvls):

        start_diff = (i+1) * sw
        stop_diff = start_diff + dw
        start_compact = stop_diff
        stop_compact  = start_compact + cw

        print("Iteration: {}".format(i))
        print("Diffuse: [{}, {}]".format(start_diff, stop_diff))
        print("Compact: [{}, {}]".format(start_compact, stop_compact))

        # diffuse
        autof[:, start_diff:stop_diff, :] = tissue
        scatt[:, start_diff:stop_diff, :] = lvl

        # compact
        autof[:, start_compact:stop_compact, :] = 0
        scatt[:, start_compact:stop_compact, :] = 255

    # last myo
    autof[:, (long - dw):, :] = tissue
    scatt[:, (long - dw):, :] = 0

    # crop the angle in both channels
    autof[int(width / 2):, long - int(dw / 2):, :] = 0
    scatt[int(width / 2):, long - int(dw / 2):, :] = 0

    # save the scattering as a separate channel in a tiff file
    autof = np.moveaxis(autof, -1, 0).astype(np.uint8)  # return to (z, y, x)
    scatt = np.moveaxis(scatt, -1, 0).astype(np.uint8)  # return to (z, y, x)
    tiff.imwrite(file=os.path.join(dir_path, "phantom_autof.tif"), data=autof)
    tiff.imwrite(file=os.path.join(dir_path, "phantom_scatt.tif"), data=scatt)
    print("Phantom generated. Saved in: \n", dir_path)

    return None


if __name__ == '__main__':

    my_parser = argparse.ArgumentParser(
        description='Generate a phantom from a autofluorescence tiff adding a fake striped scattering.')

    my_parser.add_argument('-i',
                           '--input-autof-tiffpath',
                           nargs='+',
                           help='Path of tiff file of the autofluorescence signal to use.',
                           required=True)

    main(my_parser)
