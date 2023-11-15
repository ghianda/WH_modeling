import os
import argparse
import numpy as np
import math
import meshIO as meshIO
from datetime import date
import tifffile as tiff
from scipy import ndimage
import subprocess

from skimage.morphology._skeletonize_3d_cy import fill_Euler_LUT


class BC:
    VERB = '\033[95m'
    B = '\033[94m'
    G = '\033[92m'
    Y = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def prepend_line(new_line, txt_path):
    # We read the existing lines from file in READ mode
    src = open(txt_path, 'r')
    lines = src.readlines()

    # add 'end line' to the new string
    new_line = new_line + '\n'

    # Here, we prepend the string we want to on first line
    lines.insert(0, new_line)
    src.close()

    # We again open the file in WRITE mode to overwrite
    src = open(txt_path, 'w')
    src.writelines(lines)
    src.close()


def read_carp_mesh(mesh_basepath, lon_fpath=None, _read_cpts=False, _verb=True):
    '''
    Load and return Carp files of the mesh.

    :param mesh_basepath: complete path of the mesh basename.
                          Example: /mnt/.../mesh/mesh_name  ---> means (mesh_name.pts, mesh_name.lon, mesh_name.elem...)
    :param lon_fpath: path to a different .lon file to load (insted of mesh_name.lon)
    :param _read_cpts: if True, the centroid file mesh_name.cpts will be load
    :param _verb:  bool, if True, the function print out mesh info.
    :return: carp files, [and cpts file]
    '''

    # Reads in mesh pts file
    pts = meshIO.read_pts(basename=mesh_basepath, file_pts=None)
    if _verb: print('- Mesh has', len(pts), 'nodes')

    # Reads in mesh elems file
    elems = meshIO.read_elems(basename=mesh_basepath, file_elem=None)
    if _verb: print('- Mesh has', len(elems), 'elements')

    # Reads in mesh lon
    if lon_fpath is None:
        lon = meshIO.read_lon(basename=mesh_basepath, file_lon=None)
    else:
        # load the speciific .lon file passed in input
        lon = meshIO.read_lon(file_lon=lon_fpath)

    if _verb: print('- Mesh has', len(lon), 'fibres')

    if _read_cpts:
        # Reads in mesh centroids file
        cpts = meshIO.read_cpts(basename=mesh_basepath, file_cpts=None)
        if _verb: print('- Mesh has', len(cpts), 'centroids')

    if _read_cpts:
        return pts, elems, lon, cpts
    else:
        return pts, elems, lon


def normalize_scatt(scatt_ldg, df_scatt_lvls):
    # normalize scattering from [4,... ,9] -> [3,... ,10] -> [0, 1]
    # because [4, 9] are INCLUDED into the diffuse fibrosis, so the "extremes" are ldg=3 (fib 0%, background) and ldg=10 (fibrosis 100% i.e. compact)
    min = np.min(df_scatt_lvls) - 1
    max = np.max(df_scatt_lvls) + 1
    normalized = (scatt_ldg - min) / (max - min)
    return normalized


def current_elem_is_fibrosis(scatt_ldg, df_scatt_lvls):
    # estimate the probability that the current element is fibrosis using the scattering signal,
    # and then decide if remove the fiber or not based on that probability

    # normalize scattering value between 0 an 1
    probability_of_fibrosis = normalize_scatt(scatt_ldg=scatt_ldg, df_scatt_lvls=df_scatt_lvls)

    # extract a random number between 0 and 1
    p = np.random.rand()

    # look if the random number is above or below the probability_of_fibrosis index
    if p < probability_of_fibrosis:
        # current element is fibrosis
        return True
    else:
        # current element is not fibrosis
        return False


def main(parser):

    # ===================================== INPUT and Initial Messages ================================================

    # collect arguments
    args = parser.parse_args()
    mesh_basename       = args.mesh_basename
    scattering_tiffpath = args.scattering_path
    base_path           = args.base_path
    lon_file            = args.lon_file
    _scatt_already_ds   = args.scatt_already_ds
    _vpts               = args.vpts
    _vtk                = args.vtk

    # directory of mesh files (if None, take the current directory)
    base_path = base_path if base_path is not None else os.getcwd()

    # compile absolute paths
    mesh_basepath = os.path.join(base_path, mesh_basename)
    lon_fpath = None if lon_file is None else os.path.join(base_path, lon_file)

    print(BC.Y + '*** Modelling Fibrosis on mesh {} by holes in fibers ...'.format(mesh_basename) + BC.ENDC)

    # initialize output txt file where write statistics
    txt_fpath = os.path.join(base_path, "Fiber_holes.txt")
    with open(txt_fpath, "w") as f:
        print("Execution date: ", date.today(), file=f)

    # ================================  LOAD and PREARE DATA ==========================================
    print(BC.B + 'Loading mesh and Scattering channel...' + BC.ENDC)

    # loading mesh in carp format
    pts, elem, lon, cpts = read_carp_mesh(mesh_basepath, lon_fpath=lon_fpath, _read_cpts=True, _verb=True)

    # defines pixel size of the mesh (from the segmented image)
    mesh_ps_yxz = np.array([20, 20, 20])  # um
    print('Resolution of the mesh (r,c,z)', mesh_ps_yxz, 'um')

    # load scattering tiff
    scattering_zyx = tiff.imread(scattering_tiffpath)
    scattering_yxz = np.moveaxis(scattering_zyx, 0, -1)  # move (z, y, x) to (row, col, z) = (YXZ)
    scatt_shape_yxz = scattering_yxz.shape
    print('Loaded Scattering tiff file (dtype: ', scattering_yxz.dtype, ') with shape: ', scatt_shape_yxz)

    # Defines the voxel resolution of the scattering image (old ds388 -> now 6um)
    scatt_ps_yxz = mesh_ps_yxz.copy() if _scatt_already_ds else np.array([6, 6, 6])

    # calculate the pixel_size ratio between mesh and scattering data
    mesh2scatt_res_ratio = mesh_ps_yxz / scatt_ps_yxz

    print('Resolution of the scattering data (r,c,z)', scatt_ps_yxz, 'um')
    print('Ration between mesh and scattering pixel size: ', mesh2scatt_res_ratio)

    if _scatt_already_ds:
        print('Scattering channel is already scaled to the mesh spatial resolution.')
    else:
        print('Scaling scattering channel...')

        # downsample the scattering channel down to the mesh resolution
        scattering_yxz = ndimage.zoom(scattering_yxz, zoom=1/mesh2scatt_res_ratio, prefilter=False)

        # save it in a new tiff file for the future
        phantom_ds_filepath = os.path.join(base_path, "scatt_ds.tif")
        tiff.imwrite(phantom_ds_filepath, np.moveaxis(scattering_yxz, -1, 0).astype(np.uint8))
        print('Scattering channel scaled to the mesh resolution and saved as: scatt_ds.tif')

    # check dimension
    print(BC.B + 'Checking dimension of data:' + BC.ENDC)
    print('PIXEL - Shape of Scattering data (YXZ): ', scatt_shape_yxz)
    print('PIXEL - Max coordinates of centroids (XYZ): ', np.array(cpts.max()))
    print('UM - Shape of Scattering data (YXZ) in um: ', scatt_shape_yxz * scatt_ps_yxz)
    print('UM - Max coordinates of centroids (XYZ) in um: \n', mesh_ps_yxz * cpts.max(), BC.ENDC)


    # ================================  PRE-ITERATION OPERATIONS  ==========================================

    # supports
    n_points      = len(cpts)
    magn          = math.floor(math.log10(n_points))
    empty_fiber   = np.array([0.0, 0.0, 0.0])  # fake empty fiber used to replace original fiber
    df_scatt_lvls = np.array([4, 5, 6, 7, 8, 9])  # scatt ldg defined as diffuse fibrosis

    # counters
    exception           = 0  # count exceptions while manage fibers
    compact_fib_elem    = 0  # count fiber tagged as compact fibrosis (and removed)
    diffuse_fib_elem    = 0  # count fiber tagged as diffuse fibrosis (all)
    diffuse_fib_removed = 0  # count fiber tagged as diffuse fibrosis AND REMOVED
    vessels_elem        = 0  # count fiber removed because tagged as vessels (and removed)

    # Defines a copy of the fibre file to edit
    lon_fibrosis = lon.copy()

    print(BC.B + 'Start simulating fibrotic areas inside the new .lon file... ' + BC.ENDC)

    # ================================  COMPILE NEW .LON FILES  ==========================================
    print("Progress:")
    for i in range(n_points):
        # print progress percent
        if i % 10**(magn - 1) == 0:
            print(' - {0:3.0f} %'.format(100 * i / n_points))

        # read tag of current element
        current_tag = elem.loc[i, 5]

        try:
            if current_tag == 5:
                # diffuse fibrosis -> remove fiber accordingly with scattering signal
                diffuse_fib_elem = diffuse_fib_elem + 1  # counter

                # Sees which voxel we're in
                x, y, z = np.floor(cpts.loc[i, 0]), np.floor(cpts.loc[i, 1]), np.floor(cpts.loc[i, 2])

                # read scattering signal in that position
                scatt_ldg = scattering_yxz[int(y), int(x), int(z)]

                if current_elem_is_fibrosis(scatt_ldg, df_scatt_lvls):
                    # remove the fiber
                    lon_fibrosis.loc[i] = empty_fiber
                    diffuse_fib_removed = diffuse_fib_removed + 1  # counter

            if current_tag == 6:
                # compact fibrosis -> remove fiber
                lon_fibrosis.loc[i] = empty_fiber
                compact_fib_elem = compact_fib_elem + 1  # counter

            if current_tag == 7:
                # vessels -> remove fiber
                lon_fibrosis.loc[i] = empty_fiber
                vessels_elem = vessels_elem + 1  # counter

        except ValueError as e:
            print(BC.FAIL)
            print('i={} - position: (x, y, z) = ({},{},{})'.format(i, x, y, z))
            print('FAIL with ValueError: {}'.format(e))
            print(BC.ENDC)
            exception = exception + 1
        except:
            print(BC.FAIL + 'FAIL with unknown error')
            print('i={} - position: (x, y, z) = ({},{},{})'.format(i, x, y, z))
            print(BC.ENDC)
            exception = exception + 1

    print(BC.B + 'Terminated.' + BC.ENDC)

    # =========================== STATISTICS =====================================
    # evaluate percentage of removed fibers in the entire mesh:
    perc_CF_elem  = 100 * (compact_fib_elem / n_points)
    perc_DF_elem  = 100 * (diffuse_fib_elem / n_points)
    perc_VSS_elem = 100 * (vessels_elem / n_points)
    perc_DF_fiber_removed_on_DF  = 100 * (diffuse_fib_removed / diffuse_fib_elem) if diffuse_fib_elem != 0 else 0
    perc_DF_fiber_removed_on_tot = 100 * (diffuse_fib_removed / n_points)

    # write results
    string_lines = list()
    string_lines.append('*** Successfully removed fibers on {} total elements, with:'.format(n_points))
    string_lines.append('-- {0} elements as Compact Fibrosis ({1:0.2f}%)'.format(compact_fib_elem, perc_CF_elem))
    string_lines.append('-- {0} elements as Diffuse Fibrosis ({1:0.2f}%)'.format(diffuse_fib_elem, perc_DF_elem))
    string_lines.append('-- {0} Fibers REMOVED from Diffuse Fibrosis elements ({1:0.2f}%) ({2:0.2f}% on total)'.format(
        diffuse_fib_elem, perc_DF_fiber_removed_on_DF, perc_DF_fiber_removed_on_tot))
    string_lines.append('-- {0} elements as Vessels ({1:0.2f}%)'.format(vessels_elem, perc_VSS_elem))
    string_lines.append('')
    string_lines.append('-- {} elements where an exception is occured '.format(exception))

    with open(txt_fpath, "a") as f:
        for line in string_lines:
            print(line)  # to console
            print(line, file=f)  # to file
        print("\n\n", file=f)

    # ================================  SAVE NEW .LON FILES  ==========================================

    fibrosis_lon_filepath = mesh_basepath + '_fibrosis'
    meshIO.write_lon(lonFilename=fibrosis_lon_filepath, lon=lon_fibrosis)

    # print and write output locations
    string_lines = list()
    string_lines.append('New .lon files are saved as: {}.lon '.format(fibrosis_lon_filepath))

    with open(txt_fpath, "a") as f:
        for line in string_lines:
            print(line)  # to console
            print(line, file=f)  # to file
        print("\n\n", file=f)

    # ===========================  GENERATE VECTORS FILES x MESHALYZER  ==================================

    # creation of .vpts and .vec file to meshalyzer visualization
    # (only with the real fibers vector, not sheets and perp)
    if _vpts:
        print(BC.B + ' *** Generation of .vpts and .vec file for visualization...' + BC.ENDC)
        # define new filenames of .vec and .vpts
        vec_filepath = mesh_basepath + '_fibrosis.vec'  # -> ci inserirò le componenti dei vettori (componenti)
        vpts_filepath = mesh_basepath + '_fibrosis.vpts'  # -> ci inserirò le coordinate dei cpts  (posizioni)

        # define the reducing factor of points and vectors
        scale = 100

        # scale the number of centroids
        scaled_cpts = cpts.loc[0:len(cpts):scale]
        print('Selected {} centroids from {} total.'.format(len(scaled_cpts), len(cpts)))

        # save selected centroid points into a .vpts file
        scaled_cpts.to_csv(vpts_filepath, sep=' ', header=False, index=False, mode='w')

        # add in the first line of .vpts file the number of selected centroids
        prepend_line(str(len(scaled_cpts)), vpts_filepath)
        print('Saved selected centroids in:\n -', vpts_filepath)

        # scale the number of vectors
        scaled_vec = lon_fibrosis.loc[0:len(lon_fibrosis):scale].copy()
        print('Selected {} fibers vectors from {} total'.format(len(scaled_vec), len(lon_fibrosis)))

        # Use the z-components as color:
        # [X  Y  Z  V] : X, Y, and Z are the vector components, V the scalar for the color
        # PAY ATTENTION <------------------- now I simply copied the z component
        scaled_vec[3] = scaled_vec[2]

        # save new .vec file
        scaled_vec.to_csv(vec_filepath, sep=' ', header=False, index=False, mode='w')
        print('Saved selected fibres in:\n -', vec_filepath)

    # ======================== SAVE AS VTK =============================================
    print(BC.B + "*** Generation of the mesh in VTK format collecting the new fibrosis.lon file" + BC.ENDC)
    os.system('cd {0} && meshtool collect -imsh={1} -omsh={1}_fibrosis -fib={1}_fibrosis.lon -ofmt=vtu_bin'.format(
        base_path, mesh_basename))

    # =============================================== END OF MAIN =========================================
    print(BC.Y + "*** hole_fibrosis_by_scattering.py terminated." + BC.ENDC)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Load the mesh and the scattering channel of the sample.'
                                                    'Remove fibers in elements with tag 6 and 7, and remove a '
                                                    'percentage of fibers in elements with tag 5 accordingly '
                                                    'with the intensity of the scattering signal.')
    my_parser.add_argument('-msh',
                           '--mesh-basename',
                           help='mesh basename',
                           required=True)
    my_parser.add_argument('-scatt',
                           '--scattering-path',
                           help='Path of .tiff file of scattering channel of the sample ',
                           required=True)
    my_parser.add_argument('-meshpath',
                           '--base-path',
                           help='pass the folder path of mesh files if files are not in the current directory',
                           required=False,
                           default=None)
    my_parser.add_argument('-fib',
                           '--lon-file',
                           help='Filename of the .lon file to read. \n'
                                'If passed, the script use this fibers as input instead the default file.',
                           required=False,
                           default=None)
    my_parser.add_argument('-ds',
                           action='store_true',
                           default=False,
                           dest='scatt_already_ds',
                           help='Add \'-ds\' if the scattering-path refer to a tiff file \n'
                                'already downsampled down the mesh resolution.')
    my_parser.add_argument('-vpts',
                           action='store_true',
                           default=True,
                           dest='vpts',
                           help='Add \'-vpts\' if you want to save the new fibers also in a .vec and .vpts files '
                                'for meshalyzer visualization.')
    my_parser.add_argument('-vtk',
                           action='store_true',
                           default=False,
                           dest='vtk',
                           help='Add \'-vtk\' if you want to save the mesh with the new fibers as .vtk file for '
                                'paraview visualization.')

    # run main
    main(my_parser)
