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
    # perchè [4, 9] sono cmq valori di fibrosi, quindi gli "estremi" sono 3 (fib al 0%) e 10 ( fib al 100%)
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
    mesh_basepath = args.mesh_basepath  # basepath of the mesh. Ex: "/mnt/.../FOLDER/meshbasename"
    tags          = args.tags  # list of thags where remove fibers
    lon_fpath     = args.lon_filepath  # different .lon file from the standard
    _vtk          = args.vtk

    # extract meshname
    mesh_basename = os.path.basename(mesh_basepath)

    print(BC.Y + '*** Removing Fibers on Mesh {} ***'.format(mesh_basename) + BC.ENDC)

    # generate an absolute path to the .lon file
    if lon_fpath is None:
        # generate standard .lon filepath (if not passed)
        lon_fpath = mesh_basepath + '.lon'
        print(BC.B + '- LON INPUT: the standard .lon file: {}'.format(os.path.basename(lon_fpath)) + BC.ENDC)
    else:
        print(BC.B + '- LON INPUT: selected a different .lon file: \n{}'.format(lon_fpath) + BC.ENDC)

    # list of tags on a string format
    tags_str = ', '.join(map(str, tags))
    print(BC.B + '- Removing fibers on tags: ' + BC.Y + tags_str + BC.ENDC)

    # ================================  LOAD and PREARE DATA ==========================================
    print(BC.B + 'Loading mesh ...' + BC.ENDC)

    # loading mesh in carp format
    pts, elem, lon, cpts = read_carp_mesh(mesh_basepath, lon_fpath=lon_fpath, _read_cpts=True, _verb=True)

    # ================================  PRE-ITERATION OPERATIONS  ==========================================

    # supports
    n_points      = len(cpts)
    magn          = math.floor(math.log10(n_points))
    empty_fiber   = np.array([0.0, 0.0, 0.0])  # fake empty fiber used to replace original fiber

    # counters
    exception      = 0  # count exceptions while manage fibers
    n_fibers_removed = 0  # count of fiber removed

    # Defines a copy of the fibre file to edit
    lon_modified = lon.copy()

    print(BC.B + 'Start removing fibers inside the new .lon file... ' + BC.ENDC)

    # ================================  COMPILE NEW .LON FILES  ==========================================
    print("Progress:")
    for i in range(n_points):
        # print progress percent
        if i % 10**(magn - 1) == 0:
            print(' - {0:3.0f} %'.format(100 * i / n_points))

        # read tag of current element
        current_tag = elem.loc[i, 5]

        try:
            if current_tag in tags:
                # remove the fiber
                lon_modified.loc[i] = empty_fiber
                n_fibers_removed = n_fibers_removed + 1  # counter

        except ValueError as e:
            print(BC.FAIL)
            print('i={}'.format(i))
            print('FAIL with ValueError: {}'.format(e))
            print(BC.ENDC)
            exception = exception + 1
        except:
            print(BC.FAIL + 'FAIL with unknown error')
            print('i={}'.format(i))
            print(BC.ENDC)
            exception = exception + 1

    print(BC.B + 'Terminated.' + BC.ENDC)

    # =========================== STATISTICS =====================================
    # evaluate percentage of removed fibers in the entire mesh:
    perc_fiber_removed = 100 * (n_fibers_removed / n_points)

    print(BC.B + '*** Successfully removed {2:0.2f}% of fibers ({0} fibers on {1})'.format(
        n_fibers_removed, n_points, perc_fiber_removed))
    print(' by elements with tags: ' + tags_str + BC.ENDC)

    # ================================  SAVE NEW .LON FILES  ==========================================

    new_lon_filepath = mesh_basepath + '_nofiber_tags_' + '-'.join(map(str, tags))
    meshIO.write_lon(lonFilename=new_lon_filepath, lon=lon_modified)

    # print and write output locations
    print('New .lon files are saved as: \n{}.lon '.format(new_lon_filepath))

    # ======================== SAVE AS VTK =============================================
    print(BC.B + "*** Generation of the mesh in VTK format collecting the new .lon file" + BC.ENDC)
    os.system('cd {0} && meshtool collect -imsh={1} -omsh={2} -fib={2}.lon -ofmt=vtu_bin'.format(
        os.path.dirname(mesh_basepath), mesh_basename, os.path.basename(new_lon_filepath)))

    # =============================================== END OF MAIN =========================================
    print(BC.Y + "*** remove_fibers_by_tags.py terminated." + BC.ENDC)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Load the mesh, and remove fibers from elements'
                                                    'with the selected tags.')
    my_parser.add_argument('-msh',
                           '--mesh-basepath',
                           help='mesh basepath (/path/to/the/mesh/meshbasename)',
                           required=True)
    my_parser.add_argument('-t',
                           '--tags',
                           nargs='+',
                           help='Tag (or list of tags) where remove fibers. \n'
                                'Example: -t 5 6 7',
                           required=True,
                           type=int)
    my_parser.add_argument('-lon',
                           '--lon-filepath',
                           help='Filename of the .lon file to read. \n'
                                'If passed, the script use this fibers as input instead the default file.',
                           required=False,
                           default=None)
    my_parser.add_argument('-vtk',
                           action='store_true',
                           default=False,
                           dest='vtk',
                           help='Add \'-vtk\' if you want to save the mesh with the new fibers as .vtk file for '
                                'paraview visualization.')

    # run main
    main(my_parser)
