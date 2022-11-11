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


def main(parser):

    # ===================================== INPUT and Initial Messages ================================================

    # collect arguments
    args          = parser.parse_args()
    mesh_basename = args.mesh_basename
    base_path     = args.base_path
    init_tags     = args.init_tags
    final_tag     = int(args.final_tag)
    out_name      = args.out_name
    _vtk          = args.vtk

    print(BC.Y + '*** Retagging elements in the mesh {} *** '.format(mesh_basename) + BC.ENDC)

    # directory of mesh files (if None, take the current directory)
    base_path = base_path if base_path is not None else os.getcwd()
    mesh_basepath = os.path.join(base_path, mesh_basename)

    # initialize output txt file where write statistics
    txt_fpath = os.path.join(base_path, "Retagging_elements_info.txt")
    with open(txt_fpath, "w") as f:
        print("Execution date: ", date.today(), file=f)

    # ================================  LOAD, PREPARE DATA and create a BACKUP===============================
    print(BC.B + 'Loading mesh ...' + BC.ENDC)
    pts, elem, lon = read_carp_mesh(mesh_basepath, _read_cpts=False, _verb=True)

    # Defines a copy of the elem file to edit
    elem_retagged = elem.copy()

    # create a backup of the old elem file (the new one will be overwritten)
    os.system('cd {0} && cp {1}.elem {1}.backup_before_retag'.format(base_path, mesh_basename))
    print('Backup of the .elem file created as: {}.backup_before_retag'.format(mesh_basename))

    # ================================  PRE-ITERATION OPERATIONS  ==========================================

    # supports
    n_elements    = len(elem)
    magn          = math.floor(math.log10(n_elements))

    # counters
    exception           = 0  # count exceptions while manage elements
    n_elements_retagged = 0  # count elements retagged

    # tag(s) to be replaced:
    print(BC.B + 'Tag to be replaced: ' + BC.ENDC, init_tags)
    print(BC.B + 'New tag value: ' + BC.ENDC, final_tag)
    print(BC.B + 'Start retagging elements in the .elem file... ' + BC.ENDC)

    # ================================  COMPILE NEW .LON FILES  ==========================================
    print("Progress:")
    for i in range(n_elements):
        # print progress percent
        if i % 10**(magn - 1) == 0:
            print(' - {0:3.0f} %'.format(100 * i / n_elements))

        # read tag of current element
        current_tag = elem.loc[i, 5]

        try:
            if current_tag in init_tags:

                #change the tag
                elem_retagged.loc[i, 5] = final_tag

                # increase the counter
                n_elements_retagged = n_elements_retagged + 1  # counter

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
    perc_elem_retagged  = 100 * (n_elements_retagged / n_elements)

    # write results
    string_lines = list()
    string_lines.append('*** Successfully retagged {0} elements ({1:0.2f}% of total):'.format(
        n_elements_retagged, perc_elem_retagged))
    string_lines.append('-- {} elements where an exceptions is occurred'.format(exception))

    with open(txt_fpath, "a") as f:
        for line in string_lines:
            print(line)  # to console
            print(line, file=f)  # to file
        print("\n\n", file=f)

    # ================================  SAVE NEW .ELEM FILES  ==========================================

    meshIO.write_elems(elemFilename=mesh_basepath, elem=elem_retagged)

    # print and write output locations
    string_lines = list()
    string_lines.append('Retagged elements file saved in:')
    string_lines.append(mesh_basepath)

    with open(txt_fpath, "a") as f:
        for line in string_lines:
            print(line)  # to console
            print(line, file=f)  # to file
        print("\n\n", file=f)

    # ======================== SAVE AS VTK =============================================
    if _vtk:
        print(BC.B + "*** Generation of the mesh in VTK format collecting the new .elem file" + BC.ENDC)
        os.system('cd {0} && meshtool convert -imsh={1} -ifmt=carp_txt '
                  '-omsh={1}_rettaged_as_{2}.vtk -ofmt=vtk_bin'.format(base_path, mesh_basename, final_tag))

    # =============================================== END OF MAIN =========================================
    print(BC.Y + "*** retag.py terminated." + BC.ENDC)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Load the mesh and change the tag of all the elements selected \n '
                                                    'by user input.')
    my_parser.add_argument('-msh',
                           '--mesh-basename',
                           help='mesh basename',
                           required=True)
    my_parser.add_argument('-meshpath',
                           '--base-path',
                           help='pass the folder path of mesh files if files are not in the current directory',
                           required=False,
                           default=None)
    my_parser.add_argument('-it',
                           '--init-tags',
                           nargs='+',
                           help='Tag (or list of tags) integer to be replaced with the final tag. \n'
                                'Example: -it 5 6 7',
                           required=True,
                           type=int)
    my_parser.add_argument('-ft',
                           '--final-tag',
                           help='(int) Tag to write in the selected elements.',
                           required=True,
                           type=int)
    my_parser.add_argument('-o',
                           '--out-name',
                           help='Filename of the new .elem file (if not passed, it creates a backup of the old one \n'
                                ' as MESHNAME.backup_before_retag and then overwrite the .elem file)'
                                '\n ',
                           required=False)
    my_parser.add_argument('-vtk',
                           action='store_true',
                           default=False,
                           dest='vtk',
                           help='Add \'-vtk\' if you want to save the mesh with the new tags as .vtk file for '
                                'paraview visualization.')

    # run main
    main(my_parser)
