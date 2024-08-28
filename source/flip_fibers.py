import os
import argparse
import math
import meshIO as meshIO
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
    args         = parser.parse_args()
    lon_filepath = args.lon_fiber_filepath
    flip_axis    = args.axis  # character 'x', 'y' or 'z'

    # filename and base path of the directory of mesh files
    lon_fname  = os.path.basename(lon_filepath)
    base_path  = os.path.dirname(lon_filepath)

    print(BC.Y + '*** Flipping the fibers along {} in the file: {} *** '.format(flip_axis, lon_fname) + BC.ENDC)

    # ================================  LOAD, PREPARE DATA and create a BACKUP===============================
    print(BC.B + 'Loading fibers ...' + BC.ENDC)
    lon = meshIO.read_lon(basename=None, file_lon=lon_filepath)

    # Defines a copy of the elem file to edit
    lon_inverted = lon.copy()

    # create a backup of the old lon file (the new one will be overwritten)
    backup_lon_fname = '{}.backup_before_flip_{}.lon'.format(lon_fname, flip_axis)
    os.system('cd {0} && cp {1} {2}'.format(base_path, lon_filepath, backup_lon_fname))
    print('Backup of the .lon file created as: '.format(backup_lon_fname))

    # ================================  COMPILE NEW .PTS FILES  ==========================================

    print(BC.B + 'Start flipping the Z-comp in the fibers of the mesh...' + BC.ENDC)
    
    # invert the fiber along the selected axis
    if flip_axis == 'x':
        lon_inverted[0] = -lon[0]
    elif flip_axis == 'y':
        lon_inverted[1] = -lon[1]
    if flip_axis == 'z':
        lon_inverted[2] = -lon[2]

    print('Terminated.')

    # debug check
    i = 0
    print("components of the first lon: ({}, {}, {})".format(lon.loc[i][0], lon.loc[i][1], lon.loc[i][2]))
    print("..............same inverted: ({}, {}, {})".format(lon_inverted.loc[i][0], lon_inverted.loc[i][1], lon_inverted.loc[i][2]))

    # ================================  SAVE NEW .lon FILES  ==========================================

    # overwrite the old .pts file with the new values
    print(BC.B + 'Saving the new values in the lon file...' + BC.ENDC)
    meshIO.write_lon(lonFilename=lon_filepath[:-4], lon=lon_inverted)

    # print output locations
    print(BC.B + 'Flipped fibers saved in:' + BC.ENDC)
    print(lon_filepath)

    # =============================================== END OF MAIN =========================================
    print(BC.Y + "*** flip_fibers.py terminated." + BC.ENDC)

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Load the .lon file and invert the z component. '
                                                    ' Create a new .pts file and backup the old one.')

    my_parser.add_argument('-lon',
                           '--lon-fiber-filepath',
                           help='Complete path of the .lon file of the mesh',
                           required=True)
    my_parser.add_argument('-a',
                           '--axis',
                           help='Axis to flip. Accepeted values: x, y, or z.'
                                'Ex: (x -> x component will be inverted in all fibers). '
                                'If not passed, y will be used (long axis of the heart).',
                           required=False,
                           default='y')
    # run main
    main(my_parser)
