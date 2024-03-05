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
    pts_filepath = args.points_filepath
    trasl        = args.trasl

    # filename and base path of the directory of mesh files
    points_fname  = os.path.basename(pts_filepath)
    base_path     = os.path.dirname(pts_filepath)

    print(BC.Y + '*** Flipping the points along Z in the file: {} *** '.format(points_fname) + BC.ENDC)

    # ================================  LOAD, PREPARE DATA and create a BACKUP===============================
    print(BC.B + 'Loading points ...' + BC.ENDC)
    pts = meshIO.read_pts(basename=None, file_pts=pts_filepath)

    # Defines a copy of the elem file to edit
    pts_traslated = pts.copy()

    # create a backup of the old pts file (the new one will be overwritten)
    os.system('cd {0} && cp {1} {1}.backup_before_flipZ.pts'.format(base_path, pts_filepath))
    print('Backup of the .pts file created as: {}.backup_before_flipZ.pts'.format(points_fname))

    # ================================  PRE-ITERATION OPERATIONS  ==========================================

    # supports
    n_pts = len(pts)
    magn  = math.floor(math.log10(n_pts))

    # counters
    exception = 0  # count exceptions while manage elements

    print(BC.B + 'After flipping, translate of: ' + BC.ENDC, trasl)
    print('Start flipping the points of the mesh... ')

    # ================================  COMPILE NEW .PTS FILES  ==========================================
    
    # trasl the coordinates (z = 2' axis)
    pts_traslated[2] = -pts[2] + trasl
    print('Terminated.')

    # debug check
    i = 0
    print("coordinates of the first point: ({}, {}, {})".format(pts.loc[i][0], pts.loc[i][1], pts.loc[i][2]))
    print("....... first point translated: ({}, {}, {})".format(pts_traslated.loc[i][0], pts_traslated.loc[i][1], pts_traslated.loc[i][2]))

    # ================================  SAVE NEW .pts FILES  ==========================================

    # overwrite the old .pts file with the new values
    print(BC.B + 'Saving the new values in the pts file...' + BC.ENDC)
    meshIO.write_pts(ptsFilename=pts_filepath[:-4], pts=pts_traslated)

    # print output locations
    print(BC.B + 'Rescaled points saved in:' + BC.ENDC)
    print(pts_filepath)

    # =============================================== END OF MAIN =========================================
    print(BC.Y + "*** rescale_mesh_pts.py terminated." + BC.ENDC)

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Load the mesh and flip the points along z axis. '
                                                    ' Create a new .pts file and backup the old one.')

    my_parser.add_argument('-pts',
                           '--points-filepath',
                           help='Complete path of the .pts file of the mesh',
                           required=True)
    my_parser.add_argument('-t',
                           '--trasl',
                           help='after inverting z components, mesh will be translated by this value'
                                '[to recenter the mesh, insert a value t = (bbox_z_min) + (bbox_z_max)',
                           required=True,
                           type=int)
    # run main
    main(my_parser)
