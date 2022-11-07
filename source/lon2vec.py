import os
import argparse
import numpy as np
import math
import meshIO as meshIO
from datetime import date


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

    # collect arguments
    args = parser.parse_args()

    mesh_basepath   = args.mesh_basepath  # basepath of the mesh. Ex: "/mnt/.../FOLDER/meshbasename"
    lon_filepath    = args.lon_filepath  # different .lon file from the standard
    scale           = args.scale  # the reducing factor of points and vectors
    color_component = args.color_component  # axis to use as scalar value for the color (x, y, or z) - default 2

    # extract meshname
    mesh_basename = os.path.basename(mesh_basepath)
    print(BC.Y + '*** Generating .vec and .vpts files for the mesh {} ...'.format(mesh_basename) + BC.ENDC)

    # generate an absolute path to the .lon file
    if lon_filepath is None:
        # generate standard .lon filepath (if not passed)
        lon_filepath = mesh_basepath + '.lon'
        print(BC.B + '- LON INPUT: the standard .lon file: {}'.format(os.path.basename(lon_filepath)) + BC.ENDC)
    else:
        print(BC.B + '- LON INPUT: selected a different .lon file: \n{}'.format(lon_filepath) + BC.ENDC)

    # ==========================================  LOAD MESH ===============================================

    print(BC.B + 'Loading mesh...' + BC.ENDC)
    pts, elem, lon, cpts = read_carp_mesh(mesh_basepath, lon_fpath=lon_filepath, _read_cpts=True, _verb=True)
    print(BC.B + 'Mesh successfully loaded.' + BC.ENDC)

    # ================================  GENERATE .VPTS and .VEC FILES =======================================

    print(BC.B + ' *** Generation of .vpts and .vec file for visualization...' + BC.ENDC)
    # define new filenames of .vec and .vpts
    lon_filepath_without_ext = os.path.splitext(lon_filepath)[0]
    vec_filepath = lon_filepath_without_ext + '.vec'  # -> ci inserirò le componenti dei vettori (componenti)
    vpts_filepath = lon_filepath_without_ext + '.vpts'  # -> ci inserirò le coordinate dei cpts  (posizioni)

    # scale the number of centroids
    scaled_cpts = cpts.loc[0:len(cpts):scale]
    print('Selected {} centroids from {} total.'.format(len(scaled_cpts), len(cpts)))

    # save selected centroid points into a .vpts file
    scaled_cpts.to_csv(vpts_filepath, sep=' ', header=False, index=False, mode='w')

    # add in the first line of .vpts file the number of selected centroids
    prepend_line(str(len(scaled_cpts)), vpts_filepath)
    print('Saved selected centroids in:\n -', vpts_filepath)

    # scale the number of vectors
    scaled_vec = lon.loc[0:len(lon):scale].copy()
    print('Selected {} fibers vectors from {} total'.format(len(scaled_vec), len(lon)))

    # generate the fake color
    # VEC FILE structure:
    # [X  Y  Z  V] : X, Y, and Z are the vector components, V the scalar for the color
    # [Default - Use the z-components (axis = 2) as color]
    scaled_vec[3] = scaled_vec[color_component]  # assign the color values
    print('Selected axis = {} as scalar values for the vector color.'.format(color_component))

    # save new .vec file
    scaled_vec.to_csv(vec_filepath, sep=' ', header=False, index=False, mode='w')
    print('Saved selected fibres in:\n -', vec_filepath)

    print(BC.Y + "*** lon2vec.py terminated *** " + BC.ENDC)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Generate .vpts and .vec files from the .lon file. ')

    my_parser.add_argument('-msh',
                           '--mesh-basepath',
                           help='mesh basepath (/path/to/the/mesh/meshbasename)',
                           required=True)
    my_parser.add_argument('-lon',
                           '--lon-filepath',
                           help='Path to a different .lon file to load (instead the standard one)',
                           required=False,
                           default=None)
    my_parser.add_argument('-s',
                           '--scale',
                           help='Scale factor for the number of vectors (default: 100)',
                           required=False,
                           type=int,
                           default=100)
    my_parser.add_argument('-c',
                           '--color-component',
                           help='Axis to use as color (default = 2)]. Accepted 0, 1, or 2',
                           required=False,
                           default=2)

    main(my_parser)
