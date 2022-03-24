import os
import argparse
import numpy as np
import math
import meshIO as meshIO


class Bcolors:
    VERB = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
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


def main(parser):

    # collect arguments
    args = parser.parse_args()
    mesh_basename = args.mesh_basename[0]
    R_filename    = args.R_filename[0]
    base_path     = args.base_path
    _vpts = args.vpts  # bool

    print(Bcolors.OKBLUE + '*** Filling mesh with real fibers ...' + Bcolors.ENDC)

    # directory of mesh files (if None, take the current directory)
    base_path = base_path if base_path is not None else os.getcwd()

    # compile absolute paths
    mesh_basepath = os.path.join(base_path, mesh_basename)
    R_path        = os.path.join(base_path, R_filename)

    # Reads in mesh pts file
    pts = meshIO.read_pts(basename=mesh_basepath, file_pts=None)
    print('- Mesh has', len(pts), 'nodes')

    # Reads in mesh elems file 
    elems = meshIO.read_elems(basename=mesh_basepath, file_elem=None)
    print('- Mesh has', len(elems), 'elements')

    # Reads in mesh lon  (rule-based fibers)
    lon_rb = meshIO.read_lon(basename=mesh_basepath, file_lon=None)
    print('- Mesh has', len(lon_rb), 'fibres')

    # Reads in mesh centroids file
    cpts = meshIO.read_cpts(basename=mesh_basepath, file_cpts=None)
    print('- Mesh has', len(cpts), 'centroids')

    # Reads in vector file
    R = np.load(R_path)
    print('Successfully read {}'.format(R_filename))
    print('R.npy has shape (y,x,z):', R.shape)

    # extract eigenvector components
    ev_index_orient = 2
    ev_index_sheets = 1
    ev_index_perp   = 0
    fibres_YXZ = R['ev'][..., ev_index_orient]  # quiver = fibres
    sheets_YXZ = R['ev'][..., ev_index_sheets]  # fibers sheets
    perp_YXZ   = R['ev'][..., ev_index_perp]

    # Defines the voxel resolution of the data (old ds388 -> now 6um)
    data_ps_yxz = np.array([6, 6, 6])  # um
    print('Resolution of the data (r,c,z)', data_ps_yxz, 'um')

    # defines the grane of the orientation analysis
    # (read 'Dimension of Parallelepiped' inside Orientation_INFO.txt)
    grane_yxz = np.array([16, 16, 16])
    print('Resolution of the orientation analysis (r,c,z)', grane_yxz, 'px')

    # defines resultion of the mesh and of the segmentation
    mesh_ps_yxz = np.array([20, 20, 20])  # um
    print('Resolution of the mesh (r,c,z)', mesh_ps_yxz, 'um')

    # calculate R pixel size (the resolution of the fibres)
    R_ps = data_ps_yxz * grane_yxz
    print('Resolution of the fibres (r,c,z)', R_ps, 'um')
    print()

    # pixel size of fibres
    res_x = R_ps[1]  # column
    res_y = R_ps[0]  # row
    res_z = R_ps[2]  # depth

    # Converts cpts file from um to voxels     <<<---  (doesn't round yet)
    cpts_px = cpts.copy()
    cpts_px.loc[:, 0] = cpts.loc[:, 0] / res_x  # column = x
    cpts_px.loc[:, 1] = cpts.loc[:, 1] / res_y  # row = y
    cpts_px.loc[:, 2] = cpts.loc[:, 2] / res_z  # depth = z

    # check dimension
    print('Shape of Fibres Matrix (YXZa) in px: ', fibres_YXZ.shape)
    print('Max coordinates of centroids (XYZ) in px: ', np.array(cpts_px.max()))
    print('Shape of Fibres Matrix (YXZa) in um: ', np.array(fibres_YXZ.shape[0:3]) * R_ps)
    print('Max coordinates of centroids (XYZ) in um: ', np.array(cpts.max()))

    # round the coordinates of centroids in the px space in integer numbers
    cpts_int = np.ndarray((len(cpts_px), 3)).astype(np.uint16)  # (n_points, 3)
    cpts_int[:, 0] = np.floor(np.array(cpts_px.loc[:, 0]))  # x = column
    cpts_int[:, 1] = np.floor(np.array(cpts_px.loc[:, 1]))  # y = row
    cpts_int[:, 2] = np.floor(np.array(cpts_px.loc[:, 2]))  # z = depth
    print('Evaluated integer coordinates of centroids in the px space.')
    print('Saved in cpts_int, list of {} tuple of ({}) elements'.format(len(cpts_int), cpts_int[0].shape))
    print('Max rounded coordinates of centroids (XYZ) in px: ({}, {}, {})'.format(max(cpts_int[:, 0]), max(cpts_int[:, 1]), max(cpts_int[:, 2])))
    print(type(cpts_int[:, 0]))

    # Defines a copy of the fibre file to edit
    # I will create three different .lon files, for fibers, sheets and perp
    lon_real   = lon_rb.copy()
    lon_sheets = lon_rb.copy()
    lon_perp   = lon_rb.copy()

    # Compile .lon with my data  --> LONG TASK (minutes...)
    # Iterates over the mesh centroids, computing the effective voxel location
    n_points    = len(cpts)
    magn        = math.floor(math.log10(n_points))
    exception   = 0  # count exception while insert fibers
    empty       = 0  # count mesh elements where there is empty real fiber 
    ev_xyz      = np.ndarray(3)  # temp array of real fiber vector
    fake_vector = np.array([0.01, 0.01, 0.01])  # fake sheet and perp if real-one is empty
    # NB -> I will insert the rule-based fiber where the real-one is empty

    print('*** Start compiling the new .lon files... ')
    for i in range(n_points):
        # print progress percent
        if i % 10**(magn - 1) == 0:
            print(' - {0:3.0f} %'.format(100 * i / n_points))

        # Sees which voxel we're in
        x = cpts_int[i, 0]  # Col = x
        y = cpts_int[i, 1]  # Row = y
        z = cpts_int[i, 2]  # Dep = z

        # try to find the corrispondent chunk in R of the current element centroid
        try:
            # check real fiber in this voxel
            if ~np.any(fibres_YXZ[y, x, z]):
                # real fiber is empty, i use the theoretical one
                lon_real.loc[i]   = lon_rb.loc[i]
                lon_sheets.loc[i] = fake_vector
                lon_perp.loc[i]   = fake_vector
                empty = empty + 1

            else:
                # insert real data
                try:
                    # collect real fiber components by the matrix
                    ev_xyz[0] = fibres_YXZ[y, x, z, 1]
                    ev_xyz[1] = fibres_YXZ[y, x, z, 0]
                    ev_xyz[2] = fibres_YXZ[y, x, z, 2]

                    # assign congruent direction of my real fibers
                    # by scalar product between theoretical and real fibers:
                    scalar = np.dot(ev_xyz, np.array(lon_rb.loc[i]))
                    if scalar < 0:
                        # change the direction of the real versors
                        ev_xyz              = np.negative(ev_xyz)
                        sheets_YXZ[y, x, z] = np.negative(sheets_YXZ[y, x, z])
                        perp_YXZ[y, x, z]   = np.negative(perp_YXZ[y, x, z])

                    # insert real components
                    lon_real.loc[i][[0, 1, 2]]   = np.array(ev_xyz)  # already XYZ
                    lon_sheets.loc[i][[0, 1, 2]] = sheets_YXZ[y, x, z][[1, 0, 2]]  # YXZ -> XYZ
                    lon_perp.loc[i][[0, 1, 2]]	 = perp_YXZ[y, x, z][[1, 0, 2]]  # YXZ -> XYZ

                except ValueError as e:
                    print(Bcolors.FAIL)
                    print('i={} - position: (x, y, z) = ({},{},{})'.format(i, x, y, z))
                    print('FAIL with ValueError: {}'.format(e))
                    print(Bcolors.ENDC)
                    exception = exception + 1
                except:
                    print(Bcolors.FAIL + 'FAIL with unknown error')
                    print('i={} - position: (x, y, z) = ({},{},{})'.format(i, x, y, z))
                    print(Bcolors.ENDC)
                    exception = exception + 1
        except:
            # current element is out of R dimension

    print('*** Successfully filled the real fibers on {} total elements, with:'.format(n_points))
    print('-- {} elements where real fibers was empty -> filled with rule-based ones'.format(empty))
    print('-- {} elements where the fibers isn\'t empty but an exception is occured while compile new .lon files'.format(exception))

    # Defines mapped lon filenames
    real_lon_filename 	= mesh_basepath + '_realfibers'
    sheets_lon_filename = mesh_basepath + '_sheets'
    perp_lon_filename 	= mesh_basepath + '_perp'
    
    # Writes-out mapped lon file
    meshIO.write_lon(lonFilename=real_lon_filename, lon=lon_real)
    meshIO.write_lon(lonFilename=sheets_lon_filename, lon=lon_sheets)
    meshIO.write_lon(lonFilename=perp_lon_filename, lon=lon_perp)

    print('new .lon files are saved as:')
    print('-- {}.lon  -> fibers vectors (ev=2)'.format(real_lon_filename))
    print('-- {}.lon  -> sheets vectors (ev=1)'.format(sheets_lon_filename))
    print('-- {}.lon  -> perpendicular vectors (ev=0)'.format(perp_lon_filename))
    
    # creation of .vpts and .vec file to meshalyzer visualization
    # (only with the real fibers vector, not sheets and perp)
    if _vpts:
        print(Bcolors.OKBLUE + ' *** Creation of .vpts and .vec file for visualization...' + Bcolors.ENDC)
        # define new filenames of .vec and .vpts
        vec_filepath = mesh_basepath + '.vec'  # -> ci inserirò le componenti dei vettori (componenti)
        vpts_filepath = mesh_basepath + '.vpts'  # -> ci inserirò le coordinate dei cpts  (posizioni)

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
        scaled_vec = lon_real.loc[0:len(lon_real):scale]
        print('Selected {} fibers vectors from {} total'.format(len(scaled_vec), len(lon_real)))

        # fill the new .vec array adding a new column for color plot
        # [X  Y  Z  V] : X, Y, and Z are the vector components, V the scalar for the color
        # PAY ATTENTION <------------------- now I simply copy teh z component
        scaled_vec['3'] = scaled_vec[2]

        # save new .vec file
        scaled_vec.to_csv(vec_filepath, sep=' ', header=False, index=False, mode='w')
        print('Saved selected fibres in:\n -', vec_filepath)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Extract real fibers, sheets and perpendicular vectors from R.npy and write into three different .lon files')
    my_parser.add_argument('-msh',
                           '--mesh-basename',
                           nargs='+',
                           help='mesh basename',
                           required=True)
    my_parser.add_argument('-r',
                           '--R-filename',
                           nargs='+',
                           help='R numpy filename',
                           required=True)
    my_parser.add_argument('-meshpath',
                           '--base-path',
                           help='pass the folder path of mesh files if files are not in the current directory',
                           required=False,
                           default=None)
    my_parser.add_argument('-v',
                           action='store_true',
                           default=True,
                           dest='vpts',
                           help='Add \'-v\' if you want to save the new fibers also in a .vec and .vpts files '
                                'for meshalyzer visualization.')

    main(my_parser)
