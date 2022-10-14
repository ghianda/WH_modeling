import os
import argparse
import numpy as np
import math
import meshIO as meshIO
from datetime import date


class BC:
    VERB = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
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

    print(BC.WARNING + '*** Filling mesh {} with real fibers ...'.format(mesh_basename) + BC.ENDC)

    # directory of mesh files (if None, take the current directory)
    base_path = base_path if base_path is not None else os.getcwd()

    # compile absolute paths
    mesh_basepath = os.path.join(base_path, mesh_basename)
    R_path        = os.path.join(base_path, R_filename)

    # original (rule-based) .lon file name
    rb_lon_fpath = os.path.join(base_path, mesh_basename + '.lon')

    # initialize output txt file where write statistics
    txt_fpath = os.path.join(base_path, "Fiber_statistics.txt")
    with open(txt_fpath, "w") as f:
        print("Execution date: ", date.today(), file=f)

    # ================================  LOAD MESH and FIBERS ==========================================
    print(BC.BLUE + 'Loading mesh and R files...' + BC.ENDC)

    # Reads in mesh pts file
    pts = meshIO.read_pts(basename=mesh_basepath, file_pts=None)
    print('- Mesh has', len(pts), 'nodes')

    # Reads in mesh elems file 
    elems = meshIO.read_elems(basename=mesh_basepath, file_elem=None)
    print('- Mesh has', len(elems), 'elements')

    # Reads in mesh lon  (rule-based fibers)
    lon_rb = meshIO.read_lon(file_lon=rb_lon_fpath)
    print('- Mesh has', len(lon_rb), 'fibres')

    # Reads in mesh centroids file
    cpts = meshIO.read_cpts(basename=mesh_basepath, file_cpts=None)
    print('- Mesh has', len(cpts), 'centroids')

    # Reads in vector file
    R = np.load(R_path)
    print('Successfully read {}'.format(R_filename))
    print('R.npy has shape (y,x,z):', R.shape)

    # ================================  PREPARE DATA ==========================================

    # extract eigenvector components
    ev_index_orient = 2
    ev_index_sheets = 1
    ev_index_perp   = 0
    fibres_YXZ = R['ev'][..., ev_index_orient]  # quiver = fibres
    sheets_YXZ = R['ev'][..., ev_index_sheets]  # fibers sheets
    perp_YXZ   = R['ev'][..., ev_index_perp]    # perpendic to sheets

    # Defines the voxel resolution of the data (old ds388 -> now 6um)
    data_ps_yxz = np.array([6, 6, 6])  # um
    print('Resolution of the data (r,c,z)', data_ps_yxz, 'um')

    # defines the grane of the orientation analysis
    # (read 'Dimension of Parallelepiped' inside Orientation_INFO.txt)
    grane_yxz = np.array([16, 16, 16])
    print('Resolution of the orientation analysis (r,c,z)', grane_yxz, 'px')

    # defines pixel size of the mesh (from the segmentation)
    mesh_ps_yxz = np.array([20, 20, 20])  # um
    print('Resolution of the mesh (r,c,z)', mesh_ps_yxz, 'um')

    # calculate R pixel size (the resolution of the fibres)
    R_ps_yxz = data_ps_yxz * grane_yxz
    print('Resolution of the fibres (r,c,z)', R_ps_yxz, 'um')

    # pixel size of fibres matrix
    R_ps_x = R_ps_yxz[1]  # column
    R_ps_y = R_ps_yxz[0]  # row
    R_ps_z = R_ps_yxz[2]  # depth

    # # --------------- Tarantula ---------------------------------------------
    # # Converts cpts file from um to voxels     <<<---  (doesn't round yet)
    # cpts_px = cpts.copy()
    # cpts_px.loc[:, 0] = cpts.loc[:, 0] / R_ps_x  # column = x
    # cpts_px.loc[:, 1] = cpts.loc[:, 1] / R_ps_y  # row = y
    # cpts_px.loc[:, 2] = cpts.loc[:, 2] / R_ps_z  # depth = z
    # # -------------------------------------------------------------------------

    # --------------- FTetWild---------------------------------------------
    # original cpts are not in um but in voxel!
    # Convert cpts from voxels to um  <<<--- <<<--- <<<--- <<<--- <<<--- <<<--- <<<---  (doesn't round yet)
    cpts_um = cpts.copy()
    # NB: cpts are in (X,Y,Z) format
    cpts_um.loc[:, 0] = cpts.loc[:, 0] * mesh_ps_yxz[1]  # ax=0 -> x
    cpts_um.loc[:, 1] = cpts.loc[:, 1] * mesh_ps_yxz[0]  # ax=1 -> y
    cpts_um.loc[:, 2] = cpts.loc[:, 2] * mesh_ps_yxz[2]  # ax=2 -> z
    # -------------------------------------------------------------------------

    # check dimension
    print(BC.BLUE + 'Checking dimension of data:' + BC.ENDC)
    print('PIXEL - Shape of Fibres Matrix R (YXZa) in px=96um: ', fibres_YXZ.shape)
    print('PIXEL - Max coordinates of centroids (XYZ) in px=20um [cpts]: ', np.array(cpts.max()))
    print('UM - Shape of Fibres Matrix R (YXZa) in um: ' + BC.WARNING, np.array(fibres_YXZ.shape[0:3]) * R_ps_yxz, BC.ENDC)
    print('UM - Max coordinates of centroids (XYZ) in um [cpts_um]: ' + BC.WARNING, np.array(cpts_um.max()), BC.ENDC)

    # Convert centroids coordinates in the R space
    cpts_Rspace = np.ndarray((len(cpts), 3)).astype(np.uint16)  # (n_points, 3)
    cpts_Rspace[:, 0] = np.floor(np.array(cpts_um.loc[:, 0]) / R_ps_x)  # x = column
    cpts_Rspace[:, 1] = np.floor(np.array(cpts_um.loc[:, 1]) / R_ps_y)  # y = row
    cpts_Rspace[:, 2] = np.floor(np.array(cpts_um.loc[:, 2]) / R_ps_z)  # z = depth

    print('Evaluated integer coordinates of centroids in the R pixel space.')
    print('Saved in cpts_Rspace, list of {} tuple of ({}) elements'.format(len(cpts_Rspace), cpts_Rspace[0].shape))
    print('[INT PIXEL] Max rounded coordinates of centroids (XYZ) in R px space: ({}, {}, {})'.format(
        max(cpts_Rspace[:, 0]), max(cpts_Rspace[:, 1]), max(cpts_Rspace[:, 2])))
    print('Type(cpts_Rspace[:, 0]): ', type(cpts_Rspace[:, 0]))

    # Defines a copy of the fibre file to edit
    # I will create three different .lon files, for fibers, sheets and perp
    lon_real   = lon_rb.copy()
    lon_sheets = lon_rb.copy()
    lon_perp   = lon_rb.copy()

    # ================================  COMPILE NEW .LON FILES  ==========================================

    # Compile .lon with my data  --> LONG TASK (minutes...)
    # Iterates over the mesh centroids, computing the effective voxel location
    n_points    = len(cpts)
    magn        = math.floor(math.log10(n_points))
    exception   = 0  # count exception while insert fibers
    empty       = 0  # count mesh elements where there is empty real fiber 
    ev_xyz      = np.ndarray(3)  # temp array of real fiber vector
    fake_vector = np.array([0.01, 0.01, 0.01])  # fake vector used for 'sheet' and 'perp' if the real-one is empty
    # NB -> I will insert the rule-based fiber where the real-one is empty

    print(BC.BLUE + '*** Start compiling real fibers inside the new .lon file... ' + BC.ENDC)
    for i in range(n_points):
        # print progress percent
        if i % 10**(magn - 1) == 0:
            print(' - {0:3.0f} %'.format(100 * i / n_points))

        # Sees which voxel we're in
        x = cpts_Rspace[i, 0]  # Col = x
        y = cpts_Rspace[i, 1]  # Row = y
        z = cpts_Rspace[i, 2]  # Dep = z

        # try to find the correspondent chunk in R of the current element centroid
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

    # percentage of real fibers detected for the entire mesh:
    fiber_accuracy = 100 * (1  - (empty / n_points))

    # write results
    string_lines = list()
    string_lines.append('*** Successfully filled the real fibers on {} total elements, with:'.format(n_points))
    string_lines.append('-- {} elements where real fiber was empty -> filled with rule-based ones'.format(empty))
    string_lines.append('-- {} elements where the fiber isn\'t empty but an exception is occured while compile new .lon files'.format(exception))
    string_lines.append('*** Percentage of real fibers in the mesh: {0:3.2f}%'.format(fiber_accuracy))

    with open(txt_fpath, "a") as f:
        for line in string_lines:
            print(line)  # to console
            print(line, file=f)  # to file
        print("\n\n", file=f)

    # ================================  SAVE NEW .LON FILES  ==========================================

    # Defines mapped lon filenames
    real_lon_filename 	= mesh_basepath + '_realfibers'
    sheets_lon_filename = mesh_basepath + '_sheets'
    perp_lon_filename 	= mesh_basepath + '_perp'
    
    # Writes-out mapped lon file
    meshIO.write_lon(lonFilename=real_lon_filename, lon=lon_real)
    meshIO.write_lon(lonFilename=sheets_lon_filename, lon=lon_sheets)
    meshIO.write_lon(lonFilename=perp_lon_filename, lon=lon_perp)

    # print and write output locations
    string_lines = list()
    string_lines.append('New .lon files are saved as:')
    string_lines.append('-- {}.lon  -> fibers vectors (ev=2)'.format(real_lon_filename))
    string_lines.append('-- {}.lon  -> sheets vectors (ev=1)'.format(sheets_lon_filename))
    string_lines.append('-- {}.lon  -> perpendicular vectors (ev=0)'.format(perp_lon_filename))

    with open(txt_fpath, "a") as f:
        for line in string_lines:
            print(line)  # to console
            print(line, file=f)  # to file
        print("\n\n", file=f)

    # ===========================  GENERATE VECTORS FILES x MESHALYZER  ==================================

    # creation of .vpts and .vec file to meshalyzer visualization
    # (only with the real fibers vector, not sheets and perp)
    if _vpts:
        print(BC.BLUE + ' *** Generation of .vpts and .vec file for visualization...' + BC.ENDC)
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
        scaled_vec = lon_real.loc[0:len(lon_real):scale].copy()
        print('Selected {} fibers vectors from {} total'.format(len(scaled_vec), len(lon_real)))

        # Use the z-components as color:
        # [X  Y  Z  V] : X, Y, and Z are the vector components, V the scalar for the color
        # PAY ATTENTION <------------------- now I simply copied the z component
        scaled_vec[3] = scaled_vec[2]

        # save new .vec file
        scaled_vec.to_csv(vec_filepath, sep=' ', header=False, index=False, mode='w')
        print('Saved selected fibres in:\n -', vec_filepath)

        print(BC.WARNING + "8_insert_real_fibers.py terminated." + BC.ENDC)


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
