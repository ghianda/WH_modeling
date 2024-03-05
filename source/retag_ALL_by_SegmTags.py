import os
import argparse
import numpy as np
import math
import meshIO as meshIO
from datetime import date
import tifffile as tiff
from scipy import ndimage

'''============================================================
# - created script to RETAG a mesh from the segmentation TAGS saved as PNG:
- first, save the PNG as TIFF
- in the TIFF, pixels should be saved as:
# 0 -> background
# 1 -> MYO
# 2 -> LV_pool
# 3 -> RV_pool
# 4 -> disk
# 5 -> NCF (not compact fibrosis)
# 6 -> CF (compact fibrosis)
# 7 -> other empty (vessels and holes)
#
# N-B. Here, only Myo, Cf, Ncf and Vessels wil be reatgged because INPUT MESH IS NX_TISSUE

# The script scroll the elements, find the position in the tiff, check the intensity,  
# and assign the tag accordingly (1,5,6 or 7)
==============================================================='''

# HARDCODED TAGS: change here if the tiff has different values:
class TAG:
    MYO  = 1
    NCF  = 5
    CF   = 6
    VESS = 7

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
    mesh_basepath = args.mesh_basepath
    tags_tiffpath = args.tiff_tags
    _vtk          = args.vtk

    # extract meshname folder path
    mesh_basename = os.path.basename(mesh_basepath)
    base_path = os.path.dirname(mesh_basepath)

    print(BC.Y + '*** Retagging elements in the mesh {} *** '.format(mesh_basename) + BC.ENDC)

    # initialize output txt file where write statistics
    txt_fpath = os.path.join(base_path, "Retagging_ALL_TISSUE_elements_info.txt")
    with open(txt_fpath, "w") as f:
        print("Execution date: ", date.today(), file=f)
        print("\n", file=f)
        print("Input Tagged tiff file: \n", tags_tiffpath, file=f)

    # ================================  LOAD, PREPARE DATA and create a BACKUP===============================
    print(BC.B + 'Loading mesh ...' + BC.ENDC)
    pts, elem, lon, cpts = read_carp_mesh(mesh_basepath, _read_cpts=True, _verb=True)

    # Defines a copy of the elem file to edit
    elem_retagged   = elem.copy()
    backup_filename = 'backup_before_retag_ALL'

    # create a backup of the old elem file (the new one will be overwritten)
    os.system('cd {0} && cp {1}.elem {1}.{2}'.format(base_path, mesh_basename, backup_filename))
    print('Backup of the .elem file created as: {0}.elem.{1}'.format(mesh_basename, backup_filename))

    # load scattering tiff
    tags_tiff_zyx = tiff.imread(tags_tiffpath)
    tags_tiff_yxz = np.moveaxis(tags_tiff_zyx, 0, -1)  # move (z, y, x) to (row, col, z) = (YXZ)
    tags_tiff_shape_yxz = tags_tiff_yxz.shape
    print('\nLoaded Tagged tiff file (dtype: ', tags_tiff_yxz.dtype, ') with shape: ', tags_tiff_shape_yxz)

    # defines pixel sizes
    # the points in the mesh are already "scaled" with the pixel size of segmentation (20um) - no need rescaling
    tiff_ps_yxz = np.array([20, 20, 20])
    print('Resolution of the tagged tiff data (r,c,z)', tiff_ps_yxz, 'um')

    # check dimension
    print(BC.B + 'Checking dimension of data:' + BC.ENDC)
    print('PIXEL - Shape of tagged tiff data (Y-X-Z) in px: ', tags_tiff_shape_yxz)
    print('UM    - Shape of tagged tiff data (Y-X-Z) in um: ', tags_tiff_shape_yxz * tiff_ps_yxz)
    print('CPTS  - Max coordinates of mesh centroids (X-Y-Z)  : ', np.array(cpts.max()))

    # ================================  PRE-ITERATION OPERATIONS  ==========================================

    # supports
    n_elements    = len(elem)
    magn          = math.floor(math.log10(n_elements))

    # counters
    exception           = 0  # count exceptions while manage elements
    n_elements_retagged = 0  # count elements retagged

    # just for debugging
    stringlist_elements_retagged = list()

    print(BC.B + 'Start retagging elements in the .elem file... ' + BC.ENDC)

    # ================================  COMPILE NEW .ELEM FILES  ==========================================
    print("Progress:")
    for i in range(n_elements):

        # read tag of current element
        current_tag = elem.loc[i, 5]

        try:
            # Sees which voxel we're in (in the tagged_tiff image's pixel size)
            x = np.floor(cpts.loc[i, 0] / tiff_ps_yxz[0])
            y = np.floor(cpts.loc[i, 1] / tiff_ps_yxz[1])
            z = np.floor(cpts.loc[i, 2] / tiff_ps_yxz[2])

            # read scattering signal in that position (in pixel)
            tiff_ldg = tags_tiff_yxz[int(y), int(x), int(z)]

            # HERE, I want to retag ALL elements, I can use directly the tiff_ldg as the new tag
            elem_retagged.loc[i, 5] = tiff_ldg  #

            # increase the counter - CAN BE DELETED IF IT WORKS
            n_elements_retagged = n_elements_retagged + 1  # counter

            # OLD APPROACH with IF: ---------------
            # if tiff_ldg == TAG.MYO:
            #     elem_retagged.loc[i, 5] = TAG.MYO
            #
            #     # increase the counter
            #     n_elements_retagged = n_elements_retagged + 1  # counter
            #
            #     # just for debugging
            #     stringlist_elements_retagged.append('elem={0}; tag={1}; scatt={2} ->> rettaged with {3}'.format(
            #         i, current_tag, tiff_ldg, MYO_tag))

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

        # print progress percent and debugging info
        if i % 10 ** (magn - 1) == 0:
            print(' - {0:3.0f} %'.format(100 * i / n_elements))

            # just for debugging
            stringlist_elements_retagged.append('elem={0}; tag={1}; tiff={2} ->> rettaged with {2}'.format(
                i, current_tag, tiff_ldg))

    print(BC.B + 'Terminated.' + BC.ENDC)

    # for debugging: compile the list of elements retagged:
    txt_elements_fpath = os.path.join(base_path, "List_of_elements_retagged_by_tagged_tiff.txt")
    with open(txt_elements_fpath, "w") as flist:
        print("Execution date: \n\n", date.today(), file=flist)
        for l in stringlist_elements_retagged:
            print(l, file=flist)

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
                  '-omsh={1}_ALL_rettaged.vtk -ofmt=vtk_bin'.format(base_path, mesh_basename))

    # =============================================== END OF MAIN =========================================
    print(BC.Y + "*** retag_ALL_by_SegmTags.py terminated." + BC.ENDC)


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Load the submesh NX_TISSUE and the tiff file of the segmentation'
                                                    'containing pixels segmented with TAGS values.'
                                                    'Change the tag of mesh elements mapping the tags in the tiff:'
                                                    'if tiff pixel is [1] -> tag=1 (myo)'
                                                    'if tiff pixel is [5] -> tag=5 (NCF)'
                                                    'if tiff pixel is [6] -> tag=6 (CF)'
                                                    'if tiff pixel is [7] -> tag=7 (Vessels)')
    my_parser.add_argument('-msh',
                           '--mesh-basepath',
                           help='Complete path to the mesh basename (carp format, no extension',
                           required=True)
    my_parser.add_argument('-tiff',
                           '--tiff-tags',
                           help='Path of .tiff file of with pixels segmented as tags',
                           required=True)
    my_parser.add_argument('-vtk',
                           action='store_true',
                           default=False,
                           dest='vtk',
                           help='Add \'-vtk\' if you want to save the mesh with the new tags as .vtk file for '
                                'paraview visualization.')

    # run main
    main(my_parser)
