import pandas as pd
import glob
import math
import random
import numpy as np


#########################################
# Function to read in mesh from basename
#########################################
def read_mesh(basename=None, file_pts=None, file_elem=None, file_lon=None):
    # Function to read in mesh from basename

    # Defines pts, elem and lon files from basename
    if file_pts is None:
        file_pts = glob.glob(basename + '*.pts')
        if len(file_pts) > 1:
            raise ValueError('Too many matching .pts files')
        elif len(file_pts) == 0:
            raise ValueError('No matching .pts files')
        file_pts = file_pts[0]
    if file_elem is None:
        file_elem = glob.glob(basename + '*.elem')
        if len(file_elem) > 1:
            raise ValueError('Too many matching .elem files')
        elif len(file_elem) == 0:
            raise ValueError('No matching .elem files')
        file_elem = file_elem[0]
    if file_lon is None:
        file_lon = glob.glob(basename + '*.lon')
        if len(file_lon) > 1:
            raise ValueError('Too many matching .lon files')
        elif len(file_lon) == 0:
            raise ValueError('No matching .lon files')
        file_lon = file_lon[0]

    # Read mesh files
    try:
        pts = pd.read_csv(file_pts, sep=' ', skiprows=1, header=None)
        print("Successfully read {}".format(file_pts))
    except ValueError:
        pts = None
    elem = pd.read_csv(file_elem, sep=' ', skiprows=1, usecols=(1, 2, 3, 4, 5), header=None)
    print("Successfully read {}".format(file_elem))
    lon = pd.read_csv(file_lon, sep=' ', skiprows=1, header=None)
    print("Successfully read {}".format(file_lon))

    return pts, elem, lon


#########################################
# Function to write mesh
#########################################
def write_mesh(basename, pts=None, elem=None, lon=None, shapes=None, precision_pts=None, precision_lon=None):
    # Write pts, elem and lon data to file

    # Ensure *something* is being written!
    assert ((pts is not None) and (elem is not None) and (lon is not None)), "No data given to write to file."

    # Adapt precision to default formats
    if precision_pts is None:
        precision_pts = '%.12g'
    if precision_lon is None:
        precision_lon = '%.5g'

    # Basic error checking on output file name
    if basename[-1] == '.':
        basename = basename[:-1]

    #######################
    # Writes-out pts file
    #######################
    if pts is not None:
        with open(basename + '.pts', 'w') as pFile:
            pFile.write('{}\n'.format(len(pts)))
        pts.to_csv(basename + '.pts', sep=' ', header=False, index=False, mode='a', float_format=precision_pts)
        print("pts data written to file {}.".format(basename + '.pts'))

    ######################
    # Writes-out elems file
    ######################
    # If we haven't defined a shape for our elements, set to be tets
    if shapes is None:
        shapes = 'Tt'

    if elem is not None:
        with open(basename + '.elem', 'w') as pFile:
            pFile.write('{}\n'.format(len(elem)))
        elem.insert(loc=0, value=shapes, column=0)
        elem.to_csv(basename + '.elem', sep=' ', header=False, index=False, mode='a')
        print("elem data written to file {}.".format(basename + '.elem'))
        del elem[0]  # Remove added column to prevent cross-talk problems later

    ######################
    # Writes-out lon file
    ######################
    if lon is not None:
        with open(basename + '.lon', 'w') as pFile:
            pFile.write('1\n')
        lon.to_csv(basename + '.lon', sep=' ', header=False, index=False, mode='a', float_format=precision_lon)
        print("lon data written to file {}.".format(basename + '.lon'))

    return None


#########################################
# Function to read UVC data and interpolate onto elements
#########################################

#########################################
# Function to read pts file
#########################################
def read_pts(basename=None, file_pts=None):
    # Function to read in mesh from basename

    if file_pts is None:
        file_pts = glob.glob(basename + '*.pts')
        #if len(file_pts) > 1:
         #   raise ValueError('Too many matching .pts files')
        if len(file_pts) == 0:
            raise ValueError('No matching .pts files')
        file_pts = file_pts[0]

    # Read mesh files
    pts = pd.read_csv(file_pts, sep=' ', skiprows=1, header=None)
    print("Successfully read {}".format(file_pts))

    return pts

#########################################
# Function to read cpts file
#########################################
def read_cpts(basename=None, file_cpts=None):
    # Function to read in mesh from basename

    if file_cpts is None:
        file_cpts = glob.glob(basename + '*.cpts')
        #if len(file_pts) > 1:
         #   raise ValueError('Too many matching .pts files')
        if len(file_cpts) == 0:
            raise ValueError('No matching .cpts files')
        file_cpts = file_cpts[0]

    # Read mesh files
    cpts = pd.read_csv(file_cpts, sep=' ', skiprows=1, header=None)
    print("Successfully read {}".format(file_cpts))

    return cpts



#########################################
# Function to read elems file
#########################################
def read_elems(basename=None, file_elem=None):
    # Function to read in mesh from basename

    if file_elem is None:
        file_elem = glob.glob(basename + '*.elem')
        if len(file_elem) > 1:
            raise ValueError('Too many matching .elem files')
        elif len(file_elem) == 0:
            raise ValueError('No matching .elem files')
        file_elem = file_elem[0]

    # Read mesh files
    elem = pd.read_csv(file_elem, sep=' ', skiprows=1, usecols=(1, 2, 3, 4, 5), header=None)
    print("Successfully read {}".format(file_elem))

    return elem


#########################################
# Function to read lon file
#########################################
def read_lon(basename=None, file_lon=None):
    # Function to read in mesh from basename

    if file_lon is None:
        file_lon = glob.glob(basename + '*.lon')
        if len(file_lon) > 1:
            print('*** - Too many matching .lon files. Loading the default one: {} *** '.format(basename + '.lon'))
            file_lon = [basename + '.lon']
        elif len(file_lon) == 0:
            raise ValueError('No matching .lon files')
        file_lon = file_lon[0]

    # Read mesh files
    lon = pd.read_csv(file_lon, sep=' ', skiprows=1,  header=None)
    print("Successfully read {}".format(file_lon))

    return lon


#########################################
# Function to write element file
#########################################
def write_elems(elemFilename=None, elem=None, shapes=None):
    # Write elem

    # Ensure *something* is being written!
    assert ((elem is not None)), "No data given to write to file."

    ######################
    # Writes-out elems file
    ######################
    # If we haven't defined a shape for our elements, set to be tets
    if shapes is None:
        shapes = 'Tt'

    if elem is not None:
        with open(elemFilename + '.elem', 'w') as pFile:
            pFile.write('{}\n'.format(len(elem)))
        elem.insert(loc=0, value='Tt', column=0)
        elem.to_csv(elemFilename + '.elem', sep=' ', header=False, index=False, mode='a')
        print("elem data written to file {}.".format(elemFilename + '.elem'))
        del elem[0]  # Remove added column to prevent cross-talk problems later

    return None


#########################################
# Function to write lon file
#########################################
def write_lon(lonFilename=None, lon=None):
    # Ensure *something* is being written!
    assert ((lon is not None)), "No data given to write to file."

    ######################
    # Writes-out lon file
    ######################
    if lon is not None:
        with open(lonFilename + '.lon', 'w') as pFile:
            pFile.write('1\n')
        lon.to_csv(lonFilename + '.lon', sep=' ', header=False, index=False, mode='a')
        print("lon data written to file {}.".format(lonFilename + '.lon'))

    return None


#########################################
# Function to write pts file
#########################################
def write_pts(ptsFilename=None, pts=None):
    # Ensure *something* is being written!
    assert ((pts is not None)), "No data given to write to file."

    precision_pts = '%.12g'

    ######################
    # Writes-out lon file
    ######################
    if pts is not None:
        with open(ptsFilename + '.pts', 'w') as pFile:
            pFile.write('{}\n'.format(len(pts)))
        pts.to_csv(ptsFilename + '.pts', sep=' ', header=False, index=False, mode='a', float_format=precision_pts)
        print("pts data written to file {}.".format(ptsFilename + '.pts'))

    return None
