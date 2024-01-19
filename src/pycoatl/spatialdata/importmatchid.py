import numpy as np
import pyvista as pv
from pycoatl.utils.matchidutils import read_load_file
from pycoatl.utils.matchidutils import read_matchid_csv
import pandas as pd
import os
from pycoatl.spatialdata.spatialdata import SpatialData
import platform


def return_mesh_matchid(matchid_dataframe,version):
    """Takes a pandas dataframe imported from a matchid csv export,
    theoretically any field order, and converts to a pyvista unstructured mesh.
    Needs at minimumum to contain 'x_pic','y_pic' and 'x', 'y' and 'z' coordinates.

    Args:
        matchid_dataframe (pandas dataframe): Dataframe containing matchid csv import. 
        version (str): Software version, changes the column headers
    Returns:
        pvyista unstructured grid: Mesh of the matchid coordinate data.
    """
    cells= []
    num_cells = 0
    celltypes = []
    points = []
    

    xc = matchid_dataframe[field_lookup('xc',version)].to_numpy()
    yc = matchid_dataframe[field_lookup('yc',version)].to_numpy()
    x = matchid_dataframe[field_lookup('x',version)].to_numpy()
    y = matchid_dataframe[field_lookup('y',version)].to_numpy()
    z = matchid_dataframe[field_lookup('z',version)].to_numpy()


        


    for i in range(len(xc)):
        
        points.append([x[i],-y[i],z[i]])
        cand_x = xc[i]
        cand_y = yc[i]

        spacing = np.diff(np.unique(yc))[0]

        connectivity = [i]
        try: 
            connectivity.append(np.where((xc==cand_x+spacing)*(yc == cand_y))[0][0])
        except:
            pass
            #print('No connectivity at point 1.')
        try:
            connectivity.append(np.where((xc==cand_x+spacing)*(yc == cand_y+spacing))[0][0])
        except:
            pass
            #print('No connectivity at point 2.')
        try:
            connectivity.append(np.where((xc==cand_x)*(yc == cand_y+spacing))[0][0])
        except:
            pass
            ##print('No connectivity at point 3.')
        
        if len(connectivity) <3:
            continue
        #connectivity.sort()
        connectivity = [len(connectivity)] + connectivity
        num_cells +=1
        celltypes.append(pv.CellType.POLYGON)
        cells = cells + connectivity
      

    grid = pv.UnstructuredGrid(cells,celltypes,points)

    return grid

def add_data_matchid(unstructured_grid,matchid_dataframe,fields,version):
    """Adds data from matchid_dataframe import to existing pyvista unstructured grid.
    Args:
        unstructured_grid (pvyista unstructured grid): Mesh to add fields to.
        matchid_dataframe (dataframe): Pandas dataframe with data from a particular timestep
        fields (list of str): list of fields in the data that should be added to the mesh.
    """   
    
    for field in fields:
        if field == 'v':
            unstructured_grid.point_data[field] = -matchid_dataframe[field_lookup(field,version)].to_numpy()
        else:
            unstructured_grid.point_data[field] = matchid_dataframe[field_lookup(field,version)].to_numpy()



def matchid_to_spatialdata(folder_path,load_filename,fields=['u','v','w','exx','eyy','exy'],version='2024.1',loadfile_format='Image.csv'):
    """Reads matchid data and converts to SpatialData format
    

    Args:
        folder_path (str): Path to folder containing matchid csv exports.
        load_filename (str): Path to load file of matchid data.
        fields (list, optional): List of fields to import onto the mesh, must exist in the csv data. Defaults to ['u','v','w','exx','eyy','exy'].
        version (str): Software version (2023 and 2024.1 supported)
        loadfile_format (str): Type of load file (Image.csv and Davis Export supported)
    Returns:
        SpatialData: SpatialData instance with appropriate metadata.
    """
    #Something here
    #index, load = read_load_file(load_filename)
    #load = load[1:]
    #index = index[1:]
    # Need some other way to get times, but in the absence of that this will do for now.
    #time = index-1
    index, time, load = load_lookup(loadfile_format)(load_filename)
    
    # Create metadata table
    metadata = {'data_source':'matchid', 'data_location':folder_path}

    files = os.listdir(folder_path)

    # Should maybe check indices match, but for now leaving it.
    
    path_sep = '/'
    if platform.system == 'Windows':
        path_sep = '\\'
    initial = pd.read_csv(folder_path + path_sep + files[0])
    initial_mesh = return_mesh_matchid(initial,version)

    #Assuming that the files are in order.
    data_sets = []
    for file in files:
        filename = folder_path + path_sep + file
        current_data = pd.read_csv(filename)
        #Create empty mesh to overwrite
        current_grid = pv.UnstructuredGrid()
        current_grid.copy_from(initial_mesh)
        add_data_matchid(current_grid,current_data,fields,version)
        data_sets.append(current_grid)

    mb = SpatialData(data_sets,metadata,index,time,load)

    return mb

def field_lookup(field,version):
    """As column names in the exports keep changing, nested dict to keep track

    Args:
        field (str): Field name (xc, yc, x, y, z, u, v, w, exx, eyy, exy) for now
        version (str): Software version, currently 2023 or 2024.1

    Returns:
        str: Column header to use in import
    """
    v_2024_1 = {'xc' : 'Coordinates.Image X [Pixel]',
                'yc' : 'Coordinates.Image Y [Pixel]',
                'x'  : 'coor.X [mm]',
                'y'  : 'coor.Y [mm]',
                'z'  : 'coor.Z [mm]',
                'u'  : 'disp.Horizontal Displacement U [mm]',
                'v'  : 'disp.Vertical Displacement V [mm]',
                'w'  : 'disp.Out-Of-Plane: W [mm]',
                'exx': 'strain.Strain-global frame: Exx [ ]',
                'eyy': 'strain.Strain-global frame: Eyy [ ]',
                'exy': 'strain.Strain-global frame: Exy [ ]'}
    
    v_2023   = {'xc' : 'x_pic',
                'yc' : 'y_pic',
                'x'  : 'x',
                'y'  : 'y',
                'z'  : 'z',
                'u'  : 'u',
                'v'  : 'v',
                'w'  : 'w',
                'exx': 'exx',
                'eyy': 'eyy',
                'exy': 'exy'}
    
    lookup_dict = {'2024.1':v_2024_1,'2023':v_2023}

    ## Possibly add error checking at some point

    return lookup_dict[version][field]


def load_lookup(fileformat):

    lookup_dict = {'Image.csv':read_matchid_csv,
                   'Davis'    :read_load_file}
    return lookup_dict[fileformat]

