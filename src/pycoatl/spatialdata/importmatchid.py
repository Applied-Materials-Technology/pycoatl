import numpy as np
import pyvista as pv
from pycoatl.utils.matchidutils import read_load_file
import pandas as pd
import os
from pycoatl.spatialdata.spatialdata import SpatialData
import platform


def return_mesh_matchid(matchid_dataframe):
    """Takes a pandas dataframe imported from a matchid csv export,
    theoretically any field order, and converts to a pyvista unstructured mesh.
    Needs at minimumum to contain 'x_pic','y_pic' and 'x', 'y' and 'z' coordinates.

    Args:
        matchid_dataframe (pandas dataframe): Dataframe containing matchid csv import. 

    Returns:
        pvyista unstructured grid: Mesh of the matchid coordinate data.
    """
    cells= []
    num_cells = 0
    celltypes = []
    points = []

    xc = matchid_dataframe['x_pic'].to_numpy()
    yc = matchid_dataframe['y_pic'].to_numpy()

        
    x = matchid_dataframe['x'].to_numpy()
    y = matchid_dataframe['y'].to_numpy()
    z = matchid_dataframe['z'].to_numpy()

    for i in range(len(xc)):
        
        points.append([x[i],-y[i],z[i]])
        cand_x = xc[i]
        cand_y = yc[i]

        spacing = np.diff(np.unique(yc))[0]

        connectivity = [i]
        try: 
            connectivity.append(np.where((xc==cand_x+spacing)*(yc == cand_y))[0][0])
        except:
            print('No connectivity at point 1.')
        try:
            connectivity.append(np.where((xc==cand_x+spacing)*(yc == cand_y+spacing))[0][0])
        except:
            print('No connectivity at point 2.')
        try:
            connectivity.append(np.where((xc==cand_x)*(yc == cand_y+spacing))[0][0])
        except:
            print('No connectivity at point 3.')
        
        if len(connectivity) <3:
            continue
        #connectivity.sort()
        connectivity = [len(connectivity)] + connectivity
        num_cells +=1
        celltypes.append(pv.CellType.POLYGON)
        cells = cells + connectivity
      

    grid = pv.UnstructuredGrid(cells,celltypes,points)

    return grid

def add_data_matchid(unstructured_grid,matchid_dataframe,fields):
    """Adds data from matchid_dataframe import to existing pyvista unstructured grid.
    Args:
        unstructured_grid (pvyista unstructured grid): Mesh to add fields to.
        matchid_dataframe (dataframe): Pandas dataframe with data from a particular timestep
        fields (list of str): list of fields in the data that should be added to the mesh.
    """   
    
    for field in fields:
        if field == 'v':
            unstructured_grid.point_data[field] = -matchid_dataframe[field].to_numpy()
        else:
            unstructured_grid.point_data[field] = matchid_dataframe[field].to_numpy()



def matchid_to_spatialdata(folder_path,load_filename,fields=['u','v','w','exx','eyy','exy']):
    """Reads matchid data and converts to SpatialData format
    

    Args:
        folder_path (str): Path to folder containing matchid csv exports.
        load_filename (str): Path to load file of matchid data.
        fields (list, optional): List of fields to import onto the mesh, must exist in the csv data. Defaults to ['u','v','w','exx','eyy','exy'].

    Returns:
        SpatialData: SpatialData instance with appropriate metadata.
    """
    #Something here
    index, load = read_load_file(load_filename)
    load = load[1:]
    index = index[1:]
    # Need some other way to get times, but in the absence of that this will do for now.
    time = index-1
    
    # Create metadata table
    metadata = {'data_source':'matchid', 'data_location':folder_path}

    files = os.listdir(folder_path)

    # Should maybe check indices match, but for now leaving it.
    
    path_sep = '/'
    if platform.system == 'Windows':
        path_sep = '\\'
    initial = pd.read_csv(folder_path + path_sep + files[0])
    initial_mesh = return_mesh_matchid(initial)

    #Assuming that the files are in order.
    data_sets = []
    for file in files:
        filename = folder_path + path_sep + file
        current_data = pd.read_csv(filename)
        #Create empty mesh to overwrite
        current_grid = pv.UnstructuredGrid()
        current_grid.copy_from(initial_mesh)
        add_data_matchid(current_grid,current_data,fields)
        data_sets.append(current_grid)

    mb = SpatialData(data_sets,index,time,load,metadata)

    return mb