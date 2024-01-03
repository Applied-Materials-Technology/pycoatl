import numpy as np
import pyvista as pv
import os
import sys
from pycoatl.spatialdata.spatialdata import SpatialData
from mooseherder.exodusreader import ExodusReader


def return_mesh_moose(exodus_file):
    """Returns a pyvista unstructured grid from an exodus file.
    Mesh should have a side side called 'Visible-Surface' that is 
    the surface where we want to analyse.

    Args:
        exodus_file (str): Path to exodus.e file.
    
    Returns:
        pvyista unstructured grid: Mesh of the exodus coordinate data.
    """

    return pv.read_exodus(exodus_file)['Side Sets']['Visible-Surface']

def moose_to_spatialdata(exodus_file,fields=['disp_x','disp_y','disp_z','stress_yy','creep_strain_yy','plastic_strain_yy','mechanical_strain_xx','mechanical_strain_yy']):
    """Reads a moose exported exodus file and converts to SpatialData format.
    This version doesn't require SEACAS installation, uses the netCDF reader in mooseherder

    Args:
        exodus_file (str): Path to exodus .e file. 
        fields (list, optional): List of fields to import onto the mesh, must exist in the csv data. Defaults to ['disp_x','disp_y','disp_z','stress_yy','creep_strain_yy','plastic_strain_yy','mechanical_strain_xx','mechanical_strain_yy'].

    Returns:
        SpatialData: SpatialData instance with appropriate metadata.    """
    
    # Read exodus file
    f = ExodusReader(exodus_file)
    #Check symmetry. X-symm is symmetric about the X=0 plane, and so forth
    x_symm = 'X-Symm' in f.side_set_names
    y_symm = 'Y-Symm' in f.side_set_names

    time = f.get_time()
    n_steps = len(time)
    index = np.empty(n_steps)
    load = np.empty(n_steps)

    initial_mesh = return_mesh_moose(exodus_file)
    initial_mesh.clear_point_data()

    side_node_list=f.side_set_nodes['Visible-Surface']
    surface_nodes = side_node_list-1 # Node the -1 (changes from 1 to 0 based indexing)

    

    #Iterate through file.
    data_sets = []
    for i in range(n_steps):
        # Global variables
        index[i] = i
        load[i] = -f.global_data['react_y'][i]
        # Local variables
        current_grid = pv.UnstructuredGrid()
        current_grid.copy_from(initial_mesh)
        
        for field in fields:
            current_grid[field] = f.node_data[field][i][surface_nodes]
        
        # Add in symmetry here. Y symmetry may be more tricky as it won't flip disp_y I suspect.
        if x_symm:
            reflected_grid = current_grid.reflect((1,0,0),point=(0,0,0))
            reflected_grid['disp_x'] = -1*reflected_grid['disp_x']                    
            current_grid += reflected_grid
        if y_symm:
            reflected_grid = current_grid.reflect((0,1,0),point=(0,0,0))
            reflected_grid['disp_y'] = -1*reflected_grid['disp_y']                    
            current_grid += reflected_grid
        
        data_sets.append(current_grid)
    
    # Create metadata table
    metadata = {'data_source':'moose','data_location':exodus_file}
    
    mb = SpatialData(data_sets,index,time,load,metadata)

    return mb