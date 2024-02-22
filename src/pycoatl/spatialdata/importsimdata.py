import numpy as np
import pyvista as pv
from pycoatl.spatialdata.spatialdata import SpatialData

def return_mesh_simdata(simdata,dim3):
    """Return a mesh constructed from the simdata object.

    Args:
        simdata (SimData): SimData from mooseherder exodus reader

    Returns:
        pvyista unstructured grid: Mesh of the simdata coordinate data.
    """
    # Get connectivity
    connect = simdata.connect['connect1']

    if dim3: # If 3D
        surface_nodes = simdata.side_sets[('Visible-Surface','node')]
        
        #Construct mapping from all-nodes to surface node indices
        con = np.arange(1,simdata.coords.shape[0]+1)
        mapping_inv = []
        for i in con:
            if i in surface_nodes:
                mapping_inv.append(np.where(surface_nodes==i)[0][0])
            else:
                mapping_inv.append(0)
        mapping_inv = np.array(mapping_inv)

        cells=[]
        for i in range(connect.shape[1]):
            con = connect.T[i].tolist()
            vis_con = [x for x in con if x in surface_nodes]
            if vis_con:
                cells.append([len(vis_con)]+mapping_inv[np.array(vis_con)-1].tolist())
        num_cells = len(cells)
        cells = np.array(cells).ravel()
        points = simdata.coords[surface_nodes-1]

    else:
        # Rearrange to pyvista format
        cells = np.concatenate((4*np.ones((connect.shape[1],1)),connect.T-1),axis=1).ravel().astype(int)
        num_cells = connect.shape[1]
        #Coordinates
        points = simdata.coords
        #Cell types (all polygon)
    celltypes = np.full(num_cells,pv.CellType.POLYGON,dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells,celltypes,points)
    return grid

def simdata_to_spatialdata(simdata):
    """Reads simdata from mooseherder exodus reader
     and converts to SpatialData format
    
    Args:
        simdata (SimData) : SimData produced by mooseherder exodus reader
    Returns:
        SpatialData: SpatialData instance with appropriate metadata.
    """

    # Create metadata table
    metadata = {'data_source':'SimData Exodus'}

    #Check for symmety 
    x_symm = ('X-Symm','node') in simdata.side_sets
    y_symm = ('Y-Symm','node') in simdata.side_sets

    #Check if 3D
    if 'disp_z' in simdata.node_vars:
        dim3 = True
        side_node_list=simdata.side_sets[('Visible-Surface','node')]
        surface_nodes = side_node_list-1 # Node the -1 (changes from 1 to 0 based indexing)

    else:
        dim3 = False
        # Think in 2D it treats surfaces as element blocks.
        surface_nodes = np.unique(simdata.connect['connect1'])-1

    initial_mesh = return_mesh_simdata(simdata,dim3)
    time = simdata.time
    load = -simdata.glob_vars['react_y']
    index = np.arange(len(time))

    #Iterate over times.
    data_sets = []
    for i in range(len(time)):
        #Create empty mesh to overwrite
        current_grid = pv.UnstructuredGrid()
        current_grid.copy_from(initial_mesh)
        # add only nodal variables for now.
        for data in simdata.node_vars:
            current_grid[data] = simdata.node_vars[data][surface_nodes,i]

        if x_symm:
            reflected_grid = current_grid.reflect((1,0,0),point=(0,0,0))
            reflected_grid['disp_x'] = -1*reflected_grid['disp_x']                    
            current_grid += reflected_grid
        if y_symm:
            reflected_grid = current_grid.reflect((0,1,0),point=(0,0,0))
            reflected_grid['disp_y'] = -1*reflected_grid['disp_y']                    
            current_grid += reflected_grid
        data_sets.append(current_grid)

    mb = SpatialData(data_sets,metadata,index,time,load)

    return mb