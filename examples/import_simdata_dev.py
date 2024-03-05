#
#
#

#%% import packages
from mooseherder import ExodusReader
from pathlib import Path
import numpy as np
import pyvista as pv
from pycoatl.spatialdata.spatialdata import SpatialData
import matplotlib.pyplot as plt


#%% read something
#output_file = Path('/home/rspencer/pycoatl/data/moose-sim-1_out.e')
#output_file = Path('/home/rspencer/moose_work/Viscoplastic_Creep/HVPF_Sat/Run/moose-workdir-1/moose-sim-152_out.e')
output_file = Path(r'C:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Anaconda_Python\Test\pycoatl\data\moose-sim-1_out.e')
exodus_reader = ExodusReader(output_file)

all_sim_data = exodus_reader.read_all_sim_data()
#%%
folder = 1
it = 8
best_file = '/home/rspencer/moose_work/Viscoplastic_Creep/XY_Specimen/Run/sim-workdir-{}/sim-{}_out.e'.format(folder,it)
#best_file = '/home/rspencer/moose_work/Viscoplastic_Creep/XY_Specimen/xy_creep_perz_dbl_out.e'
#best_file = '/home/rspencer/moose_work/Viscoplastic_Creep/XY_Specimen/xy_creep_perz_dbl_elastic_out.e'

exodus_reader = ExodusReader(Path(best_file))
all_sim_data = exodus_reader.read_all_sim_data()
#cur_best= simdata_to_spatialdata(all_sim_data)
# %%
connect = all_sim_data.connect['connect1']
# %%
print(np.concatenate((4*np.ones((connect.shape[1],1)),connect.T),axis=1))
# %%
cells = np.concatenate((4*np.ones((connect.shape[1],1)),connect.T),axis=1).ravel().astype(int)
points = all_sim_data.coords
celltypes = np.full(connect.shape[1],pv.CellType.POLYGON,dtype=np.uint8)

# %%
grid = pv.UnstructuredGrid(cells,celltypes,points)
# %%

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

#%%
test = return_mesh_simdata(all_sim_data,True)
# %%
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
# %%
test = simdata_to_spatialdata(all_sim_data)
# %%
simdata = all_sim_data
connect = simdata.connect['connect1']
surface_nodes = simdata.side_sets[('Visible-Surface','node')]
cells=[]
for i in range(connect.shape[1]):
    con = connect.T[i].tolist()
    vis_con = [x for x in con if x in surface_nodes]
    print(simdata.coords[np.array(vis_con)-1])
    cells.append([len(vis_con)]+(np.array(vis_con)-1).tolist())


# %%

connect = all_sim_data.connect['connect1']
surface_nodes = simdata.side_sets[('Visible-Surface','node')]
cells=[]
for i in range(1):
    con = connect.T[i].tolist()
    vis_con = [x for x in con if x in surface_nodes]
    print(simdata.coords[np.array(vis_con)-1])
    cells.append([len(vis_con)]+(np.array(vis_con)-1).tolist())
celltypes = np.full(1,pv.CellType.POLYGON,dtype=np.uint8)
points = simdata.coords[surface_nodes-1][:4]

# %%
grid = pv.UnstructuredGrid(cells,celltypes,points)
# %%
output_file = Path('/home/rspencer/moose_work/Viscoplastic_Creep/HVPF_Sat/Run/moose-workdir-1/moose-sim-152_out.e')
exodus_reader = ExodusReader(output_file)
simdata = exodus_reader.read_all_sim_data()
connect = simdata.connect['connect1']
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
    print(vis_con)
    cells.append([len(vis_con)]+mapping_inv[np.array(vis_con)-1].tolist())
#cells = [4,0,1,2,3]
celltypes = np.full(connect.shape[1],pv.CellType.POLYGON,dtype=np.uint8)
points = simdata.coords[surface_nodes-1]
# %%
grid = pv.UnstructuredGrid(cells,celltypes,points)
#grid.plot()
# %%
pl = pv.Plotter()
pl.add_mesh(grid)
pl.show()
# %%
surface_nodes = simdata.side_sets[('Visible-Surface','node')]-1
# %%
con = np.arange(1,simdata.coords.shape[0]+1)
mapping_inv = []
for i in con:
    if i in surface_nodes:
        mapping_inv.append(np.where(surface_nodes==i)[0][0])
    else:
        mapping_inv.append(0)
mapping_inv = np.array(mapping_inv)
print(mapping)
print(mapping_inv)
print(simdata.coords[mapping])
# %%
# want such that go cells[mapping] = cells in new coords.