#
#
# Will have to run Abaqus python to get a pickle of the data in some reasonable format.
# Problem with reading the abaqus generated .inp file.
#%%
import numpy as np
import pyvista as pv
import meshio
import pickle
from matplotlib import pyplot as plt
from pycoatl.spatialdata.spatialdata import SpatialData
#%%
file = r'D:\Rory\DONES_FEMU\Test_Input.inp'
test = pv.read_meshio(file)
# %%
with open(r'D:\Rory\DONES_FEMU\MeshExport.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    data = pickle.load(f,encoding='latin-1')
# %%
nl = data[0]
nx = data[1]
ny = data[2]
nz = data[3]
cells = data[4]
points = []
for i in range(len(nl)):
    points.append([data[1][i],data[2][i],data[3][i]])



#new_cells=[]
celltypes = []
for j,con in enumerate(cells):
#    xpos = np.array([nx[i] for i in con[1:]])
#    ypos = np.array([ny[i] for i in con[1:]])
#    xpos = xpos - np.mean(xpos)
#    ypos = ypos - np.mean(ypos)
#    order = np.argsort(np.arctan2(ypos,xpos))
#    new_cells.append([con[0]] + np.array(con[1:])[order].tolist())
    celltypes.append(pv.CellType.POLYGON)

grid = pv.UnstructuredGrid(cells,celltypes,points)
# %%

test_con = cells[0]

# %%
xpos = np.array([nx[i] for i in test_con[1:]])
ypos = np.array([ny[i] for i in test_con[1:]])
xpos = xpos - np.mean(xpos)
ypos = ypos - np.mean(ypos)
order = np.argsort(np.arctan2(ypos,xpos))
test_con_out = [test_con[0]] + [np.array(test_con[1:])[order]]
# %%
with open(r'D:\Rory\DONES_FEMU\DataExport.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    dataset = pickle.load(f,encoding='latin-1')
# %%
dataset[0]
# %%
dataset[1]
#%%
grid['test'] = dataset[1][-1]
grid['eyy'] = dataset[5][-1]
# %%


def abaqus_to_spatialdata(datapickle):
    """Abaqus data to spatial data format.
    Must be exported using abaqus python 2 script.
    datapickle is a nested list containing the necessary data produced by the script.
    datapickle
        meshdata
            nl, nx, ny, connectivity
        outputdata
            force, u, v, w, exx, eyy, exy
    Args:
        datapickle (str): Path to pickle file
    """

    with open(datapickle, 'rb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        dataset = pickle.load(f,encoding='latin-1')

    nl = dataset[0][0]
    cells = dataset[0][4]
    points = []
    # Build points array
    for i in range(len(nl)):
        points.append([data[1][i],data[2][i],data[3][i]])
    # Get cell types
    celltypes = []
    for con in cells:
        celltypes.append(pv.CellType.POLYGON)
    # Create mesh
    initial_mesh = pv.UnstructuredGrid(cells,celltypes,points)

    load = -dataset[1][0]
    index = np.arange(len(load))
    time = index # For now.
    data_sets = []
    for i in range(len(load)):

        # Local variables
        current_grid = pv.UnstructuredGrid()
        current_grid.copy_from(initial_mesh)
        current_grid['u'] = dataset[1][1][i]
        current_grid['v'] = dataset[1][2][i]
        current_grid['w'] = dataset[1][3][i]
        current_grid['exx'] = dataset[1][4][i]
        current_grid['eyy'] = dataset[1][5][i]
        current_grid['exy'] = dataset[1][6][i]
        
        # Add in symmetry here. Y symmetry may be more tricky as it won't flip disp_y I suspect.
       
        data_sets.append(current_grid)
    
    # Create metadata table
    metadata = {'data_source':'abaqus','data_location':datapickle}
    
    mb = SpatialData(data_sets,index,time,load,metadata)

    return mb

# %%
test = abaqus_to_spatialdata(r'D:\Rory\DONES_FEMU\DataExport.pickle')
# %%
