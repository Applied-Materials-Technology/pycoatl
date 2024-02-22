import numpy as np
import pyvista as pv
from pycoatl.spatialdata.spatialdata import SpatialData
import pickle


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
        points.append([dataset[0][1][i],dataset[0][2][i],dataset[0][3][i]])
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


    # Local variables

    initial_mesh['u'] = dataset[1][1]
    initial_mesh['v'] = dataset[1][2]
    initial_mesh['w'] = dataset[1][3]
    initial_mesh['exx'] = dataset[1][4]
    initial_mesh['eyy'] = dataset[1][5]
    initial_mesh['exy'] = dataset[1][6]
        
        # Add in symmetry here. Y symmetry may be more tricky as it won't flip disp_y I suspect.

    
    # Create metadata table
    metadata = {'data_source':'abaqus','data_location':datapickle}
    
    mb = SpatialData(data_sets,metadata,index,time,load)

    return mb