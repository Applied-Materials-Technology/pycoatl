import numpy as np
import pyvista as pv
from pycoatl.spatialdata.spatialdata import SpatialData
from pycoatl.spatialdata.tensorfield import vector_field
from pycoatl.spatialdata.tensorfield import rank_two_field


def return_mesh_simdata(simdata ,dim3: bool) -> pv.UnstructuredGrid:
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
        surface_nodes = np.unique(simdata.connect['connect1'])
        con = np.arange(1,simdata.coords.shape[0]+1)
        mapping_inv = []
        for i in con:
            if i in surface_nodes:
                mapping_inv.append(np.where(surface_nodes==i)[0][0])
            else:
                mapping_inv.append(0)
        mapping_inv = np.array(mapping_inv)
        #cells = np.concatenate((4*np.ones((connect.shape[1],1)),connect.T-1),axis=1).ravel().astype(int)
        #num_cells = connect.shape[1]
        cells=[]
        for i in range(connect.shape[1]):
            con = connect.T[i].tolist()
            vis_con = [x for x in con if x in surface_nodes]
            if vis_con:
                cells.append([len(vis_con)]+mapping_inv[np.array(vis_con)-1].tolist())
        num_cells = len(cells)
        cells = np.array(cells).ravel()
        
        #Coordinates
        points = simdata.coords[surface_nodes-1]
        #Cell types (all polygon)
    celltypes = np.full(num_cells,pv.CellType.POLYGON,dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells,celltypes,points)
    return grid

def simdata_to_spatialdata(simdata)->SpatialData:
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
    
    data_dict = {}
    for field in simdata.node_vars:
        data_dict[field] = simdata.node_vars[field][surface_nodes]

    if x_symm:
        #Reflect mesh
        initial_mesh_ref_x = initial_mesh.reflect((1,0,0),point=(0,0,0))
        # Find overlapping points
        overlap= initial_mesh.points[:,0]==initial_mesh_ref_x.points[:,0]

        for field in data_dict:
            if field == 'disp_x':
                data_dict[field] = np.concatenate((data_dict[field],-1*data_dict[field][~overlap]))
            else:
                data_dict[field] = np.concatenate((data_dict[field],data_dict[field][~overlap]))
        initial_mesh+=initial_mesh_ref_x

    if y_symm:
        #Reflect mesh
        initial_mesh_ref_y = initial_mesh.reflect((0,1,0),point=(0,0,0))
        # Find overlapping points
        overlap= initial_mesh.points[:,1]==initial_mesh_ref_y.points[:,1]

        for field in data_dict:
            if field == 'disp_y':
                data_dict[field] = np.concatenate((data_dict[field],-1*data_dict[field][~overlap]))
            else:
                data_dict[field] = np.concatenate((data_dict[field],data_dict[field][~overlap]))
        initial_mesh+=initial_mesh_ref_y

    dummy = np.zeros_like(data_dict['disp_y'])
    if dim3:
        displacement = np.stack((data_dict['disp_x'],data_dict['disp_y'],data_dict['disp_z']),axis=1)
    else:
        displacement = np.stack((data_dict['disp_x'],data_dict['disp_y'],dummy),axis=1)
    
    #Assuming symmetric strain tensor
    tensor_components = ['xx','xy','xz','xy','yy','yz','xz','yz','zz']
    #Elastic strain
    elastic_stack = []
    for comp in tensor_components:
        if any('elastic_strain_'+comp in s for s in simdata.node_vars.keys()):
            elastic_stack.append(data_dict['elastic_strain_'+comp])
        else:
            elastic_stack.append(dummy)
    elastic_strains = np.stack(elastic_stack,axis=1)
    #Plastic strain
    plastic_stack = []
    for comp in tensor_components:
        if any('plastic_strain_'+comp in s for s in simdata.node_vars.keys()):
            plastic_stack.append(data_dict['plastic_strain_'+comp])
        else:
            plastic_stack.append(dummy)
    plastic_strains = np.stack(plastic_stack,axis=1)
    #Stress
    stress_stack = []
    for comp in tensor_components:
        if any('stress_'+comp in s for s in simdata.node_vars.keys()):
            stress_stack.append(data_dict['stress_'+comp])
        else:
            stress_stack.append(dummy)
    stresses = np.stack(stress_stack,axis=1)


    data_fields = {'displacement'  :vector_field(displacement),
                   'elastic_strain':rank_two_field(elastic_strains),
                   'plastic_strain':rank_two_field(plastic_strains),
                   'stress':rank_two_field(stresses)
                   }
    mb = SpatialData(initial_mesh,data_fields,metadata,index,time,load)

    return mb