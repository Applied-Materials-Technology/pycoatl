from abaqus import *
from abaqusConstants import *
from odbAccess import *
from abaqusConstants import *
import visualization
import sys
sys.path.append(r'D:\Rory\Spyder\FEMU')
import pyffdata_ab as ff
import pickle
import numpy as np
import time

# Open ODB and get surface set.
odb = openOdb('D:\Rory\DONES_FEMU\LC-E97_Verification_Set_Run_1.odb')
instance = odb.rootAssembly.instances['DONES-1']
force_set = odb.rootAssembly.nodeSets['BTM_BC']
surface_set = odb.rootAssembly.instances['DONES-1'].nodeSets['SURF_NODES']

# Get nodal positions and labels.

nx = []
ny = []
nz = []
nl = [] 
for node in odb.steps['Step-1'].frames[0].fieldOutputs['COORD'].getSubset(region=surface_set).values:
    nx.append(node.data[0])
    ny.append(node.data[1])
    nz.append(node.data[2])
    nl.append(int(node.nodeLabel))  

# Get connectivity array if node is in surface set

con_list = []

for element in instance.elements:
    if any(x in nl for x in element.connectivity):
       # Get connectivity
       cur_con = [x for x in element.connectivity if x in nl]
       # Get indices rather than node nums
       st = set(cur_con)
       cur_con_cor = [i for i, e in enumerate(nl) if e in st]
       # Rearrange nodes to ensure they cycle the element
       xpos = np.array([nx[i] for i in cur_con_cor])
       ypos = np.array([ny[i] for i in cur_con_cor])
       xpos = xpos - np.mean(xpos)
       ypos = ypos - np.mean(ypos)
       order = np.argsort(np.arctan2(ypos,xpos))
       # Generate array
       con_list.append([len(cur_con)] + np.array(cur_con_cor)[order].tolist())
       

output_mesh_data = [nl,nx,ny,nz,con_list]

#pickle_path = 'D:\Rory\DONES_FEMU\MeshExport.pickle'
#with open(pickle_path, 'wb') as f:
#    # Pickle the 'data' dictionary using the highest protocol available.
#    pickle.dump(output_mesh_data, f, pickle.HIGHEST_PROTOCOL)
    
    
numframes = len(odb.steps['Step-1'].frames)
numnodes = len(odb.steps['Step-1'].frames[0].fieldOutputs['U'].getSubset(region=surface_set).values)

force = []

u_all = [0] * numframes
v_all = [0] * numframes
w_all = [0] * numframes

exx_all = [0] * numframes
eyy_all = [0] * numframes
exy_all = [0] * numframes

for count,frame in enumerate(odb.steps['Step-1'].frames):
    
    
    temp_f = []
    for nodeVal in frame.fieldOutputs['RF'].getSubset(region=force_set).values:
        temp_f.append(nodeVal.data[1]) #RF2
    force.append(sum(temp_f))
    
    t_disp_s = time.time()
    u = [0] * numnodes
    v = [0] * numnodes
    w = [0] * numnodes
    """
    for i,node in enumerate(frame.fieldOutputs['U'].getSubset(region=surface_set).values):
        u[i] =node.data[0]
        v[i] =node.data[1]
        w[i] =node.data[2]
       
    t_disp_e = time.time()
    print 'Disp'
    print t_disp_e - t_disp_s   
    """
    t_dispa_s = time.time()    
    nv = []
    bd = [] 
    for dataBlock in frame.fieldOutputs['U'].getSubset(region=surface_set,position=NODAL).bulkDataBlocks:
        nv.append(dataBlock.nodeLabels)
        bd.append(dataBlock.data)
    
    nvf = [item for sublist in nv for item in sublist]
    bdf = [item for sublist in bd for item in sublist]
    #print bd
    #print nvf
    for i,data in enumerate(bdf):          
        u[i] = data[0]
        v[i] = data[1]
        w[i] = data[2]

    exx = [0] * numnodes
    eyy = [0] * numnodes
    exy = [0] * numnodes
    
    nv = []
    bd = []
    for dataBlock in frame.fieldOutputs['LE'].getSubset(region=surface_set,position=ELEMENT_NODAL).bulkDataBlocks:
        nv.append(dataBlock.nodeLabels)
        bd.append(dataBlock.data)

    nvf = [item for sublist in nv for item in sublist]
    bdf = [item for sublist in bd for item in sublist]

    NodeLabels = np.array(nvf)
    Values = np.array(bdf)

    NodeLabels_unique, unq_idx = np.unique(NodeLabels, return_inverse=True)

    Values_Averaged=np.zeros((NodeLabels_unique.size,Values.shape[1]))
    unq_counts = np.bincount(unq_idx)

    for i in [0,1,3]:
        ValuesTemp = [item[i] for item in Values]
        
        unq_sum = np.bincount(unq_idx, weights=ValuesTemp)
        if i == 0:
            exx[:] =unq_sum / unq_counts
        if i == 1:
            eyy[:] =unq_sum / unq_counts
        if i == 3:
            exy[:] =unq_sum / unq_counts

    
    u_all[count] = u
    v_all[count] = v
    w_all[count] = w
    
    exx_all[count] = exx
    eyy_all[count] = eyy
    exy_all[count] = exy
    
# Convert to numpy arrays
# Note that strains are sorted into ascending nodes, whereas nodal posn and displacements are not currently.


nx = np.array(nx)
ny = np.array(ny)
nz = np.array(nz)
nl = np.array(nl)

u_all = np.array(u_all)
v_all = np.array(v_all)
w_all = np.array(w_all)

exx_all = np.array(exx_all)
eyy_all = np.array(eyy_all).squeeze()
exy_all = np.array(exy_all).squeeze()

force = np.array(force)

output_list = [force,u_all,v_all,w_all,exx_all,eyy_all,exy_all]

pickle_path = 'D:\Rory\DONES_FEMU\DataExport.pickle'
with open(pickle_path, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump([output_mesh_data, output_list], f, pickle.HIGHEST_PROTOCOL)

