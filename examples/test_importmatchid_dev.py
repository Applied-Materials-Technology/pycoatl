#%%
import numpy as np
import pyvista as pv
from pycoatl.utils.matchidutils import read_load_file
from pycoatl.utils.matchidutils import read_matchid_csv
import pandas as pd
import os
from pycoatl.spatialdata.spatialdata import SpatialData
import platform
from pycoatl.spatialdata.importmatchid import field_lookup
from pycoatl.spatialdata.tensorfield import vector_field
from pycoatl.spatialdata.tensorfield import rank_two_field
from pathlib import Path
from pycoatl.spatialdata.importmatchid import return_points_matchid
from pycoatl.spatialdata.importmatchid import matchid_to_spatialdata
import matplotlib.pyplot as plt

# %%
folder_path = Path(r'C:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Anaconda_Python\Test\pycoatl\data\matchid_dat_2024')
files = os.listdir(folder_path)
files.sort()
version='2024.1'
# Should maybe check indices match, but for now leaving it.
fields=['u','v','w','exx','eyy','exy']
#Assuming that the files are in order.
data_dict = {}
for field in fields:
    data_dict[field]= []

for file in files:
    filename = folder_path / file
    current_data = pd.read_csv(filename)

    for field in fields:
            if field == 'v':
                data_dict[field].append(-current_data[field_lookup(field,version)].to_numpy())
            else:
                data_dict[field].append(current_data[field_lookup(field,version)].to_numpy())

for field in fields:
    data_dict[field] = np.array(data_dict[field]).T

#Assemble fields
disp = np.stack((data_dict['u'],data_dict['v'],data_dict['w']),axis=1)
dummy = np.zeros_like(data_dict['exx'])
strain = np.stack((data_dict['exx'],data_dict['exy'],dummy,data_dict['exy'],data_dict['eyy'],dummy,dummy,dummy,dummy),axis=1)

displacements = vector_field(disp)
ea_strains = rank_two_field(strain)

field_dict = {'displacements':displacements,'strains':ea_strains}
#%%
input_data = np.dstack((all_sim_data.node_vars['disp_x'],all_sim_data.node_vars['disp_y'],np.zeros_like(all_sim_data.node_vars['disp_y'])))
print(input_data.shape)
input_data = np.swapaxes(input_data,1,2)

#%%
initial = pd.read_csv(folder_path / files[0])
t = return_points_matchid(initial,version)
# %%
data_folder = Path(r'c:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Projects\FY25\Eurofusion_Duct\Spec_5_RT_0-7\Exp')
load_file = Path(r'c:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Projects\FY25\Eurofusion_Duct\Spec_5_RT_0-7\Image.csv')

t = matchid_to_spatialdata(data_folder,load_file)

# %%
reader = pv.get_reader(r'c:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Projects\FY25\Eurofusion_Duct\notched\Notch_0-7.stl')
mesh_ref_0_7 = reader.read()

# %%
mesh_align,trans_mat = t.mesh_data.align(mesh_ref_0_7,return_matrix=True)
# %%
pl = pv.Plotter()
pl.add_mesh(mesh_ref_0_7)
pl.add_mesh(mesh_align)
pl.view_xz()
pl.show_axes()
pl.show()
# %%
t_start = 40
#start by reseting
x = t.mesh_data.points[:,0] + t.data_fields['displacement'].data[:,0,0]
y = t.mesh_data.points[:,1] + t.data_fields['displacement'].data[:,1,0]
z = t.mesh_data.points[:,2] + t.data_fields['displacement'].data[:,2,0]

x = t.mesh_data.points[:,0] + t.data_fields['displacement'].data[:,0,t_start]
y = t.mesh_data.points[:,1] + t.data_fields['displacement'].data[:,1,t_start]
z = t.mesh_data.points[:,2] + t.data_fields['displacement'].data[:,2,t_start]

for data_field in t.data_fields:
     cur_data = np.tile(np.expand_dims(t.data_fields[data_field].data[:,:,t_start],2),t.n_steps)
     t.data_fields[data_field].data = t.data_fields[data_field].data - cur_data

disps = np.vstack((x,y,z)).T
plt.plot(t.mesh_data.points[:,0],t.mesh_data.points[:,2])
print(np.max(t.mesh_data.points[:,2]))
t.mesh_data.points = disps
plt.plot(t.mesh_data.points[:,0],t.mesh_data.points[:,2])
print(np.max(t.mesh_data.points[:,2]))
# %%
np.tile(np.expand_dims(t.data_fields['displacement'].data[:,:,t_start],2),50).shape
# %%
def rebaseline(sd,time_step:int):
    x = sd.mesh_data.points[:,0] + sd.data_fields['displacement'].data[:,0,time_step]
    y = sd.mesh_data.points[:,1] + sd.data_fields['displacement'].data[:,1,time_step]
    z = sd.mesh_data.points[:,2] + sd.data_fields['displacement'].data[:,2,time_step]

    disps = np.vstack((x,y,z)).T
    sd.mesh_data.points= disps

    for data_field in sd.data_fields:
        cur_data = np.tile(np.expand_dims(sd.data_fields[data_field].data[:,:,time_step],2),sd.n_steps)
        t.data_fields[data_field].data = sd.data_fields[data_field].data - cur_data

# %%
rebaseline(t,0)
plt.plot(t.mesh_data.points[:,0],t.data_fields['total_strain'].data[:,0,-1],'.')
print(np.max(t.mesh_data.points[:,2]))
rebaseline(t,300)
plt.plot(t.mesh_data.points[:,0],t.data_fields['total_strain'].data[:,0,-1],'.')
print(np.max(t.mesh_data.points[:,2]))
rebaseline(t,0)
plt.plot(t.mesh_data.points[:,0],t.data_fields['total_strain'].data[:,0,-1],'.')
print(np.max(t.mesh_data.points[:,2]))

# %%
plt.plot(t.mesh_data.points[:,0],t.data_fields['total_strain'].data[:,0,-1],'.')
t.update_mesh(90)
plt.plot(t.mesh_data.points[:,0],t.data_fields['total_strain'].data[:,0,-1],'.')


# %%
