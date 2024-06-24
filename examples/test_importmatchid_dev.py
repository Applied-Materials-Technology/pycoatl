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