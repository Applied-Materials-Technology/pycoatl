#
# Testing new Lloyd's new implementation of exodus reader
#
#%%
from mooseherder.exodusreader import ExodusReader
import netCDF4 as nc
import numpy as np
from pycoatl.spatialdata.importmoose import moose_to_spatialdata


# %%
test = ExodusReader('D:\Rory\Spyder\pycoatl\data\moose_output.e')
# %%
test.node_var_names
# %%
print(test.node_data['creep_strain_yy'])
# %%
efile = 'D:\Rory\Spyder\pycoatl\data\moose_output.e'
data = nc.Dataset(efile)
# %%
efile = 'D:\Rory\Spyder\pycoatl\data\moose_output.e'
test = moose_to_spatialdata(efile)
# %%
