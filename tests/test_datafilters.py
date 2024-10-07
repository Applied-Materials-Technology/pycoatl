#
# Testing sequential / parallel running of data filters.
#
#%%
import numpy as np
from pycoatl.datafilters.datafilters import FastFilterRegularGrid
from pycoatl.spatialdata.importsimdata import simdata_to_spatialdata
import pyvista as pv
import time
from mooseherder import ExodusReader
from pathlib import Path

# %%
best_file = '/home/rspencer/pycoatl/data/sim-40_out.e'
exodus_reader = ExodusReader(Path(best_file))
all_sim_data = exodus_reader.read_all_sim_data()
cur_best= simdata_to_spatialdata(all_sim_data)
# %%
t0 = time.time()
data_filter = FastFilterRegularGrid(run_mode='sequential')
data_in = [cur_best,cur_best,cur_best]
filtered_data = data_filter.run_filter(data_in)
t1 = time.time()
print('Sequential Time: {}'.format(t1-t0))
#%%
t0 = time.time()
data_filter = FastFilterRegularGrid(run_mode='parallel')
data_in = [cur_best,cur_best,cur_best]
filtered_data = data_filter.run_filter(data_in)
t1 = time.time()
print('Parallel Time: {}'.format(t1-t0))
# %%
test = data_filter.run_filter_once(cur_best)
test.plot()
# %%
