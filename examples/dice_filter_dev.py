#%%
import numpy as np
from mooseherder import ExodusReader
from pathlib import Path
import numpy as np
from numpy._typing import NDArray
import pyvista as pv
from pycoatl.spatialdata.spatialdata import SpatialData
import matplotlib.pyplot as plt

from pycoatl.spatialdata.importmatchid import matchid_to_spatialdata
from pycoatl.spatialdata.importsimdata import simdata_to_spatialdata
from pycoatl.spatialdata.importsimdata import return_mesh_simdata
from pycoatl.spatialdata.tensorfield import rank_two_field
from pycoatl.spatialdata.tensorfield import vector_field

from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Delaunay
from scipy import interpolate
from pycoatl.datafilters.datafilters import FastFilterRegularGrid


from pyvale.imagesim.imagedefopts import ImageDefOpts
from pyvale.imagesim.cameradata import CameraData
import pyvale.imagesim.imagedef as sid

#%%
# Copying over code from LLoyd's examples in pyvale

#%%

best_file = '/home/rspencer/projects/DICe/build/tests/regression/dic_challenge_14_vsg/results/dic_challenge_14.e'
exodus_reader = ExodusReader(Path(best_file))
all_sim_data = exodus_reader.read_all_sim_data()
test = return_mesh_simdata(all_sim_data,dim3=False)
cur_best= simdata_to_spatialdata(all_sim_data)
# %%
