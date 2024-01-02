#
#
#
#%%
from pycoatl.spatialdata.importansys import read_ansys_binary
import numpy as np
import pyvista as pv
from ansys.mapdl import reader 

#%%

res = reader.read_binary('C:\ANSYS\WORK\FEMU_SO_v2\Opt_Run.rst')
# %%

test = res.grid
test['x_disp'] = res.nodal_displacement(-1)[1][:,0]
test['y_disp'] = res.nodal_displacement(-1)[1][:,1]

plastic_strain_xx = res.nodal_plastic_strain(-1)[1][:,0]
plastic_strain_yy = res.nodal_plastic_strain(-1)[1][:,0]
plastic_strain_xy = res.nodal_plastic_strain(-1)[1][:,0]
# %%
