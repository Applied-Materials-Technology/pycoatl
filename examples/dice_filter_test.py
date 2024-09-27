#%%
import numpy as np
from mooseherder import ExodusReader
from pathlib import Path
import numpy as np

import pyvista as pv
from pycoatl.spatialdata.spatialdata import SpatialData
import matplotlib.pyplot as plt
from pycoatl.spatialdata.importsimdata import simdata_to_spatialdata
from pycoatl.spatialdata.importdicedata import simdata_dice_to_spatialdata
from pycoatl.datafilters.datafilters import DiceFilter
from pycoatl.datafilters.datafilters import DiceManager
from pycoatl.datafilters.datafilters import DiceOpts

from pyvale.imagesim.imagedefopts import ImageDefOpts
from pyvale.imagesim.cameradata import CameraData
import pyvale.imagesim.imagedef as sid

#%% Import some data
#best_file = '/home/rspencer/moose_work/Viscoplastic_Creep/3P_Specimen/3p_creep_peric_sat_3d_out.e'
best_file = '/home/rspencer/pycoatl/data/moose-sim-1_out.e'
exodus_reader = ExodusReader(Path(best_file))
all_sim_data = exodus_reader.read_all_sim_data()
cur_best= simdata_to_spatialdata(all_sim_data)

#%% Image Opts
id_opts = ImageDefOpts()
id_opts.save_path = Path('/home/rspencer/pycoatl/examples/ImDef') / 'deformed_images'

# If the input image is just a pattern then the image needs to be masked to
# show just the sample geometry. This setting generates this image.
id_opts.mask_input_image = True
# Set this to True for holes and notches and False for a rectangle
id_opts.def_complex_geom = True

# If the input image is much larger than needed it can also be cropped to
# increase computational speed.
id_opts.crop_on = True
id_opts.crop_px = np.array([500,2000])

# Calculates the m/px value based on fitting the specimen/ROI within the camera
# FOV and leaving a set number of pixels as a border on the longest edge
id_opts.calc_res_from_fe = True
id_opts.calc_res_border_px = 10

# Set this to true to create an undeformed masked image
id_opts.add_static_ref = 'pad_disp'

print('-'*80)
print('ImageDefOpts:')
print(vars(id_opts))
print('-'*80)
print('')

#%% Create Camera
camera = CameraData()
# Need to set the number of pixels in [X,Y], the bit depth and the m/px

# Assume the camera has the same number of pixels as the input image unless we
# are going to crop/mask the input image
camera.num_px = id_opts.crop_px

# Based on the max grey level work out what the bit depth of the image is
camera.bits = 8

# Assume 1mm/px to start with, can update this to fit FE data within the FOV
# using the id_opts above. Or set this manually.
camera.m_per_px = 1.0e-3 # Overwritten by id_opts.calc_res_from_fe = True


# %% Define necessary inputs
input_file_name = Path('/home/rspencer/pycoatl/examples/ImDef/input.xml')
mod_file_name = input_file_name.parent /'input_mod.xml'
deformed_images = Path('/home/rspencer/pycoatl/examples/ImDef/deformed_images')
subset_file =  input_file_name.parent /'subsets_roi.txt'
output_folder = input_file_name.parent /'results'
base_image = Path('/home/rspencer/projects/pyvale/data/speckleimages/OptimisedSpeckle_2464_2056_width5.0_8bit_GBlur1.tiff')

#%%
dice_opts= DiceOpts(input_file_name,
                    mod_file_name,
                    deformed_images,
                    subset_file,
                    output_folder)
dm = DiceManager(dice_opts)

#tf= DiceFilter(base_image,id_opts,camera,dice_opts,[50,150,200])
tf= DiceFilter(base_image,id_opts,camera,dice_opts,[2,4,5])

# %%
t = tf.run_filter(cur_best)

#%%
exodus_reader = ExodusReader(Path('/home/rspencer/pycoatl/examples/ImDef/results/DICe_solution.e'))
all_sim_data = exodus_reader.read_all_sim_data()
t = simdata_dice_to_spatialdata(all_sim_data,camera.m_per_px,camera.roi_loc)
# %%
t.plot('vsg_strain',[1,1],-1)
# %%
#es = cur_best.data_fields['elastic_strain'].data[:,4,200]
#ps = cur_best.data_fields['plastic_strain'].data[:,4,200]
#cur_best.mesh_data.plot(scalars=es+ps )
# %%
#cur_best.plot('ts',[1,1],200)
# %%
#cur_best.data_fields['ts']=cur_best.data_fields['elastic_strain']+cur_best.data_fields['plastic_strain']
# %%
