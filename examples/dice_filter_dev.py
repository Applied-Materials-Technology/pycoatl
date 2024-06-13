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

import xml.etree.ElementTree as ET

#%%
# Copying over code from LLoyd's examples in pyvale

#%%
best_file = '/home/rspencer/moose_work/Viscoplastic_Creep/3P_Specimen/3p_creep_peric_sat_3d_out.e'

#best_file = '/home/rspencer/projects/DICe/build/tests/regression/dic_challenge_14_vsg/results/dic_challenge_14.e'
exodus_reader = ExodusReader(Path(best_file))
all_sim_data = exodus_reader.read_all_sim_data()
#test = return_mesh_simdata(all_sim_data,dim3=False)
cur_best= simdata_to_spatialdata(all_sim_data)
# %%
# Load image - expects a *.tiff or *.bmp that is grayscale
im_path = Path('/home/rspencer/projects/pyvale/data/speckleimages')
#im_file = 'OptimisedSpeckle_500_500_width3.0_16bit_GBlur1.tiff'
im_file = 'OptimisedSpeckle_2464_2056_width5.0_8bit_GBlur1.tiff'
im_path = im_path / im_file
print('\nLoading speckle image from path:')
print(im_path)

input_im = sid.load_image(im_path)
#%%
#---------------------------------------------------------------------------
# Changing to work with SpatialData
coords = np.array(cur_best.mesh_data.points)
disp_x = cur_best.data_fields['displacement'].data[:,0,-3:-1]
disp_y = cur_best.data_fields['displacement'].data[:,1,-3:-1]

print(f'coords.shape={coords.shape}')
print(f'disp_x.shape={disp_x.shape}')
print(f'disp_y.shape={disp_y.shape}')

#%%
#---------------------------------------------------------------------------
# INIT IMAGE DEF OPTIONS AND CAMERA
print('')
print('='*80)
print('INIT. IMAGE DEF. OPTIONS AND CAMERA')
print('')

#---------------------------------------------------------------------------
# CREATE IMAGE DEF OPTS
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
#%%
#---------------------------------------------------------------------------
# CREATE CAMERA OBJECT
camera = CameraData()
# Need to set the number of pixels in [X,Y], the bit depth and the m/px

# Assume the camera has the same number of pixels as the input image unless we
# are going to crop/mask the input image
camera.num_px = np.array([input_im.shape[1],input_im.shape[0]])
if id_opts.crop_on:
    camera.num_px = id_opts.crop_px

# Based on the max grey level work out what the bit depth of the image is
camera.bits = 8
if max(input_im.flatten()) > (2**8):
    camera.bits = 16

# Assume 1mm/px to start with, can update this to fit FE data within the FOV
# using the id_opts above. Or set this manually.
camera.m_per_px = 1.0e-3 # Overwritten by id_opts.calc_res_from_fe = True

# Can manually set the ROI location by setting the above to false and setting
# the camera.roi_loc as the distance from the origin to the bottom left
# corner of the sample [X,Y]: camera.roi_loc = np.array([1e-3,1e-3])

# Default ROI is the whole FOV but we want to set this to be based on the
# furthest nodes, this is set in FE units 'meters' and does not change FOV
camera.roi_len = sid.calc_roi_from_nodes(camera,coords)


# If we are masking an image we might want to set an optimal resolution based
# on leaving a specified number of pixels free on each image edge, this will
# change the FOV in 'meters'
if id_opts.calc_res_from_fe:
    camera.m_per_px = sid.calc_res_from_nodes(camera,coords, #type: ignore
                                            id_opts.calc_res_border_px)

# Default ROI is the whole FOV but we want to set this to be based on the
# furthest nodes, this is set in FE units 'meters' and does not change FOV
camera.roi_len = sid.calc_roi_from_nodes(camera,coords)

camera._roi_loc[0] = (camera._fov[0] - camera._roi_len[0])/2 -np.min(coords[:,0])
camera._roi_loc[1] = (camera._fov[1] - camera._roi_len[1])/2 -np.min(coords[:,1])

print('-'*80)
print('CameraData:')
print(vars(camera))
print('-'*80)
print('')
#%%
(masked_im,image_mask) = sid.get_im_mask_from_sim(camera,
                        sid.rectangle_crop_image(camera,input_im),
                        coords)
#%% For geometries without holes

border_size = 10
y = []
x_min = []
x_max = []
for j in range(0,image_mask.shape[0],20):
    edge = np.where(image_mask[j,:]==1)
    try: 
        x_min.append(edge[0][0]+border_size)
        x_max.append(edge[0][-1]-border_size)
        y.append(j)
    except IndexError:
        continue

y_roi = np.concatenate((np.flip(np.array(y)),np.array(y)))
x_roi = np.concatenate((np.array(x_min),np.flip(np.array(x_max))))
plt.plot(x_roi,y_roi)
#%%
for i in range(len(x_roi)): 
    print('{} {}'.format(x_roi[i],y_roi[i]))
#%%

#%%
#---------------------------------------------------------------------------
# PRE-PROCESS AND DEFORM IMAGES
sid.deform_images(input_im,
                camera,
                id_opts,
                coords,
                disp_x,
                disp_y,
                print_on = True)
# %%
best_file = '/home/rspencer/pycoatl/examples/ImDef/results/DICe_solution.e'
exodus_reader = ExodusReader(Path(best_file))
all_sim_data = exodus_reader.read_all_sim_data()

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(all_sim_data.node_vars['COORDINATE_X'][:,0]*camera.m_per_px -2.5,-(all_sim_data.node_vars['COORDINATE_Y'][:,0]*camera.m_per_px -10),c=all_sim_data.node_vars['VSG_STRAIN_YY'][:,1])
#ax.scatter(all_sim_data.node_vars['COORDINATE_X'][:,0]*camera.m_per_px -2.5,-(all_sim_data.node_vars['COORDINATE_Y'][:,0]*camera.m_per_px -10),c=all_sim_data.node_vars['VSG_STRAIN_YY'][:,1],vmax=5E-4,vmin=0)
#ax.scatter(coords[:,0],coords[:,1])
ax.axis('equal')

# %%
plt.hist(all_sim_data.node_vars['VSG_STRAIN_XY'][:,1],bins=100,range=(0,1E-2))
# %%
plt.hist(all_sim_data.node_vars['elastic_strain_yy'][:,-1]+all_sim_data.node_vars['plastic_strain_yy'][:,-1],bins=100,range=(0,1E-2))
# %%
xc = all_sim_data.node_vars['COORDINATE_X'][:,0]*camera.m_per_px-camera._roi_loc[0]
yc = -all_sim_data.node_vars['COORDINATE_Y'][:,0]*camera.m_per_px-camera._roi_loc[1]
zc = np.zeros(len(xc))
points = np.vstack((xc,yc,zc))
print(points.shape)
test=pv.PolyData(points.T)
test['strain_yy'] = all_sim_data.node_vars['VSG_STRAIN_YY']
#%% Parameters needed
input_file_name = Path('/home/rspencer/pycoatl/examples/ImDef/input.xml')
mod_file_name = input_file_name.parent /'input_mod.xml'
deformed_images = Path('/home/rspencer/pycoatl/examples/ImDef/deformed_images')
subset_file = Path('subsets.txt')
output_folder = input_file_name.parent /'results'
cor_param_file = Path('params.xml')

#%% 
tree = ET.parse('/home/rspencer/pycoatl/examples/ImDef/input.xml')
root = tree.getroot()

# %%
for child in root:
    print(child.tag, child.attrib)
# %%
ET.Element('ParameterList')
# %%
print(root.findall("."))
print(root.find(".//*[@name='image_folder']").set('value','test'))
print(root.find(".//*[@name='image_folder']").attrib)
# %% Create XML
# Remove any deformed images files
print(root.find('ParameterList').attrib)
def_ims = root.find(".//*[@name='deformed_images']")
for child in def_ims:
    #def_ims.remove(child)
    print(child.attrib)

# %% Get list of files.
files = []
im_folder = deformed_images.parent
for p in deformed_images.iterdir():
    files.append(p.name)

files.sort()
print(files)

# Modify xml
root.find(".//*[@name='image_folder']").set('value',str(im_folder))

def_ims = root.find(".//*[@name='deformed_images']")
for child in def_ims:
    def_ims.remove(child)
    print(child.attrib)
def_ims = root.find(".//*[@name='deformed_images']")

for file in files[1:]:
    attributes = {'name':str(file),'type':'bool','value':'true'}
    el = ET.SubElement(def_ims,'Parameter',attributes)

# %%
def_ims = root.find(".//*[@name='deformed_images']")
for child in def_ims:
    #def_ims.remove(child)
    print(child.attrib)
# %%
# Define necessary inputs
input_file_name = Path('/home/rspencer/pycoatl/examples/ImDef/input.xml')
mod_file_name = input_file_name.parent /'input_mod.xml'
deformed_images = Path('/home/rspencer/pycoatl/examples/ImDef/deformed_images')
subset_file =  input_file_name.parent /'subsets_roi.txt'
output_folder = input_file_name.parent /'results'
#cor_param_file = Path('params.xml')
#image_folder = deformed_images.parent

# Read current input file 
tree = ET.parse(input_file_name)
root = tree.getroot()

# Clear any existing deformed image paths
parent = root.find(".//*[@name='deformed_images']")
for child in parent.findall('./'):
    parent.remove(child)

# Read in deformed image paths, assumption is 0 is the reference
files = []
im_folder = deformed_images.parent
for p in deformed_images.iterdir():
    files.append(p.name)

files.sort()

# Update the subsets file
root.find(".//*[@name='subset_file']").set('value',str(subset_file))

# Update the image folder
root.find(".//*[@name='image_folder']").set('value',str(deformed_images)+'/')

# Update the reference image path
root.find(".//*[@name='reference_image']").set('value',str(files[0]))

# Update the deformed image path list
for file in files[1:]:
    attributes = {'name':str(file),'type':'bool','value':'true'}
    el = ET.SubElement(parent,'Parameter',attributes)



# Write modified XML to file
tree.write(mod_file_name)

# %% Write a subsets.txt file with ROI

# Fow now, non-hole specimens

with open(subset_file,'w') as f:
    f.write('BEGIN REGION_OF_INTEREST\n')
    f.write('  BEGIN BOUNDARY\n')
    f.write('    BEGIN POLYGON\n')
    f.write('      BEGIN VERTICES\n')
    
    for i in range(len(x_roi)):
        f.write('      {} {}\n'.format(x_roi[i],y_roi[i]))

    f.write('      END VERTICES\n')
    f.write('    END POLYGON\n')
    f.write('  END BOUNDARY\n')
    f.write('END REGION_OF_INTEREST\n')
#%% General form of the DICe filter

# All input information


# Step 1: Read a default XML file to get some information
# Need the step size to apply to ROI size.

# Step 2: Image deformation, get image mask, 
# Possibility to store image mask if it's being used repeatedly
# as that can be the time consuming bit

# Step 3: Modify DICe input files

# Step 4: Run DICe

# Step 5: Read in results, push to spatial data format

