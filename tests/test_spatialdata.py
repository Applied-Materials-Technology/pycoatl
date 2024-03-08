#
#
import pytest
from pycoatl.spatialdata.importmoose import moose_to_spatialdata
import pyvista as pv
from pycoatl.spatialdata.spatialdatawrapper import get_standard_theme
from pathlib import Path
from pycoatl.spatialdata.importmatchid import matchid_to_spatialdata
import numpy as np

def test_rotations():
    # Import data
    folderpath = Path('data/matchid_dat_2024')
    loadfile = Path('data/Image.csv')
    sd = matchid_to_spatialdata(folderpath,loadfile,version='2024.1',loadfile_format='Image.csv')
    # PLot y displacements
    sd.plot(data_field='displacements',component=[1],time_step=-1)
    #Store y displacements, sxx
    pre_r = sd.data_fields['displacements'].get_component_field(1,-1)
    pre_rs = sd.data_fields['strains'].get_component_field([0,0],-1)
    # Rotation matrix for 90deg
    t_mat = np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
    #Rotate data
    sd.rotate_data(t_mat)
    #Store post rotations
    post_r = sd.data_fields['displacements'].get_component_field(0,-1)
    post_rs = sd.data_fields['strains'].get_component_field([1,1],-1)
    # Plot y displacements (should now actually be x-displacement in original coords)
    sd.plot(data_field='displacements',component=[1],time_step=-1)
    # Check that y -> x and sxx -> syy
    assert np.all(pre_r == post_r)
    assert np.all(pre_rs == post_rs)