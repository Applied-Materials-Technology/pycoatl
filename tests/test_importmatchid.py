#
#
#

import pytest
from pycoatl.spatialdata.importmatchid import matchid_to_spatialdata
from pathlib import Path

def test_import_2023():
    folderpath = Path('data/matchid_dat_2023')
    loadfile = Path('data/matchid_load.csv')
    sd = matchid_to_spatialdata(folderpath,loadfile,version='2023',loadfile_format='Davis')
    sd.plot(data_field='displacements',component=[1],time_step=-1)
    sd.plot(data_field='strains',component=[1,1],time_step=-1)
    # Should plot the data.
    assert sd.load[-1] == pytest.approx(405.577)

def test_import_2024():
    folderpath = Path('data/matchid_dat_2024')
    loadfile = Path('data/Image.csv')
    sd = matchid_to_spatialdata(folderpath,loadfile,version='2024.1',loadfile_format='Image.csv')
    sd.plot(data_field='displacements',component=[1],time_step=-1)
    
