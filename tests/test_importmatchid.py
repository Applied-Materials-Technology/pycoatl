#
#
#

import pytest
from pycoatl.spatialdata.importmatchid import matchid_to_spatialdata

def test_import_2023():
    folderpath = 'data/matchid_dat_2023'
    loadfile = 'data/matchid_load.csv'
    sd = matchid_to_spatialdata(folderpath,loadfile,version='2023',loadfile_format='Davis')
    sd.data_sets[-1].plot(scalars='v',cpos='xy')
    # Should plot the data.
    assert sd._load[-1] == pytest.approx(405.577)

def test_import_2024():
    folderpath = 'data/matchid_dat_2024'
    loadfile = 'data/Image.csv'
    sd = matchid_to_spatialdata(folderpath,loadfile,version='2024.1',loadfile_format='Image.csv')
    sd.data_sets[-1].plot(scalars='v',cpos='xy')
