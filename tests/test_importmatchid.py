#
#
#

import pytest
from pycoatl.spatialdata.importmatchid import matchid_to_spatialdata

def test_import():
    folderpath = 'data/matchid_dat'
    loadfile = 'data/matchid_load.csv'
    sd = matchid_to_spatialdata(folderpath,loadfile)
    sd.data_sets[-1].plot(scalars='v',cpos='xy')
    # Should plot the data.
    assert sd._load[-1] == pytest.approx(405.577)
