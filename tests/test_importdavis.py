#
#

from pycoatl.spatialdata.importdavis import davis_to_spatialdata
from pycoatl.spatialdata.spatialdatawrapper import get_standard_theme

import pytest


def test_import():
    folder = r'data\davis_dat'
    loadfile = r'data\davis_load.csv'
    dd = davis_to_spatialdata(folder,loadfile)
    assert dd._load[0] == pytest.approx(197.614)
    assert dd._load[-1] == pytest.approx(1086.025)
    dd.data_sets[-1].plot(scalars='eyy',theme=get_standard_theme())
