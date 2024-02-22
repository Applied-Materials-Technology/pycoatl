#
#

from pycoatl.spatialdata.importdavis import davis_to_spatialdata
from pycoatl.spatialdata.spatialdatawrapper import get_standard_theme

import pytest


def test_import():
    folder = r'/home/rspencer/pycoatl/data/davis_dat'
    loadfile = r'/home/rspencer/pycoatl/data/davis_load.csv'
    dd = davis_to_spatialdata(folder,loadfile)
    assert dd.load[0] == pytest.approx(197.614)
    assert dd.load[-1] == pytest.approx(1086.025)
    dd.mesh_data.plot(scalars=dd.mesh_data['eyy'],theme=get_standard_theme())
