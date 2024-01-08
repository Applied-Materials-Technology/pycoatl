#
#
#

import pytest
from pycoatl.spatialdata.importmoose import moose_to_spatialdata
import pyvista as pv
from pycoatl.spatialdata.spatialdatawrapper import get_standard_theme

#def test_import():
#    efile = 'data/moose_output.e'
#    sd = moose_to_spatialdata(efile)
#    pv.global_theme.colorbar_orientation = 'vertical'
#    sd.data_sets[-1].plot(scalars='mechanical_strain_yy',cpos='xy',)
#    # Should plot the data.
# Old version of Exodus reader.

def test_new_import():
    efile = 'data/moose_output.e'
    sd = moose_to_spatialdata(efile)
    pv.global_theme.colorbar_orientation = 'vertical'
    sd.data_sets[-1].plot(scalars='mechanical_strain_yy',cpos='xy',theme=get_standard_theme())
    # Should plot the data.

def test_differentiation():
    efile = 'data/moose_output.e'
    sd = moose_to_spatialdata(efile)
    gd = sd.interpolate_to_grid()
    gd.window_differentation()
    assert 'exx' in gd.data_sets[-1].array_names
    gd.data_sets[-1].plot(scalars='eyy',theme=get_standard_theme())

def test_import2d():
    pass