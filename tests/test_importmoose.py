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
    sd.data_sets[-1].plot(scalars='mechanical_strain_yy',cpos='xy',)
    # Should plot the data.

def test_differentiation():
    efile = 'data/moose_output.e'
    sd = moose_to_spatialdata(efile)
    sd.window_differentation()
    assert 'exx' in sd.data_sets[-1].array_names
    sd.data_sets[-1].plot(scalars='eyy',theme=get_standard_theme())