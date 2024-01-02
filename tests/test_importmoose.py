#
#
#

import pytest
from pycoatl.spatialdata.importmoose import moose_to_spatialdata
import pyvista as pv

def test_import():
    efile = 'data/moose_output.e'
    sd = moose_to_spatialdata(efile)
    pv.global_theme.colorbar_orientation = 'vertical'
    sd.data_sets[-1].plot(scalars='mechanical_strain_yy',cpos='xy',)
    # Should plot the data.
