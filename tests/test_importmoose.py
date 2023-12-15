#
#
#

import pytest
from pycoatl.spatialdata.importmoose import moose_to_spatialdata

def test_import():
    efile = 'data/moose_output.e'
    sd = moose_to_spatialdata(efile)
    sd.data_sets[-1].plot(scalars='disp_y',cpos='xy')
    # Should plot the data.
