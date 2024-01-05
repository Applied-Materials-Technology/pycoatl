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

def test_metadata():
    efile = 'data/moose_output.e'
    sd = moose_to_spatialdata(efile)
    sd.add_metadata_item('test','pass')
    assert 'test' in sd._metadata.keys()
    assert 'pass' in sd._metadata.values()

    new_md = {'trial':'succeed'}
    sd.add_metadata_bulk(new_md)
    assert 'test' in sd._metadata.keys()
    assert 'succeed' in sd._metadata.values()
