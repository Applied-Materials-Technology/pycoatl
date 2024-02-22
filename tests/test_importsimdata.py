#
#

from pycoatl.spatialdata.importsimdata import simdata_to_spatialdata
from pycoatl.spatialdata.spatialdatawrapper import get_standard_theme
from pathlib import Path
from mooseherder import ExodusReader
import pytest


def test_import():
    #output_file = Path('/home/rspencer/pycoatl/data/moose-sim-1_out.e')
    output_file = Path('/home/rspencer/moose_work/Viscoplastic_Creep/HVPF_Sat/Run/moose-workdir-1/moose-sim-152_out.e')
    exodus_reader = ExodusReader(output_file)
    all_sim_data = exodus_reader.read_all_sim_data()
    dd = simdata_to_spatialdata(all_sim_data)
    #assert dd._load[0] == pytest.approx(197.614)
    #assert dd._load[-1] == pytest.approx(1086.025)
    dd.mesh_data.plot(scalars= dd.mesh_data['plastic_strain_yy'][:,-1],theme=get_standard_theme())
