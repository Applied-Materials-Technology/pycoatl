#
# Readers to be passed to the mooseherd to read in parallel
#  

from pycoatl.spatialdata.importmoose import moose_to_spatialdata

class ReadExodus():

    def __init__(self,fields=['disp_x','disp_y','disp_z','stress_yy','creep_strain_yy','plastic_strain_yy','mechanical_strain_xx','mechanical_strain_yy']):
        self.fields = fields
        
