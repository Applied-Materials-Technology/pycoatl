#
#
#

class SpatialData():
    """Spatial data from DIC and FE using PyVista
    Must be picklable. Multiprocessing requires serialisation.
    Must be able to store metadata.
    """

    def __init__(self,data_sets,index,time,load,metadata):
        """

        Args:
            data_sets (list of pyvista mesh): List of the pyvista data meshes.
            index (int array): Indices of the data sets.
            time (float array): Times 
            load (float array): _description_
            metadata (dict): _description_
        """
        self.data_sets = data_sets # List of pyvista meshes.

        self._index = index
        self._time = time
        self._load = load
        self._metadata = metadata # dict of whatever metadata we want.


    def __str__(self):
        """Make a nicely formatted string of metadata for use.
        """
        
        outstring = '**** Spatial Data Format ****\n'
        outstring += 'There are {} data sets.\n'.format(len(self.data_sets))
        outstring += 'The data has the following metadata:\n'
        for key, value in self._metadata.items():
            outstring += '{} is {}\n'.format(key,value)
        
        return outstring



    
