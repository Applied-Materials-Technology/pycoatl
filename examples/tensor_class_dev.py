#
#
#%%
import numpy as np
from mooseherder import ExodusReader
from pathlib import Path
import numpy as np
from numpy._typing import NDArray
import pyvista as pv
from pycoatl.spatialdata.spatialdata import SpatialData
import matplotlib.pyplot as plt
from pycoatl.spatialdata.importmatchid import matchid_to_spatialdata
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Sequence

#%% read something
#output_file = Path('/home/rspencer/pycoatl/data/moose-sim-1_out.e')
#output_file = Path('/home/rspencer/moose_work/Viscoplastic_Creep/HVPF_Sat/Run/moose-workdir-1/moose-sim-152_out.e')
output_file = Path(r'C:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Anaconda_Python\Test\pycoatl\data\moose-sim-1_out.e')
exodus_reader = ExodusReader(output_file)

all_sim_data = exodus_reader.read_all_sim_data()



# %%
# maybe also have scalar field
class tensor_field_base(ABC):

    @abstractmethod
    def rotate(self,rotation_matrix: npt.NDArray) -> None:
        pass

    @abstractmethod
    def get_component_field(self,component: Sequence,time_step: int)-> npt.NDArray:
        pass

    #@abstractmethod
    #def get_timestep(self,time_step: int)-> npt.NDArray:
    #    return self.data

class vector_field(tensor_field_base):

    rank = 1

    def __init__(self,input_data: npt.NDArray):

        self.data = input_data
        self.n_points = input_data.shape[0]
        self.n_steps = input_data.shape[2]

    def rotate(self, rotation_matrix: npt.NDArray) -> None:
        #rot_mat = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        if rotation_matrix.shape not in [(3,3),(4,4)]:
            raise RuntimeError('Rotation matrix is {}. Should be (3,3) or a (4,4) vtk Transformation matrix.'.format(rotation_matrix.shape))

        if rotation_matrix.shape == (4,4): # Given vtk / pyvista transformation matrix
            rotation_matrix = rotation_matrix[0:3,0:3]
        

        rot_field_0 = np.swapaxes(np.tile(rotation_matrix[0,:],(self.n_points,self.n_steps,1)),1,2)
        rot_field_1 = np.swapaxes(np.tile(rotation_matrix[1,:],(self.n_points,self.n_steps,1)),1,2)
        rot_field_2 = np.swapaxes(np.tile(rotation_matrix[2,:],(self.n_points,self.n_steps,1)),1,2)

        x_comp = vector_field.dot_product_field(self.data,rot_field_0)
        y_comp = vector_field.dot_product_field(self.data,rot_field_1)
        z_comp = vector_field.dot_product_field(self.data,rot_field_2)

        rotated_field = np.stack((x_comp,y_comp,z_comp),axis=1)
        self.data = rotated_field
    
    def get_component_field(self, component: int, time_step: int) -> npt.NDArray:
        return self.data[:,component,time_step]
    
    @staticmethod
    def dot_product_field(a: npt.NDArray,b: npt.NDArray) -> npt.NDArray: 
        """Dot product between two vector fields

        Args:
            a (npt.NDArray): Vector 1
            b (npt.NDArray): Vector 2

        Returns:
            npt.NDArray: Scalar output
        """
        return np.sum(a*b,axis=1)
    
    
class rank_two_field(tensor_field_base):

    rank = 2

    def __init__(self,input_data: npt.NDArray):

        self.data = input_data

    def rotate(self, rotation_matrix: npt.NDArray) -> None:
        return super().rotate(rot_mat)
    
    def get_component_field(self, component: Sequence , time_step: int) -> npt.NDArray:
        return self.data[:,component[0]+3*component[1],time_step]
    
    def get_principal(self):
        pass




class tensor_field():
    """Class for vector fields.
    May benefit from an abstract base class at some point that underlies both
    vector and tensor fields.
    """

    def __init__(self,input_data: npt.NDArray, name: str):
        """Init method
        """
        # data should be a n_points x 3 x n_steps for vector field
        # and n_points x 9 x n_steps for rank 2 tensor field
        self.name = name
        self.data = input_data
        if self.data.shape[1] ==3:
            self.rank=1
        elif self.data.shape[1]==9:
            self.rank=2
        else:
            print('Error. Number of points in tensor, {}, does not match rank one or two tensors.'.format(self.data.shape[1]))
        self.n_points = self.data.shape[0]
        self.n_steps = self.data.shape[2]

    def rotate(self,rotation_matrix: npt.NDArray):
        pass

    def trace(self):
        pass
    
    @staticmethod
    def dot_product(a: npt.NDArray,b: npt.NDArray) -> npt.NDArray: 
        """Dot product between two vectors

        Args:
            a (npt.NDArray): Vector 1
            b (npt.NDArray): Vector 2

        Returns:
            npt.NDArray: Scalar output
        """
        return np.sum(a*b)


    def get_step(self,i):
        return self.data[...,i]

#%% Test
input_data = np.dstack((all_sim_data.node_vars['disp_x'],all_sim_data.node_vars['disp_y'],np.zeros_like(all_sim_data.node_vars['disp_y'])))
print(input_data.shape)
input_data = np.swapaxes(input_data,1,2)
print(input_data.shape)

# %%
displacements = vector_field(input_data)
print(displacements.rank)
#print(displacements.get_component_field(0,1))
test = vector_field.dot_product_field(displacements.data,displacements.data)
print(test.shape)
#%%
dummy = np.zeros_like(all_sim_data.elem_vars[('strain_xx',1)])
strain_data = np.dstack((all_sim_data.elem_vars[('strain_xx',1)],dummy,all_sim_data.elem_vars[('strain_yy',1)],dummy,dummy,dummy,dummy,dummy,all_sim_data.elem_vars[('strain_zz',1)]))
print(strain_data.shape)
strain_data = np.swapaxes(strain_data,1,2)
print(strain_data.shape)
# %%
test_r2 = rank_two_field(strain_data)
sxx = test_r2.get_component_field([0,0],1)
print(sxx.shape)

# %%
test_input = np.ones((1,9,1))
dummy = tensor_field(test_input,'dummy')
print(dummy.rank)
rot_mat = np.array([[0.996194,0.087155,0],[-0.08715,0.996194,0],[0,0,1]])
print(rot_mat)
# %%
p=150
plt.plot([0,input_data[p,0,1]],[0,input_data[p,1,1]])
print(input_data[p,:,1])
#%%
rot_mat = np.array([[0,1,0],[-1,0,0],[0,0,1]])
rot_field_0 = np.swapaxes(np.tile(rot_mat[0,:],(displacements.n_points,displacements.n_steps,1)),1,2)
rot_field_1 = np.swapaxes(np.tile(rot_mat[1,:],(displacements.n_points,displacements.n_steps,1)),1,2)
rot_field_2 = np.swapaxes(np.tile(rot_mat[2,:],(displacements.n_points,displacements.n_steps,1)),1,2)

x_comp = vector_field.dot_product_field(displacements.data,rot_field_0)
y_comp = vector_field.dot_product_field(displacements.data,rot_field_1)
z_comp = vector_field.dot_product_field(displacements.data,rot_field_2)

rot_vec = np.stack((x_comp,y_comp,z_comp),axis=1)
print(rot_vec.shape)
# %%
rot_mat_t = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
# %%
if rot_mat_t.shape==(4,4):
    print(rot_mat_t[0:3,0:3])
# %%
displacements.rotate(rot_mat)
# %%
