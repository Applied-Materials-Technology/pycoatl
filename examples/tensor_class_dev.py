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
from pycoatl.spatialdata.importsimdata import return_mesh_simdata
from pycoatl.spatialdata.importsimdata import simdata_to_spatialdata

#%% read something
#output_file = Path('/home/rspencer/pycoatl/data/moose-sim-1_out.e')
#output_file = Path('/home/rspencer/moose_work/Viscoplastic_Creep/HVPF_Sat/Run/moose-workdir-1/moose-sim-152_out.e')
output_file = Path(r'C:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Anaconda_Python\Test\pycoatl\data\sim-40_out.e')
exodus_reader = ExodusReader(output_file)

#all_sim_data = exodus_reader.read_all_sim_data()
simdata = exodus_reader.read_all_sim_data()


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

class scalar_field(tensor_field_base):

    rank = 0

    def __init__(self,input_data: npt.NDArray):

        if input_data.shape[1] != 1:
            raise RuntimeError('Scalar fields must have shape (n,1,m), not {}.'.format(input_data.shape))
        self.data = input_data
        self.n_points = input_data.shape[0]
        self.n_steps = input_data.shape[2]

    def get_component_field(self, time_step: int) -> npt.NDArray:
        return self.data[...,time_step]
    
    def rotate(self, rotation_matrix: npt.NDArray) -> None:
        print('No rotation applied. Scalar fields are rotation invariant.')


class vector_field(tensor_field_base):

    rank = 1

    def __init__(self,input_data: npt.NDArray):

        if input_data.shape[1] != 3:
            raise RuntimeError('Vector fields must have shape (n,3,m), not {}.'.format(input_data.shape))
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
        self.n_points = input_data.shape[0]
        self.n_steps = input_data.shape[2]

    def rotate(self, rotation_matrix: npt.NDArray) -> None:

        if rotation_matrix.shape not in [(3,3),(4,4)]:
            raise RuntimeError('Rotation matrix is {}. Should be (3,3) or a (4,4) vtk Transformation matrix.'.format(rotation_matrix.shape))

        if rotation_matrix.shape == (4,4): # Given vtk / pyvista transformation matrix
            rotation_matrix = rotation_matrix[0:3,0:3]

        r11 = np.swapaxes(np.tile(rotation_matrix[0,0],(self.n_points,self.n_steps,1)),1,2)
        r12 = np.swapaxes(np.tile(rotation_matrix[0,1],(self.n_points,self.n_steps,1)),1,2)
        r13 = np.swapaxes(np.tile(rotation_matrix[0,2],(self.n_points,self.n_steps,1)),1,2)
        r21 = np.swapaxes(np.tile(rotation_matrix[1,0],(self.n_points,self.n_steps,1)),1,2)
        r22 = np.swapaxes(np.tile(rotation_matrix[1,1],(self.n_points,self.n_steps,1)),1,2)
        r23 = np.swapaxes(np.tile(rotation_matrix[1,2],(self.n_points,self.n_steps,1)),1,2)
        r31 = np.swapaxes(np.tile(rotation_matrix[2,0],(self.n_points,self.n_steps,1)),1,2)
        r32 = np.swapaxes(np.tile(rotation_matrix[2,1],(self.n_points,self.n_steps,1)),1,2)
        r33 = np.swapaxes(np.tile(rotation_matrix[2,2],(self.n_points,self.n_steps,1)),1,2)
        
        T11 = self.get_component([0,0])
        T12 = self.get_component([0,1])
        T13 = self.get_component([0,2])
        T21 = self.get_component([1,0])
        T22 = self.get_component([1,1])
        T23 = self.get_component([1,2])
        T31 = self.get_component([2,0])
        T32 = self.get_component([2,1])
        T33 = self.get_component([2,2])

        T11_r = r11*r11*T11 + r11*r12*T12 + r11*r13*T13 + r12*r11*T21 + r12*r12*T22 + r12*r13*T23 + r13*r11*T31+ r13*r12*T32+ r13*r13*T33
        T12_r = r11*r21*T11 + r11*r22*T12 + r11*r23*T13 + r12*r21*T21 + r12*r22*T22 + r12*r23*T23 + r13*r21*T31+ r13*r22*T32+ r13*r23*T33
        T13_r = r11*r31*T11 + r11*r32*T12 + r11*r33*T13 + r12*r31*T21 + r12*r32*T22 + r12*r33*T23 + r13*r31*T31+ r13*r32*T32+ r13*r33*T33
        T21_r = r21*r11*T11 + r21*r12*T12 + r21*r13*T13 + r22*r11*T21 + r22*r12*T22 + r22*r13*T23 + r23*r11*T31+ r23*r12*T32+ r23*r13*T33
        T22_r = r21*r21*T11 + r21*r22*T12 + r21*r23*T13 + r22*r21*T21 + r22*r22*T22 + r22*r23*T23 + r23*r21*T31+ r23*r22*T32+ r23*r23*T33
        T23_r = r21*r31*T11 + r21*r32*T12 + r21*r33*T13 + r22*r31*T21 + r22*r32*T22 + r22*r33*T23 + r23*r31*T31+ r23*r32*T32+ r23*r33*T33
        T31_r = r31*r11*T11 + r31*r12*T12 + r31*r13*T13 + r32*r11*T21 + r32*r12*T22 + r32*r13*T23 + r33*r11*T31+ r33*r12*T32+ r33*r13*T33
        T32_r = r31*r21*T11 + r31*r22*T12 + r31*r23*T13 + r32*r21*T21 + r32*r22*T22 + r32*r23*T23 + r33*r21*T31+ r33*r22*T32+ r33*r23*T33
        T33_r = r31*r31*T11 + r31*r32*T12 + r31*r33*T13 + r32*r31*T21 + r32*r32*T22 + r32*r33*T23 + r33*r31*T31+ r33*r32*T32+ r33*r33*T33

        rotated_tensor = np.squeeze(np.stack((T11_r,T12_r,T13_r,T21_r,T22_r,T23_r,T31_r,T32_r,T33_r),axis=1))
        self.data = rotated_tensor
    
    def get_component_field(self, component: Sequence , time_step: int) -> npt.NDArray:
        return self.data[:,3*component[0]+component[1],time_step]
    
    def get_component(self, component: Sequence) -> npt.NDArray:
        return np.swapaxes(np.atleast_3d(self.data[:,3*component[0]+component[1],:]),1,2)
    
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
#%% scalar test
ts = scalar_field(np.swapaxes(np.atleast_3d(all_sim_data.node_vars['disp_x']),1,2))
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
strain_data = np.dstack((all_sim_data.elem_vars[('strain_xx',1)],dummy,dummy,dummy,all_sim_data.elem_vars[('strain_yy',1)],dummy,dummy,dummy,all_sim_data.elem_vars[('strain_zz',1)]))
print(strain_data.shape)
strain_data = np.swapaxes(strain_data,1,2)
print(strain_data.shape)
# %%
test_r2 = rank_two_field(strain_data)
sxx = test_r2.get_component_field([0,0],1)
print(sxx.shape)
sxx_t = test_r2.get_component([0,0])
print(test_r2.data[10,:,1])
rot_mat = np.array([[0,1,0],[-1,0,0],[0,0,1]])
test_r2.rotate(rot_mat)
print(test_r2.data[10,:,1])
test_r2.rotate(rot_mat)
print(test_r2.data[10,:,1])
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
r13 = np.swapaxes(np.tile(rot_mat[2,2],(200,6,1)),1,2)
# %%
test = return_mesh_simdata(all_sim_data,False)
# %%
test_r = test.reflect((1,0,0),point=(0,0,0))
# %%
test_comb = test + test_r
test_comb.plot()
# %%
# Should fail
test_comb['test'] = np.concatenate((all_sim_data.node_vars['disp_x'],all_sim_data.node_vars['disp_x']),axis=0)
# %%
# Should fail
x_data=all_sim_data.node_vars['disp_x'][:,-1]
overlap= test.points[:,0]==test_r.points[:,0]
x_data_r = -1*x_data[~overlap]
test_comb['test'] = np.concatenate((x_data,x_data_r),axis=0)
test_comb.plot(scalars='test')

# %%
output_file = Path(r'C:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Anaconda_Python\Test\pycoatl\data\sim-40_out.e')
exodus_reader = ExodusReader(output_file)

#all_sim_data = exodus_reader.read_all_sim_data()
simdata = exodus_reader.read_all_sim_data()
test = simdata_to_spatialdata(simdata)

# %%
rot_mat_t = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
test.rotate_data(rot_mat_t)
test.plot('displacement',[1,1],-20)
# %%
