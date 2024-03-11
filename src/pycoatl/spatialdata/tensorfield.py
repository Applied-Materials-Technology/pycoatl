import numpy as np
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Sequence


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
    
    def get_fields(self, component: list[int])->list[npt.NDArray]:
        """Get component fields over all time.
        Intended to be used in differentiations.

        Args:
            component (list[int]): List of indices to get

        Returns:
            list[npt.NDArray]: List of fields corresponding to inputs.
        """
        return [self.data[:,x,:] for x in component]


    
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

    def calculate_invariant(self,n:int)->npt.NDArray:
        """Calculate a given invariant of the tensor.
        n=1 trace
        n=2 
        n=3 det
        Args:
            n (int): Int for invariant, 1,2 or 3

        Returns:
            npt.NDArray: Output
        """

        if n ==1 :
            output = self.get_component[0,0]+self.get_component[1,1]+self.get_component[2,2]
        elif n ==2:
            output = (
                self.get_component[0,0]*self.get_component[1,1]
                + self.get_component[1,1]*self.get_component[2,2]
                + self.get_component[2,2]*self.get_component[0,0]
                - self.get_component[0,1]**2
                - self.get_component[0,2]**2
                - self.get_component[1,2]**2
            )
        elif n==3:
            output = (
                self.get_component[0,0]*self.get_component[1,1]*self.get_component[2,2]
                - self.get_component[0,0]*(self.get_component[1,2]**2)
                - self.get_component[1,1]*(self.get_component[0,2]**2)
                - self.get_component[2,2]*(self.get_component[0,1]**2)
                + 2*self.get_component[0,1]*self.get_component[0,2]*self.get_component[1,2]
            )
        else:
            output = None

            return output

