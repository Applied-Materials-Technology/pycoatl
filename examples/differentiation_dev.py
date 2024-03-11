#%%
import numpy as np
import pyvista as pv
from numba import jit
from numpy._typing import NDArray
from typing import Sequence
from typing import Self
from pycoatl.spatialdata.tensorfield import scalar_field
from pycoatl.spatialdata.tensorfield import vector_field
from pycoatl.spatialdata.tensorfield import rank_two_field
from pycoatl.spatialdata.importmatchid import matchid_to_spatialdata
from pathlib import Path
from matplotlib import pyplot as plt
#%%

folderpath = Path(r'C:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Anaconda_Python\Test\pycoatl\data\matchid_dat_2024')
loadfile = Path(r'C:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Anaconda_Python\Test\pycoatl\data\Image.csv')
sd = matchid_to_spatialdata(folderpath,loadfile,version='2024.1',loadfile_format='Image.csv')
    
#%% challenging data
data_folder = Path(r'C:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Projects\FY23-24\MT2_Creep\XY\Exp2')
load_file = Path(r'C:\Users\rspencer\OneDrive - UK Atomic Energy Authority\Projects\FY23-24\MT2_Creep\XY\Image_Merge.csv')
data_xy1 = matchid_to_spatialdata(data_folder,load_file,version='2024.1',loadfile_format='Image.csv')

#%%
def get_points_neighbours(mesh,window_size=5):
    """Get the neighbouring points for a mesh.
    Initial phase of the window differentiation.
    Assumes a regular-like quad mesh. Such that surrounding each point are 
    8 others.
    Slow if window size is large.

    Args:
        mesh (pyvista unstructured mesh): Mesh file to characterise.
        window_size (int, optional): Size of the subwindow to differentiate over. Defaults to 5.

    Returns:
        array: Connectivity array, listing window indices for each point.
    """

    n_points = mesh.number_of_points
    levels = int((window_size -1)/2)
    points_array = []# = np.empty((n_points,int(window_size**2)))

    
    for point in range(n_points):
        #point = 0
        point_neighbours = mesh.point_neighbors_levels(point,levels)
        point_neighbours = list(point_neighbours)
        #print(point_neighbours)
        neighbours = [point]
        for n in point_neighbours:
            neighbours = neighbours + n
        #print(neighbours)
        points_array.append(neighbours)
    return points_array

def window_differentation(self,data_range = 'all',window_size=5):
    """Differentiate spatialdata using subwindow approach to 
    mimic DIC filter. Adds the differentiated fields into the meshes in
    the spatial data.
    Primarily intended for MOOSE FE output 


    Args:
        spatialdata (SpatialData): SpatialData instance from FE
        data_range (str, optional): Differentiate all time steps, or just last one. Should be 'all' or 'last'. Defaults to 'all'.
        window_size (int, optional): Subwindow size. Defaults to 5.
    """



    points_list = get_points_neighbours(self.data_sets[0],window_size)


    @jit(nopython=True)
    def L_Q4_n(x):
        
        return np.vstack((np.ones(x[0].shape),x[0],x[1],x[0]*x[1])).T

    @jit(nopython=True)
    def evaluate_point_dev_n(point_data,data):
        """
        Fit an calculate deformation gradient at each point.
        """
        window_spread = int((window_size - 1) /2)
        
        xdata = point_data[:,:2].T
        xbasis = L_Q4_n(xdata)
        ydata = data.ravel()

        if len(ydata)<window_size**2:
            partial_dx = np.nan
            partial_dy = np.nan
        else:
            paramsQ4, r, rank, s = np.linalg.lstsq(xbasis, ydata)
                
            px = xdata[:,int(round((window_size**2) /2))]
            partial_dx = paramsQ4[1] + paramsQ4[3]*px[1]
            partial_dy = paramsQ4[2] + paramsQ4[3]*px[0]
            
        return partial_dx, partial_dy

    @jit(nopython=True)    
    def euler_almansi_n(dudx,dudy,dvdx,dvdy):
        """
        Calculates the Euler-Almansi strain tensor from the given gradient data.
        Can implement more in future.
        """
        #exx = dudx - 0.5*(dudx**2+dvdx**2)
        exx = np.log(np.sqrt(1 + 2*dudx + dudx**2 + dudy**2))
        #eyy = dvdy - 0.5*(dvdy**2+dudy**2)
        eyy = np.log(np.sqrt(1 + 2*dvdy + dvdx**2 + dvdy**2))
        #exy = 0.5*(dudy + dvdx) - 0.5*((dudx*dudy)+(dvdx*dvdy))
        exy = dvdx*(1+dudx) + dudy*(1+dvdy)
        return exx,eyy,exy
    
    def differentiate_mesh(mesh,points_list):
        n_points = mesh.number_of_points
        dudx = np.empty(n_points)
        dvdx = np.empty(n_points)
        dudy = np.empty(n_points)
        dvdy = np.empty(n_points)

        for point in range(n_points):
            #point = 0
            neighbours = points_list[point]
            point_data = mesh.points[neighbours]
            #x_coords = point_data[:,0]
            #y_coords = point_data[:,1]
            u = mesh.point_data['disp_x'][neighbours]
            v = mesh.point_data['disp_y'][neighbours]
            
            dudx[point],dudy[point] = evaluate_point_dev_n(point_data,u)
            dvdx[point],dvdy[point] = evaluate_point_dev_n(point_data,v)

        exx,eyy,exy = euler_almansi_n(dudx,dudy,dvdx,dvdy)

        return exx,eyy,exy
    # Update meta data
    self._metadata['dic_filter'] = True
    self._metadata['window_size'] = window_size
    self._metadata['data_range'] = data_range
    # Iterate over meshes, but ignore metadata which is in posn 0
    if data_range == 'all':
        for mesh in self.data_sets:
            exx,eyy,exy = differentiate_mesh(mesh,points_list)
            mesh['exx'] =exx
            mesh['eyy'] =eyy
            mesh['exy'] =exy
    elif data_range =='last':
        mesh = self.data_sets[-1]
        exx,eyy,exy = differentiate_mesh(mesh,points_list)
        mesh['exx'] =exx
        mesh['eyy'] =eyy
        mesh['exy'] =exy
# %%
sd=data_xy1
window_size =5
points_list = get_points_neighbours(sd.mesh_data,window_size)

def window_

def L_Q4(x):
    
    return np.vstack((np.ones(x[0].shape),x[0],x[1],x[0]*x[1])).T


def evaluate_point_dev(point_data,data):
    """
    Fit an calculate deformation gradient at each point.
    """
    window_spread = int((window_size - 1) /2)
    
    xdata = point_data[:,:2].T
    xbasis = L_Q4_n(xdata)
    #print(xbasis)
    ydata = data#.ravel()
    #print(ydata)

    if len(ydata)<window_size**2:
        partial_dx = np.nan
        partial_dy = np.nan
    else:
        paramsQ4, r, rank, s = np.linalg.lstsq(xbasis, ydata)
            
        px = xdata[:,int(round((window_size**2) /2))]
        partial_dx = paramsQ4[1] + paramsQ4[3]*px[1]
        partial_dy = paramsQ4[2] + paramsQ4[3]*px[0]
        
    return partial_dx, partial_dy

mesh=sd.mesh_data
n_points = mesh.number_of_points
dudx = np.empty((n_points,data_xy1.n_steps))
dvdx = np.empty((n_points,data_xy1.n_steps))
dudy = np.empty((n_points,data_xy1.n_steps))
dvdy = np.empty((n_points,data_xy1.n_steps))

f= sd.data_fields['displacement'].get_fields([0,1])
u_all = f[0]
v_all = f[1]

for point in range(n_points):
    #point = 0
    #print(point)
    neighbours = points_list[point]
    point_data = mesh.points[neighbours]
    u = u_all[neighbours,:]
    v= v_all[neighbours,:]
    
    dudx[point],dudy[point] = evaluate_point_dev_n(point_data,u)
    dvdx[point],dvdy[point] = evaluate_point_dev_n(point_data,v)

def euler_almansi(dudx,dudy,dvdx,dvdy):
    """
    Calculates the Euler-Almansi strain tensor from the given gradient data.
    Can implement more in future.
    """
    #exx = dudx - 0.5*(dudx**2+dvdx**2)
    exx = np.log(np.sqrt(1 + 2*dudx + dudx**2 + dudy**2))
    #eyy = dvdy - 0.5*(dvdy**2+dudy**2)
    eyy = np.log(np.sqrt(1 + 2*dvdy + dvdx**2 + dvdy**2))
    #exy = 0.5*(dudy + dvdx) - 0.5*((dudx*dudy)+(dvdx*dvdy))
    exy = dvdx*(1+dudx) + dudy*(1+dvdy)
    return exx,eyy,exy

exx,eyy,exy = euler_almansi_n(dudx,dudy,dvdx,dvdy)
#%%
sd.mesh_data['eyy_base'] = eyy
# %%
