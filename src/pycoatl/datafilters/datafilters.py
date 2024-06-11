from abc import ABC, abstractmethod
from typing import Sequence
from typing import Self
from pycoatl.spatialdata.spatialdata import SpatialData
import numpy as np
from numpy._typing import NDArray
from typing import Sequence
from pycoatl.spatialdata.importsimdata import simdata_to_spatialdata
from pycoatl.spatialdata.tensorfield import rank_two_field
from pycoatl.spatialdata.tensorfield import vector_field

from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Delaunay
from scipy import interpolate
import pyvista as pv

class DataFilterBase(ABC):
    """Abstract Base Class for creating data filters.
    Such as fast spatial filter, synthetic DIC etc.
    """

    @abstractmethod
    def run_filter(self,data : SpatialData)-> SpatialData:
        pass


class FastFilterRegularGrid(DataFilterBase):
    
    def __init__(self,grid_spacing=0.2,window_size=5,strain_tensor = 'euler', exclude_limit = 30):
        
        self._grid_spacing = grid_spacing
        self._window_size = 5
        self._strain_tensor = strain_tensor
        self._exclude_limit = exclude_limit

        self.available_tensors = {'log-euler-almansi':FastFilterRegularGrid.euler_almansi}
    
    @staticmethod
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
    
    @staticmethod
    def small_strain(dudx,dudy,dvdx,dvdy):
        """
        Calculates the Euler-Almansi strain tensor from the given gradient data.
        Can implement more in future.
        """
        exx = dudx
        eyy = dvdy
        exy = 0.5*(dudy + dvdx)
        return exx,eyy,exy
    
    @staticmethod
    def excluding_mesh(x, y, nx=30, ny=30):
        """
        Construct a grid of points, that are some distance away from points (x, 
        """
        dx = x.ptp() / nx
        dy = y.ptp() / ny
        xp, yp = np.mgrid[x.min()-2*dx:x.max()+2*dx:(nx+2)*1j,
                            y.min()-2*dy:y.max()+2*dy:(ny+2)*1j]
        xp = xp.ravel()
        yp = yp.ravel()
        # Use KDTree to answer the question: "which point of set (x,y) is the
        # nearest neighbors of those in (xp, yp)"
        tree = KDTree(np.c_[x, y])
        dist, j = tree.query(np.c_[xp, yp], k=1)
        # Select points sufficiently far away
        m = (dist > np.hypot(dx, dy))
        return xp[m], yp[m]

    @staticmethod
    def interpolate_to_grid(fe_data : SpatialData,spacing : float, exclude_limit: float):
        """Interpolate the FE data onto a regular grid with spacing.

        Args:
            fe_data (SpatialData): FE Data.
            spacing (float): Spacing for the regular grid on which the data will be interpolated

        Returns:
            _type_: _description_
        """

        bounds = fe_data.mesh_data.bounds
        # Create regular grid to interpolate to
        xr = np.linspace(bounds[0],bounds[1],int((bounds[1]-bounds[0])/spacing))
        yr = np.linspace(bounds[2],bounds[3],int((bounds[3]-bounds[2])/spacing))
        zr = bounds[5]
        x,y = np.meshgrid(xr,yr,indexing='ij')
        # Add Nans to the array for outline the edges of the specimen
        
        if exclude_limit >0:
            xp,yp = FastFilterRegularGrid.excluding_mesh(fe_data.mesh_data.points[:,0], fe_data.mesh_data.points[:,1], nx=exclude_limit, ny=exclude_limit)
            zp = np.nan + np.zeros_like(xp)
            points = np.transpose(np.vstack((np.r_[fe_data.mesh_data.points[:,0],xp], np.r_[fe_data.mesh_data.points[:,1],yp])))
        
            tri = Delaunay(points)
            u_int = np.empty((x.shape[0],x.shape[1],fe_data.n_steps))
            v_int = np.empty((x.shape[0],x.shape[1],fe_data.n_steps))
            for i in range(fe_data.n_steps):
                zu = fe_data.data_fields['displacement'].data[:,0,i]
                zv = fe_data.data_fields['displacement'].data[:,1,i]
                u_int[:,:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zu,zp])(x,y)
                v_int[:,:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zv,zp])(x,y)
        
        else: # Don't use excluding mesh approach
            points = np.transpose(np.vstack((np.r_[fe_data.mesh_data.points[:,0]], np.r_[fe_data.mesh_data.points[:,1]])))
        
            tri = Delaunay(points)
            u_int = np.empty((x.shape[0],x.shape[1],fe_data.n_steps))
            v_int = np.empty((x.shape[0],x.shape[1],fe_data.n_steps))
            for i in range(fe_data.n_steps):
                zu = fe_data.data_fields['displacement'].data[:,0,i]
                zv = fe_data.data_fields['displacement'].data[:,1,i]
                u_int[:,:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zu])(x,y)
                v_int[:,:,i] = interpolate.LinearNDInterpolator(tri,np.r_[zv])(x,y)
 

        # Create pyvista mesh 
        x,y,z = np.meshgrid(xr,yr,zr)
        grid = pv.StructuredGrid(x,y,z)
        result = grid.sample(fe_data.mesh_data)
        return result, u_int, v_int
    
    @staticmethod
    def L_Q4(x:NDArray)->NDArray:
        """Reorganise x Data to perform least-squares

        Args:
            x (NDArray): _description_

        Returns:
            NDArray: _description_
        """
        return np.vstack((np.ones(x[0].shape),x[0],x[1],x[0]*x[1])).T

    @staticmethod
    def evaluate_point_dev(point_data,data,window_size):
        """
        Fit an calculate deformation gradient at each point.
        """
        #window_spread = int((window_size - 1) /2)
        
        xdata = point_data[:,:2].T
        xbasis = FastFilterRegularGrid.L_Q4(xdata)
        ydata = data

        if len(ydata)<np.power(window_size-2,2):#window_size**2:
            partial_dx = np.nan
            partial_dy = np.nan
        else:
            paramsQ4, r, rank, s = np.linalg.lstsq(xbasis, ydata)

            px = xdata[:,int(round((len(ydata)) /2))]
            partial_dx = paramsQ4[1] + paramsQ4[3]*px[1]
            partial_dy = paramsQ4[2] + paramsQ4[3]*px[0]
            
        return partial_dx, partial_dy

    @staticmethod
    def windowed_strain_calculation(grid_mesh,u_int,v_int,window_size):
    
        # Create an array of neighbour indices.
        time_steps = v_int.shape[2]
        ind_data = np.reshape(np.arange(v_int[:,:,0].size),(v_int.shape[0],v_int.shape[1]))
        ind_list = []
        levels = int((window_size -1)/2)
        for i in range(ind_data.shape[0]):
            for j in range(ind_data.shape[1]):
                ind_list.append(np.ravel(ind_data[max([0,i-levels]):min([ind_data.shape[0],i+levels+1]),max([0,j-levels]):min([ind_data.shape[1],j+levels+1])]))

        dudx = np.empty((grid_mesh.n_points,v_int.shape[2]))
        dvdx = np.empty((grid_mesh.n_points,v_int.shape[2]))
        dudy = np.empty((grid_mesh.n_points,v_int.shape[2]))
        dvdy = np.empty((grid_mesh.n_points,v_int.shape[2]))

        u_r = np.reshape(u_int,(-1,time_steps))
        v_r = np.reshape(v_int,(-1,time_steps))
        for point in range(grid_mesh.n_points):

            neighbours = ind_list[point]
            point_data = grid_mesh.points[neighbours]
            u = u_r[neighbours,:]
            v = v_r[neighbours,:]
            dudx[point,:],dudy[point,:] = FastFilterRegularGrid.evaluate_point_dev(point_data,u,window_size)
            dvdx[point,:],dvdy[point,:] = FastFilterRegularGrid.evaluate_point_dev(point_data,v,window_size)

        return dudx,dudy,dvdx,dvdy
    
    def run_filter(self,data : SpatialData)-> SpatialData:
       
        # Interpolate the data to the new grid       
        grid_mesh,u_int,v_int = FastFilterRegularGrid.interpolate_to_grid(data,self._grid_spacing,self._exclude_limit)
        
        # Perform the windowed strain calculation
        # Only Q4 for now
        dudx,dudy,dvdx,dvdy = FastFilterRegularGrid.windowed_strain_calculation(grid_mesh,u_int,v_int,self._window_size)

        # Crate new SpatialData instance to return
        time_steps = u_int.shape[2]
        u_r = np.reshape(u_int,(-1,time_steps))
        v_r = np.reshape(v_int,(-1,time_steps))
        dummy = np.zeros_like(u_r)
        displacement = np.stack((u_r,v_r,dummy),axis=1)
        data_fields = {'displacement'  :vector_field(displacement)} 


        # Apply strain tensor 
        if self._strain_tensor == 'euler':
            exx,eyy,exy = FastFilterRegularGrid.euler_almansi(dudx,dudy,dvdx,dvdy)
        elif self._strain_tensor == 'small':
            exx,eyy,exy = FastFilterRegularGrid.small_strain(dudx,dudy,dvdx,dvdy)

        strains =np.stack((exx,exy,dummy,exy,eyy,dummy,dummy,dummy,dummy),axis=1)
        data_fields['filtered_strain'] = rank_two_field(strains)
        new_metadata = data.metadata
        new_metadata['transformations'] = {'filter' : 'fast','spacing' : self._grid_spacing, 'window_size': self._window_size, 'order' : 'Q4'}
        mb = SpatialData(grid_mesh,data_fields,new_metadata,data.index,data.time,data.load)
        return mb