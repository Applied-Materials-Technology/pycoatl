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
from pycoatl.spatialdata.importsimdata import simdata_to_spatialdata
from pycoatl.spatialdata.tensorfield import rank_two_field
from pycoatl.spatialdata.tensorfield import vector_field

from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Delaunay
from scipy import interpolate
from pycoatl.datafilters.datafilters import FastFilterRegularGrid
#%%
dic_data = matchid_to_spatialdata(Path('/home/rspencer/moose_work/Viscoplastic_Creep/3P_Specimen/3P2/Exp'),
                                  Path('/home/rspencer/moose_work/Viscoplastic_Creep/3P_Specimen/3P2/Image_Avg_20.csv'),
                                  version='2024.1',loadfile_format='Image.csv')
#dic_data = matchid_to_spatialdata('/home/rspencer/moose_work/Viscoplastic_Creep/XY_Specimen/Test_XY1_Export','/home/rspencer/moose_work/Viscoplastic_Creep/XY_Specimen/Image_Test_XY1.csv',version='2024.1',loadfile_format='Image.csv')


#%%

best_file = '/home/rspencer/moose_work/Viscoplastic_Creep/3P_Specimen/3p_creep_peric_sat_3d_out.e'

exodus_reader = ExodusReader(Path(best_file))
all_sim_data = exodus_reader.read_all_sim_data()
cur_best= simdata_to_spatialdata(all_sim_data)
dic_data.align(cur_best,[0.6,1,1])

# %%
def interpolate_to_grid(data,spacing=0.2):
        """Interpolate spatial data to a regular grid with given spacing.
        Used as part of the DIC simulation.
        Primarily designed for MOOSE outputs.

        Args:
            spacing (float, optional): Grid spacing in mm. Defaults to 0.2.

        Returns:
            SpatialData: A new SpatialData instance with the interpolated data.
        """
        bounds = data.mesh_data.bounds
        # Create regular grid to interpolate to
        xr = np.linspace(bounds[0],bounds[1],int((bounds[1]-bounds[0])/spacing))
        yr = np.linspace(bounds[2],bounds[3],int((bounds[3]-bounds[2])/spacing))
        zr = bounds[5]
        x,y,z = np.meshgrid(xr,yr,zr)
        grid = pv.StructuredGrid(x,y,z)

        # Possibly want to add tag to metadata to say it's processed.
       
        data_sets_int = []
        for mesh in data.data_sets:
            result = grid.sample(mesh)
            for field in result.array_names:
                if field not in ['ObjectId','vtkGhostType','vtkValidPointMask','vtkGhostType']:
                    result[field][result['vtkValidPointMask']==False] =np.nan
            data_sets_int.append(result)

        mb_interpolated = SpatialData(data_sets_int,self._index,self._time,self._load,metadata)
        return mb_interpolated

def window_differentation(data,window_size=5):
        """Differentiate spatialdata using subwindow approach to 
        mimic DIC filter. Adds the differentiated fields into the meshes in
        the spatial data.
        Primarily intended for MOOSE FE output 


        Args:
            spatialdata (SpatialData): SpatialData instance from FE
            window_size (int, optional): Subwindow size. Defaults to 5.
        """

        def get_points_neighbours(mesh: pv.UnstructuredGrid,window_size=5)->list[int]:
            """Get the neighbouring points for a mesh.
            Initial phase of the window differentiation.
            Assumes a regular-like quad mesh. Such that surrounding each point are 
            8 others.

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

        points_list = get_points_neighbours(data.mesh_data,window_size)

        def L_Q4(x):
            return np.vstack((np.ones(x[0].shape),x[0],x[1],x[0]*x[1])).T


        def evaluate_point_dev(point_data,data):
            """
            Fit an calculate deformation gradient at each point.
            """
            window_spread = int((window_size - 1) /2)
            
            xdata = point_data[:,:2].T
            xbasis = L_Q4(xdata)
            ydata = data

            if len(ydata)<window_size**2:
                partial_dx = np.nan
                partial_dy = np.nan
            else:
                paramsQ4, r, rank, s = np.linalg.lstsq(xbasis, ydata)
                    
                px = xdata[:,int(round((window_size**2) /2))]
                partial_dx = paramsQ4[1] + paramsQ4[3]*px[1]
                partial_dy = paramsQ4[2] + paramsQ4[3]*px[0]
                
            return partial_dx, partial_dy

  
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
        
            
        dudx = np.empty((data.n_points,data.n_steps))
        dvdx = np.empty((data.n_points,data.n_steps))
        dudy = np.empty((data.n_points,data.n_steps))
        dvdy = np.empty((data.n_points,data.n_steps))

        # Get u and v data over time
        f= data.data_fields['displacement'].get_fields([0,1])
        u_all = f[0]
        v_all = f[1]

        for point in range(data.n_points):
            #point = 0
            neighbours = points_list[point]
            point_data = data.mesh_data.points[neighbours]
            u = u_all[neighbours,:]
            v= v_all[neighbours,:]
            
            dudx[point],dudy[point] = evaluate_point_dev(point_data,u)
            dvdx[point],dvdy[point] = evaluate_point_dev(point_data,v)

        exx,eyy,exy = euler_almansi(dudx,dudy,dvdx,dvdy)
        dummy = np.zeros_like(exx)
        strain = np.stack((exx,exy,dummy,exy,eyy,dummy,dummy,dummy,dummy),axis=1)
        ea_strains = rank_two_field(strain)
        data.data_fields['filter_strain'] = ea_strains

        # Update meta data
        data.metadata['dic_filter'] = True
        data.metadata['window_size'] = window_size



# %%
spacing=0.2
bounds = cur_best.mesh_data.bounds
# Create regular grid to interpolate to
xr = np.linspace(bounds[0],bounds[1],int((bounds[1]-bounds[0])/spacing))
yr = np.linspace(bounds[2],bounds[3],int((bounds[3]-bounds[2])/spacing))
zr = bounds[5]
x,y,z = np.meshgrid(xr,yr,zr)
grid = pv.StructuredGrid(x,y,z)
cur_best.mesh_data['u_int'] = cur_best.data_fields['displacement'].data[:,0,:]
cur_best.mesh_data['v_int'] = cur_best.data_fields['displacement'].data[:,1,:]
result = grid.sample(cur_best.mesh_data,1E-5)
result.plot(scalars=result['v_int'][:,-1]==0)
mask = result['v_int'][:,-1]!=0
# %%
result
# %%
cur_best.mesh_data.plot()
# %%
plt.imshow(np.reshape(result['v_int'][:,-1],(x.shape[1],x.shape[0])).T)
# %%
v_data = np.moveaxis(np.reshape(result['v_int'],(x.shape[1],x.shape[0],-1)).T,[1,2],[0,1])
ind_data = np.reshape(np.arange(len(result['v_int'][:,-1])),(x.shape[1],x.shape[0])).T
#data_stride = np.lib.stride_tricks.sliding_window_view(ind_data,[5,5])
# %%
t = np.reshape(data_stride,(-1,5,5))
#%%
ind_data = np.reshape(np.arange(len(result['v_int'])),(x.shape[1],x.shape[0]))

ind_list = []
window_size =5
levels = int((window_size -1)/2)
for i in range(ind_data.shape[0]):
    for j in range(ind_data.shape[1]):
        ind_list.append(np.ravel(ind_data[i-levels:i+levels+1,j-levels:j+levels+1]))
#%%
result=grid_mesh

def L_Q4(x):
    return np.vstack((np.ones(x[0].shape),x[0],x[1],x[0]*x[1])).T



"""
Fit an calculate deformation gradient at each point.
"""
#p = 60

dudx = np.empty((result.n_points,cur_best.n_steps))
dvdx = np.empty((result.n_points,cur_best.n_steps))
dudy = np.empty((result.n_points,cur_best.n_steps))
dvdy = np.empty((result.n_points,cur_best.n_steps))

for p in range(result.n_points):
    xdata = result.points[ind_list[p],:2].T
    xbasis = L_Q4(xdata)
    ydata = result['v_int'][ind_list[p]]
    ydata[ydata==0] = np.nan
    #print(xdata)
    if len(ydata)<window_size**2:
        partial_dx = np.nan
        partial_dy = np.nan
        #print('Here')
    else:
        paramsQ4, r, rank, s = np.linalg.lstsq(xbasis, ydata)
            
        px = xdata[:,int(round((window_size**2) /2))]
        partial_dx = paramsQ4[1] + paramsQ4[3]*px[1]
        partial_dy = paramsQ4[2] + paramsQ4[3]*px[0]
    
    dvdx[p,:] = partial_dx
    dvdy[p,:] = partial_dy

result['dvdx']= dvdx
result['dvdy']= dvdy
#print(partial_dx)
# %%
result.plot(scalars=result['dvdy'])
# %%
cur_best.plot('plastic_strain',[0,0],50)
# %%
plt.plot(result.points[:,1],result['v_int'],'.')
plt.plot(cur_best.mesh_data.points[:,1],cur_best.data_fields['displacement'].data[:,1,-1],'.')
#plt.plot(np.ravel(y),np.ravel(test_int),'.')
# %%
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
#%%
spacing=0.2
bounds = cur_best.mesh_data.bounds
# Create regular grid to interpolate to
xr = np.linspace(bounds[0],bounds[1],int((bounds[1]-bounds[0])/spacing))
yr = np.linspace(bounds[2],bounds[3],int((bounds[3]-bounds[2])/spacing))
zr = bounds[5]
x,y = np.meshgrid(xr,yr)
#test_int = interpolate.griddata(cur_best.mesh_data.points[:,:2],cur_best.data_fields['displacement'].data[:,1,-1],(x,y),'cubic')
xp,yp = excluding_mesh(cur_best.mesh_data.points[:,0], cur_best.mesh_data.points[:,1], nx=10, ny=10)
zp = np.nan + np.zeros_like(xp)
points = np.transpose(np.vstack((np.r_[cur_best.mesh_data.points[:,0],xp], np.r_[cur_best.mesh_data.points[:,1],yp])))
z = cur_best.data_fields['displacement'].data[:,1,-1]

tri = Delaunay(points)
#test_int = interpolate.CloughTocher2DInterpolator(tri,np.r_[z,zp],tol=1)
test_int = interpolate.LinearNDInterpolator(tri,np.r_[z,zp])
#test_int[mask] = np.nan
plt.contourf(x,y,np.squeeze(test_int(x,y)))
# Need a way to mask out the non-data parts. 
# Maybe dig out the KDTree code? 
# %%
x,y,z = np.meshgrid(xr,yr,zr,indexing='ij')
grid = pv.StructuredGrid(x,y,z)
result = grid.sample(cur_best.mesh_data,1E-5)
result['v_int'] = np.ravel(test_int(x,y).T)
result.plot(scalars='v_int')
# %%

# Step 1 - Interpolate
# Want to return arrays and a mesh

def interpolate_to_grid(fe_data : SpatialData,spacing : float):
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

    xp,yp = excluding_mesh(fe_data.mesh_data.points[:,0], fe_data.mesh_data.points[:,1], nx=10, ny=10)
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

    # Create pyvista mesh 
    x,y,z = np.meshgrid(xr,yr,zr)
    grid = pv.StructuredGrid(x,y,z)
    result = grid.sample(fe_data.mesh_data)
    return result, u_int, v_int
# %%
# Step 2 - Calculate the derivatives.
# Use the arrays rather than mesh.

# Create an array of neighbour indices.

def windowed_strain_calculation(grid_mesh,u_int,v_int,window_size):
    
    # Create an array of neighbour indices.
    time_steps = v_int.shape[2]
    ind_data = np.reshape(np.arange(v_int[:,:,0].size),(v_int.shape[0],v_int.shape[1]))
    ind_list = []
    levels = int((window_size -1)/2)
    for i in range(ind_data.shape[0]):
        for j in range(ind_data.shape[1]):
            # might be returning problems
            lowi = i-levels
            ind_list.append(np.ravel(ind_data[i-levels:i+levels+1,j-levels:j+levels+1]))

    def L_Q4(x):
        return np.vstack((np.ones(x[0].shape),x[0],x[1],x[0]*x[1])).T

    def evaluate_point_dev(point_data,data,window_size):
        """
        Fit an calculate deformation gradient at each point.
        """
        #window_spread = int((window_size - 1) /2)
        
        xdata = point_data[:,:2].T
        
        ydata = data

        msk = ~np.isnan(ydata[:,0])
        xbasis = L_Q4(xdata[:,msk])
        ydata = ydata[msk,:]

        if len(ydata)<1:#window_size**2:
            partial_dx = np.nan
            partial_dy = np.nan
        else:
            paramsQ4, r, rank, s = np.linalg.lstsq(xbasis, ydata)
                
            #px = xdata[:,int(round((window_size**2) /2))]
            px = xdata[:,int(round((len(ydata)) /2))]
            partial_dx = paramsQ4[1] + paramsQ4[3]*px[1]
            partial_dy = paramsQ4[2] + paramsQ4[3]*px[0]
            
        return partial_dx, partial_dy

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
        dudx[point,:],dudy[point,:] = evaluate_point_dev(point_data,u,window_size)
        dvdx[point,:],dvdy[point,:] = evaluate_point_dev(point_data,v,window_size)

    exx,eyy,exy = euler_almansi(dudx,dudy,dvdx,dvdy)
    return exx, eyy, exy

#%%
grid_mesh,u_int,v_int = interpolate_to_grid(cur_best,0.2)
exx,eyy,exy = windowed_strain_calculation(grid_mesh,u_int,v_int,5)
#%%
#Make into spatial data object
spacing = 0.2
window_size = 5
time_steps = u_int.shape[2]
u_r = np.reshape(u_int,(-1,time_steps))
v_r = np.reshape(v_int,(-1,time_steps))
dummy = np.zeros_like(u_r)
displacement = np.stack((u_r,v_r,dummy),axis=1)
data_fields = {'displacement'  :vector_field(displacement)} 
strains =np.stack((exx,exy,dummy,exy,eyy,dummy,dummy,dummy,dummy),axis=1)
data_fields['filtered_strain'] = rank_two_field(strains)
new_metadata = cur_best.metadata
new_metadata['transformations'] = {'filter' : 'fast','spacing' : spacing, 'window_size': window_size}
mb = SpatialData(grid_mesh,data_fields,new_metadata,cur_best.index,cur_best.time,cur_best.load)
#= rank_two_field(np.stack(stack,axis=1))
#%%
grid_mesh['u'] = np.reshape(u_int,(-1,217))
grid_mesh['eyy'] = eyy
# %%
plt.contourf(np.reshape(exx[:,-1],(25,100)))
#plt.contourf(v_int[:,:,-1])
# %%
window_size =3
time_steps = v_int.shape[2]
ind_data = np.reshape(np.arange(v_int[:,:,0].size),(v_int.shape[0],v_int.shape[1]))
ind_list = []
levels = int((window_size -1)/2)
for i in range(ind_data.shape[0]):
    for j in range(ind_data.shape[1]):
        ind_list.append(np.ravel(ind_data[i-levels:i+levels+1,j-levels:j+levels+1]))

#%%
point=540
neighbours = ind_list[point]
print(neighbours)
point_data = grid_mesh.points[neighbours]
t = np.reshape(u_int,(-1,217))
u = np.reshape(u_int,(-1,217))[neighbours,:]
#u = np.reshape(np.swapaxes(u_int,0,1),(-1,217))[neighbours,:]
      
plt.scatter(x=point_data[:,0],y=point_data[:,1],c=u[:,-1])
print(point_data[:,0],point_data[:,1],u[:,-1])
# %%
print(grid_mesh.points[point,:])
print(np.ravel(x)[point])
print(np.ravel(y)[point])
# %%
print(np.reshape(u_int,(-1,217))[point,-1])
print(np.reshape(np.swapaxes(u_int,0,1),(-1,217))[point,-1])
# %%
filter = FastFilterRegularGrid(0.2,5)
filtered_data = filter.run_filter(cur_best)
filtered_data.plot('filtered_strain',[1,1],-1)
# %%
