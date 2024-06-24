# Python translation of JMAN
# Originally written in matlab

#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Analytical Approach 
# JMAN - Analytical displacement data.
# By Prof. T.H. Becker, tbecker@sun.ac.za, Stellenbosch University.
# Free to use, please reference Becker et al "An approach to calculate the
# J-integral by digital image correlation displacement field measurement"
# FFEMS 2012.
# Converted to python by R.Spencer UKAEA
 
# Define material properties.
E = 1                  # Young's Modulus
v = 0.3                # Poisson ratio
G = E/(2*(1 + v))      # shear modulus
KI = 1.                 # Stress Intensity Factor (SIF)
J = (KI**2)/E             # Equivalent energy release rate
k = (3-v)/(1+v)        # Constant

# Analytical field (replace section with DIC displacment data).
# Displacement field - Anderson 3rd edition
lin = np.linspace(-1,1,100)   # Specify size and spacing of data points
gridsize = lin[1]-lin[0]
posX,posY = np.meshgrid(lin,lin)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

r,th = cart2pol(posX,posY)
uX = KI/(2*G)*np.sqrt(r/(2*np.pi))*(np.cos(th/2)*(k-np.cos(th)))
uY = KI/(2*G)*np.sqrt(r/(2*np.pi))*(np.sin(th/2)*(k-np.cos(th)))

## Strain, stress and strain energy computation. 
# Displacement gradient
uXY, uXX  = np.gradient(uX,gridsize)     
uYY, uYX  = np.gradient(uY,gridsize)
# Strain tensor 
eXX=np.array(uXX,copy=True)
eXY=1/2*(uXY+uYX)
eYY=np.array(uYY,copy=True)
# Stress tensor - linear elastic, isotropic material
sXX = (E/(1-v**2))*(uXX+v*uYY)
sYY = (E/(1-v**2))*(v*uXX+uYY)
sSS = (E/(1-v**2))*(1-v)/2*(uXY+uYX)
# Elastic strain energy
W = 0.5*(eXX*sXX+eYY*sYY+2*eXY*sSS)

#plt.contourf(eXX)

## Line Integral.
# Generate normal vector
nX = np.ones(posX.shape)

nX = np.flipud(np.tril(np.flipud(np.tril(nX)))) - np.flipud(np.triu(np.flipud(np.triu(nX))))

nY = np.array(np.transpose(nX), copy=True)

# Remove duplicates
nX[(nX*nY!=0)] = 0
# Line Integral
dS = np.ones(uXX.shape)*gridsize

JL = -(W*nX-((nX*sXX+nY*sSS)*uXX+(nX*sSS+nY*sYY)*uYX))*dS

JL[np.isnan(JL)] = 0

#plt.contourf(JL,100)

## Area Integral.
# Generate q field
celw = 5;               # Width of area contour. Has to be an odd number.
dQdx = np.ones(posX.shape)/(gridsize*celw)
dQdx = np.flipud(np.tril(np.flipud(np.tril(dQdx))))-np.flipud(np.triu(np.flipud(np.triu(dQdx))))
dQdy = np.array(np.transpose(dQdx),copy=True)
# Remove duplicates
dQdx[dQdx*dQdy!=0] = 0
# Area integral
dA = np.ones(uXX.shape)*gridsize**2
JA = ((sXX*uXX+sSS*uYX-W)*dQdx+(sYY*uYX+sSS*uXX)*dQdy)*dA
JA[np.isnan(JA)] = 0

## Integrate.
# Make contour indexing field (assumes fields are square).
mid = np.floor([(posX.shape[0]/2)-1,(posX.shape[1]/2)-1])
print(mid)
a,b = np.meshgrid(np.arange(0,uX.shape[0]),np.arange(0,uX.shape[0]))

linecon=np.round(np.max(np.dstack((np.abs(a-np.min(mid)-1),np.abs(b-np.min(mid)-1))),2)+1)
# Mask dubious high strain data around crack faces. Note, this is always
# necessary, as DIC (as well as as strain calculation) assumes a continues
# field. This mask width will have to be increased for real DIC data.

mask = np.ones(posX.shape);
mask[int(mid[0]):int(mid[1])+2,0:int(mid[1])+1] = 0

# Summation of integrals
Jl = np.empty(int(mid[0])-1)
Ja = np.empty(int(mid[0])-1)

for ii in range(1,int(mid[0])-1):
    Jl[ii] = np.sum(np.sum(mask[linecon==ii]*JL[linecon==ii]))
    areacon  = (linecon>=(ii-np.floor(celw/2)))*(linecon<=(ii+np.floor(celw/2))) 
    Ja[ii] = np.sum(np.sum(mask*JA*areacon))

# Equivalent SIF
Kl = np.sqrt(Jl*E)
Ka = np.sqrt(Ja*E)

fig = plt.figure()
ax= fig.add_subplot()
ax.plot([0,(mid[0]-1)],[KI,KI])
ax.plot(Kl)
ax.plot(Ka)

ax.set_ylabel('SIF (MPa \surdm)')
ax.set_xlabel('Contour (#)')
ax.grid()

#%% Method version of code.
def py_jman(E,v,uX,uY,xg,yg,mask_width,window_size):
    """
    Python version of Jman algorithm. Originally - # JMAN - Analytical displacement data.
    # By Prof. T.H. Becker, tbecker@sun.ac.za, Stellenbosch University.
    # Free to use, please reference Becker et al "An approach to calculate the
    # J-integral by digital image correlation displacement field measurement"
    # FFEMS 2012.
    # Converted to python by R.Spencer UKAEA
    Requires input data to be interpolated to a grid with the crack tip at the centre.
    """
    
    ## Strain, stress and strain energy computation. 
    # Displacement gradient
    
    gridsize = xg[0,1]-xg[0,0]
    
    if 1==0:
        uXY, uXX  = np.gradient(uX,gridsize)     
        uYY, uYX  = np.gradient(uY,gridsize)
    
    # MatchID type strain 
    else:
        def lagrange_Q4(x,a,b,c,d):
            return a + b*x[0] + c*x[1] + d*x[0]*x[1]

        def lagrange_Q9(x,a,b,c,d,e,f,g,h,i):
            return lagrange_Q4(x,a,b,c,d) + e*(x[0]**2) + f*(x[1]**2) + g*x[1]*x[0]**2 + h*x[0]*(x[1]**2) + i*(x[0]**2)*(x[1]**2)

        def partial_diff_Q9(x,a,b,c,d,e,f,g,h,i):
            partial_dx = b + d*x[1] + 2*e*x[0] + 2*g*x[0]*x[1] + h*x[1]**2 + 2*i*x[0]*(x[1]**2)
            partial_dy = c + d*x[0] + 2*f*x[1] + g*x[0]**2 + 2*h*x[0]*x[1] + 2*i*x[1]*(x[0]**2)
            return partial_dx, partial_dy

        def evaluate_point(xg,yg,data_map,window_centre,window_size):
            """
            Fit an calculate deformation gradient at each point.
            """
            window_spread = int((window_size - 1) /2)
            x = xg[window_centre[0]-window_spread:window_centre[0]+window_spread+1,window_centre[1]-window_spread:window_centre[1]+window_spread+1]
            y = yg[window_centre[0]-window_spread:window_centre[0]+window_spread+1,window_centre[1]-window_spread:window_centre[1]+window_spread+1]
            #print(x)
            z = data_map[window_centre[0]-window_spread:window_centre[0]+window_spread+1,window_centre[1]-window_spread:window_centre[1]+window_spread+1]

            xdata = np.vstack((x.ravel(), y.ravel()))
            ydata = z.ravel()

            paramsQ9, params_covariance = optimize.curve_fit(lagrange_Q9, xdata, ydata,p0=[0,0,0,0,0,0,0,0,0])

            partial_dx, partial_dy = partial_diff_Q9(xdata[:,int(round((window_size**2) /2))],*paramsQ9)

            return partial_dx, partial_dy

        #window_size = 3
        window_spread = int((window_size - 1) /2)
        uXX = np.empty((xg.shape[0]-(2*window_spread),xg.shape[1]-(2*window_spread)))
        uXY = np.empty((xg.shape[0]-(2*window_spread),xg.shape[1]-(2*window_spread)))
        uYX = np.empty((xg.shape[0]-(2*window_spread),xg.shape[1]-(2*window_spread)))
        uYY = np.empty((xg.shape[0]-(2*window_spread),xg.shape[1]-(2*window_spread)))
        for i in range(window_spread,xg.shape[0]-window_spread):
            for j in range(window_spread,xg.shape[1]-window_spread):
                uXX[i-window_spread,j-window_spread],uXY[i-window_spread,j-window_spread] = evaluate_point(xg,yg,uX,[i,j],window_size)
                uYX[i-window_spread,j-window_spread],uYY[i-window_spread,j-window_spread] = evaluate_point(xg,yg,uY,[i,j],window_size)

                

    #Testing
    uXYt, uXXt  = np.gradient(uX,gridsize) 
    uYYt, uYXt  = np.gradient(uY,gridsize) 
    
    #fig = plt.figure()
    #ax=fig.add_subplot(121)
    #ax.imshow(uYX)
    #ax2=fig.add_subplot(122)
    #ax2.imshow(uYXt)

    # Strain tensor 
    eXX=np.array(uXX,copy=True)
    eXY=1/2*(uXY+uYX)
    eYY=np.array(uYY,copy=True)
    # Stress tensor - linear elastic, isotropic material
    sXX = (E/(1-v**2))*(uXX+v*uYY)
    sYY = (E/(1-v**2))*(v*uXX+uYY)
    sSS = (E/(1-v**2))*((1-v)/2)*(uXY+uYX)
    # Elastic strain energy
    W = 0.5*(eXX*sXX+eYY*sYY+2*eXY*sSS)

    ## Line Integral.
    # Generate normal vector
    nX = np.ones(uXX.shape)

    nX = np.flipud(np.tril(np.flipud(np.tril(nX)))) - np.flipud(np.triu(np.flipud(np.triu(nX))))

    nY = np.array(np.transpose(nX), copy=True)

    # Remove duplicates
    nX[(nX*nY!=0)] = 0
    # Line Integral
    dS = np.ones(uXX.shape)*gridsize

    JL = -(W*nX-((nX*sXX+nY*sSS)*uXX+(nX*sSS+nY*sYY)*uYX))*dS

    JL[np.isnan(JL)] = 0

    ## Area Integral.
    # Generate q field
    celw = 5;               # Width of area contour. Has to be an odd number.
    dQdx = np.ones(uXX.shape)/(gridsize*celw)
    dQdx = np.flipud(np.tril(np.flipud(np.tril(dQdx))))-np.flipud(np.triu(np.flipud(np.triu(dQdx))))
    dQdy = np.array(np.transpose(dQdx),copy=True)
    # Remove duplicates
    dQdx[dQdx*dQdy!=0] = 0
    # Area integral
    dA = np.ones(uXX.shape)*gridsize**2
    JA = ((sXX*uXX+sSS*uYX-W)*dQdx+(sYY*uYX+sSS*uXX)*dQdy)*dA
    JA[np.isnan(JA)] = 0

    ## Integrate.
    # Make contour indexing field (assumes fields are square).
    mid = np.floor([(uXX.shape[0]/2)-1,(uXX.shape[1]/2)-1])
    a,b = np.meshgrid(np.arange(0,uXX.shape[0]),np.arange(0,uXX.shape[0]))

    linecon=np.round(np.max(np.dstack((np.abs(a-np.min(mid)-1),np.abs(b-np.min(mid)-1))),2)+1)

    # Mask dubious high strain data around crack faces. Note, this is always
    # necessary, as DIC (as well as as strain calculation) assumes a continues
    # field. This mask width will have to be increased for real DIC data.

    mask = np.ones(uXX.shape);
    #mask_width= 1
    mask[int(mid[0])+1 -mask_width:int(mid[1])+1+mask_width,0:int(mid[1])+1] = 0

    # Summation of integrals

    Jl = np.empty(int(mid[0])-1)
    Ja = np.empty(int(mid[0])-1)
    
    for ii in range(1,int(mid[0])-1):
        Jl[ii] = np.sum(np.sum(mask[linecon==ii]*JL[linecon==ii]))
        areacon  = (linecon>=(ii-np.floor(celw/2)))*(linecon<=(ii+np.floor(celw/2))) 
        Ja[ii] = np.sum(np.sum(mask*JA*areacon))

    #plt.imshow(JL)
    # Equivalent SIF
    Kl = np.sqrt(Jl*E)
    Ka = np.sqrt(Ja*E)
    
    return(Jl,Ja,Kl,Ka)

