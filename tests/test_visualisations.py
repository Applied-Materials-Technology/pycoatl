#%%
import pyvista as pv

#%%
standard_theme = pv.themes.DocumentTheme()
standard_theme.colorbar_orientation = 'vertical'
standard_theme.colorbar_vertical.position_x = 0.75
standard_theme.font.family = 'arial'
standard_theme.font.size = 20
standard_theme.camera = {'position': [0,0,1],'viewup':[0,1,0]}

# %%
