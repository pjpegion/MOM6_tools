#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import xesmf as xe
from cartopy.util import add_cyclic_point
from scipy import ndimage as nd
import sys
import os


# In[2]:

def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    #import numpy as np
    #import scipy.ndimage as nd

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]


# code is adpated from MOM6:MOM_state_initialization
def adjusthToFitBathymetry(ssh, h,bathy):
    hTolerance = 1.0e-10 #<  Tolerance to exceed adjustment criteria [Z ~> m]
    Angstrom_Z=1.0e-10
    nx=h.shape[2]
    ny=h.shape[1]
    nz=h.shape[0]
    eta=np.zeros([nz+1,ny,nx])
    eta[0,:,:]=ssh
    for k in range(nz):
        eta[k+1,:,:]=eta[k,:,:]-h[k,:,:]
    dz=eta[-1,:,:]+bathy
    for k in range(nz+1):
        eta[k,:,:]=np.where(-eta[k,:,:] > (bathy[:,:] + hTolerance),-bathy,eta[k,:,:])
    for k in range(nz-1,0,-1):
        h[k,:,:]=np.where(eta[k,:,:] < (eta[k+1,:,:] + Angstrom_Z),Angstrom_Z,eta[k,:,:] - eta[k+1,:,:])
    for i in range(nx):
        for j in range(ny):

    #   The whole column is dilated to accommodate deeper topography than
    # the bathymetry would indicate.
            if -eta[nz,j,i] < (bathy[j,i] - hTolerance):
                if eta[0,j,i] <= eta[nz,j,i]:
                    #for k in range(nz):
                    h[:,j,i] = (eta[1,j,i] + bathy[j,i]) / np.float(nz)
                else:
                    dilate = (eta[0,j,i] + bathy[j,i]) / (eta[0,j,i] - eta[nz,j,i])
                    #for k in range(nz):
                    h[:,j,i] = h[:,j,i] * dilate
    return h,eta


# In[3]:


# main program   

# specify an output resolution
ires = 'ORAS5'
ores = "mx100"
nargs=len(sys.argv[:])
print(nargs)
if (nargs != 2): 
   print('need to specify a date in YYYYMMDD format')
   os._exit(3)
cdate=sys.argv[1]
if (len(cdate)!=8):
   print('need to specify a date in YYYYMMDD format')
   os._exit(10)
print('processing ',cdate)
# specify a location to use
nemsrc     = "/scratch2/BMC/gsienkf/Philip.Pegion/UFS-coupled/ICS/source/WeightGen/TTout/"
# specifiy output directory
outdir     = "/scratch2/BMC/gsienkf/Philip.Pegion/UFS-coupled/ICS/"+ores+"/"+cdate+"/"
if not os.path.exists(outdir):
   os.mkdir(outdir)
output_file= 'ORAS5.mx100.ic.nc'
# ocean model restart location
#dirsrc     = "/scratch2/NCEPDEV/stmp1/Jeffrey.S.Whitaker/oras5/"
dirsrc     = "/scratch2/BMC/gsienkf/Philip.Pegion/reanalysis/ocean/"
# target resolution bathymetry files
bathy_file='/scratch2/BMC/gsienkf/Philip.Pegion/UFS-coupled/1deg_basedir/INPUT/topog.nc'
edits_file='/scratch2/BMC/gsienkf/Philip.Pegion/UFS-coupled/1deg_basedir/INPUT/topo_edits_011818.nc'
# open existing file to get time-stamp
res0=xr.open_dataset('/scratch2/BMC/gsienkf/Philip.Pegion/UFS-coupled/ICS/mx100/2011040100/MOM6.mx100.ic.nc')
# OPEN ESMF weights file
wgt_dir='/scratch2/BMC/gsienkf/Philip.Pegion/UFS-coupled/ICS/source/WeightGen/TTout/'
grid_file_out=xr.open_dataset(wgt_dir+'tripole.mx100.nc')
wgtsfile_t_to_t = wgt_dir+'ORAS5.Ct.to.mx100.Ct.bilinear.nc'
wgtsfile_t_to_u = wgt_dir+'tripole.mx100.Ct.to.Cu.bilinear.nc'
wgtsfile_t_to_v = wgt_dir+'tripole.mx100.Ct.to.Cv.bilinear.nc'

# OPEN 1/4 degree initial condition files
oras5=xr.open_dataset(dirsrc+"grepv2_daily_%s.nc" %cdate)

# rename lat and lons from grid files for ESMF interpolation
oras5_t_grid=oras5.rename({'longitude': 'lon', 'latitude': 'lat'})
mx100_t_grid=grid_file_out.rename({'lonCt': 'lon', 'latCt': 'lat'})
mx100_v_grid=grid_file_out.rename({'lonCv': 'lon', 'latCt': 'lat'})
mx100_u_grid=grid_file_out.rename({'lonCu': 'lon', 'latCu': 'lat'})

# rotation angles for u and v currents
ang_out= grid_file_out.anglet

rg_tt = xe.Regridder(oras5_t_grid, mx100_t_grid, 'bilinear',periodic=True,reuse_weights=True, filename=wgtsfile_t_to_t)
rg_tu = xe.Regridder(mx100_t_grid, mx100_u_grid, 'bilinear',periodic=True,reuse_weights=True, filename=wgtsfile_t_to_u)
rg_tv = xe.Regridder(mx100_t_grid, mx100_v_grid, 'bilinear',periodic=True,reuse_weights=True, filename=wgtsfile_t_to_v)


# In[5]:


nx=len(grid_file_out.ni.values)
ny=len(grid_file_out.nj.values)
nz=len(oras5.depth.values)

# define land masks
lmask_in = xr.where(xr.ufuncs.isnan(oras5['thetao_oras'][0,0,:,:]), 0.0, 1.0)
lmask_out = xr.where(grid_file_out['wet'] > 0.0, 1.0, 0.0)
# interpolate mask to new grid
lmask_interp0 = rg_tt(lmask_in.values)
lmask_interp = np.where(lmask_interp0 < 0.99, 0.0, 1.0)

#3d-copies
lmask_interp_3d=np.zeros([nz,ny,nx])
lmask_out_3d = np.zeros([nz,ny,nx])

for i in range(nz):
    lmask_interp_3d[i,:,:]=lmask_interp[:,:]
    lmask_out_3d[i,:,:]=grid_file_out['wet'].values
# fill in land points
for z in range(nz):
    tmparr=add_cyclic_point(oras5['thetao_oras'][0,z].values,axis=1)
    tnew=fill(tmparr)
    oras5['thetao_oras'][0,z]=tnew[:,0:-1]
    tmparr=add_cyclic_point(oras5['so_oras'][0,z].values,axis=1)
    tnew=fill(tmparr)
    oras5['so_oras'][0,z]=tnew[:,0:-1]
    tmparr=add_cyclic_point(oras5['uo_oras'][0,z].values,axis=1)
    tnew=fill(tmparr)
    oras5['uo_oras'][0,z]=tnew[:,0:-1]
    tmparr=add_cyclic_point(oras5['vo_oras'][0,z].values,axis=1)
    tnew=fill(tmparr)
    oras5['vo_oras'][0,z]=tnew[:,0:-1]

tmparr=add_cyclic_point(oras5['zos_oras'][0].values,axis=1)
tnew=fill(tmparr)
oras5['zos_oras'][0]=tnew[:,0:-1]
# interpolate values to new grid
new_t   = rg_tt(oras5['thetao_oras'].values)
new_s   = rg_tt(oras5['so_oras'].values)
new_sfc = rg_tt(oras5['zos_oras'].values)
new_urot= rg_tt(oras5['uo_oras'].values)
new_vrot= rg_tt(oras5['vo_oras'].values)

# rotate currents back to grid relatvie 
new_ut =   new_urot*np.cos(ang_out.values) - new_vrot*np.sin(ang_out.values)
new_vt =   new_vrot*np.cos(ang_out.values) + new_urot*np.sin(ang_out.values)

# re-stagger currents
new_u = rg_tu(new_ut)
new_v = rg_tv(new_vt)


#determine lowest valid value in ORAS5 data
bathy=xr.open_dataset(bathy_file)
edits=xr.open_dataset(edits_file)
# edit bathymetry
for i in edits.nEdits.values:
    bathy.depth[edits.jEdit[i].values,edits.iEdit[i].values]=abs(edits.zEdit[i].values)

bathy.depth[:,:]=xr.where(bathy.depth < 9.5,9.5,bathy.depth)
bathy.depth[:,:]=xr.where(bathy.depth > 6500,6500,bathy.depth)
new_bathy=bathy.depth[:,:].values
# compute layer interface depths
z_i=np.zeros(nz+1)
z_l=oras5.depth[:].values
for i in range(nz-1):
    z_i[i+1]=z_l[i]+(z_l[i+1]-z_l[i])/2.0
z_i[-1]=z_l[-1]+(z_l[-1]-z_l[-2])/2.0
# adjust columns in areas of shoaling 
for i in range(nz-1):
    new_t[0,i+1]=np.where(new_bathy > z_i[i+1],new_t[0,i+1],new_t[0,i])
    new_s[0,i+1]=np.where(new_bathy > z_i[i+1],new_s[0,i+1],new_s[0,i])
    new_u[0,i+1]=np.where(new_bathy > z_i[i+1],new_u[0,i+1],new_u[0,i])
    new_v[0,i+1]=np.where(new_bathy > z_i[i+1],new_v[0,i+1],new_v[0,i])

# In[7]:


#generate new_h
h1d=np.zeros(nz)
h1d[:]=z_i[1:]-z_i[0:-1]
new_h=np.zeros([1,nz,ny,nx])
for i in range(nz):
    new_h[0,i,:,:]=h1d[i]
# adjust heights
new_h[0,:,:,:],eta=adjusthToFitBathymetry(new_sfc[0,:,:],new_h[0,:,:,:],new_bathy)

#mask land
new_t = np.where(lmask_out_3d == 0.0, 0.0, new_t)
new_t[0,-1]=new_t[0,-2]
new_u = np.where(lmask_out_3d == 0.0, 0.0, new_u)
new_u[0,-1]=new_u[0,-2]
new_v = np.where(lmask_out_3d == 0.0, 0.0, new_v)
new_v[0,-1]=new_v[0,-2]
new_s = np.where(lmask_out_3d == 0.0, 0.0, new_s)
new_s[0,-1]=new_s[0,-2]
new_h = np.where(lmask_out_3d == 0.0, 0.0, new_h)
new_sfc = np.where(lmask_out == 0.0, 0.0, new_sfc)

# In[9]:


# set up output file grid
new_lath=grid_file_out.latCt.mean(axis=1).values
new_lonh=grid_file_out.lonCt.mean(axis=0).values
new_latq=grid_file_out.latCv.mean(axis=1).values
new_lonq=grid_file_out.lonCu.mean(axis=0).values
new_time=res0.Time.values
# create xarray DataArrays
da_t = xr.DataArray(new_t,coords=({'lath' : (['lath'], new_lath), 'lonh' : (['lonh'], new_lonh), 'Layer': (['Layer'], oras5['depth'].values),                     'Time' : (['Time'], new_time)}), dims=['Time','Layer','lath','lonh'])
da_s = xr.DataArray(new_s,coords=({'lath' : (['lath'], new_lath), 'lonh' : (['lonh'], new_lonh), 'Layer': (['Layer'], oras5['depth'].values),                      'Time' : (['Time'], new_time)}), dims=['Time','Layer','lath','lonh'])
da_h = xr.DataArray(new_h,coords=({'lath' : (['lath'], new_lath), 'lonh' : (['lonh'], new_lonh), 'Layer': (['Layer'], oras5['depth'].values),                      'Time' : (['Time'], new_time)}), dims=['Time','Layer','lath','lonh'])
da_sfc=xr.DataArray(new_sfc,coords=({'lath' : (['lath'], new_lath), 'lonh' : (['lonh'], new_lonh), 'Time' : (['Time'], new_time)}),                      dims=['Time','lath','lonh'])
da_u = xr.DataArray(new_u,coords=({'lath' : (['lath'], new_lath), 'lonq' : (['lonq'], new_lonq), 'Layer': (['Layer'], oras5['depth'].values),                      'Time' : (['Time'], new_time)}), dims=['Time','Layer','lath','lonq'])
da_v = xr.DataArray(new_v,coords=({'latq' : (['latq'], new_latq), 'lonh' : (['lonh'], new_lonh), 'Layer': (['Layer'], oras5['depth'].values),                      'Time' : (['Time'], new_time)}), dims=['Time','Layer','latq','lonh'])
# create xarray DataSets
ds_t=da_t.to_dataset(name='Temp')
ds_s=da_s.to_dataset(name='Salt')
ds_h=da_h.to_dataset(name='h')
ds_sfc=da_sfc.to_dataset(name='sfc')
ds_u=da_u.to_dataset(name='u')
ds_v=da_v.to_dataset(name='v')
# merge variables
ds_out=xr.merge([ds_t,ds_s,ds_h,ds_sfc,ds_u,ds_v])
# write to file
ds_out.to_netcdf(outdir+output_file,unlimited_dims='Time')

