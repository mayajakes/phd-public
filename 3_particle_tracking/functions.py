import os
import xarray as xr
import numpy as np
from xgcm import Grid
from scipy import interpolate
import gsw
import warnings
from datetime import timedelta
import math
import intake
import scipy
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from parcels import JITParticle, ParticleSet, Variable, StatusCode
from parcels import AdvectionRK4

from pyproj import Transformer
from pyproj import CRS

from astropy.convolution import convolve
from astropy.convolution import Box2DKernel

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

def importFloatData(floatids):
    
    floatdir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data', 'floats')

    ema = {}
    for floatid in floatids:
        input_file = os.path.join(floatdir, 'macquarie_ema-%s_qc.nc' %floatid)
        ema[floatid] = xr.open_dataset(input_file)
        
    return ema


def cum_dist(lons, lats):
    '''Cumulative distance from the first profile (km)'''
    try:
        lats, lons = xr.DataArray(lats.data, dims = 'profile'), xr.DataArray(lons.data, dims = 'profile')
    except:
        lats, lons = xr.DataArray(lats.values, dims = 'profile'), xr.DataArray(lons.values, dims = 'profile')

    lats = lats.interpolate_na(dim ='profile')
    lons = lons.interpolate_na(dim ='profile')

    dist_diff = np.concatenate((np.array([0]), gsw.distance(lons.values, lats.values)))
    dist_diff_km = dist_diff/1000
    dist_from_start = np.nancumsum(dist_diff_km)

    distance = xr.DataArray(dist_from_start, dims = 'distance')

    # check for duplicates 
    ind = np.where(np.gradient(distance)==0)[0]
    if len(ind) > 0:
        distance[ind-1] = np.nan
        distance = distance.interpolate_na(dim ='distance')

    return distance


def map_properties(ax, t, ssh_insitu, contours = True):
    ax.coastlines(color = 'silver', zorder = 0)
    ax.add_feature(cfeature.LAND, facecolor = '#fef0d9', edgecolor='k', linewidth = 0.5, zorder = 0)
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    if contours == True:
        if type(t) == np.int64 or type(t) == int:
            adt = ssh_insitu.adt[t]
        if type(t) == str:
            adt = ssh_insitu.adt.sel(time = t, method = 'nearest')
        elif type(t) == slice:
            adt = ssh_insitu.adt.sel(time = t).mean(dim  = 'time')
            
        adt.plot.contour(levels=[0.2], colors='k', linewidths = 1.2, linestyles = '-.', transform=ccrs.PlateCarree())
        adt.plot.contour(levels=[-0.65], colors='k', linewidths = 1.2, linestyles = '-', transform=ccrs.PlateCarree())


def tsDensContour(SA, CT):
    theta = gsw.pt_from_CT(SA, CT)

    # Figure out boudaries (mins and maxs)
    smin = np.nanmin(SA) - (0.01 * np.nanmin(SA))
    smax = np.nanmax(SA) + (0.01 * np.nanmax(SA))
    tmin = np.nanmin(theta) - 3 #(0.7 * np.nanmax(theta))
    tmax = np.nanmax(theta) + 3 #(0.2 * np.nanmax(theta))
    
    # Calculate how many gridcells we need in the x and y dimensions
    xdim = int(round((smax-smin)/0.1 + 2, 0))
    ydim = int(round((tmax-tmin) + 3, 0))
    
    # Create empty grid of zeros
    dens = np.zeros((ydim,xdim))

    # Create temp and salinity vectors of appropiate dimensions
    ti = np.linspace(1,ydim-1,ydim)+tmin
    si = np.linspace(1,xdim-1,xdim)*0.1+smin
    
    # Loop to fill in grid with densities
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):
            dens[j,i] = gsw.rho(si[i], ti[j], 0) - 1000

    return ti, si, dens


def to_pdens_grid(dataArray, pdens, zdim = 'pressure', dens_interval = 0.01):
    '''Generalised function to interpolate a 2D dataArray onto an even potential density grid'''

    if zdim != list(dataArray.dims)[1]:
        # second dimension must be the vertical coordinate
        dataArray = dataArray.transpose()

    if zdim != list(pdens.dims)[1]:
        # second dimension must be the vertical coordinate
        pdens = pdens.transpose()

    dens_grid = np.mgrid[np.nanmin(pdens):np.nanmax(pdens):dens_interval]

    n = len(dataArray)
    shp = (n,len(dens_grid))
    new_var = np.nan*np.ma.masked_all(shp)

    for i in range(0, n):
            g = pdens[i,:]
            dens_ind = np.where(~np.isnan(g))[0]
            dens_values = g[dens_ind]

            if np.size(dens_values)>2:
                new_var[i,:] = interpolate.interp1d(dens_values, dataArray[i,list(dens_ind)], 
                                                                    bounds_error=False)(dens_grid)
                
    on_dens = xr.DataArray(data = new_var, dims=[dataArray.dims[0], "potential_density"], 
                                        coords = dict(potential_density=("potential_density", dens_grid)),)
    on_dens = on_dens.assign_coords(dataArray[dataArray.dims[0]].coords)
    
    return on_dens


def potentialDensity(pressure, SA, CT, p_ref = 0, anomaly = True):
    '''Potential density (anomaly) referenced to the surface.
    SA = absolute salinity
    T = in-situ temperature'''
    #conservative temperature to in-situ temperature
    # T = gsw.t_from_CT(SA, CT, pressure)
    dens = gsw.pot_rho_t_exact(SA, CT, pressure, p_ref)
    if anomaly == True:
        dens = dens - 1000
    return dens

def fullErtelPV(T, S, u, v):
    '''Following Thompson et al. (2016), neglecting terms that involve the vertical velocity (w).'''

    dimz = int(np.where(np.asarray(u.dims) == 'pressure')[0])

    [N2,p_mid] = gsw.Nsquared(S, T, T.pressure, axis = dimz) 
    b = buoyancy(T, S)
    
    #interpolate to match N2 pressure mid points
    u = u.interp(pressure = p_mid[0,0])
    v = v.interp(pressure = p_mid[0,0])
    b = b.interp(pressure = p_mid[0,0])

    f = gsw.f(u.latitude)
    f = np.tile(f, (len(u.pressure), len(u.longitude), 1)).transpose()

    z = gsw.z_from_p(u.pressure, u.latitude.mean())

    dimx = int(np.where(np.asarray(v.dims) == 'longitude')[0])
    dimy = int(np.where(np.asarray(u.dims) == 'latitude')[0])

    lons = u.longitude.data
    lats = np.tile(u.latitude.mean(), len(lons))
    dx = gsw.distance(lons, lats)[0]

    lats = u.latitude.data
    lons = np.tile(u.longitude.mean(), len(lats))
    dy = gsw.distance(lons, lats)[0]

    # horizontal velocity gradients
    dvdx = np.gradient(v)[dimx] / dx
    dudy = np.gradient(u)[dimy] / dy

    # vertical shear
    dudz = np.gradient(u)[dimz] / np.gradient(z)
    dvdz = np.gradient(v)[dimz] / np.gradient(z)

    # horizontal buoyancy gradients
    dbdx = np.gradient(b)[dimx] / dx
    dbdy = np.gradient(b)[dimy] / dy

    relative_vorticity = dvdx - dudy
    absolute_vorticity = relative_vorticity + f

    ErtelPV = absolute_vorticity*N2 
    baroclinicPV = -dvdz*dbdx + dudz*dbdy

    PV = ErtelPV + baroclinicPV

    PV = xr.DataArray(PV, dims = u.dims, coords = u.coords)
    ErtelPV = xr.DataArray(ErtelPV, dims = u.dims, coords = u.coords)
    baroclinicPV = xr.DataArray(baroclinicPV, dims = u.dims, coords = u.coords)

    return PV, ErtelPV, baroclinicPV

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def EKE(u, v, t1, t2, rolling = False, window = 30):
    # mean U and V 
    start, end = str(t1.astype('M8[D]')), str(t2.astype('M8[D]'))
    
    if rolling == True:
        print(f'rolling mean: {window} days')
        u_bar = u.rolling(time = window, center = True).mean(dim = 'time')
        v_bar = v.rolling(time = window, center = True).mean(dim = 'time')
    else:
        print('mean u and v between {} and {}'.format(start, end))
        u_bar = u.sel(time = slice(start,end)).mean(dim = 'time')
        v_bar = v.sel(time = slice(start,end)).mean(dim = 'time')
        
    # calculate EKE (deviation from mean)
    EKE = 0.5*((u-u_bar)**2 + (v-v_bar)**2)
    return EKE 

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def uvBearing(u, v):
    '''Calculates the bearing (clockwise from True North) using eastward (u) and northward (v) components of velocity'''
    theta = np.rad2deg(np.arctan2(u, v))
    # theta += 360
    theta = theta % 360
    return theta
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def xyTransform(lons, lats, coords = False):
    '''convert lat and lon to Cartesian coordinates in m'''
    crs = CRS.from_epsg(3857)
    proj = Transformer.from_crs(crs.geodetic_crs, crs)

    if coords == True:
        coords = []
        for i in range(0,len(lats)):
            xx, yy = proj.transform(lats[i], lons[i])
            coords.append([xx,yy])
        return np.asarray(coords)
    
    else:
        xx, yy = proj.transform(lats, lons)
        return xx, yy
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def sshGrad(dataArray):
    '''Sea surface height gradient from satellite altimetry. 
    Also works with other spatial data variables with dimensions latitude and longitude.'''

    lons = dataArray.longitude.data
    lats = np.tile(dataArray.latitude.mean(), len(lons))
    dx = gsw.distance(lons, lats)[0]/1000

    lats = dataArray.latitude.data
    lons = np.tile(dataArray.longitude.mean(), len(lats))
    dy = gsw.distance(lons, lats)[0]/1000

    dimx = int(np.where(np.asarray(dataArray.dims) == 'longitude')[0])
    dimy = int(np.where(np.asarray(dataArray.dims) == 'latitude')[0])

    grad_x = np.gradient(dataArray)[dimx]/dx
    grad_y = np.gradient(dataArray)[dimy]/dy
    grad_total = np.sqrt(grad_y**2 + grad_x**2)

    ssh_grad = xr.DataArray(grad_total, dims = dataArray.dims, coords = dataArray.coords)

    return ssh_grad


def ssh_std(alt_cmems, start_time = None, end_time = None):
    '''Sea surface height standard deviation, H*, used in Foppert et al. (2017) to estimate EHF.'''

    mean_ssh = alt_cmems.adt.sel(time = slice(start_time, end_time)).mean(dim = 'time')
    sum_of_squares = ((alt_cmems.adt - mean_ssh)**2).sum(dim = 'time', skipna = True)
    H = np.sqrt((1/(len(alt_cmems.time)-1) * sum_of_squares))

    return H


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def okubo_weiss(u, v):

    lons = u.longitude.data
    lats = np.tile(u.latitude.mean(), len(lons))
    dx = gsw.distance(lons, lats)[0]/1000

    lats = u.latitude.data
    lons = np.tile(u.longitude.mean(), len(lats))
    dy = gsw.distance(lons, lats)[0]/1000

    ow = np.ma.masked_all(u.shape)

    for it in range(len(u.time)):
        dimx = int(np.where(np.asarray(u[it,].dims) == 'longitude')[0])
        dimy = int(np.where(np.asarray(u[it,].dims) == 'latitude')[0])
    
        # normal and shear components of strain and Okubo-Weiss parameter
        dudx = np.gradient(u[it,])[dimx] / dx
        dvdx = np.gradient(v[it,])[dimx] / dx
        
        dudy = np.gradient(u[it,])[dimy] / dy
        dvdy = np.gradient(v[it,])[dimy] / dy
    
        # Okubo-Weiss parameter (ow)
        normal_strain = dudx - dvdy
        shear_strain = dvdx + dudy
        zeta = dvdx - dudy
        ow[it,] = normal_strain**2 + shear_strain**2 - zeta**2

    ow = xr.DataArray(ow, dims = u.dims, coords = u.coords)

    return ow
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def pearsons(data1, data2, print_info = True):
    '''Pearsons correlation'''

    # remove nans 
    nansx = np.where(np.isnan(data1))[0]
    nansy = np.where(np.isnan(data2))[0]

    nans = np.unique(np.concatenate((nansx, nansy)))

    x = np.delete(data1, nans)
    y = np.delete(data2, nans)
    
    corr, pval = pearsonr(x, y)
    if print_info == True:
        print(f'Pearsons corr: {corr:.3f}, p-value: {pval:.3}')
        
    return corr, pval
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

def horizontal_divergence(ua, va):
    shp = ua.shape

    lnln, ltlt = np.meshgrid(ua.longitude.data, ua.latitude.data)
    xx, yy = xyTransform(lnln, ltlt)
    box_kernel = Box2DKernel(3)

    divag = np.ma.masked_all(shp)

    for it in range(ua.time.size):
        # ageostrophic components
        dua_dx = np.gradient(ua[it,])[1] / np.gradient(xx)[1]
        dva_dy = np.gradient(va[it,])[0] / np.gradient(yy)[0]
        dua_dx = convolve(dua_dx, box_kernel, normalize_kernel=True)
        dva_dy = convolve(dva_dy, box_kernel, normalize_kernel=True)
        
        divag[it,] = (dua_dx + dva_dy)

    divag = xr.DataArray(divag.data, dims = ["time", "latitude", "longitude"],
                      coords=dict(time=("time", ua.time.data),
                                  latitude = ("latitude", ua.latitude.data),
                                  longitude = ("longitude", ua.longitude.data),))
    
    return divag

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def DSC(T_dens, S_dens, pdens_on_d, vert_smooth = False, x_smooth = True, dens_interval = 0.01):
    '''Calculate diapycnal spiciness curvature (DSC). 
    ''' 
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    rho, alpha, beta = gsw.rho_alpha_beta(S_dens, T_dens, 0)

    if vert_smooth == True:
        # vertical smoothing to reduce noise (Shcherbina et al., 2009)
        S_dens = S_dens.rolling(potential_density = 3, center = True, min_periods = 2).mean()
        T_dens = T_dens.rolling(potential_density = 3, center = True, min_periods = 2).mean()
        pdens_on_d = pdens_on_d.rolling(potential_density = 3, center = True, min_periods = 2).mean()

    # vertical derivative of temperature with repsect to density 
    T_z = T_dens.differentiate('potential_density')

    DSC = (2*alpha*pdens_on_d*T_z.differentiate('potential_density'))

    if x_smooth == True:
        DSC_smooth = DSC.rolling(distance = 3, center = True, min_periods = 2).mean()
        return DSC, DSC_smooth
    else:
        return DSC

#--------------------------------------------------------------------------------------------------------------------------------
def load_om2(exp, variables, start_time, end_time, freq, vels = False):
    '''Load access-om2 data'''
    catalog = intake.cat.access_nri
    
    if vels == True:
        x="xu_ocean"
        y="yu_ocean"
    else:
        x="xt_ocean"
        y="yt_ocean"       

    xarray_open_kwargs=dict(chunks={x: -1, y: -1})
    xarray_combine_by_coords_kwargs=dict(
        compat="override",
        data_vars="minimal",
        coords="minimal")

    i=0
    for var in variables:
        ds_var = catalog[exp].search(
            variable = var,
            frequency=freq).to_dask(
            xarray_open_kwargs=xarray_open_kwargs,
            xarray_combine_by_coords_kwargs=xarray_combine_by_coords_kwargs)

        ds_var = ds_var.sel(time=slice(start_time, end_time))
        
        if i == 0:
            ds = ds_var.copy()
            
        ds[var] = ds_var[var]
        i+=1

    return ds
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class AgeParticle(JITParticle):  # It is a JIT particle
    age = Variable("age", initial=0)  # Variable 'age' is added with initial value 0.

def DeleteErrorParticle(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()

def CheckError(particle, fieldset, time):
    if particle.state >= 50:  # This captures all Errors
        particle.delete()

def StuckParticles(particle, fieldset, time):
    ### If the velocity is zero, the particle has gone out of bounds (set values to nan)
    u, v = fieldset.UV[time, particle.depth, particle.lat, particle.lon, particle]
    if u*v == 0 or particle.state == StatusCode.ErrorOutOfBounds: 
        particle.temp = math.nan
        particle.sal = math.nan
        particle.U = math.nan
        particle.V = math.nan
        particle.lat = math.nan
        particle.lon = math.nan
        particle.state = StatusCode.Success


def SampleTS(particle, fieldset, time):
    particle.temp = fieldset.T[time, particle.depth, particle.lat, particle.lon]
    particle.sal = fieldset.S[time, particle.depth, particle.lat, particle.lon]

def SampleUV(particle, fieldset, time):
    # attention: samples particle velocity in units of the mesh (deg/s or m/s)
    u, v = fieldset.UV[time, particle.depth, particle.lat, particle.lon, particle]
    #deg/s to m/s
    particle.U = u * (1852 * 60) 
    particle.V = v * (1852 * 60) 
    
def release_particles(fieldset, lons, lats, time, dens, n_days, filename = None, backward = False):
    pset = ParticleSet(
                fieldset=fieldset,  # the fields that the particleset uses
                pclass=JITParticle,  # define the type of particle
                lon=lons,  # release longitudes
                lat=lats, # release latitudes
                time = time, # release times
                depth = dens
    )
    
    if filename is not None:
        output_file = pset.ParticleFile(
                    name = filename,  # the name of the output file
                    outputdt=timedelta(hours=6), # the time period between consecutive out output steps
        )
    
    if backward == True:
        dt = -timedelta(hours=6)
    else:
        dt = timedelta(hours=6)
        
    pset.execute(
                [AdvectionRK4, DeleteErrorParticle], # the kernels (which defines how particles move)
                runtime=timedelta(days=n_days),  # the total length of the run in seconds
                dt=dt,  # the timestep of the kernel in seconds
                output_file=output_file, 
    )


# def originTS(fieldset, lons, lats, z, time, n_days = 10, kernels = [AdvectionRK4, SampleTS, DeleteErrorParticle], filename = f'output/Streamline_10days.zarr'):
#     '''Backward-in-time advection to extract origin TS properties'''
#     n_particles = len(z)
#     lons = np.tile(lons, n_particles)
#     lats = np.tile(lats, n_particles)
#     time = np.tile(time, n_particles)

#     SampleParticle = JITParticle.add_variables(["temp", "sal"])
#     pset = ParticleSet(fieldset=fieldset, pclass=SampleParticle, lon=lons, lat=lats, time=time, depth = z)

#     timestep = 6

#     output_file = pset.ParticleFile(name=filename, outputdt=timedelta(hours=timestep))

#     # set dt to negative for backward advection
#     dt = -timedelta(hours=timestep)

#     pset.execute(
#         kernels,
#         runtime=timedelta(days=n_days),
#         dt=dt,
#         output_file=output_file,
#     )
    
#     ds_out = xr.open_zarr(filename)
#     origin_temp = ds_out.temp[:, -1]
#     origin_sal = ds_out.sal[:, -1]
    
#     return origin_temp.data, origin_sal.data


def satgem_toDens(satgem_on_p, gw_on_p, data_vars = ['ugw', 'vgw', 'sal', 'temp'], save_to = 'satgem_on_d.nc'):

    dens = gsw.pot_rho_t_exact(satgem_on_p.sal, satgem_on_p.temp, satgem_on_p.pressure, 0) - 1000

    # First create an xgcm grid object
    grid = Grid(gw_on_p, coords={'Z': {'center':'pressure'}}, periodic=False)

    # target density grid
    dens_interval = 0.01
    dens_grid = np.mgrid[np.round(np.nanmin(dens), 2):np.round(np.nanmax(dens), 2):dens_interval]

    dims = ["time", "density", "latitude", "longitude"]

    # transform to density
    new_ds = {}
    for var in data_vars:
        try:
            var_dens = grid.transform(gw_on_p[var], 'Z', dens_grid, target_data=dens).rename({'sal': 'density'})
        except:
            var_dens = grid.transform(satgem_on_p[var], 'Z', dens_grid, target_data=dens).rename({'sal': 'density'})
        new_ds[var] = (dims, np.transpose(var_dens.data, [0, 3, 1, 2]))
            

    satgem_on_d = xr.Dataset(new_ds, coords = dict(latitude=var_dens.latitude.data,
                                                   longitude=var_dens.longitude.data,
                                                   density=var_dens.density.data,
                                                   time=var_dens.time.data))
    
    satgem_on_d.to_netcdf(save_to)


def NanStuckParticles(ds_out):
    '''Set the trajectory to nan if the particle starts in a region with no data, giving weird interpolation values.'''
    ## set all trajectory values to nan if release lat lon value is nan
    nans = np.where(np.isnan(ds_out.lat[:, 0].values))[0]
    ds_out.temp[nans, :] = np.nan
    ds_out.sal[nans, :] = np.nan
    ds_out.lat[nans, :] = np.nan
    ds_out.lon[nans, :] = np.nan

    ## set all trajectory values to nan if release T-S value is nan
    nans = np.where(np.isnan(ds_out.lat[:, -1].values))[0]
    ds_out.temp[nans, :] = np.nan
    ds_out.sal[nans, :] = np.nan
    ds_out.lat[nans, :] = np.nan
    ds_out.lon[nans, :] = np.nan

    ## set all trajectory values to nan if release T-S value is nan
    nans = np.where(np.isnan(ds_out.temp[:, 0].values))[0]
    ds_out.temp[nans, :] = np.nan
    ds_out.sal[nans, :] = np.nan
    ds_out.lat[nans, :] = np.nan
    ds_out.lon[nans, :] = np.nan

    ## set all trajectory values to nan if release T-S value is nan
    nans = np.where(np.isnan(ds_out.temp[:, -1].values))[0]
    ds_out.temp[nans, :] = np.nan
    ds_out.sal[nans, :] = np.nan
    ds_out.lat[nans, :] = np.nan
    ds_out.lon[nans, :] = np.nan

    ## set to nan if the density of the seed field value is lighter than the density of the released particles
    d_min = min(ds_out.z.values.flatten())
    rho = gsw.rho(ds_out.sal[:, 0], ds_out.temp[:, 0], 0) - 1000
    
    err = np.where(rho < 27)[0]
    ds_out.temp[err, :] = np.nan
    ds_out.sal[err, :] = np.nan
    ds_out.lat[err, :] = np.nan
    ds_out.lon[err, :] = np.nan

    ## set to nan if the density of the origin field value is lighter than the density of the released particles
    d_min = min(ds_out.z.values.flatten())
    rho = gsw.rho(ds_out.sal[:, -1], ds_out.temp[:, -1], 0) - 1000
    
    err = np.where(rho < 27)[0]
    ds_out.temp[err, :] = np.nan
    ds_out.sal[err, :] = np.nan
    ds_out.lat[err, :] = np.nan
    ds_out.lon[err, :] = np.nan

    return ds_out



def col_timeseries(ds_out, interp_data, i, t_ind, dens):
    '''ds_out = particle trajectory dataset
        interp_data = data to be interpolated onto particle trajectories
        i = trajectory column index
        t_ind = time index 
        dens = density levels
    '''
    
    times = xr.DataArray(ds_out.time[0, :t_ind+1].values, dims = ['time'], 
                         coords = dict(time = ('time', ds_out.time[0, :t_ind+1].values)))
    
    shp = (len(times), len(dens))
    arr = np.ma.masked_all(shp).data*np.nan

    lons = ds_out.lon[i:i+len(dens), :t_ind+1]
    lats = ds_out.lat[i:i+len(dens), :t_ind+1]

    col_data = xr.DataArray(arr.copy(), dims = ['time', 'potential_density'])

    j = 0
    for d in dens:
        d_interp = interp_data.sel(density = d, method = 'nearest').interp(time = times, 
                                                                           longitude = lons[j], 
                                                                           latitude = lats[j])

        d_interp_1d = [d_interp[it, it].data for it in range(len(times))]

        col_data[:,j] = np.asarray(d_interp_1d)
        j+=1

    col_data = col_data.assign_coords({'potential_density': dens, 'time': times.values})
    
    return col_data



def col_trajectory_avg(ds_out, interp_data, col_idx, t_ind, dens):
    
    shp = (len(col_idx), len(dens))
    arr = np.ma.masked_all(shp).data*np.nan

    i = 0
    for col in col_idx:
        col_data = col_timeseries(ds_out, interp_data, col, t_ind, dens)

        arr[i] = col_data.mean(dim = 'time')
        i+=1
    
    return arr


def origin_TS(ds_out, col_idx, t_ind, dens, xdim = 'latitude'):
    shp = (len(col_idx), len(dens))
    T = np.ma.masked_all(shp).data*np.nan
    S = np.ma.masked_all(shp).data*np.nan
    
    lat = ds_out.lat[col_idx, 0].values
    
    origin_temp = ds_out.temp[:, t_ind-1]
    origin_sal = ds_out.sal[:, t_ind-1]
    rho = gsw.rho(origin_sal, origin_temp, 0) - 1000
    err = np.where(rho < 27.05)[0]
    origin_temp[err] = np.nan
    origin_sal[err] = np.nan
    
    i = 0
    for col in col_idx:
        T[i] = origin_temp[col:col+len(dens)].values
        S[i] = origin_sal[col:col+len(dens)].values
        i+=1
    
    T = xr.DataArray(T, dims = [xdim, 'potential_density']).assign_coords({'potential_density': dens})
    S = xr.DataArray(S, dims = [xdim, 'potential_density']).assign_coords({'potential_density': dens})
    
    if xdim == 'latitude':
        T = T.assign_coords({'latitude':lat})
        S = S.assign_coords({'latitude':lat})
    
    return T, S

def distance_travelled(ds_out, col_idx, t_ind, dens):
    
    distance = []
    for i in range(len(ds_out.lon)):
        lon1 = ds_out.lon[i, 0:t_ind].values
        lat1 = ds_out.lat[i, 0:t_ind].values
        dist_diff = np.concatenate((np.array([0]), gsw.distance(lon1, lat1)))/1000
        distance.append(np.nancumsum(dist_diff)[-1])
    distance = np.asarray(distance)
    
    shp = (len(col_idx), len(dens))
    dist_travelled = np.ma.masked_all(shp).data*np.nan
    i = 0
    for col in col_idx:
        dist_travelled[i] = distance[col:col+len(dens)]
        i+=1
    
    lat = ds_out.lat[col_idx, 0].values
    dist_travelled = xr.DataArray(dist_travelled, dims = ['latitude', 'potential_density']).assign_coords({'potential_density': dens})
    dist_travelled = dist_travelled.assign_coords({'latitude':lat})
    
    return dist_travelled
    

def rate_of_dsc_generation(ds_out, i, t_ind, dens, fill_value = 99):
    times = xr.DataArray(ds_out.time[0, :t_ind+1].values, dims = ['time'], 
                         coords = dict(time = ('time', ds_out.time[0, :t_ind+1].values)))

    T = xr.DataArray(ds_out.temp[i:i+len(dens), :t_ind+1].T.values, 
                     dims = ['time', 'potential_density'], 
                     coords = {'potential_density': dens, 'time': times.values})
    S = xr.DataArray(ds_out.sal[i:i+len(dens), :t_ind+1].T.values, 
                     dims = ['time', 'potential_density'], 
                     coords = {'potential_density': dens, 'time': times.values})

    dens_2d = np.repeat(dens[np.newaxis, :], times.shape, axis=0)
    col_dsc = DSC(T, S, dens_2d, vert_smooth = False, x_smooth = False)
    
    # how long does it take to generate depth-average DSC magnitude grater than 1 m^3/kg?
    avg_dsc = abs(col_dsc).mean(dim = 'potential_density')
    ind = np.where(avg_dsc > 1)[0]
    
    if len(ind) > 0:
        n_days = (avg_dsc.time[0] - avg_dsc.time[ind[0]]).data.astype('timedelta64[D]')
    else:
        n_days = fill_value
    
    return n_days


def all_cols_dsc_generation(ds_out, col_idx, t_ind, dens):
    shp = (len(col_idx), len(dens))
    arr = np.ma.masked_all(shp).data*np.nan
    
    i = 0
    for col in col_idx:
        col_data = rate_of_dsc_generation(ds_out, col, t_ind, dens)

        arr[i] = col_data
        i+=1
    
    return arr

#-----------------------------------------------------------------------------------------------------------------------------------------

def grid_data(data, grid_size = 1, latlim = [-70, -35], lonlim = [-180, 185], min_obs = 5):
    lon_bins = np.arange(lonlim[0], lonlim[1], grid_size)
    lat_bins = np.arange(latlim[0], latlim[1], grid_size)

    lon_mid = (lon_bins + np.gradient(lon_bins)/2)[:-1]
    lat_mid = (lat_bins + np.gradient(lat_bins)/2)[:-1]
    
    #remove nans
    data_nonan = data.where(~np.isnan(data), drop = True)
        
    # grid using scipy
    data_mean = scipy.stats.binned_statistic_2d(data_nonan.lon, data_nonan.lat, data_nonan, 
                                                bins = [lon_bins, lat_bins], statistic='mean')[0]

    n_obs = scipy.stats.binned_statistic_2d(data_nonan.lon, data_nonan.lat, data_nonan, 
                                            bins = [lon_bins, lat_bins], statistic='count')[0]

    # must have at least 5 observations per bin
    mask = np.ma.masked_where(n_obs.transpose() < min_obs, n_obs.transpose())
    
    # to xarray
    data_mean = xr.DataArray(data_mean.transpose(), dims = ['latitude', 'longitude'], 
                                                    coords = dict(latitude = (['latitude'], lat_mid), 
                                                                  longitude = (['longitude'], lon_mid)))
    
    n_obs = xr.DataArray(n_obs.transpose(), dims = ['latitude', 'longitude'], 
                                            coords = dict(latitude = (['latitude'], lat_mid), 
                                                          longitude = (['longitude'], lon_mid)))
    
    data_mean.data[mask.mask] = np.nan
    n_obs.data[mask.mask] = np.nan
    
    return data_mean, n_obs

#-----------------------------------------------------------------------------------------------------------------------------------------

def mask_PFZ(griddata, ADT, limits = [-0.65, 0.2], pad = 0.1):
    PF, nSAF = limits
    
    # mask the PFZ for averaging
    adt_interp = ADT.interp(latitude = griddata.latitude, longitude = griddata.longitude)
    mask1 = np.ma.masked_where((adt_interp < PF-pad), adt_interp)
    mask2 = np.ma.masked_where((adt_interp > nSAF+pad), adt_interp)

    # grid_data_copy = griddata.transpose().copy()
    grid_data_copy = griddata.copy()
    grid_data_copy.data[mask1.mask] = np.nan
    grid_data_copy.data[mask2.mask] = np.nan
    
    return grid_data_copy

#-----------------------------------------------------------------------------------------------------------------------------------------

def plot_circmupolar(data, masked_data = None, adt = None, adt_contours = [-0.65, 0.2], contourf = True, clabel = '', **kwargs):
    extent = [-180, 180, -90, -36]

    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes(projection=ccrs.SouthPolarStereo())

    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor= None, facecolor='k')

    ax.add_feature(land)
    ax.set_extent(extent, ccrs.PlateCarree())

    gls = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.6,
                        color='gray', alpha=0.6, zorder=10, xlocs=range(-180,180,30))

    gls.xlabel_style = {'size': 10, 'color': 'k'}
    gls.ylabel_style = {'size': 0, 'color': 'k'}

    r_extent = 6521311
    r_extent *= 1.005  

    # set the plot limits
    ax.set_xlim(-r_extent, r_extent)
    ax.set_ylim(-r_extent, r_extent)

    # Prep circular boundary
    circle_path = mpath.Path.unit_circle()
    circle_path = mpath.Path(circle_path.vertices.copy() * r_extent, circle_path.codes.copy())

    #set circular boundary
    ax.set_boundary(circle_path)
    # draw longitude labels
    plt.draw() 

    if masked_data is not None:
        if contourf == True:
            data.plot.contourf(ax = ax, alpha = 0.4, transform = ccrs.PlateCarree(), add_colorbar = False, **kwargs)
            masked_data.plot.contourf(ax = ax, transform = ccrs.PlateCarree(), cbar_kwargs = dict(label = clabel), **kwargs)
                                     
        else:
            data.plot(ax = ax, alpha = 0.4, transform = ccrs.PlateCarree(), add_colorbar = False, **kwargs)
            masked_data.plot(ax = ax, transform = ccrs.PlateCarree(), cbar_kwargs = dict(label = clabel), **kwargs)
    else:
        if contourf == True:
            data.plot.contourf(ax = ax, transform = ccrs.PlateCarree(), cbar_kwargs = dict(label = clabel), **kwargs)
        else:
            data.plot(ax = ax, transform = ccrs.PlateCarree(), cbar_kwargs = dict(label = clabel), **kwargs)
        
    if adt is not None:
        adt.plot.contour(ax = ax, levels=adt_contours[0], colors='w', linewidths = 0.8, linestyles = '-', transform=ccrs.PlateCarree())
        adt.plot.contour(ax = ax, levels=adt_contours[1], colors='w', linewidths = 0.8, linestyles = '-.', transform=ccrs.PlateCarree())
        
    return fig, ax
        