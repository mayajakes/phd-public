import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import scipy
import gsw
import math

from scipy.signal import savgol_filter
import pyproj
import warnings

from astropy.convolution import convolve
from astropy.convolution import Box2DKernel

import src.interpolation as interp
import src.stats as stats
import src.settings as settings
import src.velocities as vel
import src.importData as imports

def findRSperiod(float_num):
    '''Find the profile index where the float rapid sampling ends.'''
    nan_index = np.where(np.isnan(float_num.hpid) == True)
    over250 = np.where(float_num.profile > 251)
    rs_end = np.intersect1d(nan_index, over250)
    if len(rs_end) == 0:
        rs = int(float_num.hpid[-1].data)
    else:
        rs = rs_end[0]
    return slice(0,rs)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def cum_dist(lons, lats):
    '''Cumulative distance from the first profile (km)'''
    # use lat an lon positions corresponding to the start of each profile (e.g. surface position for down profiles and bottom of down profile for up profiles)?
    lats, lons = xr.DataArray(lats.data, dims = 'profile'), xr.DataArray(lons.data, dims = 'profile')
    lats = lats.interpolate_na(dim ='profile')
    lons = lons.interpolate_na(dim ='profile')

    dist_diff = np.concatenate((np.array([0]), gsw.distance(lons.values, lats.values)))
    dist_diff_km = dist_diff/1000
    dist_from_start = np.nancumsum(dist_diff_km)

    distance = xr.DataArray(dist_from_start, dims = 'distance')

    return distance

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def tsDensContour(SA, CT):
    '''Density grid for T-S diagram'''
    # Figure out boudaries (mins and maxs)
    smin = np.nanmin(SA) - (0.01 * np.nanmin(SA))
    smax = np.nanmax(SA) + (0.01 * np.nanmax(SA))
    tmin = np.nanmin(CT) - (0.5 * np.nanmax(CT))
    tmax = np.nanmax(CT) + (0.2 * np.nanmax(CT))
    
    # Calculate how many gridcells we need in the x and y dimensions
    xdim = int(round((smax-smin)/0.1 + 2, 0))
    ydim = int(round((tmax-tmin) + 3, 0))
    
    # Create empty grid of zeros
    dens = np.zeros((ydim,xdim))

    # Create temp and salt vectors of appropiate dimensions
    ti = np.linspace(1,ydim-1,ydim)+tmin
    si = np.linspace(1,xdim-1,xdim)*0.1+smin
    
    # Loop to fill in grid with densities
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):
            dens[j,i] = gsw.rho(si[i], ti[j], 0) - 1000

    return ti, si, dens

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def speed(u, v):
    '''Calculate current speed (m/s)
    INPUT: 
    u = zonal velocity (m/s)
    v = meridional velocity (m/s)'''
    return np.sqrt(u**2 + v**2)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def eddyKineticEnergy(u, v, u_bar, v_bar):
    '''Calculate eddy kinetic energy
    INPUT: 
    u = zonal velocity (m/s)
    v = meridional velocity (m/s)
    u_bar, v_bar = mean u and v'''

    # ke = 0.5*(u**2 + v**2)
    # mke = 0.5*(u_bar**2 + v_bar**2)
    return 0.5*((u-u_bar)**2 + (v-v_bar)**2)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def satellite_eke(ugos, vgos, start_time, end_time):
    '''Calculate eddy kinetic energy (EKE) from satellite geostrophic velocities.
    INPUTS:
    ugos, vgos = surface geostrophic velocities derived from satellite sea surface height data
    start_time, end_time = start and end time for calculation of mean velocities (u_bar and v_bar)'''

    start, end = str(start_time.astype('M8[D]')), str(end_time.astype('M8[D]'))
    print('mean u and v between {} and {}'.format(start, end))

    # mean U and V 
    u_bar = ugos.sel(time = slice(start,end)).mean(dim = 'time')
    v_bar = vgos.sel(time = slice(start,end)).mean(dim = 'time')
    # calculate EKE (deviation from mean)
    EKE = eddyKineticEnergy(ugos, vgos, u_bar, v_bar)

    return EKE


# ------------------------------------------------------------------------------------------------------------------------------------------------------

def calcEKE(float_num, floatid, alt_ds = None, altimetry = True, floats = False, abs_v = None, interp_to_flt = False, smooth_vels = False):
    '''Calculate eddy kinetic energy EKE from u and v (either from float data or altimetry)'''
    rs = findRSperiod(float_num)

    if altimetry == True:
        # find 3 year mean surface horizontal velocities from 2 years prior to float deployment
        # changing the mean window doesn't seem to affect the value (little interannual)
        start_time = float_num.time[0].values - np.timedelta64(365*2,'D')
        end_time = start_time + np.timedelta64(365*3,'D')

        u, v = alt_ds.ugos, alt_ds.vgos # multiply by 100 to convert m/s to cm/s

        if interp_to_flt == True:
            print('mean u and v between {} and {}'.format(start, end))
            start, end = str(start_time.astype('M8[D]')), str(end_time.astype('M8[D]'))
            # surface altimetry u and v at float locations 
            u_alt = u.interp(latitude=float_num.latitude, longitude=float_num.longitude)
            v_alt = v.interp(latitude=float_num.latitude, longitude=float_num.longitude)
            # mean surface horizontal velocities at float positions
            u_bar = u_alt.sel(time = slice(start,end)).mean(dim = 'time')
            v_bar = v_alt.sel(time = slice(start,end)).mean(dim = 'time')
            # interpolate to find surface altimetry u and v at locations and times of the float profiles
            u = u_alt.interp(time = float_num.time)
            v = v_alt.interp(time = float_num.time)
            # calculate EKE (deviation from 1-yr mean)
            eke = eddyKineticEnergy(u, v, u_bar, v_bar)

            EKE = []
            for i in range(0, len(float_num.profile)):
                value = eke.isel(time = i, latitude = i, longitude = i).values
                EKE.append(value.tolist())
        else:
            EKE = satellite_eke(u, v, start_time, end_time)

    if floats == True:
        # use absolute velocities interpolated onto potential density 
        u = interp.varToDens(abs_v.u_abs, float_num  = float_num, floatid = floatid)
        v = interp.varToDens(abs_v.v_abs, float_num  = float_num, floatid = floatid)
        # u, v = vel.setAbsVelToNan(floatid, u), vel.setAbsVelToNan(floatid, v)
        # u, v = u_dens*100, v_dens*100 # convert m/s to cm/s

        if smooth_vels == True:
            u = vel.smooth_prof_by_prof(u, window = 75, print_info = False)
            v = vel.smooth_prof_by_prof(u, window = 75, print_info = False)
        
        # rapid sampling mean
        u_bar = u[rs].mean(dim = 'distance',skipna = True)
        v_bar = v[rs].mean(dim = 'distance',skipna = True)

        EKE = eddyKineticEnergy(u, v, u_bar, v_bar)

    return EKE

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def sshGrad(dataArray):
    '''Sea surface height gradient from satellite altimetry. 
    Also works with other spatial data variables with dimensions latitude and longitude.'''
    # lnln, ltlt = np.meshgrid(dataArray.longitude.data, dataArray.latitude.data)
    # xx, yy = xyTransform(lnln, ltlt, coords = False) 
    # grad_x = np.gradient(dataArray)[1] / np.gradient(xx)[1]
    # grad_y = np.gradient(dataArray)[0] / np.gradient(yy)[0]

    lons = dataArray.longitude.data
    lats = np.tile(dataArray.latitude.mean(), len(lons))
    dx = gsw.distance(lons, lats)[0]

    lats = dataArray.latitude.data
    lons = np.tile(dataArray.longitude.mean(), len(lats))
    dy = gsw.distance(lons, lats)[0]

    dimx = int(np.where(np.asarray(dataArray.dims) == 'longitude')[0])
    dimy = int(np.where(np.asarray(dataArray.dims) == 'latitude')[0])

    grad_x = np.gradient(dataArray)[dimx]/dx
    grad_y = np.gradient(dataArray)[dimy]/dy
    grad_total = np.sqrt(grad_y**2 + grad_x**2)

    if len(np.gradient(dataArray)) > 2:
        # convert to xarray format
        ssh_grad = xr.DataArray(data = grad_total, dims = ["time", "latitude", "longitude"],coords = dict(time = (["time"], dataArray.time.data), longitude=(["longitude"], 
        dataArray.longitude.data),latitude=(["latitude"], dataArray.latitude.data),))
    else:
        # convert to xarray format
        ssh_grad = xr.DataArray(data = grad_total, dims = ["latitude", "longitude"],coords = dict(longitude=(["longitude"], 
        dataArray.longitude.data),latitude=(["latitude"], dataArray.latitude.data),))
    
    return ssh_grad

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def potentialDensity(pressure, SA, CT, p_ref = 0, anomaly = True):
    '''Potential density (anomaly) referenced to the surface'''
    dens = gsw.pot_rho_t_exact(SA, CT, pressure, p_ref)
    if anomaly == True:
        dens = dens - 1000
    return dens


# ------------------------------------------------------------------------------------------------------------------------------------------------------

def MLD(pdens, criteria = 0.05, pref = 10, pref_max = 30, return_drho = False):
    '''Calculate mixed layer depth using criteria from Dove et al. (2021):  density difference greater than 0.05 kg/m3 
    from a 10 dbar (surface) reference level.
    Input: xarray DataArray of potential density'''

    mld = []
    drho = xr.zeros_like(pdens)*np.nan

    for i in range(0,len(pdens)):
        ind_nonan = np.where(~np.isnan(pdens[i]))[0]
        pdens_nonan = pdens[i,ind_nonan]

        if len(pdens_nonan) == 0: # empty profile
            mld.append(np.nan)
        # if the first non nan value is at a depth greater than pref_max, set as nan
        elif pdens_nonan[0].pressure.data > pref_max:
            mld.append(np.nan)
        else:
            # find the pressure level at which the density difference from 10dbar reference is greater than 0.05 kg/m3.
            pd0 = pdens_nonan.sel(pressure = pref, method = 'nearest')
            dens_diff = pdens[i] - pd0
            drho[i] = dens_diff
            mask = (dens_diff >= criteria) & (dens_diff.pressure > pd0.pressure)
            
            if len(pdens[i].pressure[mask]) == 0:
                mld.append(np.nan)
            else:
                mld.append(pdens[i].pressure[mask][0].data.tolist())
            
    mld = xr.DataArray(np.asarray(mld), dims = pdens.dims[0], coords = pdens[pdens.dims[0]].coords)

    if return_drho == True:
        return mld, drho
    else:
        return mld

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def MLD_float(float_num, floatid, plot = True):
    '''Calculate mixed layer depth using criteria from Dove et al. (2021):  density difference greater than 0.05 kg/m3 
    from a 10 dbar (surface) reference level.'''

    CT, SA = settings.remove_bad_T_S(float_num, floatid)
    pdens = potentialDensity(float_num.pressure, SA, CT, p_ref = 0)

    mld = MLD(pdens, criteria = 0.05, pref = 10)

    if plot == True:
        rs = findRSperiod(float_num)
        dist = cum_dist(float_num.longitude, float_num.latitude)

        fig, ax = plt.subplots(figsize = (10,3))
        ax.scatter(dist[rs], mld[rs], c = 'grey', s= 15)
        ax.plot(dist[rs], mld[rs], c = 'k', alpha = 0.6)

        settings.tickLocations(ax)
        if floatid == 8490: 
            settings.tickLocations(ax, major = 200)

        ax.set_ylim(0,200)
        ax.invert_yaxis()
        ax.grid(True)
        ax.set_title(floatid)
        ax.set_ylabel('MLD (m)')
        ax.set_xlabel('distance (km)')

    return mld

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def MLD_on_dens(pdens, criteria = 0.05, pref = 10, dz = 0):
    # dz = pressure value added to bottom of mixed layer

    mld = MLD(pdens, criteria = criteria, pref = pref)
    
    dens_mld = []
    for j in range(0, len(pdens)):
        ind = np.where(pdens.pressure == mld[j]+dz)[0].tolist()
        if len(ind) == 0:
            dens_mld.append(np.nan)
        else:
            dens_mld.append(pdens[j,ind].values[0])

    dens_mld = xr.DataArray(dens_mld, dims = pdens.dims[0], coords = pdens[pdens.dims[0]].coords)
            
    return dens_mld

# ------------------------------------------------------------------------------------------------------------------------------------------------------
    
def dynamicHeight(CT, SA, p = 500, p0 = 1500, steric = False):
    '''Calculates dynamic height or steric height referenced to 1500 dbar.
    Outputs: 
    1. dynamic height & dynamic height at 500 dbar relative to 1500 dbar
    or
    2. steric height & steric height at 50 dbar'''
    pressure = CT.pressure

    if p0 == 1500:
        p2 = pressure[-1] - 100
    else:
        p2 = pressure[-1]
    
    # taking off the surface (top 10 m) and the bottom 100 m 
    SA = SA.sel(pressure = slice(10,p2))
    CT = CT.sel(pressure = slice(10,p2))
    pres = pressure.sel(pressure = slice(10,p2))

    pres = np.tile(pres,(len(CT), 1))

    dist = CT.distance

    dyn_m = gsw.geo_strf_dyn_height(SA, CT, pres, p_ref=p0, axis=1)
    const_grav = 9.7963  # Griffies, 2004.
    steric_height = dyn_m/const_grav

    steric_h = xr.DataArray(data=steric_height.data, dims=["distance", "pressure"], coords=dict(
        distance=(["distance"], dist.data),
        pressure=(["pressure"], pres[0].data)),)

    dyn_m = xr.DataArray(data=dyn_m.data, dims=["distance", "pressure"], coords=dict(
        distance=(["distance"], dist.data),
        pressure=(["pressure"], pres[0].data)),)

    # steric height at 50 dbar
    steric_50 = steric_h.sel(pressure = 50)
    # 500 dbar referenced to 1500 dbar
    dyn_m_ref = dyn_m.sel(pressure = p)/10 # divide by 10 to convert to dyn m (1 dyn m = 10 m2/s2)

    if steric == True:
        return steric_h, steric_50
    else:
        return dyn_m, dyn_m_ref

# ------------------------------------------------------------------------------------------------------------------------------------------------------
def N2(CT, SA, lat, smooth = False, window = 75):
    '''Output: Buoyancy frequency (N squared) at pressure midpoints'''
    f = gsw.f(lat)
    p = CT.pressure
    f = np.tile(f, (len(p)-1,1)).transpose()

    [N2,p_midarray_n2] = gsw.Nsquared(SA, CT, p, axis = 1) # axis = dimension along which pressure increases
    
    N2 = xr.DataArray(N2.data, dims = CT.dims, coords = CT[CT.dims[0]].coords)
    N2 = N2.assign_coords({"pressure": p_midarray_n2[0]})
    # N2 = N2.interp(pressure = CT.pressure)

    if smooth == True:
        N2 = vel.smooth_prof_by_prof(N2, window = window)

    return N2

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def N2_float(float_num, floatid, smooth = False, window = 75, by_dist = True):
    '''Output: Buoyancy frequency (N squared) at pressure midpoints'''
    f = gsw.f(float_num.latitude)
    f = np.tile(f, (len(float_num.pressure)-1,1)).transpose()
    CT, SA = settings.remove_bad_T_S(float_num, floatid)

    [N2,p_midarray_n2] = gsw.Nsquared(SA, CT, float_num.pressure, axis = 1) # axis = dimension along which pressure increases
    
    if by_dist == True:
        dist = cum_dist(float_num.latitude, float_num.longitude)
        N2 = xr.DataArray(data = N2, dims = ["distance", "pressure"],coords = dict(pressure=(["pressure"], 
            p_midarray_n2[0]),distance=(["distance"], dist.data),))
    else:
        N2 = xr.DataArray(data = N2, dims = ["profile", "pressure"],coords = dict(pressure=(["pressure"], 
            p_midarray_n2[0]),profile=(["profile"], float_num.profile.data),))

    if smooth == True:
        N2 = vel.smooth_prof_by_prof(N2, window = window)

    return N2

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def relativeVort(float_num, altimetry, interp_to_float = True):
    '''Calculate relative vorticity from satellite altimetry using the gradient wind balance'''
    start = float_num.time.values[0]
    end = float_num.time.values[-1]
    start_time = str(start.astype('M8[D]'))
    end_time = str(end.astype('M8[D]'))

    lon = slice(145,175)
    lat = slice(-60,-50) 
    t = slice(start_time, end_time)

    adt = altimetry.adt.sel(longitude = lon, time = t)

    gradvel = gradientWind(adt)
    qgvb = vortBalance(gradvel)

    if interp_to_float == False:
        return qgvb.zeta
    else:
        vort = interp.interpToFloat(float_num, qgvb.zeta)
        return vort

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def OkuboWeiss(float_num, altimetry, interp_to_float = False):
    start = float_num.time.values[0] - np.timedelta64(7, 'D')
    end = float_num.time.values[-1] + np.timedelta64(7, 'D')
    start_time = str(start.astype('M8[D]'))
    end_time = str(end.astype('M8[D]'))

    lon = slice(145,175)
    lat = slice(-60,-50) 
    t = slice(start_time, end_time)

    adt = altimetry.adt.sel(longitude = lon, latitude = lat, time = t)

    gradvel = gradientWind(adt)
    qgvb = vortBalance(gradvel)

    if interp_to_float == False:
        return qgvb.okubo_weiss

    else:
        ow = interp.interpToFloat(float_num, qgvb.okubo_weiss)
        return ow

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def calc_dx_dy(float_num, coord = True):
    '''Calculate dy and dx (relative to the lat lon of the first profile) from lat and lon coordinates'''
    
    # Smooth trajectory
    lons, lats = interp.gaussianFilter(float_num.longitude), interp.gaussianFilter(float_num.latitude)
    
    lon0 = np.floor(lons[0])
    lat0 = np.floor(lats[0])
    
    xx = []
    yy = []
    coords = []
    
    for i in range(0,len(lats)):
        lat1 = lats[i]
        lon1 = lons[i]
        
        dx = (lon1-lon0)*40000*math.cos((lat1+lat0)*math.pi/360)/360
        dy = (lat1-lat0)*40000/360
        
        coords.append([dx,dy])

        xx.append(dx)
        yy.append(dy)
            
    if coord == False:
        return np.array(xx), np.array(yy)
    else:
        return np.array(coords)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def xyTransform(lons, lats, coords = False):
    '''Convert lon and lat to x and y (m)'''

    crs = pyproj.CRS.from_epsg(3857)
    proj = pyproj.Transformer.from_crs(crs.geodetic_crs, crs)

    if coords == True:
        coords = []
        for i in range(0,len(lats)):
            xx, yy = proj.transform(lats[i], lons[i])
            coords.append([xx,yy])
        return np.asarray(coords)
    
    else:
        xx, yy = proj.transform(lats, lons)
        return xx, yy

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def floatCurvature(float_num, transform = True, remove_outliers = False, smooth_gps = False):
    '''Calcualte the curvature of the trajectory using dy and dx coordindates, following Bower & Rossby (1989)'''
    lons, lats = float_num.longitude, float_num.latitude
    time = float_num.time

    if smooth_gps == True:
        lons, lats = interp.gaussianFilter(float_num.longitude), interp.gaussianFilter(float_num.latitude)
        t = time.astype(float)
        # set NaT to nan before smoothing
        nat_ind = np.where(np.isnat(time))[0]
        t[nat_ind] = np.nan
        time = interp.gaussianFilter(t).astype('datetime64[ns]')

    if transform == True: # using different transormation from lon lat to x an y
        coordinates = xyTransform(lons, lats, coords = True) 
    else:
        coordinates = calc_dx_dy(float_num)

    dt = np.gradient(time).astype('timedelta64[s]')
    x_t = np.gradient(coordinates[:, 0])/dt.astype(float)
    y_t = np.gradient(coordinates[:, 1])/dt.astype(float)

    # Smooth first derivatives
    x_t = savgol_filter(x_t, 9, 3, axis=-1, mode='interp')
    y_t = savgol_filter(y_t, 9, 3, axis=-1, mode='interp')

    xx_t = np.gradient(x_t)/dt.astype(float)
    yy_t = np.gradient(y_t)/dt.astype(float)

    # Smooth second derivatives
    # xx_t = savgol_filter(xx_t, 13, 3, axis=-1, mode='interp')
    # yy_t = savgol_filter(yy_t, 13, 3, axis=-1, mode='interp')
    
    curvature = []
    
    for i in range(0, len(lons)):
        a = x_t[i] * yy_t[i] -  y_t[i] * xx_t[i] 
        b = ((x_t[i] * x_t[i]) + (y_t[i] * y_t[i]))**(3/2)

        if b == 0:
            b = np.nan
            
        c = a/b
        curvature.append(c)

    if remove_outliers == True:
        # Outliers remove using Tukey's box plot method
        curvature = stats.detectOutliers(np.asarray(curvature), remove = True)[0]
    
    return np.array(curvature)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def radiusOfCurvature(float_num, floatid, remove_outliers = False):
    '''Another method to extract the curvature of the float trajectory using the radius of curvature.'''
    
    rs = findRSperiod(float_num)
    lnln, ltlt = np.meshgrid(float_num.longitude.data, float_num.latitude.data)

    if floatid == 8490:
        lnln, ltlt = interp.gaussianFilter(float_num.longitude, ind = rs), interp.gaussianFilter(float_num.latitude, ind = rs)
    else:
        lnln, ltlt = interp.gaussianFilter(float_num.longitude), interp.gaussianFilter(float_num.latitude)

    xx, yy = xyTransform(lnln, ltlt)

    # first derivative
    dydx = np.gradient(yy) / np.gradient(xx)
    dydx = savgol_filter(dydx, 13, 3, axis=-1, mode='interp')

    # second derivative
    d2ydx2 = np.gradient(dydx) / np.gradient(xx)
    d2ydx2 = savgol_filter(d2ydx2, 13, 3, axis=-1, mode='interp')

    curvature = []

    for i in range(0, len(lnln)-3):
        a = (1 + (dydx[i:i+3]**2))**(3/2)
        b = d2ydx2[i:i+3]

        R = np.nanmean(a)/np.nanmean(b)
        curv = 1/R

        curvature.append(curv)

    return np.array(curvature)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def surfaceFlowCurv(ssh, transform = True, xr_array = False, uv = False):
    dims = ssh.dims

    lat = ssh[dims[1]][:] if dims[1] in ssh.dims else None
    lon = ssh[dims[2]][:] if dims[2] in ssh.dims else None
    lnln, ltlt = np.meshgrid(lon.data, lat.data)
    # if transform = False, calc_dx_dy same method as float curvature
    # or xx, yy = np.meshgrid(lon.data, lat.data) 

    if transform == True:
        xx, yy = xyTransform(lnln, ltlt)
    
    shp = ssh.shape
    gravity = gsw.grav(ltlt, p=0)
    fcor = gsw.f(ltlt)

    box_kernel = Box2DKernel(3)
    kappa = np.ma.masked_all(shp)
    ugeos, vgeos = np.ma.masked_all(shp), np.ma.masked_all(shp)

    for it in range(ssh[dims[0]].size):
        # first derivatives
        detadx = np.gradient(ssh[it,])[1] / np.gradient(xx)[1]
        detady = np.gradient(ssh[it,])[0] / np.gradient(yy)[0]
        # boxcar filter
        detadx = convolve(detadx, box_kernel, normalize_kernel=True)
        detady = convolve(detady, box_kernel, normalize_kernel=True)

        # second derivatives
        d2etadx2 = np.gradient(detadx)[1] / np.gradient(xx)[1]
        d2etady2 = np.gradient(detady)[0] / np.gradient(yy)[0]
        d2etadxdy = np.gradient(detadx)[0] / np.gradient(yy)[0]
        # boxcar filter
        d2etadx2 = convolve(d2etadx2, box_kernel, normalize_kernel=True)
        d2etady2 = convolve(d2etady2, box_kernel, normalize_kernel=True)
        d2etadxdy = convolve(d2etadxdy, box_kernel, normalize_kernel=True)

        kappa[it,] = (-(d2etadx2*detady**2) -(d2etady2*detadx**2) + (2*d2etadxdy*detadx*detady)) / (detadx**2 + detady**2)**(3/2)
        ugeos[it,] = -(gravity / fcor) * (detady)
        vgeos[it,] = (gravity / fcor) * (detadx)

    if xr_array == True: # make lowercase for CMEMS 
        kappa = xr.DataArray(data=kappa, dims=["time", "latitude", "longitude"], coords=dict(
            time=(["time"], ssh[dims[0]].data),
            latitude=(["latitude"], ssh[dims[1]].data),
            longitude=(["longitude"], ssh[dims[2]].data)),)

        gvel = xr.Dataset(data_vars=dict(ugeos=(["time", "latitude", "longitude"], ugeos),
                                          vgeos=(["time", "latitude", "longitude"], vgeos),),
                                coords=dict(
                                    time=("time", ssh[dims[0]].data),
                                    latitude = ("latitude", ssh[dims[1]].data),
                                    longitude = ("longitude", ssh[dims[2]].data),))

    if uv == True:
        if xr_array == True:
            return gvel
        else:
            return ugeos, vgeos
    else:
        return kappa

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def gradientWind(ssh):
    '''Gradient wind balance: incorporating surface flow curvature to extract ageostrophic velocities.'''

    kappa = surfaceFlowCurv(ssh, transform = True, xr_array = False)
    ugeos, vgeos =  surfaceFlowCurv(ssh, transform = True, xr_array = False, uv = True)
    
    dims = ssh.dims

    lat = ssh[dims[1]][:] if dims[1] in ssh.dims else None
    lon = ssh[dims[2]][:] if dims[2] in ssh.dims else None
    lnln, ltlt = np.meshgrid(lon.data, lat.data)

    shp = ssh.shape
    fcor = gsw.f(ltlt)

    xpos = ugeos < 0
    ypos = vgeos < 0

    orientation = np.arctan(vgeos / ugeos)
    orientation[xpos] = np.arctan(vgeos[xpos] / ugeos[xpos]) + np.pi
    orientation[xpos & ypos] = np.arctan(vgeos[xpos & ypos] / ugeos[xpos & ypos]) - np.pi

    Vgeos = np.sqrt(ugeos**2 + vgeos**2)

    if ssh.ndim != 2:
        fcor = np.broadcast_to(fcor, shp)

    Vgrad = np.ma.masked_all(shp).flatten()
    fcor, Vgeos, kappa = fcor.flatten(), Vgeos.flatten(), kappa.flatten()

    for i in range(len(Vgrad)):
        Vgrad[i] = 2*Vgeos[i] / (1 + np.sqrt(1 + ((4*kappa[i]*Vgeos[i])/fcor[i])))

    data = {}
    data['ugeos'], data['vgeos'], data['ori'] = ugeos, vgeos, orientation
    data['Vgrad'], data['Vgeos'] = Vgrad.reshape(shp), Vgeos.reshape(shp)
    data['ugrad'], data['vgrad'] = data['Vgrad'] * np.cos(orientation), data['Vgrad'] * np.sin(orientation)

    gradvel = xr.Dataset(data_vars=dict(Vgrad=(["time", "latitude","longitude"], data['Vgrad']),
                                Vgeos=(["time", "latitude","longitude"], data['Vgeos']),
                                ugeos = (["time", "latitude","longitude"], ugeos),
                                 vgeos = (["time", "latitude","longitude"], vgeos),
                                  ugrad = (["time", "latitude","longitude"],  data['Vgrad'] * np.cos(orientation)),
                                  vgrad = (["time", "latitude","longitude"], data['Vgrad'] * np.sin(orientation)),
                                ori = (["time", "latitude","longitude"], orientation),),
                    coords=dict(
                        time=("time", ssh[dims[0]].data),
                        latitude = ("latitude", ssh[dims[1]].data),
                        longitude = ("longitude", ssh[dims[2]].data),))

    return gradvel

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def vortBalance(gradvel):
    '''Calculate terms in the vorticity balance from gradient wind velocities.'''

    dims = gradvel.dims
    shp = gradvel.vgrad.shape

    lnln, ltlt = np.meshgrid(gradvel.longitude.data, gradvel.latitude.data)
    xx, yy = xyTransform(lnln, ltlt)
    box_kernel = Box2DKernel(3)

    fcor = gsw.f(ltlt)
    beta = np.gradient(fcor)[0] / np.gradient(yy)[0]

    zeta = np.ma.masked_all(shp)
    divag, residual = zeta.copy(), zeta.copy()
    fdwdz, betav, ugradzeta = zeta.copy(), zeta.copy(), zeta.copy()
    ow, normal_strain, shear_strain  = zeta.copy(), zeta.copy(), zeta.copy()

    ua = gradvel.ugrad - gradvel.ugeos
    va = gradvel.vgrad - gradvel.vgeos
    
    # time (seconds)
    dt = np.gradient(gradvel['time'][:]).astype('timedelta64[s]')
    deltat = []
    for i in range(0, len(gradvel.time)):
        t = (gradvel.time[i] - gradvel.time[0])/1e+9
        deltat.append(int(t))

    for it in range(gradvel.time.size):
        dvdx = np.gradient(gradvel.vgrad[it,])[1] / np.gradient(xx)[1]
        dudy = np.gradient(gradvel.ugrad[it,])[0] / np.gradient(yy)[0]
        dvdx = convolve(dvdx, box_kernel, normalize_kernel=True)
        dudy = convolve(dudy, box_kernel, normalize_kernel=True)

        zeta[it,] = dvdx - dudy
        
        dzetadx = np.gradient(zeta[it,])[1] / np.gradient(xx)[1] #dimension 1: longitude
        dzetady = np.gradient(zeta[it,])[0] / np.gradient(yy)[0] #dimension 0: latitude
        dzetadx = convolve(dzetadx, box_kernel, normalize_kernel=True)
        dzetady = convolve(dzetady, box_kernel, normalize_kernel=True)

        # ageostrophic components
        dua_dx = np.gradient(ua[it,])[1] / np.gradient(xx)[1] # (ua[it,])[0]
        dva_dy = np.gradient(va[it,])[0] / np.gradient(yy)[0]
        dua_dx = convolve(dua_dx, box_kernel, normalize_kernel=True)
        dva_dy = convolve(dva_dy, box_kernel, normalize_kernel=True)
        
        divag[it,] = (dua_dx + dva_dy)
        fdwdz[it,] = -fcor * (-divag[it,])
        betav[it,] = beta * gradvel.vgrad[it,]
        ugradzeta[it,] = (gradvel.ugrad[it,] * dzetadx) + (gradvel.vgrad[it,] * dzetady)

        # normal and shear components of strain and Okubo-Weiss parameter
        dudx = np.gradient(gradvel.ugrad[it,])[1] / np.gradient(xx)[1]
        dudx = convolve(dudx, box_kernel, normalize_kernel=True)
        dvdy = np.gradient(gradvel.vgrad[it,])[0] / np.gradient(yy)[0]
        dvdy = convolve(dvdy, box_kernel, normalize_kernel=True)

        # Okubo-Weiss parameter (ow)
        normal_strain[it,] = dudx - dvdy
        shear_strain[it,] = dvdx + dudy
        ow[it,] = normal_strain[it,]**2 + shear_strain[it,]**2 - zeta[it,]**2

    if len(np.unique(dt)) > 1:
        dt = np.tile(np.gradient(deltat)[:, None, None], (1, gradvel.latitude.size, gradvel.longitude.size))
        dzetadt = np.gradient(zeta)[0] / dt #dimension 0 here is time becuase were not computing for individual time steps 
    else:
        dzetadt = np.gradient(zeta)[0] / np.unique(dt).astype('float')
    
    qgvb = xr.Dataset(data_vars=dict(zeta=(["time", "latitude", "longitude"], zeta.data),
                                     dzetadt=(["time", "latitude", "longitude"], dzetadt.data),
                                    fdwdz=(["time", "latitude", "longitude"], fdwdz.data),
                                    ugradzeta=(["time", "latitude", "longitude"], ugradzeta.data),
                                    betav=(["time", "latitude", "longitude"], betav.data),
                                    divag=(["time", "latitude", "longitude"], divag.data),
                                    okubo_weiss = (["time", "latitude", "longitude"], ow.data),
                                    shear_strain = (["time", "latitude", "longitude"], shear_strain.data),
                                    normal_strain = (["time", "latitude", "longitude"], normal_strain.data),),
                    coords=dict(
                        time=("time", gradvel.time.data),
                        latitude = ("latitude", gradvel.latitude.data),
                        longitude = ("longitude", gradvel.longitude.data),))

    return qgvb
    

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def potentialVorticity(float_num, floatid, smooth = True, window = 75, on_dens = True):
    '''Approximaiton of PV from the float data. Doe not take into account relative vorticity.'''

    CT, SA = settings.remove_bad_T_S(float_num, floatid)

    [IPV,p_midarray_ipv] = gsw.IPV_vs_fNsquared_ratio(SA, CT, float_num.pressure, p_ref=1600, axis = 1)
    f = gsw.f(float_num.latitude)
    f = np.tile(f, (len(float_num.pressure)-1,1)).transpose()

    if smooth == True:
        N_squared = N2_float(float_num, floatid, smooth = True, window = window, by_dist = True)
    else:
        N_squared = N2_float(float_num, floatid, smooth = False, by_dist = True)

    PV = IPV * (f*N_squared)

    # Simplified calculation of PV 
    PV = N_squared*f/9.81

    if on_dens == True:
        PV_on_dens = interp.varToDens(PV, float_num  = float_num, floatid = floatid, by_dist = True, PV = True)

        return PV_on_dens     
    else:
        return PV 
    
    
def ErtelPV_approximation(N2, dvdx, f, dvdz, dbdx):
    # As in Thompson et al. (2016), Naveira Garabato et al. (2019) and Archer et al. (2020)

    PV = (f + dvdx)*N2 - dvdz*dbdx

    return PV


# ------------------------------------------------------------------------------------------------------------------------------------------------------

def PV_gradient(float_num, floatid, smooth = False):
    PV = potentialVorticity(float_num, floatid, smooth = True, on_dens = True, dist = True)
    PV_gradient = PV.differentiate("distance")
    if smooth == True:
        PV_gradient = settings.smooth(PV_gradient, box_size = 3, return_xr = False)
        PV_gradient = xr.DataArray(data=PV_gradient, dims=["distance", "potential_density"], 
                                        coords=dict(distance=(["distance"], PV.distance.data),
                                        potential_density=(["potential_density"], PV.potential_density.data)),)
    return PV_gradient


# ------------------------------------------------------------------------------------------------------------------------------------------------------

def velocity_error(float_num, floatid, abs_v, alt_cmems):
    '''Average speed in the mixed layer (from floats) minus interpolated satellite geostrophic velocities.'''
    rs = findRSperiod(float_num)
    dist = cum_dist(float_num.longitude, float_num.latitude)
    mld = MLD_float(float_num, floatid, plot = False)
    mld = xr.DataArray(mld[rs]).interpolate_na(dim = 'dim_0', use_coordinate = True)

    flt_speed = speed(abs_v.u_abs, abs_v.v_abs)

    mld_speed = []
    for i in range(0, len(mld)):
        value = flt_speed[i].sel(pressure = slice(0,mld[i])).mean(dim = 'pressure', skipna = True) 
        mld_speed.append(value.data)

    mld_speed = xr.DataArray(mld_speed).interpolate_na(dim = 'dim_0', use_coordinate = True)

    alt_speed = speed(alt_cmems.ugos, alt_cmems.vgos)
    speed_interp = interp.interpToFloat(float_num, alt_speed)
    speed_interp = xr.DataArray(speed_interp[rs], dims = 'distance', coords = dict(distance = (["distance"], dist[rs].data)))

    error = mld_speed.data - speed_interp.data
    return error

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def buoyancy(CT, SA, rho_ref = 'mean'):
    g = 9.81
    # in-situ density
    rho = gsw.rho(SA, CT, 0)

    if rho_ref == 'mean':
        b = g*(1 - rho/rho.mean())
    else:
        rho_ref = 1027.5
        b = 9.81*(1 - rho/rho_ref)
    return b

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def RiNumber(N2, b = None, lat = None, u = None, v = None, smooth_vels = False, window = 7):
    '''Ri = N2 / (dudz**2 + dvdz**2). Relative strength of vertical stratification vs vertical velocity shear.'''

    if N2.dims[0] == 'distance':
        dx = np.gradient(N2.distance)
        dx = np.tile(dx, (len(N2.pressure),1)).transpose()

    if u is not None:
        # mask out nans in N2 
        N2 = N2.interp(pressure = u.pressure)
        mask = np.isnan(N2).data
        
        u.data[mask] = np.nan
        v.data[mask] = np.nan

        if smooth_vels == True:
            u = vel.smooth_prof_by_prof(u, window = window, print_info = False)
            v = vel.smooth_prof_by_prof(v, window = window, print_info = False)
            N2 = vel.smooth_prof_by_prof(N2, window = window, print_info = False)

        z = gsw.z_from_p(u.pressure, lat[1])
        dudz = np.gradient(u)[1]/np.gradient(z)
        dvdz = np.gradient(v)[1]/np.gradient(z)
        dudz = xr.DataArray(dudz, dims = u.dims, coords = u.coords)
        dvdz = xr.DataArray(dvdz, dims = u.dims, coords = u.coords)

        if len(N2) != len(dudz):
            print('N2 to even x grid')
            dx = np.gradient(u.distance)[0]
            N2 = interp.even_dist_grid(N2, int(dx))
            
        shear = (dudz**2 + dvdz**2)
        Ri = abs(N2 / shear)

        mask = np.ma.masked_where(Ri == np.inf, Ri)
        Ri.data[mask.mask] = np.nan
        # mask = np.ma.masked_where(Ri < 0, Ri)
        # Ri.data[mask.mask] = 0

        return Ri, shear

    elif lat is not None:
        ### Inverse Richardson number (Ri^-1 = f^2 N^2 / b_grad^2), assuming thermal wind balance (Siegelman, 2020).
        f = gsw.f(lat)
        f = np.tile(f, (len(N2.pressure),1)).transpose()

        b_grad = np.gradient(b[:,0:-1])[0]/dx
        b_grad = xr.DataArray(b_grad, dims = N2.dims, coords = N2.coords)
        b_grad_smoothed = vel.smooth_prof_by_prof(b_grad, window = 7)

        Ri = (f**2 * N2) / b_grad_smoothed**2
        return Ri, b_grad_smoothed, N2

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def fillNans(ds, method = 'linear'):
    '''fill nans by linear interpolation or nearest value'''

    if len(np.where(np.isinf(ds))[0]) > 0:
        ds = ds.replace([np.inf, -np.inf], np.nan, inplace=True)

    if method == 'nearest':
        ds_nonan = ds.fillna(method = 'bfill')
        ds_nonan = ds_nonan.fillna(method = 'ffill')
        
    elif method == 'linear':
        ds_nonan = ds.interpolate(method='linear', limit_direction ='both', axis=0) 
        
        return ds_nonan

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def DSC(CT, SA, pdens = None, vert_smooth = False, x_smooth = True, dens_interval = 0.01):
    '''Calculate diapycnal spiciness curvature (DSC). 
    CT = conservative temperature
    SA = absolute salinity
    ''' 
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    potential_temp = gsw.pt_from_CT(SA, CT)

    if pdens is None:
        print('calculating density...')
        pdens = potentialDensity(SA.pressure, SA, CT)
    else:
        pdens = pdens

    if 'potential_density' not in CT.dims:
        # interpolate onto density surfaces
        print('interpolating to density grid')
        theta_on_d = interp.to_pdens_grid(potential_temp, pdens, dens_interval = dens_interval)
        CT_on_d = interp.to_pdens_grid(CT, pdens, dens_interval = dens_interval)
        S_on_d = interp.to_pdens_grid(SA, pdens, dens_interval = dens_interval)
    else:
        theta_on_d = potential_temp
        CT_on_d = CT
        S_on_d = SA

    rho, alpha, beta = gsw.rho_alpha_beta(S_on_d, CT_on_d, 0)

    ## remember -- this is density anomaly (potential density - 1000 kg/m^-3)
    pdens_on_d = interp.to_pdens_grid(pdens, pdens, dens_interval = dens_interval)
    pdens_on_d = pdens_on_d.interp(potential_density = S_on_d.potential_density)

    if vert_smooth == True:
        # vertical smoothing to reduce noise (Shcherbina et al., 2009)
        S_on_d = S_on_d.rolling(potential_density = 3, center = True, min_periods = 2).mean()
        theta_on_d = theta_on_d.rolling(potential_density = 3, center = True, min_periods = 2).mean()
        pdens_on_d = pdens_on_d.rolling(potential_density = 3, center = True, min_periods = 2).mean()

    # vertical derivative of temperature with repsect to density 
    T_z = theta_on_d.differentiate('potential_density')

    # azTz = alpha.differentiate('potential_density')*T_z
    # aTzz = alpha*T_z.differentiate('potential_density')

    DSC = (2*alpha*pdens_on_d*T_z.differentiate('potential_density'))

    if x_smooth == True:
        DSC_smooth = DSC.rolling(distance = 3, center = True, min_periods = 2).mean()
        return DSC, DSC_smooth
    else:
        return DSC, theta_on_d, S_on_d


# ------------------------------------------------------------------------------------------------------------------------------------------------------


def direct_lateral_heat_flux(CT, rho, velocity, mld, order = 4, window = 125):
    '''Direct estimate of lateral heat flux associated with thermohaline intrusions.
    CT = conservative temperature
    rho = potential density
    For cross-front lateral heat flux, velocity = v_rot (cross-front velocity). 
    mld = mixed layer depth (remove the mixed layer measurements to focus on interior features)
    
    Polynomial fit of tracer profile against density is used to obtain a 'mean' profile (as in Bieito et al. 2024). 
    Separates density compensated intrusion-scales (10-100 m) from large scales. 
    Adjust the order and window size depending on the data and scales of interest. 

    '''
    T_mean, T_anom, v_mean, v_anom, heat_flux = (xr.zeros_like(CT)*np.nan for _ in range(5))

    for prof in range(len(CT)):
        # temperature profile against density
        T_rho = xr.DataArray(CT[prof].data, dims = ['density'], coords = dict(density = ('density', rho[prof].data)))
        #remove mixed layer
        ml_idx = np.where(CT[prof].pressure <= mld[prof])[0]
        T_rho[ml_idx] = np.nan
        nonans = np.where(~np.isnan(T_rho))[0]
        T = T_rho.dropna(dim = 'density')
        
        # temperature anomalies
        if len(T) > 10:
            # divide by density after polynomial fitting
            T_mean[prof, nonans] = interp.gaussianFilter(T, window = window, order = order, interp_na = False)
            T_anom[prof, nonans] = T.data - T_mean[prof, nonans]

        # velocity 
        v_sel = velocity[prof].copy()
        #remove mixed layer
        v_sel[ml_idx] = np.nan
        nonans = np.where(~np.isnan(v_sel))[0]
        v = v_sel.dropna(dim = 'pressure')
        
        # velocity anomalies
        if len(v) > 10:
            # divide by density after polynomial fitting
            v_mean[prof, nonans] = interp.gaussianFilter(v, window = window, order = order, interp_na = False)
            v_anom[prof, nonans] = v - v_mean[prof]
        
        # covert to a heat flux in kW/m^2
        heat_flux[prof, nonans] = (v_anom[prof, nonans]*T_anom[prof, nonans]*1027.5*4000)/1000


    ds = xr.Dataset(data_vars=dict(heat_flux=(["distance", "pressure"], heat_flux.data),
                                   T_anom=(["distance", "pressure"], T_anom.data),
                                   T_mean=(["distance", "pressure"], T_mean.data),
                                   v_anom=(["distance", "pressure"], v_anom.data),
                                   v_mean=(["distance", "pressure"], v_mean.data),),
                             coords=dict(
                                 pressure=("pressure", heat_flux.pressure.data),
                                 distance = ("distance", heat_flux.distance.data)),)

    return ds


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def mean_below_ml(dataArray, mld, zmax = 27.7, zdim = 'potential_density', integrate = False, median = False):
    '''Average in profile data below the mixed layer to a certain maximum density or pressure (zmax)'''
    # dz = pressure difference added to the base of the mixed layer e.g. dz = 10 adds 10 dbar onto the base of the mixed layer before doing the average

    lst = []
    for i in range(0, len(dataArray)):
        if zdim == 'potential_density':
            if integrate == True:
                d = np.trapz(dataArray[i].sel(potential_density = slice(mld[i], zmax)).dropna(dim = 'potential_density'))
            elif median == True:
                d = dataArray[i].sel(potential_density = slice(mld[i], zmax)).median(dim = zdim, skipna = True)
            else:
                d = dataArray[i].sel(potential_density = slice(mld[i], zmax)).mean(dim = zdim, skipna = True)
        else:
            if median == True:
                d = dataArray[i].sel(pressure = slice(mld[i], zmax)).median(dim = zdim, skipna = True)
            else:
                d = dataArray[i].sel(pressure = slice(mld[i], zmax)).mean(dim = zdim, skipna = True)
        lst.append(d)

    lst = np.asarray(lst)
    return lst

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def average_in_density_on_pgrid(data, pdens, dens_min, dens_max, integrate = False, median = False):
    '''Average between density range when data is gridded on pressure'''

    avg_data = []
    dens_copy = pdens.copy()
    for j in range(0, len(dens_copy)):
        dens_copy[j][np.isnan(dens_copy[j])] = 0
        max_idx, value = stats.find_nearest(dens_copy[j], dens_max[j])
        min_idx, value = stats.find_nearest(dens_copy[j], dens_min[j])

        if integrate == True:
            d = np.trapz(data[j][min_idx:max_idx].dropna(dim = 'pressure'))
        elif median == True:
            d = data[j][min_idx:max_idx].median(skipna=True).data
        else:
            d = data[j][min_idx:max_idx].mean(skipna=True).data

        avg_data.append(float(d))

    da = xr.DataArray(np.asarray(avg_data), dims = data.dims[0], coords = data.coords[data.dims[0]].coords)
    return da

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def percent_below_mL(data, mld, condition = '<= 1', zmax = None, dim = ('pressure', 'potential_density')):

    condition_met = []

    for i in range(0, len(data)):
        mld_prof = mld[i]

        if dim == 'pressure':
            if zmax is None:
                zmax = float(np.nanmax(data[i].pressure.data))
            data_sel = data[i].sel(pressure = slice(mld_prof,zmax))

        elif dim == 'potential_density':
            if zmax is None:
                zmax = float(np.nanmax(data[i].potential_density.data))
            data_sel = data[i].sel(potential_density = slice(mld_prof,zmax))

        if type(condition) == list:
            string1 = 'data_sel' + condition[0]
            string2 = 'data_sel' + condition[1]
            ind = np.where(eval(string1) & eval(string2))[0]
        else:
            string = 'data_sel' + condition
            ind = np.where(eval(string))[0]
            
        if len(data_sel[~np.isnan(data_sel)]) > 0:
            condition_met.append((len(ind)/len(data_sel))*100)
        else:
            condition_met.append(np.nan)

    return np.asarray(condition_met)