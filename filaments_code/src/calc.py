#!/usr/bin/env python
# coding: utf-8

# ### List of functions 
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import scipy
import gsw
import math

from scipy.signal import savgol_filter
from pyproj import Transformer
from pyproj import CRS
import warnings

from astropy.convolution import convolve
from astropy.convolution import Box2DKernel

import src.interpolation as interp
import src.stats as stats
import src.settings as settings
import src.velocities as vel
import src.importData as imports

def findRSperiod(float_num):
    '''find the profile index in where the rapid sampling ends'''
    nan_index = np.where(np.isnan(float_num.hpid) == True)
    over250 = np.where(float_num.profile > 251)
    rs_end = np.intersect1d(nan_index, over250)
    if len(rs_end) == 0:
        rs = int(float_num.hpid[-1].data)
    else:
        rs = rs_end[0]
    return slice(0,rs)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def distFromStart(float_num):
    '''Cumulative distance from the first profile (km)'''
    lat = float_num.latitude
    lon = float_num.longitude

    dist_between_profiles = np.concatenate((np.array([0]), gsw.distance(lon.values, lat.values)))
    dist_between_profiles_km = dist_between_profiles/1000
    dist_from_start = np.nancumsum(dist_between_profiles_km)

    # add to float dataset as a coordinate (dimension)
    distance = xr.DataArray(dist_from_start, dims = 'distance')
    float_num = float_num.assign_coords(distance=("distance", distance.data))

    return float_num.distance


def cum_dist(lons, lats):
    dist_diff = np.concatenate((np.array([0]), gsw.distance(lons.values, lats.values)))
    dist_diff_km = dist_diff/1000
    dist_from_start = np.nancumsum(dist_diff_km)

    distance = xr.DataArray(dist_from_start, dims = 'distance')

    return distance

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def tsDensContour(SA, CT):
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
    s = np.sqrt(u**2 + v**2)
    return s

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def calcEKE(float_num, floatid, alt_ds = None, altimetry = True, floats = False, abs_v = None, interp_to_flt = False, smooth_vels = False):
    '''Calculate eddy kinetic energy EKE from u and v (either from float data or altimetry)'''
    rs = findRSperiod(float_num)

    if altimetry == True:
        # find 3 year mean surface horizontal velocities from 2 years prior to float deployment
        # changing the mean window doesn't seem to affect the value (little interannual)
        start_time = float_num.time[0].values - np.timedelta64(365*2,'D')
        end_time = start_time + np.timedelta64(365*3,'D')
        start, end = str(start_time.astype('M8[D]')), str(end_time.astype('M8[D]'))

        # alt_ds.adt.sel(time = start)
        print('mean u and v between {} and {}'.format(start, end))
        u, v = alt_ds.ugos, alt_ds.vgos # multiply by 100 to convert m/s to cm/s

        if interp_to_flt == True:
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
            eke = 0.5*((u-u_bar)**2 + (v-v_bar)**2)

            EKE = []
            for i in range(0, len(float_num.profile)):
                value = eke.isel(time = i, latitude = i, longitude = i).values
                EKE.append(value.tolist())
        else:
            # mean U and V 
            u_bar = u.sel(time = slice(start,end)).mean(dim = 'time')
            v_bar = v.sel(time = slice(start,end)).mean(dim = 'time')
            # calculate EKE (deviation from mean)
            EKE = 0.5*((u-u_bar)**2 + (v-v_bar)**2)

    if floats == True:
        # use absolute velocities interpolated onto potential density 
        u = interp.varToDens(abs_v.u_abs, float_num  = float_num, floatid = floatid)
        v = interp.varToDens(abs_v.v_abs, float_num  = float_num, floatid = floatid)
        u, v = vel.setAbsVelToNan(floatid, u), vel.setAbsVelToNan(floatid, v)
        # u, v = u_dens*100, v_dens*100 # m/s to cm/s

        if smooth_vels == True:
            u = vel.smooth_prof_by_prof(u, window = 75, print_info = False)
            v = vel.smooth_prof_by_prof(u, window = 75, print_info = False)
        
        # rapid sampling mean
        u_bar = u[rs].mean(dim = 'distance',skipna = True)
        v_bar = v[rs].mean(dim = 'distance',skipna = True)

        # ke = 0.5*(u**2 + v**2)
        # mke = 0.5*(u_bar**2 + v_bar**2)
        EKE = 0.5*((u-u_bar)**2 + (v-v_bar)**2)

    return EKE

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def rollingEKE(float_num, floatid, alt_ds = None, altimetry = True, floats = False, abs_v = None, window = 5, min_prof = 3):

    if altimetry == True:
        u, v = alt_ds.ugos, alt_ds.vgos
        # interpolate altimetry velocities onto float profile locations and times
        u_alt = interp.interpToFloat(float_num, u, return_xr = True)
        v_alt = interp.interpToFloat(float_num, v, return_xr = True)

        # centered rolling mean along the float track
        u_bar = u_alt.rolling(profile = window, center = True, min_periods = min_prof).mean()
        v_bar = v_alt.rolling(profile = window, center = True, min_periods = min_prof).mean()

        eke = 0.5*((u_alt-u_bar)**2 + (v_alt-v_bar)**2)

        return eke
    
    if floats == True:
        u_smooth = vel.smooth_prof_by_prof(abs_v.u_abs, window = 75)
        v_smooth = vel.smooth_prof_by_prof(abs_v.v_abs, window = 75)

        u = interp.varToDens(u_smooth, float_num = float_num, floatid = floatid)
        v = interp.varToDens(v_smooth, float_num = float_num, floatid = floatid)

        # remove erroneous velocity profiles
        u, v = vel.setAbsVelToNan(floatid, u), vel.setAbsVelToNan(floatid, v)

        # centered rolling mean along isopycnals
        u_bar = u.rolling(distance = window, center = True, min_periods = min_prof).mean()
        v_bar = v.rolling(distance = window, center = True, min_periods = min_prof).mean()

        eke = 0.5*((u-u_bar)**2 + (v-v_bar)**2)
        flt_speed = speed(u,v)

        return eke, flt_speed

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def sshGrad(dataArray):
    '''Sea surface height gradient from satellite altimetry. 
    Also works with other spatial data variables with dimensions latitude and longitude.'''

    deg_to_m = 111195.
    grad_x = dataArray.differentiate('longitude')/deg_to_m
    grad_y = dataArray.differentiate('latitude')/deg_to_m
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

def TS_anom(float_num, floatid, pdens = True, by_dist = True, rs = True, window = None, min_prof = 3):
    '''Calculate temperature anomalies on either pressure of density surfaces.
      Mean temperature is taken during the rapid sampling period or as a rolling average is window is not None.'''

    if pdens == True:
        zdim = 'potential_density'
        newFlt = interp.interpOnDens(float_num, floatid, rs = True, by_dist = by_dist)
        xdim = newFlt.CT.dims[0]
    else:
        zdim = 'pressure'
        if by_dist == True:
            xdim = 'distance'
            newFlt = settings.distanceAsCoord(float_num, float_num = None, rs = True)
        else:
            newFlt = float_num
            xdim = 'profile'

    CT, SA = settings.remove_bad_T_S(newFlt, floatid)

    if rs == True:
        ind = findRSperiod(float_num)
        CT, SA = CT[ind], SA[ind]

    # remove rapid sampling mean on each z level.
    T_bar = CT.mean(dim = xdim, skipna = True)
    S_bar = SA.mean(dim = xdim, skipna = True)
    T_anom = CT - T_bar
    S_anom = SA - S_bar

    if window != None:
        # remove rolling mean on each z level
        Tmean_rolling = CT.rolling(profile = window, center = True, min_periods = min_prof).mean(dim = xdim, skipna = True)
        T_anom = CT - Tmean_rolling
        Smean_rolling = SA.rolling(profile = window, center = True, min_periods = min_prof).mean(dim = xdim, skipna = True)
        S_anom = SA - Smean_rolling

    # to xarray dataset
    anomalies = xr.Dataset(data_vars=dict(CT=([xdim, zdim], T_anom.data),
                                          SA=([xdim, zdim], S_anom.data)))
    
    anomalies = anomalies.assign_coords(CT[zdim].coords)
    anomalies = anomalies.assign_coords(CT[xdim].coords)

    return anomalies  

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def potentialDensity(pressure, SA, CT, p_ref = 0, anomaly = True):
    '''Potential density (anomaly) referenced to the surface'''
    dens = gsw.pot_rho_t_exact(SA, CT, pressure, p_ref)
    if anomaly == True:
        dens = dens - 1000
    return dens

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def densityLayerThickness(float_num, floatid, dmin, dmax, plot = True):
    '''Calculates thicnkess of the isopycnal layer between two density classes (dmin and dmax)'''
    rs = findRSperiod(float_num)
    dist = distFromStart(float_num)
    
    newFloat = settings.distanceAsCoord(float_num)
    CT, SA = settings.remove_bad_T_S(newFloat, floatid)

    pdens = potentialDensity(newFloat.pressure, SA, CT)
    
    dens_thickness = []

    for i in range(0, len(float_num.profile)):
        ind = np.where(np.logical_and(pdens[i] > dmin, pdens[i] < dmax))[0].tolist()
        if len(ind) < 1:
            dens_thickness.append(np.nan)
        else:
            thickness = pdens[i][ind].pressure[-1] - pdens[i][ind].pressure[0]
            dens_thickness.append(thickness.values)

    if plot == True:
        fig, ax = plt.subplots(figsize = (10,3))
        ax.plot(dist[rs], dens_thickness[rs])
        ax.set_ylim(0,400)
        ax.grid()

        settings.tickLocations(ax)
        if floatid == 8490: 
            settings.tickLocations(ax, major = 200)

        ax.set_ylabel(f'density layer thicnkess (m)')
        ax.set_xlabel(f'distance (km)')
        plt.legend([f'{dmin}-{dmax}'])

    return dens_thickness

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def MLD(pdens, criteria = 0.05, pref = 10):
    '''Calculate mixed layer depth using criteria from Dove et al. (2021):  density difference greater than 0.05 kg/m3 
    from a 10 dbar (surface) reference level.
    Input: xarray DataArray of potential density'''

    mld = []

    for i in range(0,len(pdens)):
        ind_nonan = np.where(~np.isnan(pdens[i]))[0]
        pdens_nonan = pdens[i,ind_nonan]

        if len(pdens_nonan) == 0: # empty profile
            mld.append(np.nan)
        # if the first non nan value is at a depth greater than 20 dbar, set as nan
        elif pdens_nonan[0].pressure > 20:
            mld.append(np.nan)
        else:
            # find the pressure level at which the density difference from 10dbar reference is greater than 0.05 kg/m3.
            pd0 = pdens_nonan.sel(pressure = pref, method = 'nearest')
            dens_diff = pdens_nonan - pd0
            mask = (dens_diff >= criteria) & (dens_diff.pressure > pd0.pressure)
            
            if len(pdens_nonan.pressure[mask]) == 0:
                mld.append(np.nan)
            else:
                mld.append(pdens_nonan.pressure[mask][0].data.tolist())
            
    mld = xr.DataArray(np.asarray(mld), dims = pdens.dims[0], coords = pdens[pdens.dims[0]].coords)

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
        dist = distFromStart(float_num)

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

def MLD_on_dens(float_num, floatid):
    CT, SA = settings.remove_bad_T_S(float_num, floatid)
    pdens = potentialDensity(float_num.pressure, SA, CT)
    mld = MLD(pdens, criteria = 0.05, pref = 10)
    
    dens_mld = []
    for j in range(0, len(float_num.SA)):
        ind = np.where(float_num.pressure == mld[j])[0].tolist()
        if len(ind) == 0:
            dens_mld.append(np.nan)
        else:
            dens_mld.append(pdens[j,ind].values[0])

    dens_mld = xr.DataArray(dens_mld, dims = float_num.SA.dims[0], coords = float_num.SA[float_num.SA.dims[0]].coords)
            
    return dens_mld

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def find_T_mld(float_num, floatid, mld, plot = True):
    '''Find the temperature at the base of the mixed layer'''
    T_mld = []
    for j in range(0, len(float_num.profile)):
        ind = np.where(float_num.pressure == mld[j])[0].tolist()
        if len(ind) == 0:
            T_mld.append(np.nan)
        else:
            T_mld.append(float_num.T[j,ind].values[0])

    if plot == True:
        rs = findRSperiod(float_num)
        dist = distFromStart(float_num)

        fig, ax = plt.subplots(figsize = (10,3))
        ax.scatter(dist[rs], T_mld[rs], s = 10)
        ax.plot(dist[rs], T_mld[rs])

        settings.tickLocations(ax)
        if floatid == 8490: 
            settings.tickLocations(ax, major = 200)

        ax.set_ylim(1,8)
        ax.grid(True)
        ax.set_title(floatid)
        ax.set_ylabel('T at MLD (°C)')
        ax.set_xlabel('distance (km)')

    return T_mld

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def dynamicHeight(float_num, floatid, p0 = 1500, steric = False):
    '''Calculates dynamic height or steric height referenced to 1500 dbar.
    Outputs: 
    1. dynamic height & dynamic height at 500 dbar relative to 1500 dbar
    or
    2. steric height & steric height at 50 dbar'''
    CT, SA = settings.remove_bad_T_S(float_num, floatid)
    pres = float_num.pressure
    
    # taking off the surface (top 10 m) and the bottom 100 m 
    SA = SA.sel(pressure = slice(10,1500))
    CT = CT.sel(pressure = slice(10,1500))
    pres = float_num.pressure.sel(pressure = slice(10,1500))

    p = np.tile(pres,(len(CT), 1))
    try:
        dist = float_num.distance
    except:
        dist = distFromStart(float_num)

    dyn_m = gsw.geo_strf_dyn_height(SA, CT, p, p_ref=p0, axis=1)
    const_grav = 9.7963  # Griffies, 2004.
    steric_height = dyn_m/const_grav

    steric_h = xr.DataArray(data=steric_height.data, dims=["distance", "pressure"], coords=dict(
        distance=(["distance"], dist.data),
        pressure=(["pressure"], pres.data)),)

    dyn_m = xr.DataArray(data=dyn_m.data, dims=["distance", "pressure"], coords=dict(
        distance=(["distance"], dist.data),
        pressure=(["pressure"], pres.data)),)

    # steric height at 50 dbar
    steric_50 = steric_h.sel(pressure = 50)
    #500 dbar referenced to 1500 dbar
    dyn_m_500_1500 = dyn_m.sel(pressure = 500)/10 # divide by 10 to convert to dyn m (1 dyn m = 10 m2/s2)

    if steric == True:
        return steric_h, steric_50
    else:
        return dyn_m, dyn_m_500_1500

# ------------------------------------------------------------------------------------------------------------------------------------------------------
def N2(CT, SA, lat, smooth = True, window = 75):
    '''Output: Buoyancy frequency (N squared) at pressure midpoints'''
    f = gsw.f(lat)
    p = CT.pressure
    f = np.tile(f, (len(p)-1,1)).transpose()

    [N2,p_midarray_n2] = gsw.Nsquared(SA, CT, p, axis = 1) # axis = dimension along which pressure increases
    
    N2 = xr.DataArray(data = N2, dims = ["distance", "pressure"],coords = dict(pressure=(["pressure"], 
        p_midarray_n2[0]),distance=(["distance"], CT.distance.data),))

    if smooth == True:
        N2 = vel.smooth_prof_by_prof(N2, window = window)

    return N2


def N2_float(float_num, floatid, smooth = True, window = 75, by_dist = True):
    '''Output: Buoyancy frequency (N squared) at pressure midpoints'''
    f = gsw.f(float_num.latitude)
    f = np.tile(f, (len(float_num.pressure)-1,1)).transpose()
    CT, SA = settings.remove_bad_T_S(float_num, floatid)

    [N2,p_midarray_n2] = gsw.Nsquared(SA, CT, float_num.pressure, axis = 1) # axis = dimension along which pressure increases
    
    if by_dist == True:
        dist = distFromStart(float_num)
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

def getIndexList(float_num, week_number):
    calcWeeknum(float_num)
    week_list = list(float_num.week.values)
    return [i for i in range(len(week_list)) if week_list[i] == week_number]

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def calcWeeknum(float_num):

    t1 = float_num.isel(time = 0).time
    week_num = []

    for i in range(0,len(float_num.time)):
        t2 = float_num.isel(time = i).time
        diff = t2 - t1
        days = diff.values.astype('timedelta64[D]')
        days /= np.timedelta64(1, 'D')

        if days >= 0:
            week_num.append(int(days // 7))
        else:
            week_num.append(None)

    week_num = xr.DataArray(week_num, dims = 'profile')
    float_num['week'] = week_num

    return float_num['week']

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
    
#------------------------------------------------------------------------------------------------------------------------------------------

def depth_avg_spice_std(float_num, floatid, below_ml = False):

    CT, SA = settings.remove_bad_T_S(float_num, floatid)

    spice = gsw.spiciness1(SA, CT)
    spice = interp.varToDens(spice, float_num  = float_num, floatid = floatid)
    spice_std = spice.rolling(distance=5, center=True, min_periods = 3).std()

    if below_ml == True:
        #average spice std below the mixed layer
        mld_dens = MLD_on_dens(float_num, floatid)
        mld = xr.DataArray(mld_dens, dims = ['distance'], coords = dict(distance = (["distance"], spice_std.distance.data)))

        spice_below_ml = []
        for i in range(0, len(mld)):
            value = spice_std[i].sel(potential_density = slice(mld[i], spice_std[i].potential_density[-1])).mean(dim = 'potential_density') 
            spice_below_ml.append(value.data)
        
        spice_avg = xr.DataArray(spice_below_ml, dims = ['distance'], coords = dict(distance=(["distance"], spice_std.distance.data)))

    else:
        
        spice_avg = spice_std.mean(dim = 'potential_density', skipna = True)

    return spice_avg

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def depth_avg_eke(float_num, floatid, abs_v, rolling = True, window = 15, min_prof = 9):
    '''depth-averaged EKE from the float velocities. Rolling mean or rapid sampling mean.'''
    if rolling == True:
        eke = rollingEKE(float_num, floatid, altimetry = False, floats = True, abs_v = abs_v, centre = True, window = window, min_prof = min_prof)
        eke_avg = eke.mean(dim = 'potential_density', skipna = True)

    else:
        eke = calcEKE(float_num, floatid, altimetry = False, floats = True, abs_v = abs_v, smooth = True)[0]
        eke_avg = eke.mean(dim = 'potential_density', skipna = True)

    return eke_avg
    
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def estimateDiffusivity(k_ds, float_num, var = 'diffusivity',  by_dist = True, zdim = 'depth'):
    '''Uses dataset developed by Cole et al. (2015) to find a temporally-averaged (2005-2015) 
    estimate of subsurface diffusivity (k) along the float track. 
    Diffusivity is calculated from salinty gradients from Argo floats and subsurface velocity fluctuations from ECCO2 state estimates'''

    k_flt = interp.interpToFloat(float_num, k_ds[var], location_only = True, zdim = zdim)

    if by_dist == True:
        dist = distFromStart(float_num)           
        k_flt_xr = xr.DataArray(data=k_flt, dims=["distance", "depth"], coords=dict(
        distance=(["distance"], dist.data),
        depth=(["depth"], k_flt.depth.data)),)
    else:
        k_flt_xr = xr.DataArray(data=k_flt, dims=["profile", "depth"], coords=dict(
        profile=(["profile"], float_num.profile.data),
        depth=(["depth"], k_flt.depth.data)),)

    return k_flt_xr

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def ssh_std(alt_cmems, float_num, start_time = None, end_time = None, interp_to_flt = False):
    '''Sea surface height standard deviation, H*, used in Foppert et al. (2017) to estimate EHF.'''

    if start_time is None:
        start_time = float_num.time[0].values - np.timedelta64(365*2,'D')
        end_time = start_time + np.timedelta64(365*3,'D')

    start, end = str(start_time.astype('M8[D]')), str(end_time.astype('M8[D]'))

    alt_cmems.adt.sel(time = start)
    print('mean ADT between {} and {}'.format(start, end))

    mean_ssh = alt_cmems.adt.sel(time = slice(start, end)).mean(dim = 'time')
    sum_of_squares = ((alt_cmems.adt - mean_ssh)**2).sum(dim = 'time', skipna = True)
    H = np.sqrt((1/(len(alt_cmems.time)-1) * sum_of_squares))

    if interp_to_flt == True:
        H = interp.interpToFloat(float_num, H, location_only = True)
        
    return H

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
    dist = distFromStart(float_num)
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

def RiNumber(N2, b, lat, u = None, v = None, window = 55):
    '''Inverse Richardson number (Ri^-1 = f^2 N^2 / b_grad^2), assuming thermal wind balance (Siegelman, 2020). '''

    f = gsw.f(lat)
    f = np.tile(f, (len(N2.pressure),1)).transpose()
    dx = np.gradient(N2.distance)
    dx = np.tile(dx, (len(N2.pressure),1)).transpose()

    b_grad = np.gradient(b[:,0:-1])[0]/dx
    b_grad = xr.DataArray(b_grad, dims = N2.dims, coords = N2.coords)
    b_grad_smoothed = vel.smooth_prof_by_prof(b_grad, window = window)

    Ri = (f**2 * N2) / b_grad_smoothed**2

    if u is not None:
        # Ri = N2/(u_z**2 + v_z**2)
        dudz = u.interp(pressure = N2.pressure).differentiate('pressure')
        dvdz = v.interp(pressure = N2.pressure).differentiate('pressure')
        Ri = abs(N2 / (dudz**2 + dvdz**2))
        mask = np.ma.masked_where(Ri == np.inf, Ri) 
        Ri.data[mask.mask] = np.nan
        Ri = Ri.rolling(pressure = window, center = True, min_periods = 3).mean(skipna = True)
        # Ri = vel.smooth_prof_by_prof(Ri, window = window)
    return Ri, b_grad_smoothed

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def get_all_vars(float_num, floatid, alt_cmems, fsle_max, diffusivity_ds, SST, ow, velocity = None, pandas_df = False):
    rs = findRSperiod(float_num)
    # float vars
    spice_avg = depth_avg_spice_std(float_num, floatid)
    curv_float = floatCurvature(float_num, floatid, transform = True, remove_outliers = True)
    dyn_m_500 = dynamicHeight(float_num)[1]
    dens_thickness = np.asarray(densityLayerThickness(float_num, floatid, 27.0, 27.2, plot = False))

    PV_grad = PV_gradient(float_num, floatid, smooth = True)
    depth_avg_PV_gradient = abs(PV_grad).mean(dim = 'potential_density', skipna = True)

    mld = MLD_float(float_num, floatid, plot = False)
    mld_da = xr.DataArray(mld, dims = 'profile', coords = dict(profile = (["profile"], float_num.profile.data)))
    mld_variability = mld_da.rolling(profile=4, center=True).std()

    if floatid == 8490:
        shp = spice_avg.shape
        eke_avg = np.empty(shp)*np.nan
        avg_speed = np.empty(shp)*np.nan
        vel_shear = np.empty(shp)*np.nan
        depth_avg_vel_shear = np.empty(shp)*np.nan
        northward_v = np.empty(shp)*np.nan
    else:
        eke_avg = depth_avg_eke(float_num, floatid, velocity, centre = False)
        u, v = vel.setAbsVelToNan(floatid, velocity.u_abs), vel.setAbsVelToNan(floatid, velocity.v_abs)
        avg_speed = speed(u, v).mean(dim = 'pressure')
        vel_shear = vel.velShearSection(float_num, floatid, remove_odd_profiles = True, plot = False)
        depth_avg_vel_shear = vel_shear.sel(pressure = slice(200,1600)).mean(dim = 'pressure', skipna = True)
        northward_v = vel.setAbsVelToNan(floatid, velocity.v_abs).mean(dim = 'pressure', skipna = True)
    
    # altimetry vars
    ssh_grad = interp.interpToFloat(float_num, sshGrad(alt_cmems.adt))
    ssh = interp.interpToFloat(float_num, alt_cmems.adt)
    ssh_change = ssh - np.nanmean(ssh[0:5])
    sla = interp.interpToFloat(float_num, alt_cmems.sla)
    sst = interp.interpToFloat(float_num, SST)
    sst_grad = interp.interpToFloat(float_num, sshGrad(SST))

    alt_speed = speed(alt_cmems.ugos, alt_cmems.vgos)
    alt_speed = interp.interpToFloat(float_num, alt_speed)

    ssh_da = xr.DataArray(ssh, dims = 'profile', coords = dict(profile = (["profile"], float_num.profile.data)))
    ssh_variability = ssh_da.rolling(profile=5, center=True, min_periods = 3).std()

    # eke_interp = np.asarray(calcEKE(float_num, floatid, alt_ds = alt_cmems, altimetry = True, interp_to_flt = True))
    eke_interp = rollingEKE(float_num, floatid, alt_ds = alt_cmems, altimetry = True, centre = False, window = 15, min_prof = 9)
    okubo_weiss = interp.interpToFloat(float_num, ow)

    curv_flow = surfaceFlowCurv(alt_cmems.adt, transform = True, xr_array = True)
    curv_flow_interp = interp.interpToFloat(float_num, curv_flow)
    FSLE = interp.interpToFloat(float_num, fsle_max)

    #other data sources (diffusivity estimates from Cole et al. (2015))
    diffusivity = estimateDiffusivity(diffusivity_ds, float_num, by_dist = False)
    k_top_500m = diffusivity.sel(depth = slice(0,1000)).mean(dim = 'depth', skipna = True)

    # to dataset
    ds = xr.Dataset(data_vars=dict(depth_avg_spice_std=(["profile"], spice_avg[rs].data),
                            mld =(["profile"], mld[rs]),
                            mld_std = (["profile"], mld_variability[rs].data),
                            dens_layer_m = (["profile"], dens_thickness[rs]),
                            dynamic_height = (["profile"], dyn_m_500[rs].data),
                            
                            sea_surface_height = (["profile"], ssh[rs]),
                            sea_surface_temp = (["profile"], sst[rs]),
                            sst_gradient = (["profile"], sst_grad[rs]),
                            surface_speed = (["profile"], alt_speed[rs]),
                            ssh_gradient =(["profile"], ssh_grad[rs]),
                            ssh_std =(["profile"], ssh_variability[rs].data),
                            ssh_change = (["profile"], ssh_change[rs]),
                            sla =(["profile"], sla[rs]),
                            latitude = (["profile"], float_num.latitude[rs].data),
                            
                            PV_gradient = (["profile"], depth_avg_PV_gradient[rs].data),
                            float_eke=(["profile"], eke_avg[rs].data),
                            float_v_abs = (["profile"], northward_v[rs].data),
                            surface_eke=(["profile"], eke_interp[rs].data),
                            diffusivity=(["profile"], k_top_500m[rs].data),
                            FSLE = (["profile"], abs(FSLE)[rs]),
                            okubo_weiss = (["profile"], okubo_weiss[rs]),

                            curv_float = (["profile"], curv_float[rs]),
                            curv_surf_flow = (["profile"],  curv_flow_interp[rs]),

                            depth_avg_vel_shear = (["profile"], depth_avg_vel_shear[rs].data),
                            depth_avg_speed = (["profile"], avg_speed[rs].data),
                            ),
                coords=dict(
                    profile=("profile", float_num.profile[rs].data),))

    if pandas_df == True:
        return ds.to_dataframe()
    else:
        return ds

# ------------------------------------------------------------------------------------------------------------------------------------------------------

    
def getDatasets(float_num, floatid):
    # import extra data files
    # 1. absolute velocity 
    if floatid != 8490:
        datadir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data', 'floats', 'absolute_velocity')
        abs_v = imports.importNetCDF(datadir, 'abs_vel_%s.nc' %floatid, datatype ='by_profile')
    
    # 2. fsle & okuubo weiss
    datadir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data')
    if floatid != 8495:
        alt_cmems = imports.importNetCDF(datadir, 'CMEMS.nc', datatype ='altimetry')
        fsle_ds = imports.importNetCDF(datadir, 'FSLE.nc')
        fsle_ds = fsle_ds.rename({'lat':'latitude', 'lon':'longitude'})
        ow = imports.importNetCDF(datadir, 'okubo_weiss.nc', datatype ='altimetry').okubo_weiss
    else:
        alt_cmems = imports.importNetCDF(datadir, 'CMEMS_recent.nc', datatype ='altimetry')
        fsle_ds = imports.importNetCDF(datadir, 'FSLE_recent.nc')
        fsle_ds = fsle_ds.rename({'lat':'latitude', 'lon':'longitude'})
        ow = OkuboWeiss(float_num, alt_cmems, interp_to_float = False)
    
    # 3. diffusivity estimates
    input_file = os.path.join(datadir, 'diffusivity', 'ArgoTS_eddydiffusivity_20052015_1deg.nc')
    ds = xr.open_dataset(input_file)
    
    k_ds_Cole_2015 = xr.Dataset(data_vars=dict(mixing_length = (["latitude", "longitude", "depth"], ds.mixing_length.data),
                                  diffusivity = (["latitude", "longitude", "depth"], ds.diffusivity.data),
                                 sal_grad = (["latitude", "longitude", "depth"], ds.salinity_gradient.data),
                                 vel_std = (["latitude", "longitude", "depth"], ds.velocity_std.data),),
                        coords=dict(
                            latitude=("latitude", ds.latitude.data),
                            longitude = ("longitude", ds.longitude.data),
                            density = ("density", ds.density.data),
                            depth = ("depth", ds.depth.data),))

    # 4. SST data 
    sst = imports.importNetCDF(datadir, 'METOFFICE-SST-L4.nc')
    sst = sst.rename({'lat':'latitude', 'lon':'longitude'})
    sst_deg = sst.analysed_sst - 273.15 # From Kelvin to Celcius
    
    if floatid != 8490:
        ds = get_all_vars(float_num, floatid, alt_cmems, fsle_ds.fsle_max, k_ds_Cole_2015, sst_deg, ow, velocity = abs_v, pandas_df = True)
    else:
        ds = get_all_vars(float_num, floatid, alt_cmems, fsle_ds.fsle_max, k_ds_Cole_2015, sst_deg, ow, pandas_df = True) 
    
    return ds

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

def DSC(CT, SA, pdens = None, vert_smooth = False, x_smooth = True):
    '''Calculate diapycnal spiciness curvature (DSC). 
    '''
    # If $\alpha_{z}T_{z}$ << $\alpha T_{zz}$, then DSC can be well approximated 
    # as $2\alpha\rho T_{zz}$ or $2\beta\rho S_{zz}$ (Scherbina et al., 2009).

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if pdens is None:
        print('calculating density...')
        pdens = potentialDensity(SA.pressure, SA, CT)
    else:
        pdens = pdens

    if 'potential_density' not in CT.dims:
        # interpolate onto density surfaces
        print('interpolating to density grid')
        T_on_d = interp.to_pdens_grid(CT, pdens)
        S_on_d = interp.to_pdens_grid(SA, pdens)
    else:
        T_on_d = CT
        S_on_d = SA

    rho, alpha, beta = gsw.rho_alpha_beta(S_on_d, T_on_d, 0)

    pdens_on_d = interp.to_pdens_grid(pdens, pdens)
    pdens_on_d = pdens_on_d.interp(potential_density = S_on_d.potential_density)

    if vert_smooth == True:
        # vertical smoothing to reduce noise (Shcherbina et al., 2009)
        S_on_d = S_on_d.rolling(potential_density = 3, center = True, min_periods = 2).mean()
        T_on_d = T_on_d.rolling(potential_density = 3, center = True, min_periods = 2).mean()
        pdens_on_d = pdens_on_d.rolling(potential_density = 3, center = True, min_periods = 2).mean()

    # vertical derivative of temperature with repsect to density 
    T_z = T_on_d.differentiate('potential_density')

    # azTz = alpha.differentiate('potential_density')*T_z
    # aTzz = alpha*T_z.differentiate('potential_density')

    DSC = (2*alpha*pdens_on_d*T_z.differentiate('potential_density'))

    if x_smooth == True:
        DSC_smooth = DSC.rolling(distance = 3, center = True, min_periods = 2).mean()
        return DSC, DSC_smooth
    else:
        return DSC, T_on_d, S_on_d


def mean_below_ml(dataArray, mld, zmax = 27.7, zdim = 'potential_density'):
    '''Average in density space values below the mixed layer to a certain maximum density (zmax)'''

    lst = []
    for i in range(0, len(dataArray)):
        if zdim == 'potential_density':
            mean = dataArray[i].sel(potential_density = slice(mld[i], zmax)).mean(dim = zdim, skipna = True)
        else:
            mean = dataArray[i].sel(pressure = slice(mld[i], zmax)).mean(dim = zdim, skipna = True)
        lst.append(mean)

    lst = np.asarray(lst)
    return lst