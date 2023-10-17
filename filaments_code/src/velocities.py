import matplotlib.pyplot as plt 
import numpy as np
import os
import xarray as xr
import matplotlib
from matplotlib.ticker import (MultipleLocator, MaxNLocator, StrMethodFormatter)
import math
import warnings
import gsw
import datetime
import stat
import pyproj

from geographiclib.geodesic import Geodesic
from scipy.signal import savgol_filter
from timeit import default_timer as timer

import imp
import src.importData as imports
import src.interpolation as interp
import src.settings as settings
import src.stats as stats
import src.calc as calc

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def avgProfilePairs(float_num, u, v, vel = True, positions = False):
    ''' Take the average velocities for the up and down casts (profile pairs).
    u and v could be relative or absolute velocities (xarray format)'''
    lons, lats = interp.gaussianFilter(float_num.longitude, interp_na = True), interp.gaussianFilter(float_num.latitude, interp_na = True)


    if vel == True:
        arr_u = [u[0:2].mean(dim = 'profile')]
        arr_v = [v[0:2].mean(dim = 'profile')]

        for i in range(0,len(float_num.profile)-3,2):
            dive_u = u[i+2:i+4].mean(dim = 'profile')
            dive_v = v[i+2:i+4].mean(dim = 'profile')
            arr_u.append(dive_u)
            arr_v.append(dive_v)

        combined_u = xr.concat(arr_u, dim='profile')
        combined_v = xr.concat(arr_v, dim='profile')

        return combined_u, combined_v
    
    if positions == True:
        arr_lat = [np.mean(lats[0:2])]
        arr_lon = [np.mean(lons[0:2])]

        for i in range(0,len(float_num.profile)-3,2):
            dive_lat = np.mean(lats[i+2:i+4])
            dive_lon = np.mean(lons[i+2:i+4])
            arr_lat.append(dive_lat)
            arr_lon.append(dive_lon)
            
        return arr_lat, arr_lon

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def avgShear(float_num, u, v, surf = slice(0, 100), mid = slice(300, 800), deep = slice(1200, 1600)):
    ''' This function is used within plotVelMap to calculate the average velocities within different depth ranges (e.g. surface, mid, deep and full depth).
    Input:
    float_num (e.g. ema[8489])
    u and v (absolute or relative velocities in Xarray format)
    
    Default values:
    surface = 0-100 m 
    mid-depth = 300-800 m 
    deep = 1200-1600 m

    OUTPUT: 
    surfu, surfv, midu, midv, deepu, deepv, fullu, fullv
    '''
    combined_u, combined_v = avgProfilePairs(float_num, u, v)

    surfu = combined_u.sel(pressure = surf).mean(dim = 'pressure')
    surfv = combined_v.sel(pressure = surf).mean(dim = 'pressure')

    midu = combined_u.sel(pressure = mid).mean(dim = 'pressure')
    midv = combined_v.sel(pressure = mid).mean(dim = 'pressure')

    deepu = combined_u.sel(pressure = deep).mean(dim = 'pressure')
    deepv = combined_v.sel(pressure = deep).mean(dim = 'pressure')

    fullu = combined_u.mean(dim = 'pressure')
    fullv = combined_v.mean(dim = 'pressure')

    return surfu, surfv, midu, midv, deepu, deepv, fullu, fullv

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def calcGradVel(float_num, alt_cmems):
    '''Calculate gradient wind velocities from satellite altimetry'''
    start = float_num.time.values[0]
    end = float_num.time.values[-1]
    start_time = str(start.astype('M8[D]'))
    end_time = str(end.astype('M8[D]'))

    lon = slice(145,175)
    lat = slice(-60,-50) 
    t = slice(start_time, end_time)

    # Copernicus
    adt = alt_cmems.adt.sel(longitude = lon, time = t)
    kappa = calc.surfaceFlowCurv(adt, transform = True, xr_array = False)
    ugeos, vgeos =  calc.surfaceFlowCurv(adt, transform = True, xr_array = False, uv = True)
    gradvel = calc.gradientWind(adt, kappa, ugeos, vgeos)
    
    return gradvel

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def plotVelMap(float_num, u, v, profiles, alt_cmems, xlim, ylim, scale = 1, fig_size = (14,6), 
                    surf = slice(0,250), deep = slice(1200, 1500), dots = False, gradvel_vectors = False):
    '''Function for making a quiver plot of float velocities along the trajectory. Different color arrows correspond to different depths.

    INPUTS: 
    profiles - e.g. np.arange(70,86,1) corresponding to the velocity shear dives in plotShear(float_num, dives). These will be marked as red dots on the map.
    set xlim and ylim (e.g. -55.5, -53.9 and 150.5, 153 respectively)
    scale of the quiver arrows can be changed (default = 1)
    *select gradvel_vectors = False to remove flow velocity vectors.

     '''
    cols = ['#a1dab4','#41b6c4','#225ea8', '#253494']
    
    dives = []
    for i in range(0,len(profiles),2):
        dives.append(int(profiles[i]/2))
    
    rs = calc.findRSperiod(float_num)
    surfu, surfv, midu, midv, deepu, deepv, fullu, fullv = avgShear(float_num, u, v, surf = surf, deep = deep)
    rs_dives = slice(0,int(len(float_num.profile[rs])/2))
    dive_lat, dive_lon = avgProfilePairs(float_num, u, v, vel = False, positions = True)

    fig, ax = plt.subplots(figsize = fig_size)
    plt.scatter(dive_lon[rs_dives], dive_lat[rs_dives], c= 'grey', zorder = 2)
    ax.quiver(dive_lon[rs_dives], dive_lat[rs_dives], surfu[rs_dives], surfv[rs_dives], color = cols[0], scale = scale, width = 0.006, zorder = 3)
    ax.quiver(dive_lon[rs_dives], dive_lat[rs_dives], deepu[rs_dives], deepv[rs_dives], color = cols[3], scale = scale, width = 0.006, zorder = 3)
    
    if np.diff(xlim)[0] < 3.5:
        ax.set_xticks(np.arange(xlim[0],xlim[1]+0.5,0.5))
    else:
        ax.set_xticks(np.arange(xlim[0],xlim[1]+0.5,1))
    # ax.legend(['0-250m', '1200-1500 m'], loc = 'lower right', fontsize = 12)

    ax.set_facecolor('whitesmoke')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    # red dots for profiles/dives
    if dots == True: 
        for d in dives:
            ax.scatter(dive_lon[d], dive_lat[d],c = 'r', s = 20, alpha = 0.6, zorder = 3)
            
    # add altimetry (avg ssh during selected profiles)
    levels = np.arange(-0.7, 0.5, 0.1)
    start = float_num.time[profiles[0]:profiles[-1]].values[0]
    end = float_num.time[profiles[0]:profiles[-1]].values[-1] 
    if np.isnat(start):
        start = float_num.time[profiles[0]:profiles[-1]].values[1]
    if np.isnat(end):
        end = float_num.time[profiles[0]:profiles[-1]].values[-2]

    start_time, end_time = str(start.astype('M8[D]')), str(end.astype('M8[D]'))
    
    msl = alt_cmems.adt.sel(time = slice(start_time, end_time)).mean(dim = 'time')
    CS = msl.plot.contour(colors = 'gray',linewidths = 1.5, alpha = 0.5, levels = levels, zorder = 1)
    plt.clabel(CS, inline = True, fontsize=10, fmt = '%1.1f')

    if gradvel_vectors == True: 
        gradvel = calcGradVel(float_num, alt_cmems)
        gradvel_mean = gradvel.Vgrad.sel(time = slice(start_time, end_time)).mean(dim = 'time')
        u_mean = gradvel.ugrad.sel(time = slice(start_time, end_time)).mean(dim = 'time')
        v_mean = gradvel.vgrad.sel(time = slice(start_time, end_time)).mean(dim = 'time')
        
        ax.quiver(gradvel_mean.longitude,  gradvel_mean.latitude, u_mean, v_mean, color = 'grey', 
                        scale = scale+1, width = 0.004, zorder = 1, alpha = 0.5) 

    return ax

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def plotShear(float_num, u, v, dives, c_speed = True):
    
    dz = 50
    cols = ['#a1dab4','#41b6c4','#225ea8', '#253494']
    combined_u, combined_v = avgProfilePairs(float_num, u, v)
    dive_lat, dive_lon = avgProfilePairs(float_num, u, v, vel = False, positions = True)
    speed = 0.5*(combined_u**2 + combined_v**2)
    
    cm = matplotlib.cm.plasma
    if c_speed == True:
        v_max = 0.3
        f_size = (16,10)
        clabel = 'speed (m $s^{-1}$)'
    else:
        v_max = 1600
        f_size = (16,8)
        clabel = 'pressure (dbar)'

    norm = matplotlib.colors.Normalize(vmin = 0, vmax = v_max)
    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
    
    if len(dives) > 4:
        fig, axs = plt.subplots(2, 4, figsize = f_size)
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(1, 4, figsize = (14,6))
        axs = axs.flatten()
    
    i = 0
    for d in dives:
        u = combined_u[d].dropna(dim = 'pressure').sel(pressure = slice(0,1600,int(dz/2)))
        v = combined_v[d].dropna(dim = 'pressure').sel(pressure = slice(0,1600,int(dz/2)))

        # Data for quiver colormap
        if c_speed == True:
            c = speed[d].dropna(dim = 'pressure').sel(pressure = slice(0,1600,int(dz/2)))
            y = u.pressure.values
            x = np.ones((len(y),1))
            x_ref, y_ref = 0.92, 150
        else:
            c = float_num.pressure.sel(pressure = slice(0,1600,int(dz/2)))
            y = np.tile(dive_lat[d], len(u))
            x = np.tile(dive_lon[d], len(u))
            x_ref, y_ref = 150, -54

        axs[i].set_facecolor('whitesmoke')
        axs[i].quiver(x, y, u, v, scale = 2, width = 0.006, color = cm(norm(c)))

        # depth-averaged
        # axs[i].quiver(x_ref, y_ref, combined_u[d].mean(dim = 'pressure'),combined_v[d].mean(dim = 'pressure'), 
        #               scale = 1, width = 0.008, color = cols[2])

        # reference arrows
        # axs[i].quiver(0.92, 1600, 0, 0.15, scale = 1, width = 0.006, color = 'grey')
        # axs[i].quiver(0.92, 1600, 0.15, 0, scale = 1, width = 0.006, color = 'grey')

        # axs[i].text(0.955, 1620, 'u', color = 'grey')
        # axs[i].text(0.915, 1390, 'v', color = 'grey')
        # axs[0].text(0.925, 1510, '0.15', color = 'dimgrey', fontsize = 11)
        axs[i].invert_yaxis()

        if c_speed == True:
            axs[i].set_xlim(0.9, 1.1)
        else:
            axs[i].scatter(dive_lon[d], dive_lat[d], c = 'r', s = 20, alpha = 0.6)
            axs[i].set_xlim(145, 160)
            axs[i].set_ylim(-56, -51)
        
        axs[i].set_title(f'profile {d*2} & {d*2+1}')
        # axs[i].xaxis.set_ticklabels([])

        i += 1

    plt.tight_layout(w_pad = 0.5)
    cax = plt.axes([1, 0.15, 0.013, 0.7])
    clb = plt.colorbar(sm, cax=cax, extend = 'both', label = clabel)

    print('reference arrows are 15 cm/s')

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def plotSpeedTimeseries(float_num, floatid, u, v, vspan_lim = None):
    ''' Plot surface speed timeseries using the float relative veocities (averaged profile pairs).
    To shade a part of the timeseries, enter the x limits in vspan_lim.''' 
    rs = calc.findRSperiod(float_num)
    dist = calc.distFromStart(float_num)

    combined_u, combined_v = avgProfilePairs(float_num, u, v)
    speed = calc.speed(combined_u, combined_v)

    surf_speed = speed.sel(pressure = slice(0,1600)).mean(dim = 'pressure')
    # surf_speed2 = speed.sel(pressure = slice(0,500)).mean(dim = 'pressure')

    rs_dives = slice(0,int(len(float_num.profile[rs])/2))

    dist_dives = [dist[0:2].values.mean()]

    for i in range(0,len(float_num.profile)-3,2):
        dive_d = dist[i+2:i+4].values.mean()
        dist_dives.append(dive_d)

    fig, ax = plt.subplots(figsize = (10,3))

    plt.plot(dist_dives[rs_dives],surf_speed[rs_dives], c='orange')
    # plt.plot(dist_dives[rs_dives],surf_speed2[rs_dives])

    settings.tickLocations(ax)
    if floatid == 8490: 
        settings.tickLocations(ax, major = 200)

    if vspan_lim != None:
        plt.axvspan(vspan_lim[0], vspan_lim[-1], color='grey', alpha=0.1, lw=0)

    plt.axhline(y = 0.05, linestyle = '--', c = 'grey')
    plt.grid(True)
    plt.ylabel('speed (m $s^{-1}$)')
    plt.xlabel('distance (km)')
    # plt.legend(['0-100 m', '0-500 m'])

    return surf_speed[rs_dives]

#-------------------------------------------------------------------------------------------------------------------------------------------------------
# For velocity offset calculations 

def CTD_dt(float_num, prof):
    '''Calculates the time difference between surfacing profiles according to the time recorded by the CTD on the float at every depth level.'''
    # CTD: if nan values are present, trace back in time using constant velocity from the first valid datetime.
    if len(np.where(~np.isnat(float_num.ctd_t[prof].values))[0]) < 1:
        print(f'profile {prof}: no ctd time data in down cast')
        return np.nan
    if len(np.where(~np.isnat(float_num.ctd_t[prof+1].values))[0]) < 1:
        print(f'profile {prof+1}: no ctd time data in up cast')
        return np.nan

    dt = np.diff(float_num.ctd_t[prof].values).astype('timedelta64[s]')
    index = np.where(~np.isnat(dt))[0][0] 
    
    t1 = float_num.ctd_t[prof].values[index] - np.timedelta64((dt[index].astype('int')*index),'s')
    t2 = float_num.ctd_t[prof+1].values[0]
    
    if np.isnat(t2):
        dt2 = np.diff(float_num.ctd_t[prof+1].values).astype('timedelta64[s]')
        index = np.where(~np.isnat(dt2))[0][0] 
        t2 = float_num.ctd_t[prof+1].values[index] + np.timedelta64((dt2[index].astype('int')*index),'s')

    ctd_dt = (t2-t1).astype('timedelta64[s]')
    
    return ctd_dt

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def erroneous_rel_vels(velocity, floatid):
    '''Remove additional velocity spikes / erroneous profiles'''
    if floatid == 8489:
        odd_profile = [40, 41, 84, 88, 90, 160, 161, 172, 173, 240, 241, 256]
        new = stats.delOddProfiles(velocity, odd_profile)

    if floatid == 8492:
        odd_profile = [84, 88, 118]
        new = stats.delOddProfiles(velocity, odd_profile)

    if floatid == 8493:
        odd_profile = [6, 7, 34, 35, 40, 41, 88, 89, 94, 120, 121, 220, 221, 250]
        new = stats.delOddProfiles(velocity, odd_profile)

    return new


def calcVelOffset(float_num, floatid, u_rel, v_rel, prof):
    '''Calculates the velocity offset using the time difference between the float surfacing from the CTD and the GPS.
    Relative velocities through the water column are used to calculate the predicted position lat lon postiion when the float resurfaces.'''
    # time difference at the surface for both gps and ctd

    if prof % 2 != 0:
        warnings.warn("Odd number profile entered. Profile before has been used to produce the correct offset", UserWarning)
        prof -= 1

    u_rel = erroneous_rel_vels(u_rel, floatid)
    v_rel = erroneous_rel_vels(v_rel, floatid)

    lons = float_num.longitude
    lats = float_num.latitude
    
    dt_gps = (float_num.time[prof+1] - float_num.time[prof]).values.astype('timedelta64[s]')
    dt_ctd = CTD_dt(float_num, prof)
    
    # fill in nan values at the surface and the bottom with nearest value
    u1, v1 = fillnans(u_rel[prof]), fillnans(v_rel[prof])
    u2, v2 = fillnans(u_rel[prof+1]), fillnans(v_rel[prof+1])
    dt1 = fillnans(np.diff(float_num.ctd_t[prof].values).astype('timedelta64[s]'))
    dt2 = fillnans(np.diff(float_num.ctd_t[prof+1].values).astype('timedelta64[s]'))
    
    # calculate dx and dy (in metres) at every depth from the CTD
    dx1, dy1, dist1 = subsurface_dxdy(u1, v1, dt1)
    dx2, dy2, dist2 = subsurface_dxdy(u2, v2, dt2)

    # if a profile contains no velocity data, set profile pair to nan
    if len(np.where(~np.isnan(dist1))[0]) < 1:
        print(f'profile {prof}: no velocity data in down cast')
        return np.nan, np.nan
    if len(np.where(~np.isnan(dist1))[0]) < 1:
        print(f'profile {prof+1}: no velocity data in up cast')
        return np.nan, np.nan
    
    # integrate relative velocities through the water column with resepct to time 
    meanu = np.nansum(np.concatenate((dx1, dx2)).tolist())/dt_ctd.astype('float')
    meanv = np.nansum(np.concatenate((dy1, dy2)).tolist())/dt_ctd.astype('float')
    # should be the same as taking the depth-avg relative velocity measured at each pressure level?
    # meanu = np.nanmean(np.concatenate((u1, u2)).tolist())
    # meanv = np.nanmean(np.concatenate((v1, v2)).tolist())
    # print(meanu1, meanu)
    # print(meanv1, meanv)
    
    # calculate GPS displacement 
    dx_gps, dy_gps = surface_dxdy(lons[prof], lats[prof], lons[prof+1], lats[prof+1])
    
    if lats[prof+1] <  lats[prof]:
        # print(f'profile {prof} to {prof+1}: dy is negative')
        dy_gps *= -1
    if lons[prof+1] <  lons[prof]:
        # print(f'profile {prof} to {prof+1}: dx is negative')
        dx_gps *= -1
    
    # the depth-integrated velocity according to the GPS 
    u_gps = dx_gps/dt_gps.astype('float')
    v_gps = dy_gps/dt_gps.astype('float')

    # calculate the velocity offset (gps - ctd)
    u_offset = u_gps - meanu
    v_offset = v_gps - meanv
    
    return u_offset, v_offset 

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def surface_dxdy(lon1, lat1, lon2, lat2):
    '''Calculates the x and y distances in m between two lat and lon coordinates'''
    lons = [lon1, lon2]
    lats = [lat1, lat1]
    dx = gsw.distance(lons, lats)
    
    lons = [lon1, lon1]
    lats = [lat1, lat2]
    dy = gsw.distance(lons, lats)
    
    return dx, dy

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def fillnans(var):
    '''Fill nan values at the surface and the bottom of each profile with the nearest value to obtain assume a constant flow'''
    lst = []
    for i in range(0, len(var)):
        if np.isnan(var[i]):
            if len(var[~np.isnan(var)]) < 1:
                lst.append(np.nan)
            elif i < 10:
                lst.append(var[~np.isnan(var)][0])
            elif i > len(var) - 20:
                lst.append(var[~np.isnan(var)][-1])
            else:
                lst.append(var[i-1])
        else:
            lst.append(var[i])
    return np.asarray(lst)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def uvBearing(u, v):
    '''Calculates the bearing (clockwise from True North) using eastward (u) and northward (v) components of velocity'''
    theta = np.rad2deg(np.arctan2(u, v))
    theta += 360
    theta = theta % 360
    return theta

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def ctd_lat_lon(lat1, lon1, bearings, dist):
    ''' Use velocity bearings and distance between ctd measurements to find the lat and lon of the subsequent measurement'''
    geod = Geodesic.WGS84 
    # geod = pyproj.Geod(ellps='WGS84')

    ctd_lat, ctd_lon = [], []
    for i in range(0, len(bearings)):
        g = geod.Direct(lat1, lon1, bearings[i], dist[i])
        lat2, lon2 = g['lat2'], g['lon2']
        ctd_lat.append(lat2)
        ctd_lon.append(lon2)

        lat1, lon1 = lat2, lon2
    
    return ctd_lat, ctd_lon

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def subsurface_dxdy(u, v, dt):
    '''Calculates dx and dy in metres using the current speed at each depth and the bearing (direction of current) '''

    speed = calc.speed(u,v)

    # replace nans with the time difference of the nearest measurement (assume constant speed). 
    if len(np.where(~np.isnan(dt))[0]) < 1:
        return np.nan, np.nan, np.nan

    index = 0
    dt = np.abs(np.insert(dt, index, dt[0]))

    # calculate the distance between each measurement using (distance = speed x time). 
    dist = []
    for i in range(0, len(dt)):
        distance = float(speed[i]) * dt[i].astype('float')
        dist.append(distance)

    bearings = uvBearing(u, v)

    # calculate dx and dy using trigonometry rules. 
    dx = []
    dy = []
    for i in range(0, len(bearings)):
            if 0 < bearings[i] < 90:
                # top right quarter
                theta = 90 - bearings[i]
                x = np.cos(theta * np.pi / 180)*dist[i]
                y = np.sin(theta * np.pi / 180)*dist[i]
                dx.append(x)
                dy.append(y)

            if 90 < bearings[i] < 180:
                # bottom right quarter
                theta = bearings[i] - 90
                x = np.cos(theta * np.pi / 180)*dist[i]
                y = np.sin(theta * np.pi / 180)*dist[i]
                dx.append(x)
                dy.append(y*-1)

            if 180 < bearings[i] < 270:
                # bottom left quarter
                theta = 270 - bearings[i]
                x = np.cos(theta * np.pi / 180)*dist[i]
                y = np.sin(theta * np.pi / 180)*dist[i]
                dx.append(x*-1)
                dy.append(y*-1)

            if 270 < bearings[i] < 360:
                # top left quarter
                theta = bearings[i] - 270
                x = np.cos(theta * np.pi / 180)*dist[i]
                y = np.sin(theta * np.pi / 180)*dist[i]
                dx.append(x*-1)
                dy.append(y)
                
    return dx, dy, dist

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def calcAbsoluteVelocity(float_num, floatid, u_rel, v_rel, prof):
    '''Calculates absolute velocity = relative velocity + velocity offset.
    Returns absolute velocity for both down and up profiles.'''
    u_offset, v_offset = calcVelOffset(float_num, floatid, u_rel, v_rel, prof)
    u_rel1, v_rel1 = fillnans(u_rel[prof]), fillnans(v_rel[prof])
    u_rel2, v_rel2 = fillnans(u_rel[prof+1]), fillnans(v_rel[prof+1])

    # absolute velocity
    u_abs1 = u_rel1 + u_offset
    v_abs1 = v_rel1 + v_offset
    u_abs2 = u_rel2 + u_offset
    v_abs2 = v_rel2 + v_offset

    return u_abs1, v_abs1, u_abs2, v_abs2

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def plotSubsurfaceTrajectory(float_num, floatid, u_rel, v_rel, prof, abs_vels = True, print_offset = False):
    '''plot the subsurface trajectory between profiles (from the surface down to 1600 m and back up to the surface).
    Use either absolute velocities (relative velocities adjusted with velocity offset) - set abs_vels = True.
    Or use relative velocities (no offset) - set abs_vels = False. '''

    if prof % 2 != 0:
        raise ValueError('Error: enter an even profile number starting from 0')

    # DOWN profile
    u_rel1, v_rel1 = fillnans(u_rel[prof]), fillnans(v_rel[prof])
    lon1_down = float_num.longitude[prof].values
    lat1_down = float_num.latitude[prof].values

    if abs_vels == True:
        u_offset, v_offset = calcVelOffset(float_num, floatid, u_rel, v_rel, prof)
        print('u offset: {}, v offset: {}'.format(u_offset, v_offset))
    else:
        # set the offset to zero
        u_offset, v_offset = 0, 0

    # absolute velocity
    u1 = u_rel1 + u_offset
    v1 = v_rel1 + v_offset

    bearings1 = uvBearing(u1, v1)
    dt1 = fillnans(np.diff(float_num.ctd_t[prof].values).astype('timedelta64[s]'))
    dx1, dy1, dist1 = subsurface_dxdy(u1, v1, dt1)
    ctd_lats1, ctd_lons1 = ctd_lat_lon(lat1_down, lon1_down, bearings1, dist1)

    # UP profile
    u_rel2, v_rel2 = fillnans(u_rel[prof+1]), fillnans(v_rel[prof+1])
    u2 = u_rel2 + u_offset
    v2 = v_rel2 + v_offset

    # use final position of previous profile for lat1 and lon1 (end of down start of up profile)
    lon1_up = np.asarray(ctd_lons1)[~np.isnan(ctd_lons1)][-1]
    lat1_up = np.asarray(ctd_lats1)[~np.isnan(ctd_lats1)][-1]

    bearings2 = uvBearing(u2, v2)
    dt2 = fillnans(np.diff(float_num.ctd_t[prof+1].values).astype('timedelta64[s]'))
    dx2, dy2, dist2 = subsurface_dxdy(u2, v2, dt2)
    ctd_lats2, ctd_lons2 = ctd_lat_lon(lat1_up, lon1_up, bearings2, np.abs(dist2))

    # PLOT on map 
    fig, ax = plt.subplots(figsize = (7,4))
    im = plt.scatter(ctd_lons1, ctd_lats1, c = float_num.pressure, s = 60)
    im = plt.scatter(ctd_lons2, ctd_lats2, c = np.flip(float_num.pressure), s = 60)
    plt.scatter(float_num.longitude[prof:prof+2], float_num.latitude[prof:prof+2], c = 'r')
    plt.colorbar(im, label = 'pressure (dbar)')
    ax.set_xlabel(u'Longitude [\N{DEGREE SIGN}E]')
    ax.set_ylabel(u'Latitude [\N{DEGREE SIGN}N]')

    # set axis limits
    ctd_lats = np.concatenate((ctd_lats1, ctd_lats2)).tolist()
    lats = np.concatenate((float_num.latitude[prof:prof+2].values, ctd_lats))

    y_min = np.nanmin(lats) - 0.005
    y_max = np.nanmax(lats) + 0.005
    plt.ylim(y_min, y_max)

    ctd_lons = np.concatenate((ctd_lons1, ctd_lons2)).tolist()
    lons = np.concatenate((float_num.longitude[prof:prof+2].values, ctd_lons))
    
    x_min = np.nanmin(lons) - 0.005
    x_max = np.nanmax(lons) + 0.005
    plt.xlim(x_min, x_max)

    if print_offset == True:
        ax.text(0.05, 0.24, 'offset ($m$ $s^{-1}$)', transform = ax.transAxes, fontsize = 12)
        ax.text(0.06, 0.16, f'u = {u_offset[0]:.3f}', transform = ax.transAxes, fontsize = 12)
        ax.text(0.06, 0.08, f'v = {v_offset[0]:.3f}', transform = ax.transAxes, fontsize = 12)

    ax.text(0.04, 0.9, f'profile no.: {prof+1} & {prof+1+1}', transform = ax.transAxes, fontsize = 12)

    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    
    if abs_vels == True:
        ax.set_title('abs = relative + offset')
    else:
        ax.set_title('relative velocities')

    return fig

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def location_at_bottom(surface_lat, surface_lon, ctd_time, prof, u_abs, v_abs):
    '''Get the estimated lat lon location at the bottom of a down profile using the abosolute velocities.
    Surface lat and lon positions of float from GPS.
    ctd_time = time recorded by the ctd as the float is profiling under water.'''
        
    #even profile index = down profile
    if prof % 2 != 0:
        raise ValueError('Error: enter an even profile number starting from 0')
    
    nonans = ~np.isnan(u_abs[prof])
    len_nonans = len(u_abs[prof].data[nonans])
    nonats = ~np.isnat(ctd_time[prof])
    len_nonat = len(ctd_time[prof].data[nonats])

    if (len_nonans > 1) & (len_nonat > 1):
        lon1_down = surface_lon[prof].values
        lat1_down = surface_lat[prof].values

        bearings1 = uvBearing(u_abs[prof], v_abs[prof])
        dt1 = fillnans(np.diff(ctd_time[prof].values).astype('timedelta64[s]'))

        dx1, dy1, dist1 = subsurface_dxdy(u_abs[prof], v_abs[prof], dt1)
        ctd_lats1, ctd_lons1 = ctd_lat_lon(lat1_down, lon1_down, bearings1, dist1)

        bottom_lat = ctd_lats1[-1]
        bottom_lon = ctd_lons1[-1]
    else:
        bottom_lat, bottom_lon = np.nan, np.nan

    return bottom_lat, bottom_lon



def location_at_middepth(surface_lat, surface_lon, ctd_time, prof, u_abs, v_abs):
    '''Get the estimated lat lon location at the middle of a profile using the abosolute velocities.
    Surface lat and lon positions of float from GPS.
    ctd_time = time recorded by the ctd as the float is profiling under water.'''

    nonans = ~np.isnan(u_abs[prof])
    len_nonans = len(u_abs[prof].data[nonans])
    nonats = ~np.isnat(ctd_time[prof])
    len_nonat = len(ctd_time[prof].data[nonats])

    if (len_nonans > 1) & (len_nonat > 1):

        lon_gps = surface_lon[prof].values
        lat_gps = surface_lat[prof].values

        vel_bearings = uvBearing(u_abs[prof], v_abs[prof])
        dt = fillnans(np.diff(ctd_time[prof].values).astype('timedelta64[s]'))

        dx, dy, dist1 = subsurface_dxdy(u_abs[prof], v_abs[prof], dt)
        ctd_lats, ctd_lons = ctd_lat_lon(lat_gps, lon_gps, vel_bearings, dist1)

        mid_ind = int(len(ctd_lats)/2)

        mid_lat = ctd_lats[mid_ind]
        mid_lon = ctd_lons[mid_ind]
    else:
        mid_lat, mid_lon = np.nan, np.nan

    return mid_lat, mid_lon

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def createAbsVelDataset(float_num, floatid, u_rel, v_rel, by_dist = False, save_file = False, filename = 'abs_vel_%s_extra_qc') :
    '''Create an xarray dataset with absolute velocity data and save it to a file'''
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    rs = calc.findRSperiod(float_num)
    dist = calc.distFromStart(float_num)
    # dist = calc.cum_dist(brng_lons, brng_lats) # TO DO: Use the distance calculated from brng lat and lons 

    shp = u_rel[rs].shape
    u_abs = np.ma.masked_all(shp)
    v_abs = np.ma.masked_all(shp)

    for prof in range(0, len(float_num.profile[rs]), 2):
        # print(prof)
        u_abs1, v_abs1, u_abs2, v_abs2 = calcAbsoluteVelocity(float_num, floatid, u_rel, v_rel, prof)
        
        u_abs[prof,] = u_abs1
        u_abs[prof+1,] = u_abs2
        
        v_abs[prof,] = v_abs1
        v_abs[prof+1,] = v_abs2
    
    if by_dist == True:
        folder = 'by_distance'
        abs_vel = xr.Dataset(data_vars=dict(u_abs =(["distance", "pressure"], u_abs.data),
                                v_abs =(["distance", "pressure"], v_abs.data),),
                coords=dict(
                    distance =("distance", dist[rs].data),
                    pressure = ("pressure", float_num.pressure.data),))
    else:
        folder = 'by_profile'
        abs_vel = xr.Dataset(data_vars=dict(u_abs =(["profile", "pressure"], u_abs.data),
                                    v_abs =(["profile", "pressure"], v_abs.data),),
                    coords=dict(
                        profile =("profile", float_num.profile[rs].data),
                        pressure = ("pressure", float_num.pressure.data),))

    abs_vel['u_abs'].attrs = {'units':'m $s^{-1}$', 'long_name':'zonal absolute velocity'}
    abs_vel['v_abs'].attrs = {'units':'m $s^{-1}$', 'long_name':'meridional absolute velocity'}
    abs_vel['pressure'].attrs = {'units':'dbar', 'long_name':'sea water pressure'}
    abs_vel.attrs = {'creation_date':str(datetime.datetime.now()), 'author':'Maya Jakes', 'email':'maya.jakes@utas.edu.au'}

    if save_file == True:
        # Save to NETCDF
        datadir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data', 'floats', 'absolute_velocity')
        name = filename %floatid + '.nc' 
        # name = 'abs_vel' + f'_{floatid}_' + '.nc' 
        settings.save_to_netCDF(abs_vel, os.path.join(datadir, folder), name)

    return abs_vel

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# set odd velocity profiles to nan
def setAbsVelToNan(floatid, data):
    '''Deal with stange absolute velocity profiles that loop back on themselves due to similar start and end positions.
    TO DO: Calculate a way to automate this - identify surface lat and lons that are close together'''
    if floatid == 8489:
        odd_profile = [198, 199]
        new = stats.delOddProfiles(data, odd_profile)
    if floatid == 8492:
        odd_profile = [40, 41, 44, 45, 80, 81, 120, 121, 136, 137, 200, 201, 234, 235, 240, 241, 242, 243]
        new = stats.delOddProfiles(data, odd_profile)
    if floatid == 8493:
        odd_profile = [72, 73, 80, 81, 200, 201, 240, 241]
        new = stats.delOddProfiles(data, odd_profile)
    if floatid == 8495:
        odd_profile = [40, 80, 160, 240]
        new = stats.delOddProfiles(data, odd_profile)
    if floatid == 7789:
        odd_profile = [60, 100, 122, 134, 136]
        new = stats.delOddProfiles(data, odd_profile)
    else:
        new = data.copy() 
    return new
        
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def calcSpeedAndRotation(float_num, floatid, remove_odd_profiles = True, plot = True):
    '''Calculate depth-averaged speed and velocity rotation from 200-1500 dbar'''

    rs = calc.findRSperiod(float_num)
    dist = calc.distFromStart(float_num)
    
    datadir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data', 'floats', 'absolute_velocity')
    abs_v = imports.importNetCDF(datadir, 'abs_vel_%s.nc' %floatid, datatype ='by_profile')
    
    speed = calc.speed(abs_v.u_abs, abs_v.v_abs)
    avg_speed = speed.mean(dim='pressure')
    
    vel_200_1500 = []
    for i in range(0, len(abs_v.profile)):
        bearings = uvBearing(abs_v.u_abs[i].values, abs_v.v_abs[i].values, positive = True)

        diff = abs(bearings[100] - bearings[750]) # 100 (200 dbar) 750 (1500 dbar)
        if diff > 180:
            diff = abs(360 - diff)

        vel_200_1500.append(diff)
    
    if remove_odd_profiles == True:
        avg_speed = setAbsVelToNan(floatid, avg_speed)
        vel_200_1500 = setAbsVelToNan(floatid, np.asarray(vel_200_1500))

    if plot == True:
        # depth-averaged speed
        fig, ax = plt.subplots(figsize = (10,3))
        plt.plot(dist[rs], avg_speed[rs], color = 'orange')
        plt.grid()
        plt.ylim(-0.02,0.3)
        plt.ylabel('speed (m $s^{-1}$)')
        plt.xlabel('distance (km)')

        settings.tickLocations(ax)
        if floatid == 8490: 
            settings.tickLocations(ax, major = 200)

        # rotation 200 - 1500 dbar
        fig, ax = plt.subplots(figsize = (10,3))
        plt.plot(dist[rs], vel_200_1500)
        plt.grid()
        plt.ylim(-12,180)
        plt.ylabel('\u03B1 (\xb0)')
        plt.title('200 dbar - 1500 dbar')
        plt.xlabel('distance (km)')

        settings.tickLocations(ax)
        if floatid == 8490: 
            settings.tickLocations(ax, major = 200)
        
    return speed, avg_speed, vel_200_1500

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def velShearSection(float_num, floatid, ref = 200, plot = True, remove_odd_profiles = True, smooth = False):
    '''Plot velocity shear (angle from 200 dbar) through the water column. Positive values indicate clockwise rotation, 
    negative values indicate anticlockwise rotation  with depth.'''
    rs = calc.findRSperiod(float_num)
    dist = calc.distFromStart(float_num)

    datadir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data', 'floats', 'absolute_velocity')
    abs_v = imports.importNetCDF(datadir, 'abs_vel_%s.nc' %floatid, datatype ='by_profile')

    if smooth == True:
        u = smooth_prof_by_prof(abs_v.u_abs, window = 75)
        v = smooth_prof_by_prof(abs_v.v_abs, window = 75)
    else:
        u, v = abs_v.u_abs, abs_v.v_abs

    shear = np.ma.masked_all(u.shape)
    for i in range(0, len(u)):
        bearings = uvBearing(u[i].values, v[i].values)
        diff = bearings - bearings[int(ref/2)] # angle from 200 dbar

        for j in range(0, len(diff)):
            if diff[j] < -180:
                diff[j] = diff[j] + 360
            if diff[j] > 180:
                diff[j] = diff[j] - 360
                
        shear[i,:] = diff

    vel_shear = xr.DataArray(data=shear, dims=["distance","pressure"], coords=dict(
        distance=(["distance"], dist[rs].data),
        pressure=(["pressure"], abs_v.v_abs.pressure.data)),)

    if remove_odd_profiles == True:
        vel_shear = setAbsVelToNan(floatid, vel_shear)
    
    if plot == True:
        fig, ax = plt.subplots(figsize = (10,3))
        vel_shear.plot(x = 'distance', cmap = 'RdBu_r', vmin = -60, vmax = 60, alpha = 0.8, cbar_kwargs = dict(label = '\u03B1 (\xb0)', extend = 'both'))
        ax.invert_yaxis()
        ax.set_ylabel('pressure (dbar)')
        ax.set_xlabel('distance (km)')
        plt.title('velocity shear (angle from {} dbar)'.format(ref))

    return vel_shear

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def floatTrackBearing_old_method(float_num, smooth_gps = True, window = 9):
    ''' Calculates the bearing (clockwise from True North) between each lat and lon position.
    ==============================================================================
    INPUTS:
    lons = 1D array of longitude values
    lats = 1D array of latitude values
    
    OUTPUT:
    1D array of bearings (in degrees)
    '''
    if smooth_gps == True:
        lons = interp.gaussianFilter(float_num.longitude, window = window, interp_na = True)
        lats = interp.gaussianFilter(float_num.latitude, window = window, interp_na = True)
    else:
        lons = xr.DataArray(float_num.longitude.data)
        lons = lons.interpolate_na(dim = 'dim_0').values

        lats = xr.DataArray(float_num.latitude.data)
        lats = lats.interpolate_na(dim = 'dim_0').values

    bearing = []
    for i in range(0, len(lats)-1):
        lat1, lat2 = lats[i], lats[i+1]
        lon1, lon2 = lons[i], lons[i+1]

        geodesic = pyproj.Geod(ellps='WGS84')  # WGS84 is the reference coordinate system (Earth's centre of mass) used by GPS. 
        # Inverse computation to calculate the forward and back azimuths and distance from two lat and lon coordinates. 
        fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)

        # if the angle is negative (anticlockwise from N), add it to 360 to get the bearing clockwise from N.
        fwd_azimuth += 360
        fwd_azimuth = fwd_azimuth % 360

        if fwd_azimuth == 0:
            bearing.append(np.nan)
        else:
            bearing.append(fwd_azimuth)

    return np.asarray(bearing)



def floatTrackBearing(floatid, lats, lons, ctd_time, u_abs, v_abs):
    ''' Calculates the bearing (clockwise from True North) corresponding to the depth-integrated strean direction associated with each profile.
    ==============================================================================
    INPUTS:
    lons = 1D array of longitude values
    lats = 1D array of latitude values
    abs_vels = absolute velocities
    ctd_time = time recorded by ctd on float
    
    OUTPUT:
    1D array of bearings (in degrees)
    '''

    u = setAbsVelToNan(floatid, u_abs)
    v = setAbsVelToNan(floatid, v_abs)

    u = erroneous_rel_vels(u, floatid)
    v = erroneous_rel_vels(v, floatid)

    # latitude and longitude positions for where the bearing is taken from (surface of down profile and bottom of down profile)
    brng_lats = []
    brng_lons = []

    bearing = np.zeros(lons.shape)*np.nan
    for i in range(0, len(bearing)-1, 2):
        # bearing from surface gps to bottom of down
        lat1, lon1 = lats[i], lons[i]

        brng_lats.append(lat1)
        brng_lons.append(lon1)
    
        lat2, lon2 = location_at_bottom(lats, lons, ctd_time, i, u, v)

        brng_lats.append(lat2)
        brng_lons.append(lon2)

        geodesic = pyproj.Geod(ellps='WGS84')
        fwd_azimuth = geodesic.inv(lon1, lat1, lon2, lat2)[0]

        fwd_azimuth += 360
        fwd_azimuth = fwd_azimuth % 360

        if fwd_azimuth == 0:
            bearing[i] = np.nan
        else:
            bearing[i] = fwd_azimuth

        # bearing from bottom of down to resurface
        lat3, lon3 = lats[i+1], lons[i+1]

        fwd_azimuth = geodesic.inv(lon2, lat2, lon3, lat3)[0]

        fwd_azimuth += 360
        fwd_azimuth = fwd_azimuth % 360

        if fwd_azimuth == 0:
            bearing[i+1] = np.nan
        else:
            bearing[i+1] = fwd_azimuth
    
    brng_lats = np.asarray(brng_lats)
    brng_lons = np.asarray(brng_lons)

    # TO DO: extract time at the bottom to be used for the time associated with the up profiles?

    return bearing, brng_lats, brng_lons


# ------------------------------------------------------------------------------------------------------------------------------------------------------

def get_streamline_paths(paths):
    '''Extract x and y values from a path e.g. SSH streamline'''

    x_values = paths[0].vertices[:,0]
    y_values = paths[0].vertices[:,1]
    
    for ind in range(1, len(paths)):

        x = paths[ind].vertices[:,0]
        y = paths[ind].vertices[:,1]

        x_values = np.concatenate((x_values, x))
        y_values = np.concatenate((y_values, y))
        
    return(x_values, y_values)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def rotate_to_ssh(adt, obs, lons, lats, obs_times, smooth_contours = False, window = 15):
    '''Rotate velocities with respect to the orientation of SSH streamlines.
    obs = float_num or triaxus_vertical_cast'''

    bearing = []
    geodesic = pyproj.Geod(ellps='WGS84')

    times_no_nat = interp.interp_nats(obs_times, obs_times)
    adt_interp_t = adt.interp(time = times_no_nat)

    # interpolate adt onto lat and lon observation coords
    adt_interp = stats.temporalError(obs, adt, method = 'interp')[0]

    for i in range(0, len(adt_interp)):

        # ssh_contours = np.arange(np.floor(np.nanmin(adt_interp_t[i])-0.2), np.ceil(np.nanmax(adt_interp_t[i])+0.2), 0.01)
        ssh_contours = np.arange(-0.9, 0.4, 0.01)
        CS = adt_interp_t[i].plot.contour(levels = ssh_contours, alpha = 0.6)
        
        # find nearest SSH contour to float profile location 
        contour_ind, contour_val = stats.find_nearest(ssh_contours, [float(adt_interp[i])])

        # extract all x an y positions for that contour
        paths = CS.collections[contour_ind].get_paths()
        x, y = get_streamline_paths(paths)

        if smooth_contours == True:
            x = savgol_filter(x, window, 3, axis=-1, mode='interp')
            y = savgol_filter(y, window, 3, axis=-1, mode='interp')

        #calcualte the x-coordinate with the shortest distance from float position
        dist = []
        for k in range(0, len(x)):
            distance = geodesic.inv(x[k], y[k], lons[i], lats[i])[2]
            dist.append(distance)

        ind = np.where(dist == np.nanmin(dist))[0]

        if len(ind) > 1:
            ind = int(ind[0])
        elif len(ind) == 0:
            ind = np.nan
        else:
            ind = int(ind)

        if ~np.isnan(ind):
            try:
                # x and y position of SSH streamline nearest to float
                x1, y1 = x[ind], y[ind]
                # x and y position of SSH streamline ahead of float
                x2, y2 = x[ind+1], y[ind+1]
            except:
                # x and y position of SSH streamline behind float
                x1, y1 = x[ind-1], y[ind-1]
                # x and y position of SSH streamline nearest to float
                x2, y2 = x[ind], y[ind]

            # calcaulte the bearing from one contour coordinate to the next 
            fwd_azimuth, back_azimuth, distance = geodesic.inv(x1, y1, x2, y2)

            fwd_azimuth += 360
            fwd_azimuth = fwd_azimuth % 360
            if fwd_azimuth == 0:
                bearing.append(np.nan)
            else:
                bearing.append(fwd_azimuth)
            
        else:
            bearing.append(np.nan)

    ssh_brngs = np.asarray(bearing)

    u, v = obs.u, obs.v
    ssh_brngs_2d = np.tile(ssh_brngs,(len(v[v.dims[1]]), 1)).transpose()

    speed = calc.speed(u, v)
    velocity_bearing = uvBearing(u, v)

    theta = ssh_brngs_2d - velocity_bearing
    theta = (theta + 180) % 360 - 180

    u_rot = speed * np.cos(theta*np.pi/180)
    v_rot = speed * np.sin(theta*np.pi/180)

    return ssh_brngs, u_rot, v_rot

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def rotate_velocities_old_method(float_num, floatid, u, v, along_stream = None, smooth_vels = False, smooth_gps = True, window = 9):
    ''' Rotate velocities from eastward (u) and northward (v) to along-track (u_rot) and cross-track (v_rot), respectively.
        velocity_bearing = bearing of the velocities clockwise from True North
        stream_bearing = bearing from one profile to the next clockwise from True North

     ==============================================================================
     INPUT:
     u = eastward velocity component (could be 1D or 2D)
     v = northward velocity component (could be 1D or 2D)
     lons = 1D array of longitude values
     lats = 1D array of latitude values
     
     OUTPUT:
     Rotated velocities with one less observation in x than the input velocities (no forward azimuth from the last lat lon position)
     u_rot = along-stream velocity
     v_rot = cross-stream velocity 
     ==============================================================================
    '''
    rs = calc.findRSperiod(float_num)
    
    # u, v = setAbsVelToNan(floatid, u), setAbsVelToNan(floatid, v)

    speed = calc.speed(u, v)
    velocity_bearing = uvBearing(u, v)

    if along_stream is None:
        along_stream = floatTrackBearing_old_method(float_num, smooth_gps = smooth_gps, window = window)[rs]
    # make 1D array into 2D in the same shape as velocity bearings
    stream_bearing = np.tile(along_stream,(len(float_num.pressure), 1)).transpose()

    # find the angle between the velocity bearing and the along stream direction
    theta = stream_bearing - velocity_bearing
    theta = (theta + 180) % 360 - 180

    # calculate u and v using this new angle (converting degrees to radians)
    u_rot = speed * np.cos(theta*np.pi/180)
    v_rot = speed * np.sin(theta*np.pi/180)

    if smooth_vels == True: 
        u_rot = smooth_prof_by_prof(u_rot, window = 75, print_info = False)
        v_rot = smooth_prof_by_prof(v_rot, window = 75, print_info = False)

    return u_rot, v_rot, theta



def rotate_velocities(floatid, u_abs, v_abs, lats, lons, ctd_time, along_stream = None, smooth_vels = False):
    ''' Rotate velocities from eastward (u) and northward (v) to along-track (u_rot) and cross-track (v_rot), respectively.
        velocity_bearing = bearing of the velocities clockwise from True North
        stream_bearing = bearing from surface gps position to bottom of profile (downcast) and bottom of profile to resurface position (upcast).

     ==============================================================================
     INPUT:
     u_abs = eastward absolute velocity component (could be 1D or 2D)
     v_abs = northward absolute velocity component (could be 1D or 2D)
     lons = 1D array of longitude values
     lats = 1D array of latitude values
     ctd_time = subsurface time recorded by ctd on float
     
     OUTPUT:
     u_rot = along-stream velocity
     v_rot = cross-stream velocity 
     ==============================================================================
    '''
    
    u = setAbsVelToNan(floatid, u_abs)
    v = setAbsVelToNan(floatid, v_abs)

    u = erroneous_rel_vels(u, floatid)
    v = erroneous_rel_vels(v, floatid)

    speed = calc.speed(u, v)
    velocity_bearing = uvBearing(u, v)

    if along_stream is None:
        along_stream, brng_lats, brng_lons = floatTrackBearing(floatid, lats, lons, ctd_time, u, v)

    # make 1D array into 2D in the same shape as velocity bearings
    stream_bearing = xr.DataArray(np.tile(along_stream,(len(u.pressure), 1)).transpose(), dims = velocity_bearing.dims, 
                                                                    coords = velocity_bearing.coords)

    # find the angle between the velocity bearing and the along stream direction
    theta = stream_bearing - velocity_bearing
    theta = (theta + 180) % 360 - 180

    # calculate u and v using this new angle (converting degrees to radians)
    u_rot = speed * np.cos(theta*np.pi/180)
    v_rot = speed * np.sin(theta*np.pi/180)

    if smooth_vels == True: 
        u_rot = smooth_prof_by_prof(u_rot, window = 75, print_info = False)
        v_rot = smooth_prof_by_prof(v_rot, window = 75, print_info = False)

    return u_rot, v_rot, theta


# ------------------------------------------------------------------------------------------------------------------------------------------------------

def createRotVelDataset(float_num, floatid, u_abs, v_abs, ctd_time, by_dist = False, save_file = False):
    '''Create an xarray dataset with rotated velocity data and save it to a file'''
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    rs = calc.findRSperiod(float_num)
    # dist = calc.distFromStart(float_num)

    lats, lons = float_num.latitude[rs], float_num.longitude[rs]

    u = setAbsVelToNan(floatid, u_abs)
    v = setAbsVelToNan(floatid, v_abs)

    u = erroneous_rel_vels(u, floatid)
    v = erroneous_rel_vels(v, floatid)

    along_stream, brng_lats, brng_lons = floatTrackBearing(floatid, lats, lons, ctd_time, u, v)
    dist = calc.cum_dist(brng_lons, brng_lats)

    u_rot, v_rot, theta = rotate_velocities(floatid, u, v, lats, lons, ctd_time, along_stream = along_stream, smooth_vels = False)

    if by_dist == True:
        folder = 'by_distance'
        rot_vel = xr.Dataset(
            data_vars=dict(
                u_rot =(["distance", "pressure"], u_rot.data),
                v_rot =(["distance", "pressure"], v_rot.data),
                theta =(["distance", "pressure"], theta.data),
                stream_brng = (["distance"], along_stream.data)),
            coords=dict(
                brng_lats = ("latitude", brng_lats),
                brng_lons = ("longitude", brng_lons),
                distance =("distance", dist[rs].data),
                pressure = ("pressure", float_num.pressure.data),))
    else:
        folder = 'by_profile'
        rot_vel = xr.Dataset(data_vars=dict(u_rot =(["profile", "pressure"], u_rot.data),
                                    v_rot =(["profile", "pressure"], v_rot.data),
                                    theta =(["profile", "pressure"], theta.data),
                                    stream_brng = (["profile"], along_stream.data)),
                    coords=dict(
                        brng_lats = ("latitude", brng_lats),
                        brng_lons = ("longitude", brng_lons),
                        profile =("profile", float_num.profile[rs].data),
                        pressure = ("pressure", float_num.pressure.data),))

    rot_vel['u_rot'].attrs = {'units':'m $s^{-1}$', 'long_name':'along-track velocity'}
    rot_vel['v_rot'].attrs = {'units':'m $s^{-1}$', 'long_name':'cross-track velocity'}
    rot_vel['pressure'].attrs = {'units':'dbar', 'long_name':'sea water pressure'}
    rot_vel.attrs = {'creation_date':str(datetime.datetime.now()), 'author':'Maya Jakes', 'email':'maya.jakes@utas.edu.au'}

    if save_file == True:
        # Save to NETCDF
        datadir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data', 'floats', 'rotated_velocity')
        name = 'rot_vel' + f'_{floatid}' + '.nc' 
        settings.save_to_netCDF(rot_vel, os.path.join(datadir, folder), name)

    return rot_vel


# ------------------------------------------------------------------------------------------------------------------------------------------------------

# def plot_along_strm_dir(float_num, alt_cmems, prof1, prof2, window = 13, u = None, v = None, scale = 1):
#     '''Plot a quiver map showing the along-track bearing at each float profile.'''
#     rs = calc.findRSperiod(float_num)
    
#     start = float_num.time.values[prof1]
#     end = float_num.time.values[prof2]

#     start_time, end_time = str(start.astype('M8[D]')), str(end.astype('M8[D]'))

#     msl = alt_cmems.adt.sel(time = slice(start_time, end_time)).mean(dim = 'time').sel(
#                                                         latitude = slice(-56, -51), longitude = slice(148.7, 153.2))

#     levels = np.arange(-0.8,0.4,0.1)

#     # UNSMOOTHED 
#     along_stream = floatTrackBearing(float_num, smooth_gps = False)
#     angle_rad = np.deg2rad(along_stream)
#     y1 = 0.3 * np.cos(angle_rad) 
#     x1 = 0.3 * np.sin(angle_rad) 

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,6))
#     ax1.scatter(float_num.longitude[rs], float_num.latitude[rs], c= 'slategrey', alpha = 0.3)
#     ax1.scatter(float_num.longitude[prof1:prof2], float_num.latitude[prof1:prof2])
#     ax1.quiver(float_num.longitude[prof1:prof2], float_num.latitude[prof1:prof2], x1[prof1:prof2], y1[prof1:prof2], color = 'tab:red', scale = 3)

#     CS = msl.plot.contour(ax = ax1, colors = 'dimgrey', alpha = 0.5, linewidths = 1, levels = levels, zorder = 3)
#     plt.clabel(CS, inline=True, fontsize=11, fmt = '%1.1f')
    
#     # ax1.text(0.68, 0.04, f'prof {prof1}-{prof2}', transform = ax1.transAxes)

#     # ax1.set_title('stream bearing (unsmoothed GPS)')
#     ax1.set_title('unsmoothed GPS')
#     ax1.set_ylabel(u'Latitude [\N{DEGREE SIGN}N]')
#     ax1.set_xlabel(u'Longitude [\N{DEGREE SIGN}E]')

#     # SMOOTHED 
#     lons = interp.gaussianFilter(float_num.longitude, window = window, interp_na = True)
#     lats = interp.gaussianFilter(float_num.latitude, window = window, interp_na = True)

#     along_stream = floatTrackBearing(float_num, smooth_gps = True, window = window)
#     angle_rad = np.deg2rad(along_stream)
#     y2 = 0.3 * np.cos(angle_rad) 
#     x2 = 0.3 * np.sin(angle_rad) 

#     ax2.scatter(float_num.longitude[rs], float_num.latitude[rs], c= 'slategrey', alpha = 0.3)
#     ax2.scatter(lons[prof1:prof2], lats[prof1:prof2])
#     ax2.quiver(lons[prof1:prof2], lats[prof1:prof2], x2[prof1:prof2], y2[prof1:prof2], color = 'tab:red', scale =3)
    

#     CS = msl.plot.contour(ax = ax2, colors = 'dimgrey', alpha = 0.5, linewidths = 1, levels = levels, zorder = 3)
#     plt.clabel(CS, inline=True, fontsize=11, fmt = '%1.1f')
    
#     # ax2.text(0.68, 0.04, f'prof {prof1}-{prof2}', transform = ax2.transAxes)
    
#     ax2.set_ylabel(' ')
#     # plt.title('stream bearing (smoothed GPS)')
#     ax2.set_title('smoothed GPS')
#     ax2.set_xlabel(u'Longitude [\N{DEGREE SIGN}E]')
    
#     if u is not None:
#         xmin, xmax = np.nanmin(lons[prof1:prof2]), np.nanmax(lons[prof1:prof2])
#         ymin, ymax = np.nanmin(lats[prof1:prof2]), np.nanmax(lats[prof1:prof2])

#         # UNSMOOTHED
#         fig2, (ax1, ax2) = plt.subplots(1, 2, figsize = (11,4.5))
#         ax1.scatter(float_num.longitude[prof1:prof2], float_num.latitude[prof1:prof2], c = 'slategrey')
#         # surface velocity 
#         ax1.quiver(float_num.longitude[prof1:prof2], float_num.latitude[prof1:prof2], u[prof1:prof2].sel(pressure = 200), 
#                 v[prof1:prof2].sel(pressure = 200),  color = 'cyan', scale = scale+1)

#         # float track bearing
#         ax1.quiver(float_num.longitude[prof1:prof2], float_num.latitude[prof1:prof2], x1[prof1:prof2], y1[prof1:prof2], color = 'tab:red', scale = scale)

#         # ax1.legend(['gps', 'velocity at 200 dbar', 'stream bearing'], fontsize = 12, loc = 'lower right')
#         # ax1.set_title(f'prof {prof1}-{prof2}')
#         ax1.text(0.05, 0.9, f'prof {prof1}-{prof2}', transform = ax1.transAxes)
#         ax1.set_xlim(xmin - 0.3, xmax + 0.4)
#         ax1.set_ylim(ymin - 0.3, ymax + 0.5)
#         ax1.set_ylabel(u'Latitude [\N{DEGREE SIGN}N]')
#         ax1.set_xlabel(u'Longitude [\N{DEGREE SIGN}E]')

#         # SMOOTHED
#         ax2.scatter(lons[prof1:prof2], lats[prof1:prof2], c = 'slategrey')
#         # surface velocity 
#         ax2.quiver(float_num.longitude[prof1:prof2], float_num.latitude[prof1:prof2], u[prof1:prof2].sel(pressure = 200), 
#                 v[prof1:prof2].sel(pressure = 200),  color = 'cyan', scale = scale+1)

#         # float track bearing
#         ax2.quiver(lons[prof1:prof2], lats[prof1:prof2], x2[prof1:prof2], y2[prof1:prof2], color = 'tab:red', scale = scale)

#         plt.legend(['gps', 'velocity at 200 dbar', 'stream bearing'], fontsize = 13, bbox_to_anchor=(0.65,  1.15), ncol = 3)
#         # ax2.set_title(f'prof {prof1}-{prof2}')
#         ax2.text(0.05, 0.9, f'prof {prof1}-{prof2}', transform = ax2.transAxes)
#         ax2.set_xlim(xmin - 0.3, xmax + 0.4)
#         ax2.set_ylim(ymin - 0.3, ymax + 0.5)
#         ax2.set_xlabel(u'Longitude [\N{DEGREE SIGN}E]')

#         # plt.tight_layout(w_pad = 1.2)

#         return fig, fig2
#     else:
#         return fig


def plot_along_strm_dir(float_num, along_stream, brng_lats, brng_lons, alt_cmems, prof1, prof2, u = None, v = None, scale = 1):
    '''Plot a quiver map showing the along-track bearing at each float profile.'''
    rs = calc.findRSperiod(float_num)
    
    start = float_num.time.values[prof1]
    end = float_num.time.values[prof2]
    start_time, end_time = str(start.astype('M8[D]')), str(end.astype('M8[D]'))

    msl = alt_cmems.adt.sel(time = slice(start_time, end_time)).mean(dim = 'time').sel(latitude = slice(-56, -51), longitude = slice(148.7, 153.2))
    levels = np.arange(-0.8,0.4,0.1)

    # along_stream, brng_lats, brng_lons = floatTrackBearing(floatid, float_num.latitude[rs], 
    #                                                    float_num.longitude[rs], ctd_time[rs], u_abs, v_abs)
    angle_rad = np.deg2rad(along_stream)
    y1 = 0.3 * np.cos(angle_rad) 
    x1 = 0.3 * np.sin(angle_rad)

    fig, ax= plt.subplots(figsize = (5.5,6))
    ax.scatter(float_num.longitude[rs], float_num.latitude[rs], c= 'slategrey', alpha = 0.3)
    ax.scatter(brng_lons[prof1:prof2], brng_lats[prof1:prof2])
    ax.quiver(brng_lons[prof1:prof2], brng_lats[prof1:prof2], x1[prof1:prof2], y1[prof1:prof2], color = 'tab:red', scale = 2.5)

    CS = msl.plot.contour(colors = 'dimgrey', alpha = 0.5, linewidths = 1, levels = levels, zorder = 3)
    plt.clabel(CS, inline=True, fontsize=11, fmt = '%1.1f')
    
    # ax.text(0.68, 0.04, f'prof {prof1}-{prof2}', transform = ax.transAxes)
    ax.set_title(f'prof {prof1}-{prof2}', fontsize = 15)
    ax.set_ylabel(u'Latitude [\N{DEGREE SIGN}N]')
    ax.set_xlabel(u'Longitude [\N{DEGREE SIGN}E]')

    if u is not None:
        xmin, xmax = np.nanmin(brng_lons[prof1:prof2]), np.nanmax(brng_lons[prof1:prof2])
        ymin, ymax = np.nanmin(brng_lats[prof1:prof2]), np.nanmax(brng_lats[prof1:prof2])

        # UNSMOOTHED
        fig2, ax = plt.subplots(figsize = (5,4))
        ax.scatter(brng_lons[prof1:prof2], brng_lats[prof1:prof2],c = 'slategrey')
        # surface velocity 
        ax.quiver(brng_lons[prof1:prof2], brng_lats[prof1:prof2], u[prof1:prof2].sel(pressure = 200), 
                v[prof1:prof2].sel(pressure = 200),  color = 'cyan', scale = scale+1)

        # float track bearing
        ax.quiver(brng_lons[prof1:prof2], brng_lats[prof1:prof2], x1[prof1:prof2], y1[prof1:prof2], color = 'tab:red', scale = scale-0.5)

        ax.legend(['float location', 'velocity at 200 dbar', 'stream bearing'], fontsize = 12, loc = 'lower left')
        ax.text(0.05, 0.9, f'prof {prof1}-{prof2}', transform = ax.transAxes)
        ax.set_xlim(xmin - 0.3, xmax + 0.4)
        ax.set_ylim(ymin - 0.4, ymax + 0.4)
        ax.set_ylabel(u'Latitude [\N{DEGREE SIGN}N]')
        ax.set_xlabel(u'Longitude [\N{DEGREE SIGN}E]')

        return fig, fig2
    else:
        return fig


# ------------------------------------------------------------------------------------------------------------------------------------------------------

def remove_inertial(ema, floatids, abs_vels, inertial, rotate = True, save_file = False, filename = 'vel_rot_minus_inert'):
    '''Remove inertial velocities (provided by Ajitha) from absolute velocities. Inertial velocities provided in xarray DataSet
       lat and lons are different for the inertial velocities (mid-points?) 
       I used the original lat and lons to compute distance and be able to subtract from abs_vels
    '''
    if inertial is None:
        datadir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data', 'floats')
        inertial = {}
        for floatid in floatids:
            float_num = ema[floatid]
            inert = imports.importNetCDF(datadir, 'inertial-%s.nc' %int(str(floatid)[2:]), datatype ='inertial_velocities')
            inertial[floatid] = settings.distanceAsCoord(inert, float_num, rs = True)

    datadir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data', 'floats', 'absolute_velocity', 'by_distance')
            
    if len(floatids) > 1:
        vel_minus_inertial = {}
        vel_rot_minus_inert = {}
        for floatid in floatids:
            vel_minus_inertial[floatid] = (abs_vels[floatid].u_abs - inertial[floatid].u_inert).to_dataset(name = 'u')
            vel_minus_inertial[floatid]['v'] = abs_vels[floatid].v_abs - inertial[floatid].v_inert

            if rotate == True:
                u_rot, v_rot, theta = rotate_velocities(ema[floatid], floatid, vel_minus_inertial[floatid].u, vel_minus_inertial[floatid].v)
                vel_rot_minus_inert[floatid] = u_rot.to_dataset(name = 'u')
                vel_rot_minus_inert[floatid]['v'] = v_rot

                if save_file == True: 
                    os.chmod(os.path.join(datadir), stat.S_IWUSR | stat.S_IRUSR)
                    vel_rot_minus_inert[floatid].to_netcdf(os.path.join(datadir, filename + '_%s.nc' %floatid))

    else:
        floatid = floatids[0]
        vel_minus_inertial = (abs_vels.u_abs - inertial.u_inert).to_dataset(name = 'u')
        vel_minus_inertial['v'] = abs_vels.v_abs - inertial.v_inert

        if rotate == True:
            u_rot, v_rot, theta = rotate_velocities(ema[floatid], floatid, vel_minus_inertial.u, vel_minus_inertial.v)
            vel_rot_minus_inert = u_rot.to_dataset(name = 'u')
            vel_rot_minus_inert['v'] = v_rot
        
            if save_file == True:
                os.chmod(os.path.join(datadir), stat.S_IWUSR | stat.S_IRUSR)
                name = filename + f'_{floatid}' + '.nc' 
                settings.save_to_netCDF(vel_rot_minus_inert, datadir, name)

    return vel_minus_inertial

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def smooth_prof_by_prof(data, window = 75, print_info = False):
    '''Smooth xr dataArray profile by profile using a Savitz-Golay 1D filter that fits a 3rd degree polynomial to a rolling window of points.
        Default window length = 75'''
    data_smooth = np.zeros(data.shape)*np.nan

    for prof in range(0, len(data)):
        d_new = data[prof]
        
        if len(np.where(np.isnan(d_new))[0]) == len(d_new):
        # if len(d_new[~np.isnan(d_new)])== 0:
            if print_info == True:
                print(prof, 'empty profile')
        else:
            if np.isnan(data).any(): # if a nan value is present in the profile, fill with the nearest value
                d_new = d_new.interpolate_na(dim = 'pressure', method = 'nearest', fill_value="extrapolate")
            
            data_smooth[prof] = savgol_filter(d_new, window, 3, axis=-1, mode='interp')

    data_smooth = xr.DataArray(data=data_smooth, dims = [data.dims[0], data.dims[1]], coords = data.coords)

    return data_smooth

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def vert_vel_deriv(float_num, floatid, u, v, smooth_vels = True, save_file = False, filename = None):
    '''Use float_num dataset with variables that have had inertial oscillations removed 
    (i.e. via half inertial pair averaging) and then interpolated back onto the original times/locations.
    u and v are velocities rotated to along-stream and cross-stream.'''
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start = timer()

    t0 = float_num.time[0]
    lat = float_num.latitude[1]
    z = gsw.z_from_p(float_num.pressure, lat)

    # dist = calc.distFromStart(float_num)
    dist = float_num.distance

    # CT, SA = settings.remove_bad_T_S(float_num, floatid)
    CT, SA = float_num.CT, float_num.SA
    
    drho_dt, drho_dz, dv_dz, du_dz = (np.zeros(u.shape)*np.nan for _ in range(4))
    du_dx, dv_dx, d2u_dzdx, d2v_dzdx = (np.zeros(v.shape)*np.nan for _ in range(4))
    
    if smooth_vels == True:
        u = smooth_prof_by_prof(u)
        v = smooth_prof_by_prof(v)
        
    for i in range(0, len(u)-2):
        arr = [i, i+1, i+2]
        prof = i+1
        
        t = (float_num.ctd_t[arr] - t0).values.astype('timedelta64[s]') #from CTD
        d = dist[arr]*1000
        time = (float_num.time[arr] - t0).values.astype('timedelta64[s]') #from gps
        # c = d[-1]/np.nanmean(np.nanmax(time).astype(float))

        u_arr = u[arr]
        v_arr = v[arr]

        if len(np.unique(np.where(~np.isnan(u_arr))[0])) > 1:
            # need at least two data values to compute the gradient across profiles
            u_bar = np.nanmean(u_arr, axis = 0)
            v_bar = np.nanmean(v_arr, axis = 0)
        else:
            u_bar, v_bar = np.nan, np.nan

        rho = calc.potentialDensity(float_num.pressure, SA, CT)[arr] + 1000
        rho = rho.interpolate_na(dim = 'pressure', method = 'linear', fill_value="extrapolate")
        rho_bar = np.nanmean(rho, axis = 0)

        drho_dz[prof] = np.gradient(rho_bar)/np.gradient(z)
    
        nans = np.where(np.isnan(u_arr[1]))[0]
        if len(nans) == len(z):
            print(prof, 'empty velocity profile')
        else:  
            du_dx[prof] = np.gradient(u_arr)[0][1]/np.gradient(d)[1]
            dv_dx[prof] = np.gradient(v_arr)[0][1]/np.gradient(d)[1]
            du_dz[prof] = np.gradient(u_bar)/np.gradient(z)
            dv_dz[prof] = np.gradient(v_bar)/np.gradient(z)
            d2u_dzdx[prof] = np.gradient(du_dx[prof])/np.gradient(z)
            d2v_dzdx[prof] = np.gradient(dv_dx[prof])/np.gradient(z)
            

            for iz in range(0, len(z)):
              # First derivatives: gradient of middle profile using central differencing
                ii = np.where(~np.isnat(t[:,iz]))[0].tolist()
                if len(ii) > 1:
                    # drho_dx = np.gradient(rho[ii,iz])[1]/np.gradient(d)[1]
                    # drho_dt[prof, iz]  = (np.gradient(rho[ii,iz])[1] / np.gradient(t[ii,iz]/np.timedelta64(1, 's'))[1]) - c*drho_dx
                    drho_dt[prof, iz]  = np.gradient(rho[ii,iz])[1] / np.gradient(t[ii,iz]/np.timedelta64(1, 's'))[1]
                    

    ds = xr.Dataset(data_vars=dict(drho_dt=(["distance", "pressure"], drho_dt),
                                   drho_dz=(["distance", "pressure"], drho_dz),
                                   dv_dz=(["distance", "pressure"], dv_dz),
                                   du_dz=(["distance", "pressure"], du_dz),
                                   d2u_dzdx=(["distance", "pressure"], d2u_dzdx),
                                   d2v_dzdx=(["distance", "pressure"], d2v_dzdx),),
                             coords=dict(
                                 pressure=("pressure", float_num.pressure.data),
                                 distance = ("distance", dist.data)),)
    
    end = timer()
    print(f"Elapsed time: {(end - start)/60} minutes") 

    if save_file == True:
        dir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data', 'floats', 'vertical_velocity', 'by_distance')
        name = filename + f'_{floatid}' + '.nc' 
        settings.save_to_netCDF(ds, dir, name)               
    
    return ds                             

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def calc_w(float_num, floatid, u, v, deriv_ds, smooth_vels = True, smooth_gps = True, by_dist = True, save_file = False, filename = None):
    '''Calculates each term in the vertical velocity equation from Phillips & Bindoff (2014).
    Takes three profiles at a time and calculates derivatives centred on the middle profile (centered difference scheme)
    u and v are rotated velocities, with inertial oscillations removed through 1/2 inertial pair averaging.'''
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    g = 9.81
    t0 = float_num.time[0]
    # rs = calc.findRSperiod(float_num)

    # dist = calc.distFromStart(float_num)
    dist = float_num.distance

    # CT, SA = settings.remove_bad_T_S(float_num, floatid)
    CT, SA = float_num.CT, float_num.SA

    # smooth gps positions in the same way as is used in rotate_velocities
    K = calc.floatCurvature(float_num, transform = True, remove_outliers = True, smooth_gps = smooth_gps)
    
    w_tchng_a, w_tchng_b, w_rot, w_curv, w_accel = (np.zeros(u.shape)*np.nan for _ in range(5))
    
    if smooth_vels == True:
        u = smooth_prof_by_prof(u)
        v = smooth_prof_by_prof(v)

    for i in range(0, len(u)-2):
        prof = i+1 
        arr = [i, i+1, i+2]

        u_new = u[arr]
        v_new = v[arr]
        
        f = gsw.f(float_num.latitude)[arr][1]
        d = dist[arr]*1000
        time = (float_num.time[arr] - t0).values.astype('timedelta64[s]')
        c = d[-1]/np.nanmean(np.nanmax(time).astype(float))
        
        rho = calc.potentialDensity(float_num.pressure, SA, CT)[arr] + 1000
        rho = rho.interpolate_na(dim = 'pressure', method = 'linear', fill_value="extrapolate", limit=3)
        
        rho_bar = np.nanmean(rho, axis = 0)
        rho_bar = xr.DataArray(rho_bar, dims = ['pressure']).interpolate_na(dim = 'pressure', method = 'linear', 
                                                                fill_value="extrapolate", limit=3)
        
        drho_dz = deriv_ds.drho_dz[prof].copy()
        drho_dz[np.where(drho_dz == 0)] = np.nan
        
        w_tchng_a[prof] = -deriv_ds.drho_dt[prof] / drho_dz
        w_tchng_b[prof] = (-(c * f/g) * rho_bar * deriv_ds.dv_dz[prof]) / drho_dz
        w_rot[prof] = ((f * rho[1] / g) * (u_new[1] * deriv_ds.dv_dz[prof] - v_new[1] * deriv_ds.du_dz[prof])) / drho_dz
        w_curv[prof] = ((K[prof] * c * rho[1] / g) * (u_new[1] * deriv_ds.dv_dz[prof] - v_new[1] * deriv_ds.du_dz[prof])) / drho_dz
        w_accel[prof] = ((-rho[1] * c / g) * (u_new[1] * deriv_ds.d2u_dzdx[prof] + v_new[1] * deriv_ds.d2v_dzdx[prof])) / drho_dz   

        # w_tchng_a = setAbsVelToNan(floatid, w_tchng_a)
        # w_tchng_b = setAbsVelToNan(floatid, w_tchng_b)
        # w_rot = setAbsVelToNan(floatid, w_rot)
        # w_curv = setAbsVelToNan(floatid, w_curv)
        # w_accel = setAbsVelToNan(floatid, w_accel)

        if by_dist == True:
            folder = 'by_distance'
            
            vert_vel = xr.Dataset(data_vars=dict(w_tchng_a=(["distance", "pressure"], w_tchng_a.data),
                                                w_tchng_b=(["distance", "pressure"], w_tchng_b.data),
                                                w_rot=(["distance", "pressure"], w_rot.data),
                                                w_curv=(["distance", "pressure"], w_curv.data),
                                                w_accel=(["distance", "pressure"], w_accel.data),),
                            coords=dict(
                                pressure=("pressure", float_num.pressure.data),
                                distance = ("distance", dist.data)),)

        else:
            folder = 'by_profile'
            vert_vel = xr.Dataset(data_vars=dict(w_tchng_a=(["profile", "pressure"], w_tchng_a.data),
                                                w_tchng_b=(["profile", "pressure"], w_tchng_b.data),
                                                w_rot=(["profile", "pressure"], w_rot.data),
                                                w_curv=(["profile", "pressure"], w_curv.data),
                                                w_accel=(["profile", "pressure"], w_accel.data),),
                coords=dict(
                    pressure=("pressure", float_num.pressure.data),))
    
    # calculate total vertical velocity by performing a nansum
    ds_no_nan = vert_vel.fillna(0)
    w_total = ds_no_nan.w_tchng_a + ds_no_nan.w_tchng_b + ds_no_nan.w_rot + ds_no_nan.w_curv + ds_no_nan.w_accel
    
    # turn zeros back to nan
    mask = np.ma.masked_where(w_total == 0, w_total) 
    w_total.data[mask.mask] = np.nan
    # w_total = setAbsVelToNan(floatid, w_total)
    vert_vel['w_total'] = w_total

    if save_file == True: 
        # filename = 'vert_vel'
        dir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data', 'floats', 'vertical_velocity', folder)
        name = filename + f'_{floatid}' + '.nc' 
        settings.save_to_netCDF(vert_vel, dir, name)  
        
    return vert_vel

# ------------------------------------------------------------------------------------------------------------------------------------------------------














