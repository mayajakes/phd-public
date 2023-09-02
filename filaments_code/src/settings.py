# General settings
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import stat
import imp
import src.calc as calc
import src.interpolation as interp
import src.stats as stats
import src.velocities as vel
import pandas as pd
import datetime

from matplotlib.ticker import MultipleLocator
from astropy.convolution import convolve
from astropy.convolution import Box2DKernel
from astropy.convolution import Gaussian2DKernel
from scipy.signal import savgol_filter

from scipy.signal import butter, lfilter, filtfilt

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def remove_bad_T_S(float_num, floatid):

    if floatid == 8490:
        odd_profile = 119
        CT, SA = stats.delOddProfiles(float_num.CT, odd_profile), stats.delOddProfiles(float_num.SA, odd_profile)
    if floatid == 8492:
        odd_profile = [109, 177]
        CT, SA = stats.delOddProfiles(float_num.CT, odd_profile), stats.delOddProfiles(float_num.SA, odd_profile)
    else:
        CT, SA = float_num.CT, float_num.SA

    if CT.dims[0] == 'profile':
        CT = CT.assign_coords({"profile": np.arange(0, len(CT))})
        SA = SA.assign_coords({"profile": np.arange(0, len(CT))})

    return CT, SA

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def smooth(data, box_size = 3, gaussian = False, window = 13, return_xr = False):
    '''Apply a smoothing filter to either 2D or 1D dataArrays'''

    if len(data.shape) == 2:
        if gaussian == True:
            kernel = Gaussian2DKernel(x_stddev=1)
            data_smooth = convolve(data, kernel)
        else:
            box_kernel = Box2DKernel(box_size)
            data_smooth = convolve(data, box_kernel)

    elif len(data.shape) == 1:
        data_smooth = savgol_filter(data, window, 3, axis=-1, mode='interp')

    if return_xr == True:
        try:
            data_smooth = xr.DataArray(data=data_smooth, dims = [data.dims[0], data.dims[1]], coords = data.coords)
        except:
            data_smooth = xr.DataArray(data=data_smooth)

    return data_smooth

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def save_figure(fig, my_path, name, dpi = 300, pad = 0.2):
    '''example: 
    my_path = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd','figures',str(floatid), 'altimetry', 'CMEMS')
    name = 'float-{}-weeks-{}.png'.format(floatid, weeks[0])
    pad = space around figure in inches'''
    strFile = os.path.join(my_path, name)
    # remove old figure
    if os.path.isfile(strFile):
        os.remove(strFile)
    fig.savefig(strFile, dpi = dpi, bbox_inches='tight', pad_inches= pad)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def save_to_netCDF(ds, dir, name):
    print('saving to netcdf...')
    os.chmod(dir, stat.S_IRWXO)#stat.S_IWUSR | stat.S_IRUSR)
    # os.chmod(os.path.join(dir, name), stat.S_IRWXO)
    ds.to_netcdf(os.path.join(dir, name), mode='w')                      

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def frontSSH(reference = ('KimOrsi', 'SokRin')):
    if reference == 'KimOrsi':
        # Kim & Orsi (2014)
        SAF = np.arange(-0.1,0.2,0.1)
        PF = np.arange(-0.7,-0.5,0.1)
        SACCF = np.arange(-1,-0.8,0.1)

    if reference == 'SokRin':
        # Sokolov & Rintoul (2009) 
        SAF = np.arange(1.5,1.9,0.1)
        PF = np.arange(1,1.3,0.1)
        SACCF = np.arange(0.7,0.9,0.1)
    
    return SAF, PF, SACCF

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# to do : make this function generalised to work for different datasets
def distanceAsCoord(ds, float_num = None, rs = False, xdim = 'profile', zdim = 'pressure'):
    '''create new xarray dataset with distance as the x dimension (not profile).
    Second input (float_num) is used to calculate distance and rs period'''
    if float_num is not None:
        dist = calc.distFromStart(float_num)
    else:
        dist = calc.cum_dist(ds.longitude, ds.latitude)
        # dist = calc.distFromStart(ds)

    if rs == True:
        try: 
            prof = calc.findRSperiod(ds)
        except:
            prof = calc.findRSperiod(float_num)  
    else:
        prof = slice(0, len(dist))

    dict = {}
    for var in list(ds.data_vars):
        if var != 'distance':
            if len(ds[var].dims) < 2 and xdim in list(ds[var].dims):
                dict[var] = (("distance"), ds[var][prof].data, ds[var].attrs)
            elif len(ds[var].dims) == 2 and ds[var].dims[0] == xdim:
                dict[var] = (("distance", zdim), ds[var][prof].data, ds[var].attrs)
            elif len(ds[var].dims) == 2 and ds[var].dims[1] == xdim:
                dict[var] = (("distance", zdim), ds[var].data.transpose()[prof], ds[var].attrs)

    for coord in list(ds.coords):
        dict[coord] = (ds[coord].dims, ds[coord].data, ds[coord][prof].attrs)
    
    dict['distance'] = (("distance"), dist[prof].data)

    new_ds = xr.Dataset(dict)

    return new_ds

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def tickLocations(ax, minor = 50, major = 100):
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.xaxis.set_major_locator(MultipleLocator(major))

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def half_inertial_averaging(data, float_num, dim = 'profile'):
    '''Perform half-inertial pair averaging of EM-APEX float data. 
    Removes the effect of inertial oscillations if consecutive down or up profiles are approximately half and inertial period apart.'''
    
    d_no_inert = xr.zeros_like(data)
    
    if dim == 'time':
        if len(data.shape)>1:
            d_no_inert[:,:] = 'NaT'
        else:
            d_no_inert[:] = 'NaT'
    else:
        d_no_inert = d_no_inert*np.nan
    
    for prof in range(0,len(data)-2):
        prof2 = prof+2
        
        if dim == 'time':
            if len(data.shape)>1:
                # 2D dataArray
                t1 = data[prof].astype(np.float64)
                t2 = data[prof2].astype(np.float64)
                
                stack = np.vstack((t1, t2))
                mean_t = np.nanmean(stack, axis = 0)
                
                nats = np.where(mean_t < 0)[0]
                mean_datetime = np.asarray(mean_t.astype('datetime64[ns]'))
                mean_datetime[nats] = 'NaT'

            else:
                # 1D dataArray
                t1 = float_num.time[prof]
                t2 = float_num.time[prof2]
                
                stack = np.vstack((t1, t2)).astype(np.float64)
                mean_t = np.asarray(np.nanmean(stack))

                mean_datetime = mean_t.astype('datetime64[ns]')
                if mean_t < 0:
                    mean_datetime = 'NaT'
            
            if len(data.shape)>1:
                d_no_inert[prof, :] = mean_datetime
            else:
                d_no_inert[prof] = mean_datetime
            
        else:
            if len(data.shape)>1:
                d_no_inert[prof, :] = data[[prof,prof2], :].mean(dim = dim,  skipna = True)
            else:
                d_no_inert[prof] = data[[prof,prof2]].mean(dim = dim, skipna = True)
        
    return d_no_inert

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def create_sub_inertial_ds(float_num, floatid, abs_vels, rot_vels, ctd_time, savedir):
    ''' Half inertial pair averaging of all data variables into new dataset - this does not interpolate back onto original times. Use the imports.sub_inertial_ds function for this.'''

    print('Commencing half-inertial pair averaging...')
    rs = calc.findRSperiod(float_num)
    CT, SA = remove_bad_T_S(float_num, floatid)

    u = vel.erroneous_rel_vels(abs_vels.u_abs, floatid)
    v = vel.erroneous_rel_vels(abs_vels.v_abs, floatid)
    u_abs = vel.setAbsVelToNan(floatid, u).interpolate_na('profile', method = 'linear', max_gap = 3)
    v_abs = vel.setAbsVelToNan(floatid, v).interpolate_na('profile',  method = 'linear', max_gap = 3)

    # rotated velocities
    along_interp = rot_vels.u_rot.interpolate_na('profile',  method = 'linear', max_gap = 3)
    cross_interp = rot_vels.v_rot.interpolate_na('profile', method = 'linear', max_gap = 3)

    CT_no_inert = half_inertial_averaging(CT[rs].interpolate_na('profile', method = 'linear', max_gap = 3), float_num)
    SA_no_inert = half_inertial_averaging(SA[rs].interpolate_na('profile', method = 'linear', max_gap = 3), float_num)

    u_abs_no_inert = half_inertial_averaging(u_abs, float_num, dim = 'profile')
    v_abs_no_inert = half_inertial_averaging(v_abs, float_num, dim = 'profile')

    along_no_inert = half_inertial_averaging(along_interp, float_num, dim = 'profile')
    cross_no_inert = half_inertial_averaging(cross_interp, float_num, dim = 'profile')

    ctd_t_no_inert = half_inertial_averaging(ctd_time[rs], float_num, dim = 'time')
    t_no_inert = half_inertial_averaging(float_num.time[rs], float_num, dim = 'time') 
    #TO DO: Should we be setting the time of the up profile to where the float starts ascending (bottom of down profile?) #or the mid-point of each profile?
    
    lat_mid = half_inertial_averaging(float_num.latitude[rs], float_num, dim = 'latitude')
    lon_mid = half_inertial_averaging(float_num.longitude[rs], float_num, dim = 'longitude')
    
    ds_no_inertial = xr.Dataset(data_vars=dict(CT=(["time", "pressure"], CT_no_inert.data),
                                  SA = (["time", "pressure"], SA_no_inert.data),
                                  u_abs = (["time", "pressure"], u_abs_no_inert.data, {'description':'absolute eastward velocity'}),
                                  v_abs = (["time", "pressure"], v_abs_no_inert.data, {'description':'absolute northward velocity'}),
                                  u_rot = (["time", "pressure"], along_no_inert.data, {'description':'along-stream absolute velocity'}),
                                  v_rot = (["time", "pressure"], cross_no_inert.data, {'description':'cross-stream absolute velocity'}),
                                  ctd_t = (["time", "pressure"], ctd_t_no_inert.data),),
                         coords = dict(pressure = ('pressure', float_num.pressure.data), 
                                       time = ('time', t_no_inert.data, {'description':'time_mid'}), 
                                       latitude = (["latitude"], lat_mid.data, {'description':'lat_mid'}),
                                       longitude = (["longitude"], lon_mid.data, {'description':'lon_mid'})), 
                        attrs=dict(description=f"{floatid}: Dataset of variables with inertial oscillations removed using half inertial pair averaging"),)
    
    ds_no_inertial.attrs = {'creation_date':str(datetime.datetime.now()), 'author':'Maya Jakes', 'email':'maya.jakes@utas.edu.au'}

    print('saving file')
    filename = 'ds_no_inertial'
    name = filename + f'_{floatid}_extra_qc' + '.nc' 
    save_to_netCDF(ds_no_inertial, savedir, name)

    return ds_no_inertial


# ------------------------------------------------------------------------------------------------------------------------------------------------------

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3, phase_lock = True):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if phase_lock == True: #filters forwards and backwards (zero phase change)
        y = filtfilt(b, a, data, method = 'gust')
    else:
        y = lfilter(b, a, data)
    return y

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def band_pass_filter(data, fs, lowcut, highcut, dim = 'time', order = 3, phase_lock = True, plot = False):
    '''
    Apply a band pass filter to 1D dataArray

    INPUTS:
    data = array with regular time or distance interval
    fs = sampling rate (in cycles per day or km) e.g. 6 cycles per day (timestep of 4 hours)
    lowcut = low frequency cutoff (e.g. 1 cycle every 10 days = 1/10 or 0.1)
    highcut = high frequency cutoff (e.g. 1 cycle every 2 days = 1/2 or 0.5)
    order = order of the filter (default = 3)

    '''
    # interpolate nans
    x = data.interpolate_na(dim = dim, fill_value="extrapolate")
    # nans = np.where(np.isnan(x))[0]
    # x = x.dropna(dim = dim)

    nsamples = len(x) 
    ndays = nsamples/fs

    t = np.linspace(0, ndays, nsamples, endpoint=False)
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=order, phase_lock = phase_lock)

    if plot == True:
        fig, ax = plt.subplots(figsize = (11,3))
        plt.plot(t, data)
        plt.plot(t, x, linestyle = '--')

        plt.plot(t, y, c = 'k', alpha = 0.7)
        plt.xlabel('time (days)')
        plt.grid(True)

        plt.legend(['original data', 'nans interpolated', 'filtered'])

    y = xr.DataArray(data = y, dims = x.dims, coords = x.coords)

    return x, y







