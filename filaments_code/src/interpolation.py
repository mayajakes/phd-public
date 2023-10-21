from scipy import interpolate
import gsw
import numpy as np
import xarray as xr
import traces
import pandas as pd
from datetime import timedelta, datetime
from scipy.signal import savgol_filter

import src.importData as imports
import src.calc as calc
import src.stats as stats
import src.settings as settings
import src.velocities as vel
import src.concat as ct
import warnings


def to_pdens_grid(dataArray, pdens, zdim='pressure', dens_interval=0.01):
    '''Generalised function to interpolate a 2D dataArray onto an even potential density grid'''

    if zdim != list(dataArray.dims)[1]:
        # second dimension must be the vertical coordinate
        dataArray = dataArray.transpose()

    if zdim != list(pdens.dims)[1]:
        # second dimension must be the vertical coordinate
        pdens = pdens.transpose()

    dens_grid = np.mgrid[np.nanmin(pdens):np.nanmax(pdens):dens_interval]

    n = len(dataArray)
    shp = (n, len(dens_grid))
    new_var = np.nan*np.ma.masked_all(shp)

    for i in range(0, n):
        g = pdens[i, :]
        dens_ind = np.where(~np.isnan(g))
        dens_values = g[dens_ind]

        if np.size(dens_values) != 0:
            new_var[i, :] = interpolate.interp1d(dens_values, 
                                        dataArray.values[i, list(dens_ind)],
                                        bounds_error=False)(dens_grid)

    on_dens = xr.DataArray(data=new_var, dims=[dataArray.dims[0], "potential_density"],
                           coords=dict(potential_density=("potential_density", dens_grid)),)
    on_dens = on_dens.assign_coords(dataArray[dataArray.dims[0]].coords)

    return on_dens


def varToDens(dataArray, float_num=None, floatid=None, pdens=None, dens_interval=0.01, by_dist=True, rs=True, PV=False):
    '''Interpolate any 2D xarray data variable from the float data to an even potential denisty grid'''

    if rs == True:
        try:
            ind = calc.findRSperiod(float_num)
        except:
            ind = slice(0, len(dataArray))
    else:
        ind = slice(0, len(dataArray))

    try:
        dist = dataArray.distance[ind]
    except:
        print('calculating distance')
        dist = calc.distFromStart(float_num)[ind]

    if pdens is None:
        print('calculating density')
        CT, SA = settings.remove_bad_T_S(float_num, floatid)
        p = float_num.pressure
        if PV == True:
            pres_mid_points = (p - np.gradient(p)[0])[1:]
            interp_SA = new_pressure_grid(SA, pres_mid_points)
            interp_T = new_pressure_grid(CT, pres_mid_points)
            pdens = calc.potentialDensity(pres_mid_points, interp_SA, interp_T)
        else:
            pdens = calc.potentialDensity(p, SA, CT)

    dens_grid = np.mgrid[np.nanmin(pdens):np.nanmax(pdens):dens_interval]

    n = len(dataArray)
    shp = (n, len(dens_grid))
    new_var = np.nan*np.ma.masked_all(shp)

    for i in range(0, n):
        g = pdens[i, :]
        dens_ind = np.where(~np.isnan(g))
        dens_values = g[dens_ind]

        if np.size(dens_values) != 0:
            new_var[i, :] = interpolate.interp1d(dens_values, dataArray.values[i, list(dens_ind)],
                                                 bounds_error=False)(dens_grid)

    if by_dist == False:
        on_dens = xr.DataArray(data=new_var, 
                               dims=[dataArray.dims[0], "potential_density"],
                               coords=dict(potential_density=("potential_density", dens_grid)),)
        
        on_dens = on_dens.assign_coords(dataArray[dataArray.dims[0]].coords)

    else:
        on_dens = xr.DataArray(data=new_var, dims=["distance", "potential_density"],
                               coords=dict(potential_density=("potential_density", dens_grid),
                                           distance=("distance", dist[ind].data)),)

    return on_dens

# ------------------------------------------------------------------------------------------------------------------------------------------------------


def interpOnDens(float_num, floatid, dens_interval=0.01, pdens=None, rs=False, by_dist=True):
    '''interpolate all 2D variables from the float onto a potential denisty grid'''

    if by_dist == True:
        dist = calc.distFromStart(float_num)
        dim0 = 'distance'
    else:
        dim0 = 'profile'

    rs_ind = calc.findRSperiod(float_num)

    if pdens is None:
        print('calculating density')
        CT, SA = settings.remove_bad_T_S(float_num, floatid)
        pdens = calc.potentialDensity(float_num.pressure, SA, CT)

    if len(pdens) == len(float_num.profile[rs_ind]) or rs == True:
        n = len(float_num.profile[rs_ind])
        dist = dist[rs_ind]
        prof = float_num.profile[rs_ind]
    else:
        n = len(float_num.profile)

    dens_grid = np.mgrid[np.nanmin(pdens):np.nanmax(pdens):dens_interval]
    shp = (n, len(dens_grid))

    new_flt = {}
    for var in list(float_num.data_vars):
        new_var = np.zeros(shp)*np.nan
        if len(float_num[var].shape) == 2:
            for i in range(0, n):
                g = pdens[i, :]
                dens_ind = np.where(~np.isnan(g))
                dens_values = g[dens_ind]

                if np.size(dens_values) != 0:
                    new_var[i, :] = interpolate.interp1d(dens_values, float_num[var].values[i, list(dens_ind)],
                                                         bounds_error=False)(dens_grid)

        new_flt[var] = ((dim0, "potential_density"),
                        new_var, float_num[var].attrs)

    if by_dist == True:
        on_dens = xr.Dataset(new_flt, coords=dict(potential_density=("potential_density", dens_grid),
                                                  distance=('distance', dist.data)),)
    else:
        on_dens = xr.Dataset(new_flt, coords=dict(potential_density=("potential_density", dens_grid),
                                                  profile=('profile', prof.data)),)

    return on_dens


# ------------------------------------------------------------------------------------------------------------------------------------------------------


def new_pressure_grid(dataArray, new_grid, zdim='pressure'):
    '''Interpolate 2D data to new even pressure grid'''

    # current pressure grid
    pressure_values = dataArray[zdim]

    n = len(dataArray)
    shp = (n, len(new_grid))
    new_da = np.nan*np.ma.masked_all(shp)

    for i in range(0, n):
        new_da[i, :] = interpolate.interp1d(pressure_values, dataArray.values[i, :],
                                            bounds_error=False)(new_grid)

    if zdim == 'pressure':
        new_da = xr.DataArray(data=new_da, dims=[dataArray.dims[0], 'pressure'],
                              coords=dict(pressure=('pressure', new_grid.data),))
    elif zdim == 'depth':
        new_da = xr.DataArray(data=new_da, dims=[dataArray.dims[0], 'depth'],
                              coords=dict(depth=('depth', new_grid.data),))

    new_da = new_da.assign_coords(dataArray[dataArray.dims[0]].coords)

    return new_da

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# TO DO : check this function


def dens_to_pres(dataArray, float_num, floatid, dens_interval=0.01, pdens=None):
    on_dens = interpOnDens(float_num, floatid, dens_interval, pdens)
    press_grid = float_num.pressure.values

    shp = (len(dataArray), len(press_grid))
    new_da = np.nan*np.ma.masked_all(shp)

    for i in range(0, len(dataArray)):
        g = on_dens.P[i, :]
        press_ind = np.where(~np.isnan(g))[0]
        press_values = g[press_ind]

        if np.size(press_values) != 0:
            new_da[i, :] = interpolate.interp1d(press_values, dataArray.values[i, list(press_ind)],
                                                bounds_error=False)(press_grid)

    new_da = xr.DataArray(data=new_da, dims=[dataArray.dims[0], 'pressure'],
                          coords=dict(pressure=('pressure', press_grid),))

    new_da = new_da.assign_coords(dataArray[dataArray.dims[0]].coords)

    return new_da

# ------------------------------------------------------------------------------------------------------------------------------------------------------


def even_dist_grid(dataArray, dist_interval):
    '''Interpolate 1D or 2D data to an even distance grid. 
    2D dataArray must have dimension 0 = distance.
    dist_interval in km'''

    new_grid = np.arange(
        dataArray.distance[0], dataArray.distance[-1], dist_interval)

    if len(dataArray.dims) > 1:
        shp = (len(new_grid), len(dataArray[0]))
        new_da = np.nan*np.ma.masked_all(shp)
        for i in range(0, len(dataArray[0])):
            new_da[:, i] = interpolate.interp1d(dataArray.distance.values, dataArray.values[:, i],
                                                bounds_error=False)(new_grid)

        new_da = xr.DataArray(data=new_da, dims=[dataArray.dims[0], dataArray.dims[1]],
                              coords=dict(dataArray[dataArray.dims[1]].coords,
                                          distance=("distance", new_grid)),)
    else:
        shp = (len(new_grid),)
        new_da = np.nan*np.ma.masked_all(shp)
        new_da = interpolate.interp1d(dataArray.distance.values, dataArray.values,
                                      bounds_error=False)(new_grid)

        new_da = xr.DataArray(data=new_da, dims=['distance'],
                              coords=dict(distance=("distance", new_grid)),)

    return new_da


# ------------------------------------------------------------------------------------------------------------------------------------------------------

def gaussianFilter(DataArray, ind=None, window=9, order=3, interp_na=True):
    '''Smoothes curve by taking 3 days of float data and fitting a 3rd order polynomial to the points, then moves the window along'''

    if interp_na == True:
        data = DataArray.interpolate_na(
            dim=DataArray.dims[0], use_coordinate=False)
    else:
        data = DataArray

    if ind != None:
        data = data[ind]

    data_smooth = savgol_filter(data, window, order, axis=-1, mode='interp')

    return data_smooth

# ------------------------------------------------------------------------------------------------------------------------------------------------------


def interpToFloat(float_num, data, location_only=False, time_only=False, zdim=None):
    '''interpolate spatial satellite data to float locations and times

    Input: 
    float_num (e.g. ema[floatid])
    Xarray Dataset - coordinates of latitude, longitude and time

    Output: 
    Interpolated values of the satellite field on float locations and/or times

    '''
    lon = float_num.longitude
    lat = float_num.latitude
    t = float_num.time

    lst = []
    if time_only == True:
        to_float = data.interp(time=t)
        for i in range(0, len(float_num.time)):
            value = to_float.isel(time=i).values
            lst.append(value.tolist())

        new_da = xr.DataArray(data=np.asarray(lst), dims=[float_num.time.dims[0], data.latitude.dims[0], data.longitude.dims[0]],
                              coords=dict(data.latitude.coords))

        new_da = new_da.assign_coords(data.longitude.coords)
        new_da = new_da.assign_coords(float_num.time.coords)

    elif location_only == True:
        to_float = data.interp(latitude=lat, longitude=lon)
        for i in range(0, len(float_num.hpid)):
            value = to_float.isel(latitude=i, longitude=i).values
            lst.append(value.tolist())

        try:
            new_da = xr.DataArray(data=np.asarray(lst), dims=[float_num.hpid.dims[0], data.time.dims[0]],
                                  coords=dict(data.time.coords))
        except:
            # if data has no time dimension e.g. bathymetry
            if zdim is not None:
                new_da = xr.DataArray(np.asarray(lst), dims=[float_num.hpid.dims[0], zdim],
                                      coords=dict(data[zdim].coords))
            else:
                new_da = xr.DataArray(np.asarray(
                    lst), dims=float_num.hpid.dims[0])

    else:
        # interpolate both location and time
        to_float = data.interp(latitude=lat, longitude=lon, time=t)
        for i in range(0, len(float_num.hpid)):
            value = to_float.isel(time=i, latitude=i, longitude=i).values
            lst.append(value.tolist())

        new_da = xr.DataArray(data=np.asarray(lst), dims=float_num.hpid.dims,
                              coords=float_num.hpid.coords)

    return new_da

# ------------------------------------------------------------------------------------------------------------------------------------------------------


def regular_time_grid(data, original_time_da, new_timestep):
    '''
    Interpolate data onto a regular time interval for signal processing.

    INPUTS:
    data = 1D data to be interpolated 
    original_time_da = e.g. float_num.time[rs] (current time dimension)
    new_timestep = new even time interval (in hours)

    OUTPUT:
    new_da = new interpolated xr.DataArray with regular timesteps

    '''
    warnings.filterwarnings("ignore")

    start = original_time_da.values[0].astype('M8[h]')
    end = original_time_da.values[-1].astype('M8[h]')
    start_time = pd.Timestamp(start).to_pydatetime()
    end_time = pd.Timestamp(end).to_pydatetime()

    my_dict = {}
    for i in range(0, len(original_time_da)):
        if ~np.isnat(original_time_da.values[i]):
            my_dict[pd.Timestamp(original_time_da.values[i]
                                 ).to_pydatetime()] = data.values[i]

    ts = traces.TimeSeries(my_dict)
    ts_even = ts.sample(sampling_period=timedelta(hours=new_timestep), start=start_time,
                        end=end_time, interpolate='linear',)

    new_data = []
    new_time = []
    for i in range(0, len(ts_even)):
        new_time.append(ts_even[i][0])
        if ts_even[i][1] == None:
            new_data.append(np.nan)
        else:
            new_data.append(ts_even[i][1])

    new_da = xr.DataArray(data=np.asarray(new_data), dims=['time'],
                          coords=dict(time=('time', np.asarray(new_time)),))

    return new_da

# ------------------------------------------------------------------------------------------------------------------------------------------------------


def interp_nats(data, time_grid):
    '''Interpolation of NaTs in a time variable'''

    nats = np.where(np.isnat(time_grid))[0]
    t = time_grid.astype(np.float64)
    t[nats] = np.nan
    t_interp = xr.DataArray(t.data).interpolate_na(dim='dim_0', fill_value="extrapolate")
    t_interp = t_interp.astype('datetime64[ns]')

    # # change time coords to interpolated grid
    if len(data.shape) > 1:
        new_da = xr.DataArray(data.data, dims=['time', data.dims[1]], coords=dict(
            time=('time', t_interp.data)))
        new_da = new_da.assign_coords(data[data.dims[1]].coords)
    else:
        new_da = xr.DataArray(t_interp.data, dims=[
                              'time'], coords=dict(time=('time', t_interp.data)))

    return new_da

# ------------------------------------------------------------------------------------------------------------------------------------------------------


def interp_time(data, new_time):
    '''Interpolate data to a new time grid.'''

    dtype = data.dtype
    if dtype == 'datetime64[ns]':
        data = data.astype(np.float64)
        for i in range(0, len(data)):
            data[i][np.where(data[i] < 0)[0]] = np.nan

    time_grid = data.time

    new_da = interp_nats(data, time_grid)
    new_t = interp_nats(new_time, new_time)

    new_da = new_da.interp(time=new_t)

    if dtype == 'datetime64[ns]':
        new_da = new_da.astype('datetime64[ns]')

    return new_da

# ------------------------------------------------------------------------------------------------------------------------------------------------------


def even_time_grid_2d(data, old_time_grid, new_timestep, zdim='pressure'):
    '''Interpolate 2D data to an even time grid.'''

    new_da = interp_nats(data, old_time_grid)
    old_time_grid = new_da.time

    x = regular_time_grid(new_da[:, 10], old_time_grid, new_timestep)

    d = np.zeros((len(x), len(new_da[zdim])))*np.nan

    i = 0
    for p in data[zdim]:
        if zdim == 'pressure':
            d[:, i] = regular_time_grid(new_da.sel(
                pressure=p), old_time_grid, new_timestep)
        elif zdim == 'potential_density':
            d[:, i] = regular_time_grid(new_da.sel(
                potential_density=p), old_time_grid, new_timestep)
        elif zdim == 'depth_cell':
            d[:, i] = regular_time_grid(new_da.sel(
                depth_cell=p), old_time_grid, new_timestep)
        i += 1

    if zdim == 'pressure':
        d = xr.DataArray(d, dims=['time', 'pressure'],
                         coords=dict(time=('time', x.time.data),
                                     pressure=('pressure', new_da.pressure.data)))

    elif zdim == 'potential_density':
        d = xr.DataArray(d, dims=['time', 'potential_density'],
                         coords=dict(time=('time', x.time.data),
                                     potential_density=('potential_density', new_da.potential_density.data)))

    elif zdim == 'depth_cell':
        d = xr.DataArray(d, dims=['time', 'depth_cell'],
                         coords=dict(time=('time', x.time.data),
                                     depth_cell=('depth_cell', new_da.depth_cell.data)))

    return d


# ------------------------------------------------------------------------------------------------------------------------------------------------------

# def grid_and_smooth(data_dict, floatids, dist_interval = 3, window = 3, min_window = 1, interp_nans = False, max_gap = 14, vert_smooth = False):
#     '''Evenely grid in distance, then smooth using a rolling average on z levels. Concatenates multiple floats.'''

#     # horizontal gridding and horizontal smoothing
#     d_even_dist = {}
#     d_smooth = {}

#     flt_dist_loc = []
#     for floatid in floatids:
#         if vert_smooth == True:
#             d = vel.smooth_prof_by_prof(data_dict[floatid], window = 75, print_info = False)
#         else:
#             d = data_dict[floatid]

#         if interp_nans == True:
#             # drop duplicates in distancea and interpolate to fill nans
#             d = d.drop_duplicates(dim = 'distance', keep='first')
#             d = d.interpolate_na(dim = 'distance', max_gap = max_gap)

#         d_even_dist[floatid] = even_dist_grid(d, dist_interval)
#         # if interp_nans == True:
#         #     d_even_dist[floatid] = d_even_dist[floatid].interpolate_na(dim = 'distance', max_gap = max_gap)

#         d_smooth[floatid] = d_even_dist[floatid].rolling(distance = window, center = True, min_periods = min_window).mean()
#         flt_dist_loc.append(d_even_dist[floatid].distance[-1].data)

#     flt_dist_loc = np.cumsum(np.asarray(flt_dist_loc))[:-1]
#     d_even_concat = ct.joinFloats(d_even_dist, 'distance', new_dim = True)
#     d_smooth_concat = ct.joinFloats(d_smooth, 'distance', new_dim = True)

#     return d_even_concat, d_smooth_concat


def grid_and_smooth(data_dict, floatids, dist_interval=3, window=3, min_window=1, interp_nans=False, max_gap=14, vert_smooth=False):
    '''Evenely grid in distance, then smooth using a rolling average on z levels. Concatenates multiple floats.'''

    # horizontal gridding and horizontal smoothing
    d_even_dist = {}
    d_smooth = {}

    flt_dist_loc = []
    for floatid in floatids:
        if vert_smooth == True:
            d = vel.smooth_prof_by_prof(
                data_dict[floatid], window=75, print_info=False)
        else:
            d = data_dict[floatid]

        if interp_nans == True:
            # drop duplicates in distancea and interpolate to fill nans
            d = d.drop_duplicates(dim='distance', keep='first')
            d = d.interpolate_na(dim='distance', max_gap=max_gap)

        d_even_dist[floatid] = even_dist_grid(d, dist_interval)
        # if interp_nans == True:
        #     d_even_dist[floatid] = d_even_dist[floatid].interpolate_na(dim = 'distance', max_gap = max_gap)

        d_smooth[floatid] = d_even_dist[floatid].rolling(
            distance=window, center=True, min_periods=min_window).mean()
        flt_dist_loc.append(d_even_dist[floatid].distance[-1].data)

    flt_dist_loc = np.cumsum(np.asarray(flt_dist_loc))[:-1]
    d_even_concat = ct.joinFloats(d_even_dist, 'distance', new_dim=True)
    d_smooth_concat = ct.joinFloats(d_smooth, 'distance', new_dim=True)

    return d_even_concat, d_smooth_concat, d_smooth, d_even_dist
