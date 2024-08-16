
import xarray as xr
import imp
import numpy as np
import os
from scipy import interpolate

import src.calc as calc
import src.importData as imports
import src.interpolation as interp
import src.settings as settings

def concatenated_flts(data, floatids, interp_to_flt = False, interp_to_dens = False, pdens = None, new_dim = True, rs = True,
                        save_ds = False, datadir = None, filename = None):
    '''concatenate data from more than one float and plot together by cumulative distance
    INPUTS:
    data - could be a satellite field that will be interpolated onto the float locations and times
           interp_to_flt == True
           or 
           could be a dictionary of rs data (e.g. spice) for more than one float
    
    '''
    ema = imports.importFloatData(floatids)
    
    d = {}
    d_rs = {}
    end_dist = {}
    for floatid in floatids:
        newFloat = settings.distanceAsCoord(ema[floatid], rs = rs)
        end_dist[floatid] = newFloat.distance[-1].data

        if rs == True:
            rapid_s = calc.findRSperiod(ema[floatid])
        else:
            rapid_s = slice(0,len(data[floatid]))

        if interp_to_flt == True:
            d[floatid] = interp.interpToFloat(newFloat, data)
            d_rs[floatid] = d[floatid][rapid_s]

        else:
            d_rs[floatid] = data[floatid][rapid_s]

    total_dist = new_dist(d_rs)

    if interp_to_dens == True:
        ds_concat = joinFloats(d_rs, 'distance')
        xr_ds = toDens(ds_concat, floatids, pdens = pdens, dist = True)

        flt_ind = np.where(xr_ds.distance == 0)[0].tolist()
        flt_ind.append(len(xr_ds))
        d_rs = {}
        i = 0
        for floatid in floatids:
            d_rs[floatid] = xr_ds[flt_ind[i]:flt_ind[i+1]]
            i+=1

        if new_dim == True:
            xr_ds['distance'] = total_dist
        
    else:
        combined_data = np.concatenate([d_rs[x] for x in d_rs], 0)
        if len(d_rs[floatids[0]].dims) < 2:
            xr_ds = xr.DataArray(data = combined_data, dims = ['distance'],
                        coords = dict(distance=('distance', total_dist.data),))
        else:
            xr_ds = xr.DataArray(data = combined_data, dims = ['distance', 'pressure'],
                                    coords = dict(distance=('distance', total_dist.data),
                                                pressure = ('pressure', d_rs[floatid].pressure.data)))

    if save_ds == True:
        xr_ds.to_netcdf(os.path.join(datadir, filename))

    return xr_ds, end_dist, d_rs

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def new_dist(data_dict):
    '''Cumulative concatenated distance'''
    data_list = list(data_dict.values())

    end_dist = []
    for i in range(0, len(data_list)):
        end_dist.append(data_list[i].distance[-1].values)

    if len(data_list) == 4:
        # calculate the total distance in km
        total_dist = np.concatenate((data_list[0].distance.data, 
                            end_dist[0] + data_list[1].distance.data,
                            end_dist[0] + end_dist[1] + data_list[2].distance.data,
                            end_dist[0] + end_dist[1] + end_dist[2] + data_list[3].distance.data), 0)

    elif len(data_list) == 3:
        total_dist = np.concatenate((data_list[0].distance.data,
                                end_dist[0]  + data_list[1].distance.data,
                                end_dist[0]  + end_dist[1] + data_list[2].distance.data), 0)
    
    return total_dist

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def joinFloats(data_dict, dim_name, new_dim = False):
    ''' Concatenate data from separate floats 
    e.g. data_list = [ema[8489].SA[rs_8489], ema[8492].SA[rs_8492], ema[8493].SA[rs_8493]]
            dim_name = 'profile'   '''

    data_list = list(data_dict.values())
    concat_ds = xr.concat(data_list, dim = dim_name)

    if new_dim == True:
        if dim_name == 'distance':
            concat_ds['distance'] = new_dist(data_dict)

    return concat_ds

# ------------------------------------------------------------------------------------------------------------------------------------------------------
def toDens(concat_ds, pdens, dist = True):
    '''Concatenated float data into density space
    INPUTS: 
    concat_ds = Concatenated dataArray containing more than one float (joined using joinFloats)
    pdens = Concatenated density dataset for the same floats

    '''
    dens_grid = np.mgrid[np.nanmin(np.nanmin(pdens)):np.nanmin(np.nanmax(pdens)):0.01]

    if dist == True:
        new_var =  np.nan*np.ones((len(concat_ds.distance),len(dens_grid)))
        n = len(concat_ds.distance)
    else:
        new_var =  np.nan*np.ones((len(concat_ds.profile),len(dens_grid)))
        n = len(concat_ds.profile)

    for i in range(0, n):
            g = pdens[i,:]
            dens_ind = np.where(~np.isnan(g))
            dens_values = g[dens_ind]

            if np.size(dens_values)!=0:
                new_var[i,:] = interpolate.interp1d(dens_values, concat_ds.values[i,list(dens_ind)], 
                                        bounds_error=False)(dens_grid)
                
    
    if dist == True:
        on_dens = xr.DataArray(data = new_var, dims=["distance", "potential_density"], 
                                    coords = dict(potential_density=("potential_density", dens_grid),
                                        distance = ("distance", concat_ds.distance.data)),)
    else: 
        on_dens = xr.DataArray(data = new_var, dims=["profile", "potential_density"], 
                                    coords = dict(potential_density=("potential_density", dens_grid),
                                        profile = ("profile", concat_ds.profile.data)),)
    
    return on_dens

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# def toDens(concat_ds, floatids, pdens = None, dist = True):
#     '''Concatenated float data into density space
#     INPUTS: 
#     1. Concatenated dataArray containing more than one float (joined using joinFloats)
#     2. Concatenated temperature data for more than one float
#     3. Concatenated salinity data for more than one float 

#     '''
#     ema = imports.importFloatData(floatids)
#     T_rs = {}
#     S_rs = {}
#     for floatid in floatids:

#         if pdens is None:
#             rs = calc.findRSperiod(ema[floatid])
#             if dist == True:
#                 dim = 'distance'
#                 d = settings.distanceAsCoord(ema[floatid])
#                 if len(concat_ds.pressure) != len(ema[floatid].pressure): 
#                     pres_mid_points = concat_ds.pressure
#                     S_rs[floatid] = interp.new_pressure_grid(d.SA, pres_mid_points)[rs]
#                     T_rs[floatid] = interp.new_pressure_grid(d.CT, pres_mid_points)[rs]
#                 else:
#                     T_rs[floatid] = d.CT[rs]
#                     S_rs[floatid] = d.SA[rs]
                    
#             else:
#                 dim = 'profile'
#                 if len(concat_ds.pressure) != len(ema[floatid].pressure): 
#                     pres_mid_points = concat_ds.pressure
#                     S_rs[floatid] = interp.new_pressure_grid(ema[floatid].SA, pres_mid_points)[rs]
#                     T_rs[floatid] = interp.new_pressure_grid(ema[floatid].CT, pres_mid_points)[rs]
#                 else:
#                     T_rs[floatid] = ema[floatid].CT[rs]
#                     S_rs[floatid] = ema[floatid].SA[rs]  

#         concat_T = joinFloats(T_rs, dim)
#         concat_S = joinFloats(S_rs, dim)

#         print('calculating density')
#         pdens = calc.potentialDensity(concat_T.pressure, concat_S, concat_T)

#     else:
#         pdens = pdens

#     dens_grid = np.mgrid[np.nanmin(np.nanmin(pdens)):np.nanmin(np.nanmax(pdens)):0.01]

#     if dist == True:
#         new_var =  np.nan*np.ones((len(concat_ds.distance),len(dens_grid)))
#         n = len(concat_ds.distance)
#     else:
#         new_var =  np.nan*np.ones((len(concat_ds.profile),len(dens_grid)))
#         n = len(concat_ds.profile)

#     for i in range(0, n):
#             g = pdens[i,:]
#             dens_ind = np.where(~np.isnan(g))
#             dens_values = g[dens_ind]

#             if np.size(dens_values)!=0:
#                 new_var[i,:] = interpolate.interp1d(dens_values, concat_ds.values[i,list(dens_ind)], 
#                                         bounds_error=False)(dens_grid)
                
    
#     if dist == True:
#         on_dens = xr.DataArray(data = new_var, dims=["distance", "potential_density"], 
#                                     coords = dict(potential_density=("potential_density", dens_grid),
#                                         distance = ("distance", concat_ds.distance.data)),)
#     else: 
#         on_dens = xr.DataArray(data = new_var, dims=["profile", "potential_density"], 
#                                     coords = dict(potential_density=("potential_density", dens_grid),
#                                         profile = ("profile", concat_ds.profile.data)),)
    
#     return on_dens

