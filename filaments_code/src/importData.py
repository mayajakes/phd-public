'''Functions for importing data'''

import os
import xarray as xr
import src.interpolation as interp
import src.calc as calc

def importFloatData(floatids):
    
    floatdir = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd', 'data', 'floats')

    ema = {}
    for floatid in floatids:
        input_file = os.path.join(floatdir, 'macquarie_ema-%s_qc.nc' %floatid)
        ema[floatid] = xr.open_dataset(input_file)
        
    return ema

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def importNetCDF(datadir, filename, datatype = None):
    
    if datatype != None:
        input_file = os.path.join(datadir, datatype, filename)
    else:
        input_file = os.path.join(datadir, filename)
    
    dataset = xr.open_dataset(input_file)

    return dataset

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def sub_inertial_ds(ema, floatids, datadir, xdim = ('distance', 'profile', 'time')):
    # import subinertial dataset and interpolate back onto original times

    ds_no_inertial = {}
    ds = {}
    CT, SA = {}, {}

    for floatid in floatids:
        float_num = ema[floatid]
        rs = calc.findRSperiod(float_num)
        dist = calc.distFromStart(float_num)
        
        file_2 = os.path.join(datadir, 'ds_no_inertial_%s.nc' %floatid)
        ds_no_inertial[floatid] = xr.open_dataset(file_2)
        
        # interpolate back to original x coordinates
        CT_no_inert = interp.interp_time(ds_no_inertial[floatid].CT, float_num.time[rs])
        SA_no_inert = interp.interp_time(ds_no_inertial[floatid].SA, float_num.time[rs])
        
        if floatid != 8490:
            u_no_inert = interp.interp_time(ds_no_inertial[floatid].u, float_num.time[rs])
            v_no_inert = interp.interp_time(ds_no_inertial[floatid].v, float_num.time[rs])

        ctd_t_no_inert = interp.interp_time(ds_no_inertial[floatid].ctd_t, float_num.time[rs])

        ds[floatid] = xr.Dataset(data_vars=dict(CT=([xdim, "pressure"], CT_no_inert.data),
                                    SA = ([xdim, "pressure"], SA_no_inert.data),
                                    ctd_t = ([xdim, "pressure"], ctd_t_no_inert.data),),
                            coords = dict(pressure = ('pressure', float_num.pressure.data), 
                                        time = ('time', float_num.time[rs].data, {'description':'time'}), 
                                        distance = ('distance', dist[rs].data),
                                        latitude = (["latitude"], float_num.latitude[rs].data, {'description':'latitude'}),
                                        longitude = (["longitude"], float_num.longitude[rs].data, {'description':'longitude'})), 
                            attrs=dict(description=f"{floatid}: Dataset of variables with inertial oscillations removed using half inertial pair averaging, then interpolated back onto original times/locations"),)
        
        if floatid != 8490:
            if xdim == 'profile':
                ds[floatid]['u'] = xr.DataArray(u_no_inert.data, dims = [xdim, "pressure"],attrs = {'description':'absolute eastward'})
                ds[floatid]['v'] = xr.DataArray(v_no_inert.data, dims = [xdim, "pressure"], attrs = {'description':'absolute northward'})
            
            else:
                ds[floatid]['u'] = xr.DataArray(u_no_inert.data, dims = [xdim, "pressure"], coords = ds[floatid][xdim].coords, attrs = {'description':'absolute eastward'})
                ds[floatid]['v'] = xr.DataArray(v_no_inert.data, dims = [xdim, "pressure"], coords = ds[floatid][xdim].coords, attrs = {'description':'absolute northward'})
                    
        CT[floatid], SA[floatid] = ds[floatid].CT, ds[floatid].SA

    
    return ds

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def triaxus_data(datadir, tow_list, folder):
    '''Import Trisxus data from the voyage.
    Some tows are classed as separate but run along the same transect, so these are combined together.'''
    triaxus_cast = {}

    for i in range(0, len(tow_list)):
        if tow_list[i] not in ['02_005', '08_003', '09_002', '09_003']:
            file = 'in2018_v05_'+ tow_list[i] +'AvgCast.nc'
            
            # combine tows
            if tow_list[i] == '02_002':
                file1 = 'in2018_v05_'+ tow_list[i] +'AvgCast.nc'
                file2 = 'in2018_v05_'+ tow_list[i+1] +'AvgCast.nc'
                triaxus_cast_1 = importNetCDF(datadir, file1, datatype = folder)
                triaxus_cast_2 = importNetCDF(datadir, file2, datatype = folder)
                triaxus_cast[tow_list[i]] = xr.concat([triaxus_cast_1, triaxus_cast_2], dim = 'time')
                
            elif tow_list[i] == '08_002':
                file1 = 'in2018_v05_'+ tow_list[i] +'AvgCast.nc'
                file2 = 'in2018_v05_'+ tow_list[i+1] +'AvgCast.nc'
    #             file3 = 'in2018_v05_'+ lst[i+2] +'AvgCast.nc'
    #             file4 = 'in2018_v05_'+ lst[i+3] +'AvgCast.nc'
                triaxus_cast_1 = importNetCDF(datadir, file1, datatype = folder)
                triaxus_cast_2 = importNetCDF(datadir, file2, datatype = folder)
    #             triaxus_cast_3 = imports.importNetCDF(datadir, file3, datatype = folder)
    #             triaxus_cast_4 = imports.importNetCDF(datadir, file4, datatype = folder)
                triaxus_cast[tow_list[i]] = xr.concat([triaxus_cast_1, triaxus_cast_2], dim = 'time')

            else:
                triaxus_cast[tow_list[i]] = importNetCDF(datadir, file, datatype = folder)

    return triaxus_cast