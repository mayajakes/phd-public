import xarray as xr
import warnings
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

import imp
import src.calc as calc


def detectOutliers(data, remove = False, show = False):
    '''Tukey's box plot method.
    https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755'''

    warnings.filterwarnings("ignore")

    if show == True:
        print('maximum before: {}'.format(np.nanmax(data)))

    data2 = data.copy()

    q1 = np.nanquantile(data2, 0.25)
    q3 = np.nanquantile(data2, 0.75)
    iqr = q3-q1

    inner_fence = 1.5*iqr
    outer_fence = 3*iqr

    #inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence

    #outer fence lower and upper end
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence

    outliers_prob = []
    outliers_poss = []
    for index, x in enumerate(data2):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers_prob.append(index)
    for index, x in enumerate(data2):
        if x <= inner_fence_le or x >= inner_fence_ue:
            outliers_poss.append(index)
        
    if remove == True:
        values = [x for x in abs(data2[outliers_prob])]
        data2[outliers_prob] = np.nan
        if show == True:
            print('maximum after: {}'.format(np.nanmax(data2)))
        return data2, values

    else:
        return outliers_prob, outliers_poss


# ------------------------------------------------------------------------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def spearmans(floatid, data1, data2, plot = False):

    if plot == True:
        fig, ax = plt.subplots(figsize = (4,3))

        ax.scatter(data1,data2)
        ax.set_title('Float {}'.format(floatid))
        ax.set_xlabel('data 1')
        ax.set_ylabel('data 2')


    corr, pval = spearmanr(data1, data2, nan_policy = 'omit')
    if len(data1.shape) == 1:
        print(f'Spearmans corr: {corr:.3f}, p-value: {pval:.5f}')
    return corr, pval

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def variance(data):
    # Mean of the data
    mean = np.nanmean(data)
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    variance = np.nanmean(deviations)
    return variance

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def std(data):
    return np.sqrt(variance(data))

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def find_nearest(array, value):
    array = np.asarray(array)
    # if type(value) != int:
    #     idx = []
    #     for v in value:
    #         i = (np.abs(array - v)).argmin()
    #         idx.append(i)
    # else:
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def delOddProfiles(data, odd_profile):
    '''odd_profile could be one profile or a list of profiles e.g. [198, 199]'''
    data[odd_profile] = np.nan
    
    return data

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def temporalError(float_num, dataArray, method = ('nearest', 'interp'), rs = True):
    '''find the value at the time stamp before and after the float profile as an indication of temporal error.
    Returns: 
    1. Nearest value to float time
    2. Lowerlim (time stamp before)
    3. Upperlim (time stamp after)'''
    
    if rs == True:
        try:
            rs = calc.findRSperiod(float_num)
        except:
            rs = slice(0, len(float_num.latitude))
    else:
        rs = slice(0, len(float_num.latitude))

    # interpolate onto float profile locations
    lon = float_num.longitude[rs]
    lat = float_num.latitude[rs]
    t = float_num.time[rs]

    lower_t = t - np.timedelta64(1,'D')
    upper_t = t + np.timedelta64(1,'D')

    upper = xr.zeros_like(float_num.time[rs].astype(float))*np.nan
    lower = xr.zeros_like(float_num.time[rs].astype(float))*np.nan

    if method == 'interp':
        interp_val = xr.zeros_like(float_num.time[rs].astype(float))*np.nan

        to_float = dataArray.interp(latitude=lat, longitude=lon, time = t)
        lower_val = dataArray.interp(latitude=lat, longitude=lon, time = lower_t)
        upper_val = dataArray.interp(latitude=lat, longitude=lon, time = upper_t)
        
        if len(to_float.shape) > 1:
            for i in range(0, len(float_num.time[rs])):
                interp_val[i] = to_float.isel(latitude = i, longitude = i, time = i)
                upper[i] = upper_val.isel(latitude = i, longitude = i, time = i)
                lower[i] = lower_val.isel(latitude = i, longitude = i, time = i)
            
            return interp_val, lower, upper
        else:
            return to_float, lower_val, upper_val

    elif method == 'nearest':
        nearest = xr.zeros_like(float_num.time[rs].astype(float))*np.nan
        to_float = dataArray.interp(latitude=lat, longitude=lon)

        nearest_val = to_float.sel(time = float_num.time[rs], method = 'nearest')
        lower_val = to_float.sel(time = lower_t, method = 'nearest')
        upper_val = to_float.sel(time = upper_t, method = 'nearest')

        for i in range(0, len(float_num.time[rs])):
            nearest[i] = nearest_val.isel(latitude = i, longitude = i, time = i)
            upper[i] = upper_val.isel(latitude = i, longitude = i, time = i)
            lower[i] = lower_val.isel(latitude = i, longitude = i, time = i)

        return nearest, lower, upper

# ------------------------------------------------------------------------------------------------------------------------------------------------------

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