
import os 
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean
import scipy
import scipy.signal
from scipy.optimize import curve_fit
import xarray as xr
import numpy as np
import matplotlib.ticker as ticker
import gsw

import imp 
import src.calc as calc
import src.importData as imports
import src.interpolation as interp
import src.settings as settings
import src.stats as stats
import src.concat as ct


def trajectory_sections_and_TS(ema, floatids, sections, alt_cmems,  lons = (148,159), lats = (-57, -51)):
    '''ema: dictionary containing floats
       floatids: IDs of the floats you want to plot
       sections: parts of the float trajectory you would like to plot in white and show the T-S signature for (e.g. [slice(,), slice(,), ...])'''
    mpl.rcParams['xtick.labelsize'] = 13 
    mpl.rcParams['ytick.labelsize'] = 13
    # colors for distance markers every 100 km
    d_colors = ['#FFE0E0','#FFB2B2', '#F28A8A', '#E85555', '#D84747', '#C93636', '#963333', '#7F1A1A', '#691515', '#5B1313', '#440000', '#2E0000', '#040000']
    lonmin, lonmax = lons
    latmin, latmax = lats
    # Set SSH contours 
    levels = np.arange(-0.8,0.4,0.1)

    if len(floatids) > 1:
        cols = len(floatids)
        fig, axs = plt.subplots(nrows = 2, ncols=cols, figsize = (14.5,6.5)) # (15,9) 
        axs = axs.flatten()
        j = len(floatids)
    else:
        cols = len(sections)
        if cols == 3:
            fig, axs = plt.subplots(nrows = 2, ncols=cols, figsize = (11,6.5)) # (15,9)
        elif cols == 4:
            fig, axs = plt.subplots(nrows = 2, ncols=cols, figsize = (14,6.5)) # (15,9)
        axs = axs.flatten()
        j = len(sections)

    i = 0
    while i < cols:
        for floatid in floatids:
            float_num = ema[floatid]
            rs = calc.findRSperiod(float_num)
            dist = calc.cum_dist(float_num.longitude, float_num.latitude)[rs]
            float_num = settings.distanceAsCoord(ema[floatid])
            
            lst = []
            # find index of nearest float profile for every 100 km 
            for l in range(100,1400,100):
                value = stats.find_nearest(dist, l)[0]
                lst.append(value)

            # Trajectory sections
            if type(sections[i]) == list:
                start = float_num.time.values[sections[i][0]][0]
                end = float_num.time.values[sections[i][1]][-1]
                if np.isnat(end):
                    end = float_num.time.values[sections[i][1]][-2]
            else:
                start = float_num.time.values[sections[i]][0]
                end = float_num.time.values[sections[i]][-1]
                if np.isnat(end):
                    end = float_num.time.values[sections[i]][-2]
            start_time, end_time = str(start.astype('M8[D]')), str(end.astype('M8[D]'))

            # mean sea level during float section
            msl = alt_cmems.adt.sel(time = slice(start_time, end_time)).mean(dim = 'time')

            im = msl.sel(longitude = slice(lonmin, lonmax), latitude = slice(latmin, latmax)).plot(ax = axs[i], 
                                                                                                alpha = 0.3, 
                                                                                                vmin = -1, vmax = 0.5,
                                                                                                add_colorbar=False)


            CS = msl.plot.contour(ax = axs[i], colors = 'dimgrey', alpha = 0.5, linewidths = 1, levels = levels, zorder = 3)
            plt.clabel(CS, inline=True, fontsize=11, fmt = '%1.1f')

            # Float trajectory map with points coloured in acccording to time slice
            axs[i].scatter(float_num.longitude, float_num.latitude, s = 40, c= 'slategrey',zorder = 2, alpha = 0.4)
            if type(sections[i]) == list:
                axs[i].scatter(float_num.longitude[sections[i][0]], float_num.latitude[sections[i][0]], s = 30, c='w', zorder = 2, alpha = 0.7)
                axs[i].scatter(float_num.longitude[sections[i][1]], float_num.latitude[sections[i][1]], s = 30, c='w', zorder = 2, alpha = 0.7)
            else:
                axs[i].scatter(float_num.longitude[sections[i]], float_num.latitude[sections[i]], s = 30, c='w', zorder = 2, alpha = 0.7)

            # distance markers
            if dist[-1] < 800:
                axs[i].scatter(float_num.longitude[lst[0:7]], float_num.latitude[lst[0:7]], s = 25, c = d_colors[0:7], zorder = 3)
            if dist[-1] < 900:
                axs[i].scatter(float_num.longitude[lst[0:8]], float_num.latitude[lst[0:8]], s = 25, c = d_colors[0:8], zorder = 3)
            elif 900 < dist[-1] < 1000:
                axs[i].scatter(float_num.longitude[lst[0:9]], float_num.latitude[lst[0:9]], s = 25, c = d_colors[0:9], zorder = 3)
            else:
                axs[i].scatter(float_num.longitude[lst], float_num.latitude[lst], c = d_colors, s = 25, zorder = 3)

            axs[i].text(0.02, 0.03, f'{start_time}', transform = axs[i].transAxes, fontsize = 11)
            axs[i].text(0.66, 0.03, f'{end_time}', transform = axs[i].transAxes, fontsize = 11)
            axs[i].set_xticks(np.arange(148,159, 2)) 
            axs[i].set_xlabel('')
            axs[i].set_ylabel('')
            axs[i].set_title(f'EM-{floatid}')

            # T-S diagrams on bottom 3 panels
            CT, SA = settings.remove_bad_T_S(float_num, floatid)

            if type(sections[i]) == list:
                sal = xr.concat([SA[sections[i][0]], SA[sections[i][1]]], dim = 'distance')
                temp = xr.concat([CT[sections[i][0]], CT[sections[i][1]]], dim = 'distance') 
            else:
                sal = SA[sections[i]]
                temp = CT[sections[i]]  

            ti, si, dens = calc.tsDensContour(sal, temp)
            n = len(sal)
            colors = plt.cm.rainbow(np.linspace(0,1,n))

            for k in range(n):
                im = axs[j].plot(sal[k], temp[k], color=colors[k], alpha = 0.5)

            # add density contours
            CS = axs[j].contour(si, ti, dens, levels = np.arange(26.4, 27.9, 0.2), colors='darkgrey', alpha = 0.6)
            plt.clabel(CS, inline=1, fontsize=11, fmt = '%1.1f')

            ymin = np.floor(np.nanmin(temp) - 0.05)
            ymax = np.ceil(np.nanmax(temp) - 0.1)
            xmin, xmax = 33.8, 35
            # axs[j].set_ylim(ymin,ymax)
            axs[j].set_xlim(xmin,xmax)
            axs[j].set_ylim(-0.3,8)
            # axs[j].set_xlim(33.75,35)
            axs[j].set_xticks(np.arange(34,34.8, 0.25)) 
            axs[j].grid(True)

            axs[j].set_xlabel('Salinity (psu)')
            axs[4].set_ylabel('\u03B8 (\N{DEGREE SIGN}C)')
            axs[j].text(0.35, 0.05,'{} profiles'.format(n),transform=axs[j].transAxes, fontsize = 12)

            i += 1
            j += 1

    plt.tight_layout(w_pad = 0.3, h_pad = 1)
    return axs


def surface_var_and_TS(float_num, floatid, section, alt_cmems, pcolor_ds, p_range = slice(0,1600), lon_range = [148, 155], lat_range = [-57, -51], **kwargs):

    levels = np.arange(-0.8,0.4,0.1)

    fig, axs = plt.subplots(nrows = 1, ncols=2, figsize = (10,4))
    axs = axs.flatten()

    start = float_num.time.values[section][0]
    end = float_num.time.values[section][-1]
    start_time, end_time = str(start.astype('M8[D]')), str(end.astype('M8[D]'))

    # mean sea level during float section
    msl = alt_cmems.adt.sel(time = slice(start_time, end_time)).mean(dim = 'time')
    mean_field = pcolor_ds.sel(time = slice(start_time, end_time)).mean(dim = 'time')
    
    im = mean_field.sel(longitude = slice(lon_range[0], lon_range[1]), latitude = slice(lat_range[0], lat_range[1])).plot(ax = axs[0], 
                                                                        **kwargs,add_colorbar=False)

    CS = msl.plot.contour(ax = axs[0], colors = 'dimgrey', alpha = 0.5, linewidths = 1.2, levels = levels, zorder = 3)
    plt.clabel(CS, inline=True, fontsize=11, fmt = '%1.1f')

    # Float trajectory map with points coloured in acccording to time slice
    axs[0].scatter(float_num.longitude, float_num.latitude, s = 50, c= 'slategrey',zorder = 2, alpha = 0.4)
    axs[0].scatter(float_num.longitude[section], float_num.latitude[section], s = 48, c='w', zorder = 2, alpha = 0.7)
    
    axs[0].text(0.03, 0.03, f'{start_time}', transform = axs[0].transAxes)
    axs[0].text(0.67, 0.03, f'{end_time}', transform = axs[0].transAxes)
    
    axs[0].set_xlabel(u'Longitude [\N{DEGREE SIGN}E]')
    axs[0].set_ylabel(u'Latitude [\N{DEGREE SIGN}N]')
    axs[0].set_title(f'EM-{floatid}')

    # T-S diagram
    CT, SA = settings.remove_bad_T_S(float_num, floatid)
        
    sal = SA[section].sel(pressure = p_range) #.sel(pressure = slice(150,300))
    temp = CT[section].sel(pressure = p_range) #.sel(pressure = slice(150,300))
    ti, si, dens = calc.tsDensContour(sal, temp)
    n = len(sal)
    colors = plt.cm.rainbow(np.linspace(0,1,n))

    for k in range(n):
        im = axs[1].plot(sal[k], temp[k], color=colors[k], alpha = 0.7)

    # add density contours
    CS = axs[1].contour(si, ti, dens, levels = np.arange(26.4, 27.9, 0.2), colors='darkgrey', alpha = 0.7)
    plt.clabel(CS, inline=1, fontsize=12, fmt = '%1.1f')
    
    axs[1].set_ylim(0.5,8)
    axs[1].set_xlim(33.75,35)
    axs[1].set_title(floatid)
    
    axs[1].text(0.35, 0.05,'{} profiles'.format(n),transform=axs[1].transAxes)
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def TSdiagrams(floatids, rs = True, save_fig = False):
    ema = imports.importFloatData(floatids)

    fig, axs = plt.subplots(3, 2, sharey = True, sharex = True, figsize = (10,12))
    axs = axs.flatten()

    i = 0 

    for floatid in floatids:
        float_num = ema[floatid]
        CT = float_num.CT
        SA = float_num.SA

        prof_num = np.tile(float_num.profile,(len(float_num.pressure),1))
        dist = calc.cum_dist(float_num.longitude, float_num.latitude)
        dist_km = np.tile(dist,(len(float_num.pressure),1))
        
        if rs == True:
            section = calc.findRSperiod(float_num)
            max_ = 1200
            colormap = plt.cm.get_cmap('tab20c', 12)
            ymax = 8
        else:
            section = section = slice(0, len(dist))
            max_ = 5000
            colormap = 'tab20c'
            ymax = 10
        
        for j in range(0,len(dist[section])):
            
            im = axs[i].scatter(SA[j], CT[j], c = dist_km[:,j], cmap = colormap, vmin = 0, vmax = max_)

        axs[i].set_title('Float '+ str(floatid))
        axs[i].set_ylim(0,ymax)
        axs[i].set_xlim(33.6,35)
        axs[i].set_xticks(np.arange(33.6,35.2, 0.2))
        axs[i].tick_params(axis='x', labelrotation = 45)
        axs[i].set_xlabel('SA')
        axs[i].set_ylabel('CT')
        axs[i].grid(True)

        if i % 2 != 0:
            axs[i].set_ylabel('')

        if 0 <= i < 4:
            axs[i].set_xlabel('')

        i += 1

    cax = plt.axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cax, extend = 'both')
    cbar.set_label('distance (km)')
    plt.show()
    
    if save_fig == True:
        if rs == True:
            sec = 'rs'
        else:
            sec = 'all'
        my_path = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd','figures','T-S')
        fig.savefig(os.path.join(my_path, 'TSdiag-{}.png'.format(sec)))


# ------------------------------------------------------------------------------------------------------------------------------------------------------

def colTrajectory(ax2, surf_data, float_num, floatid, alt_cmems, rs = True, msl_contours = True, line_plot = False, save_fig = False, my_path = None, name = None, **kwargs):
    dist = calc.cum_dist(float_num.longitude, float_num.latitude)
    latmin, latmax, lonmin, lonmax = -60, -50, 148, 173
    size = (9,10)
    profiles = slice(0,len(float_num.profile))
    cbar_size = [0.94, 0.12, 0.02, 0.45]

    start = float_num.time.values[0]
    end = float_num.time.values[-1]
    if rs == True:
        profiles = calc.findRSperiod(float_num)
        end = float_num.time[profiles].values[-1]
        latmin, latmax, lonmin, lonmax = -56.6, -51, 148.5, 155.5
        size = (6,10)
        if line_plot == True:
            size = (6,8)
            cbar_size = [0.94, 0.125, 0.02, 0.475]

    start_time, end_time = str(start.astype('M8[D]')), str(end.astype('M8[D]'))

    ssh = np.arange(-0.7, 0.5, 0.1)
    msl = alt_cmems.adt.sel(latitude = slice(latmin, latmax), longitude = slice(lonmin, lonmax), time = slice(start_time, end_time)).mean(dim = 'time')

    if line_plot == True:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = size, gridspec_kw={'height_ratios': [0.8, 2],
                                                                            'hspace':0.26})

        # ax1.scatter(dist[profiles],surf_data[profiles], s = 25)
        ax1.plot(dist[profiles],surf_data[profiles], linewidth = 2)
        

    # im = ax2.scatter(float_num.longitude, float_num.latitude, c = 'slategrey', s = 30, alpha = 0.7)
    im2 = ax2.scatter(float_num.longitude[profiles], float_num.latitude[profiles], c = surf_data[profiles], **kwargs, zorder = 2)

    if msl_contours == True:
        CS = msl.plot.contour(ax = ax2, colors = 'gray', linewidths = 1.5, alpha = 0.5, levels = ssh, zorder = 1)
        plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')

    if line_plot == True:
        cbar_ax = fig.add_axes(cbar_size)
        fig.colorbar(im2, cax=cbar_ax, extend = 'both')

    ax2.set_ylim(latmin, latmax)
    ax2.set_xlim(lonmin, lonmax)
    ax2.set_xlabel(u'Longitude [\N{DEGREE SIGN}E]')
    ax2.set_ylabel(u'Latitude [\N{DEGREE SIGN}N]')

    if save_fig == True:
        plt.savefig(os.path.join(my_path, 'float-%s-%s.png' %floatid %name), bbox_inches='tight') 
    
    if line_plot == True:
        return ax1, ax2
    else:
        return im2, msl

# ------------------------------------------------------------------------------------------------------------------------------------------------------


def concatTickValues(data_dict):
    '''Input: a list of the last distance value of each float.'''

    # find indices where new float starts
    data_ct = ct.joinFloats(data_dict, 'distance', new_dim = False)
    flt_ind = np.where(data_ct.distance == 0)[0].tolist()
    flt_ind.remove(flt_ind[0])
    # concatenate with total distance
    data = ct.joinFloats(data_dict, 'distance', new_dim = True)
    flt_dist_loc = data.distance[flt_ind].data

    if len(data_dict) == 4:
        ticks =[0,300,600,flt_dist_loc[0], flt_dist_loc[0]+300, flt_dist_loc[0]+600, flt_dist_loc[0]+900, flt_dist_loc[0]+1200, 
                flt_dist_loc[1], flt_dist_loc[1]+300, flt_dist_loc[1]+600, flt_dist_loc[1]+900, 
                flt_dist_loc[2], flt_dist_loc[2]+300, flt_dist_loc[2]+600]

        values = [0,300,600,0,300,600,900,1200, 0, 300, 600, 900, 0, 300, 600]

    elif len(data_dict) == 3:

        ticks =[0,300,600,flt_dist_loc[0], flt_dist_loc[0]+300, flt_dist_loc[0]+600, flt_dist_loc[0]+900,
                flt_dist_loc[1], flt_dist_loc[1]+300, flt_dist_loc[1]+600]

        values = [0,300,600, 0, 300, 600, 900, 0, 300, 600]

    return ticks, values, flt_dist_loc

