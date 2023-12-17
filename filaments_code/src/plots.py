
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

def sectionsOnP(float_num, floatid, z_max = None, TS = True, UV = False, rs = True, save_fig = False):
    
    newFloat = settings.distanceAsCoord(float_num)
    CT, SA = settings.remove_bad_T_S(newFloat, floatid)

    if rs == True: 
        section = calc.findRSperiod(float_num)
        lw = 1
    else:
        section = slice(0, len(newFloat.distance))
        lw = 0.6

    if z_max == None:
        z_max = int(float_num.pressure[-1])

    pdens = calc.potentialDensity(newFloat.pressure, SA, CT)
    levels = np.arange(26.8,np.nanmax(pdens),0.2)
    
    if TS == True:
        fig, (ax1, ax2) = plt.subplots(2,1, figsize = (12,8))

        if rs == True:
            min_, max_ = 1, 5
        else:
            min_, max_ = 0, 8

        # Temperature
        newFloat.T[section].sel(pressure = slice(0, z_max)).plot(ax = ax1, x = 'distance', y='pressure',cbar_kwargs={'label':'°C'},
                         vmin = min_, vmax = max_, cmap = cmocean.cm.thermal)
        CS = pdens[section].plot.contour(ax = ax1, x = 'distance', colors = 'snow', linewidths = lw, linestyles = 'solid', levels = levels, alpha = 0.7)
        plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')
        ax1.invert_yaxis()
        ax1.set_ylabel('pressure (dbar)')
        ax1.set_xlabel('')
        ax1.set_title('Float {}'.format(floatid))

        # Salinity
        newFloat.S[section].sel(pressure = slice(0, z_max)).plot(ax = ax2, x = 'distance', y='pressure',cbar_kwargs={'label':'PSU'},
                         vmin = 33.5, vmax = 34.5, cmap = cmocean.cm.haline)
        CS = pdens[section].plot.contour(ax = ax2, x = 'distance', colors = 'k', linewidths = lw, linestyles = 'solid', levels = levels, alpha = 0.5)
        plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')
        ax2.invert_yaxis()
        ax2.set_ylabel('pressure (dbar)')
        ax2.set_xlabel('distance (km)')

        if save_fig == True:
            datatype = 'TS'
            if rs == True:
                section = 'rs'
            else:
                section = 'all'

            my_path = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd','figures','T-S')
            name = 'float-{}-{}-onP-{}.png'.format(floatid, datatype, section)
            settings.save_figure(fig, my_path, name)


    if UV == True:
        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True, figsize = (10,8))

        # Zonal velocity
        newFloat.u[section].plot(ax = ax1, x = 'distance', y='pressure',cbar_kwargs={'label':'U (m/s)'}, 
                                 cmap = cmocean.cm.balance)
        ax1.invert_yaxis()
        ax1.set_ylabel('pressure (dbar)')
        ax1.set_xlabel('')
        ax1.set_title('Float {}'.format(floatid))

        # Meridional velocity
        newFloat.v[section].plot(ax = ax2, x = 'distance', y='pressure',cbar_kwargs={'label':'V (m/s)'}, 
                                 cmap = cmocean.cm.balance)
        ax2.invert_yaxis()
        ax2.set_ylabel('pressure (dbar)')
        ax2.set_xlabel('distance (km)')
    
        if save_fig == True:
            datatype = 'UV'
            if rs == True:
                section = 'rs'
            else:
                section = 'all'
            my_path = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd','figures','velocities', 'abs')
            name = 'float-{}-{}-onP-{}.png'.format(floatid, datatype, section)
            settings.save_figure(fig, my_path, name)

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# IDENTIFY ACC FRONTS from float data 

def identifyPF(float_num, floatid, z_max = None, panel = False, ax = None, save_fig = False, my_path = None, **kwargs):
    '''Identify the Polar Front using the criteria: presence of the 2 °C isotherm between 200-300 m depth '''
    try:
        rs = calc.findRSperiod(float_num)
        newFloat = settings.distanceAsCoord(float_num)
    except:
        rs = slice(0, len(float_num.CT))
        newFloat = float_num
    
    CT, SA = settings.remove_bad_T_S(newFloat, floatid)
    dist = newFloat.distance

    pdens = calc.potentialDensity(newFloat.pressure, SA, CT)
    levels = np.arange(26.8,np.nanmax(pdens),0.2)

    if z_max == None:
        z_max = int(float_num.pressure[-1])

    # find where the 2 degree isotherm is found between 200-300 m depth 
    ind = np.where(np.logical_and(newFloat.CT[rs].sel(pressure = slice(200,300)) > 1.95, 
                                  newFloat.CT[rs].sel(pressure = slice(200,300)) < 2.04))[0].tolist()
    
    # print the number of profiles where this criteria is met  
    print('Float {}: PF criteria met in {} out of {} profiles during rapid sampling'.format(floatid, len(np.unique(ind)), len(CT)))

    if panel == False: 
        fig, ax = plt.subplots(figsize = (8,3))

    # plot the T section with the 2 degree isotherm, 200 & 300 m dashed lines and red markers indicating the where the criteria is met
    im = newFloat.CT[rs].sel(pressure = slice(0,z_max)).plot(ax = ax, x = 'distance', **kwargs, cmap = cmocean.cm.thermal, 
                                                      add_colorbar = False, alpha = 0.9)

    # plot potential density contours
    CS = pdens[rs].plot.contour(ax = ax, x = 'distance', colors = 'k', linewidths = 1, linestyles = 'solid', levels = levels, alpha = 0.5)
    plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')

    for j in range(0, len(np.unique(ind))):
        ax.axvline(x = dist[np.unique(ind)[j]], ymin = 0, ymax = 0.07, color = 'red', alpha = 0.8)
    
    ax.axhline(y = 200, color = 'white', linestyle = '--')
    ax.axhline(y = 300, color = 'white', linestyle = '--')

    if panel == False: 
        settings.tickLocations(ax)
        if floatid == 8490: 
            settings.tickLocations(ax, major = 200)
               
    levels = np.arange(2,2.05,0.1)
    newFloat.CT[rs].plot.contour(ax = ax, x = 'distance', colors = 'snow', alpha = 0.7, linewidths = 1.8, 
                                 levels = levels, linestyles = 'solid', zorder = 4)

    ax.invert_yaxis()
    ax.set_xlabel('distance (km)')
    ax.set_ylabel('pressure (dbar)')
    ax.set_title('EM-{}'.format(floatid))
    
    if panel == False: 
        cax = plt.axes([0.93, 0.14, 0.02, 0.73])
        clb = plt.colorbar(im, cax=cax, extend = 'both', label='T (°C)')
    else:
        return im 

    if save_fig == True:
        name = '{}-PF-identification-{}m'.format(floatid, z_max) 
        settings.save_figure(fig, my_path, name)



# ------------------------------------------------------------------------------------------------------------------------------------------------------

def identifySAF(float_num, floatid, z_max = None, panel = False, ax = None, save_fig = False, my_path = None, **kwargs):
    '''Identify the Sub Antarctic Front using the criteria: salinity minimum waters (< 34.2) at depths greater than 400 m (Kim & Orsi, 2014)'''
    rs = calc.findRSperiod(float_num)
    newFloat = settings.distanceAsCoord(float_num)
    CT, SA = settings.remove_bad_T_S(newFloat, floatid)
    dist = newFloat.distance

    if z_max == None:
        z_max = int(float_num.pressure[-1])

    # calculate potential density
    pdens = calc.potentialDensity(newFloat.pressure, SA, CT)
    levels = np.arange(26.8,np.nanmax(pdens),0.2)
    
    if panel == False: 
        fig, ax = plt.subplots(figsize = (8,3))
    
    # plot salinity shading
    im = newFloat.S[rs].sel(pressure = slice(0,z_max)).plot(ax = ax, x = 'distance', cmap = cmocean.cm.haline, **kwargs, 
                                                      add_colorbar = False, alpha = 0.8)
    
    # plot potential density contours
    CS = pdens[rs].plot.contour(ax = ax, x = 'distance', colors = 'k', linewidths = 1, linestyles = 'solid', levels = levels, alpha = 0.5)
    plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')
    
    if panel == False: 
        settings.tickLocations(ax)
        if floatid == 8490: 
            settings.tickLocations(ax, major = 200)
    
    # plot the 34.2 psu isohaline
    isoline = np.arange(34.2,34.25,0.1)
    newFloat.S[rs].plot.contour(ax = ax, x = 'distance', colors = 'snow', alpha = 0.9, linewidths = 1.8, 
                                 levels = isoline, linestyles = 'solid')

    # find indices that meet SAF criteria
    ind = np.where(newFloat.S[rs].sel(pressure = slice(400,int(float_num.pressure[-1]))) < 34.2)[0].tolist()
     # print the number of profiles where this criteria is met  
    print('Float {}: SAF criteria met in {} out of {} profiles during rapid sampling'.format(floatid, len(np.unique(ind)), len(float_num.profile[rs])))

    # plot blue markers where the criteria is met
    for j in range(0, len(np.unique(ind))):
        ax.axvline(x = dist[np.unique(ind)[j]], ymin = 0, ymax = 0.07, color = 'blue', alpha = 0.7)
    
    # boundary marking 400 m depth
    ax.axhline(y = 400, color = 'white', linestyle = '--')
    
    ax.invert_yaxis()
    ax.set_xlabel('distance (km)')
    ax.set_ylabel('pressure (dbar)') 
    ax.set_title('EM-{}'.format(floatid))

    if panel == False: 
        cax = plt.axes([0.93, 0.14, 0.02, 0.73])
        clb = plt.colorbar(im, cax=cax, extend = 'both', label='S (psu)')
    else:
        return im 
    
    if save_fig == True:
        name = '{}-SAF-identification-{}m'.format(floatid, z_max) 
        settings.save_figure(fig, my_path, name)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot float data with density as y coordinate

def sectionsOnD(float_num, floatid, T = True, S = True, U = False, V = False, rs = True, save_fig = False):
    
    on_dens = interp.interpOnDens(float_num)
    dist = calc.cum_dist(float_num.longitude, float_num.latitude)
    
    if rs == True: 
        section = calc.findRSperiod(float_num)
    else:
        section = slice(0, len(dist))
    
    if T and S == True:
        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True, figsize = (10,8))

        if rs == True:
            min_, max_ = 1, 7
        else:
            min_, max_ = 0, 8

        # Temperature
        on_dens.T[section].plot(ax = ax1, x = 'distance',cbar_kwargs={'label':'°C'},
                         vmin = min_, vmax = max_, cmap = cmocean.cm.thermal)
        ax1.invert_yaxis()
        ax1.set_ylabel('\u03C3 (kg $m^{-3}$)')
        ax1.set_xlabel('')
        ax1.set_title('Float {}'.format(floatid))

        # Salinity
        on_dens.S[section].plot(ax = ax2, x = 'distance', cbar_kwargs={'label':'PSU'},
                         vmin = 33.5, vmax = 34.5, cmap = cmocean.cm.haline)
        ax2.invert_yaxis()
        ax2.set_ylabel('\u03C3 (kg $m^{-3}$)')
        ax2.set_xlabel('distance (km)')

        if save_fig == True:
            name = 'TS'
            if rs == True:
                sec = 'rs'
            else:
                sec = 'all'
            my_path = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd','figures','T-S')
            fig.savefig(os.path.join(my_path, 'float-{}-{}-onD-{}.png'.format(floatid, name, sec)))

    if U and V == True:
        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True, figsize = (10,8))

        # Zonal velocity
        on_dens.U[section].plot(ax = ax1, x = 'distance', cbar_kwargs={'label':'U (m/s)'}, 
                                 vmin = -1, vmax = 1, cmap = cmocean.cm.balance)
        ax1.invert_yaxis()
        ax1.set_ylabel('\u03C3 (kg $m^{-3}$)')
        ax1.set_xlabel('')
        ax1.set_title('Float {}'.format(floatid))

        # Meridional velocity
        on_dens.V[section].plot(ax = ax2, x = 'distance', cbar_kwargs={'label':'V (m/s)'}, 
                                 vmin = -1, vmax = 1, cmap = cmocean.cm.balance)
        ax2.invert_yaxis()
        ax2.set_ylabel('\u03C3 (kg $m^{-3}$)')
        ax2.set_xlabel('distance (km)')
    
        if save_fig == True:
            name = 'uv'
            if rs == True:
                sec = 'rs'
            else:
                sec = 'all'
            my_path = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd','figures','velocities', 'abs')
            fig.savefig(os.path.join(my_path, 'float-{}-{}-onD-{}.png'.format(floatid, name, sec)))


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DYNAMIC HEIGHT vs SSH 

def dynH_ADT(float_num, floatid, alt_cmems, steric = False, panel = False, ax = None):

    try:
        rs = calc.findRSperiod(float_num)
    except:
        rs = slice(0, len(float_num.CT))

    dist = calc.cum_dist(float_num.longitude, float_num.latitude)

    interp_val, lower, upper = stats.temporalError(float_num, alt_cmems.adt, method = 'interp', rs = True)

    dyn_m, dyn_m_500 = calc.dynamicHeight(float_num, floatid)
    steric_h, steric_50 = calc.dynamicHeight(float_num, floatid, steric = True)

    if panel == False:
        fig, ax = plt.subplots(figsize = (10,4))

    # plot ADT with temporal error shading 
    ax.plot(dist[rs], interp_val, alpha = 0.7, label = 'adt (m)')
    ax.plot(dist[rs], upper, c = 'grey', alpha = 0.4)
    ax.plot(dist[rs], lower, c = 'grey', alpha = 0.4)
    ax.fill_between(dist[rs], lower, upper, alpha=0.2)

    if panel == False:
        print('Spatially interpolated ADT onto float locaitons, with temporal error bars showing values at time stamps before and after')
        settings.tickLocations(ax)
        if floatid == 8490: 
            settings.tickLocations(ax, major = 200)
        
    ax.set_ylabel('adt (m)', color= 'tab:blue', fontweight = 'bold')
    ax.grid()
    ax.set_xlabel('distance (km)')
    
    if floatid == 8493:
        ax.set_ylim(-0.9, -0.1)
    else:
        ax.set_ylim(-0.8,0)

    if steric == True:
        st_h = steric_50 - 1.5
        st_h[rs].plot(ax = ax, c = 'tab:purple', linewidth = 2, label = 'steric height - offset')
        if panel == False:
            plt.legend()
            plt.title(floatid)
            print('Steric Height = dyn_h anomaly / 9.7963 m/s2. Compare to ADT by substracting an offset value (e.g. 1.5)')

    else:
        ax2 = ax.twinx()
        dyn_m_500[rs].plot(ax = ax2, c = 'tab:orange', linewidth = 2, alpha = 0.8)
        if floatid == 8493:
            ax2.set_ylim(0.40, 0.57)
        else:
            ax2.set_ylim(0.41, 0.69)

        if panel == True:
            return ax2
        else:
            plt.title(f'EM-{floatid}')
            ax2.set_ylabel('dynamic height (dyn m)', color="tab:orange", fontweight = 'bold')
            print('dynamic height is at 500 db relative to 1500 dbar (units of dyn m)')

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def surfVectors(float_num, floatid, alt_cmems, gradvel, pcolor_data, ind_list, extent, fig_size = (9,8), cbar_label = None, **kwargs):
    '''Plot a map of the flow field (geostrophic and ageostrophic) with float track, averaged during part of the float track 
    given by ind_list.

    Inputs: 
    pcolor_data - could be relative vorticity, sea level anomaly, divergence (any altimetry data file)
    gradvel - gradient wind datafile from calc.gradientWind(adt, kappa, ugeos, vgeos). Used for geostrophic and ageostrophic velocities.
    ind_list - index list of profiles (could be a week index or between two distances). Used for time averaging and to get change in dynamic H and SSH. '''
    
    # settings
    lonmin, lonmax, latmin, latmax = extent
    levels = np.arange(-0.9,0.5,0.1)
    F = np.arange(-0.4,-0.1,0.1)

    # change in dynaimc H and SSH 
    dyn_m, dyn_m_500 = calc.dynamicHeight(float_num)
    ssh = interp.interpToFloat(float_num, alt_cmems.adt)
    deltaDH = dyn_m_500[ind_list][-1].data - dyn_m_500[ind_list][0].data
    deltaSSH = ssh[ind_list][-1] - ssh[ind_list][0]

    start = float_num.time.values[ind_list][0]
    end = float_num.time.values[ind_list][-1]
    start, end = str(start.astype('M8[D]')), str(end.astype('M8[D]'))

    # calculate ageostrophic velocities
    ua = gradvel.ugrad - gradvel.ugeos
    va = gradvel.vgrad - gradvel.vgeos
    ageos = gradvel.Vgrad - gradvel.Vgeos
    ageos_ratio = ageos/gradvel.Vgrad

    # time-mean fields 
    a_mean = ageos_ratio.sel(time = slice(start, end)).mean(dim = 'time')
    vel_mean = gradvel.Vgrad.sel(time = slice(start, end)).mean(dim = 'time')
    msl = alt_cmems.adt.sel(time = slice(start, end)).mean(dim = 'time')
    mdata = pcolor_data.sel(time = slice(start, end)).mean(dim = 'time')

    # time-mean velocities
    ua_mean = ua.sel(time = slice(start, end)).mean(dim = 'time')
    va_mean = va.sel(time = slice(start, end)).mean(dim = 'time')
    u_mean = gradvel.ugrad.sel(time = slice(start, end)).mean(dim = 'time')
    v_mean = gradvel.vgrad.sel(time = slice(start, end)).mean(dim = 'time')

    # plot figure
    fig, ax = plt.subplots(figsize = fig_size)
    im = mdata.sel(latitude = slice(latmin,latmax), longitude = slice(lonmin,lonmax)).plot(**kwargs, add_colorbar = False)

    plt.quiver(a_mean.longitude,  a_mean.latitude, ua_mean, va_mean, scale = 0.2, alpha = 0.7,  width = 0.003)
    plt.quiver(vel_mean.longitude,  vel_mean.latitude, u_mean, v_mean, color = 'silver', scale = 4, alpha = 0.7, width = 0.003) 

    CS = msl.plot.contour(colors = 'grey', linewidths = 1.5, linestyles = 'solid', levels = F)
    plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')
    CS = msl.plot.contour(colors = 'grey', linewidths = 1.5, linestyles = 'solid', levels = levels, alpha = 0.4)
    plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')

    ax.scatter(float_num.longitude, float_num.latitude, s = 50, c= 'slategrey', zorder = 3)
    ax.scatter(float_num.longitude[ind_list], float_num.latitude[ind_list], s = 30, c='w', zorder = 4)
    ax.text(0, 1.02, f'$\Delta$ dyn_h: {deltaDH:.3} m', transform = ax.transAxes)
    ax.text(0.72, 1.02, f'$\Delta$ ssh: {deltaSSH:.3} m', transform = ax.transAxes)
    plt.title(floatid)

    clb = plt.colorbar(im, extend = 'both')
    clb.set_label(cbar_label, fontsize = 20)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def weekPanel(floatid, float_num, dataArray, ssh, weeks, contours, col = None, label = None, rotate = False, save_fig = False, name = None, my_path = None, **kwargs):

    fig, axs = plt.subplots(nrows = 1, ncols=3, sharey = True, sharex = True, figsize = (12,4)) # (16,9)
    axs = axs.flatten()

    i = 0
    for week in weeks:
        
        ind_list = calc.getIndexList(float_num, week)

        start = float_num.time.values[ind_list][0]
        end = float_num.time.values[ind_list][-1]
        start_time = str(start.astype('M8[D]'))
        end_time = str(end.astype('M8[D]'))

        wk_mean = dataArray.sel(time = slice(start_time, end_time)).mean(dim = 'time')
        msl = ssh.sel(time = slice(start_time, end_time)).mean(dim = 'time')

        if week > 17:
            lonmin, lonmax = 158, 172
            latmin, latmax = -60, -53
        else:
            lonmin, lonmax = 148, 162
            latmin, latmax = -57.5, -50.5

        im = wk_mean.sel(longitude = slice(lonmin,lonmax), latitude = slice(latmin,latmax)).plot(ax = axs[i], **kwargs, add_colorbar=False)

        levels = np.arange(contours[0]-0.5,contours[-1]+0.5,0.1)
        CS = msl.plot.contour(ax = axs[i], colors = 'lightgrey', alpha = 0.5, linewidths = 1.8, linestyles = 'solid', levels = levels)
        plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')
        CS = msl.plot.contour(ax = axs[i], colors = col, alpha = 0.4, linewidths = 1.8, linestyles = 'solid', levels = contours)
        plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')

        # Float trajectory map with points coloured in acccording to time slice
        axs[i].scatter(float_num.longitude, float_num.latitude, s = 20, c= 'slategrey', zorder = 2, alpha = 0.6)
        axs[i].scatter(float_num.longitude[ind_list], float_num.latitude[ind_list], s = 20, c='w', zorder = 2, alpha = 0.9)
        
        axs[i].set_title('week: {}'.format(week))
        if 0 <= i < 3:
            axs[i].set_xlabel('')
            
        if 1 <= i < 3 or 4 <= i < 6:
            axs[i].set_ylabel('')
        
        i += 1

    plt.tight_layout(w_pad = 1)
        
    cax = plt.axes([1, 0.15, 0.015, 0.7])
    clb = plt.colorbar(im, cax=cax, extend = 'both')
    if label is not None:
        if rotate == True:
            clb.set_label(label, labelpad=-30, y=1.1, rotation=0, fontsize = 20)
        else:
            clb.set_label(label, fontsize = 15)
        # clb.ax.set_title(label,y=1.13, fontsize = 20)
    plt.show()

    if save_fig == True:
        name = '{}-{}-weeks-{}'.format(name, floatid, weeks[0]) 
        settings.save_figure(fig, my_path, name)
        

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Temperature anomalies

def plot_T_anomaly(float_num, floatid, zdim = ('pressure', 'potential_density'), figsize = (10,4), vmax = 1.5,
                                                                                        save_fig = False, my_path = None):
    
    rs = calc.findRSperiod(float_num)
    if zdim == 'potential_density':
        pdens = True
    else:
        pdens = False
    
    anomalies = calc.TS_anom(float_num, floatid, pdens = pdens, by_dist = True)

    fig, ax = plt.subplots(figsize = figsize)
    anomalies.CT[rs].plot(x = 'distance', vmin = vmax*-1, vmax = vmax, cmap = 'RdBu_r', cbar_kwargs={'label':'$CT^\prime$ (°C)'})
    
    settings.tickLocations(ax)

    if floatid == 8490: 
        settings.tickLocations(ax, major = 200)

    plt.gca().invert_yaxis()
    ax.set_ylabel(zdim)
    # ax.set_ylim(26.4, 27.8)
    ax.set_xlabel('distance (km)')
    plt.title('EM-{}'.format(floatid))
    plt.tight_layout(h_pad = 0.5)

    # cax = plt.axes([1, 0.24, 0.02, 0.67])
    # plt.colorbar(im, cax=cax, extend = 'both', label = 'T Anomaly (°C)')

    if save_fig == True:
        name = 'float-%s-anomalies.png' %floatid
        settings.save_figure(fig, my_path, name)

    return ax

# ------------------------------------------------------------------------------------------------------------------------------------------------------


def plotCurvature(float_num, floatid, curv_data, flow = False, save_fig = False, my_path = None):
    
    dist = calc.cum_dist(float_num.longitude, float_num.latitude)
    k = curv_data.copy()

    if flow == True:
        title = 'surface flow curvature - {}'
        name =  '%s-flow-curvature.png' %floatid
    else:
        title = 'Float {}: Curvature'
        name = '%s-float-curvature.png' %floatid

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (9,10), gridspec_kw={'height_ratios': [1.2, 2],
                                                                        'hspace':0.3})
    ax1.scatter(dist,k)
    ax1.plot(dist,k)
    ax1.axhline(y=0, color='grey', linestyle='--')
    ax1.set_ylim(np.nanmin(k)-5e-6,np.nanmax(k)+5e-6)
    ax1.set_title(title.format(floatid))
    ax1.set_xlabel('distance (km)')

    im2 = ax2.scatter(float_num.longitude, float_num.latitude, c = k, vmin = -3e-5, vmax = 3e-5, cmap = cmocean.cm.balance)
    cbar_ax = fig.add_axes([0.92, 0.13, 0.02, 0.4])
    cbar = plt.colorbar(im2, cax = cbar_ax)
    ax2.set_xlabel('longitude')
    ax2.set_ylabel('latitude')
    fig.subplots_adjust(wspace = 0.2)

    if save_fig == True:
        settings.save_figure(fig, my_path, name)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def plotSurfVars(float_num, floatid, alt_cmems, save_fig = False, my_path = None):
    rs = calc.findRSperiod(float_num)
    dist = calc.cum_dist(float_num.longitude, float_num.latitude)

    name = '%s-surface-vars-rs.png' %floatid
    adt = interp.interpToFloat(float_num, alt_cmems.adt)
    sla = interp.interpToFloat(float_num, alt_cmems.sla)
    vort = calc.relativeVort(float_num, alt_cmems, interp_to_float = True)
    eke = calc.calcEKE(float_num, alt_cmems, altimetry = True, interp_to_flt = True)
    
    # comes up with an error for float 8490 - difficulty smoothing the contorted trajectory. Just use rapid sampling
    curv_float = calc.floatCurvature(float_num, floatid, transform = True, remove_outliers = True)

    fig, ax = plt.subplots(figsize = (12,5))
    ax.plot(dist[rs],vort[rs], color="#e41a1c")
    ax.set_ylabel('Relative Vorticity', color="#e41a1c", fontweight = 'bold')
    ax.axhline(y=0, color='grey', linestyle='--', alpha = 0.5)
    if floatid != 8490:
        ax.set_xticks(np.arange(0,1400,100))
    ax.set_ylim(np.nanmin(vort[rs])-0.2e-5, (np.nanmin(vort[rs])*-1)+0.2e-5) # -0.6e-5,0.6e-5
    ax.grid(True)
    ax.set_title("Float {}".format(floatid))

    ax2 = ax.twinx()
    ax2.plot(dist[rs],curv_float[rs], color="#4daf4a")
    ax2.set_ylabel('Float Curvature', color="#4daf4a", fontweight = 'bold')
    ax2.set_ylim(np.nanmin(curv_float[rs])-0.2e-4, (np.nanmin(curv_float[rs])*-1)+0.2e-4) # -6e-5,6e-5
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    ax3 = ax.twinx()
    ax3.plot(dist[rs],sla[rs], color="#377eb8")
    ax3.set_ylabel('Sea Level Anomaly (m)', color="#377eb8", fontweight = 'bold')
    ax3.spines['right'].set_position(("axes", 1.12))
    ax3.set_ylim(np.nanmin(sla[rs])-0.1, (np.nanmin(sla[rs])*-1)+0.1) # -0.3,0.3

    ax4 = ax.twinx()
    ax4.plot(dist[rs], eke[rs], color="#984ea3", label ='EKE')
    ax4.set_ylabel('EKE ($m^{2}$ $s^{-2}$)', color="#984ea3", fontweight = 'bold')
    # ax4.set_yscale("log")
    ax4.set_ylim(-0.01,0.6)
    ax4.spines['right'].set_position(("axes", 1.25)) # 1.23

    # plt.axvspan(200, 300, color='grey', alpha=0.2, lw=0)
    # plt.axvspan(700, 850, color='grey', alpha=0.2, lw=0)
    
    fig.tight_layout(pad=1.5)

    if save_fig == True:
        settings.save_figure(fig, my_path, name)


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

# -----------------------------------------------------------------------------------------------------------------------------------

def plotUVonD(float_num, floatid, save_fig = False):
    on_dens = interp.interpOnDens(float_num)
    rs = calc.findRSperiod(float_num)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (10,10))

    on_dens.U[rs].plot(ax = ax1, x = 'distance', cbar_kwargs={'label':'u (m $s^{-1}$)'}, vmin = -1, vmax = 1,  cmap = cmocean.cm.balance)
    ax1.set_xlabel('')
    ax1.set_ylabel('\u03C3 (kg $m^{-3}$)')
    ax1.invert_yaxis()
    ax1.set_title('Float {}'.format(floatid))


    on_dens.V[rs].plot(ax = ax2, x = 'distance', cbar_kwargs={'label':'v (m $s^{-1}$)'}, vmin = -1.5, vmax = 1.5,  cmap = cmocean.cm.balance)
    ax2.invert_yaxis()
    ax2.set_ylabel('\u03C3 (kg $m^{-3}$)')
    ax2.set_xlabel('distance (km)')

    if save_fig == True:
        my_path = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd','figures','velocities')
        plt.savefig(os.path.join(my_path, 'float-%s-uv-onD-rs.png' %floatid), bbox_inches='tight') 

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def EKEsection(float_num, floatid, alt_ds = None, save_fig = False):
    rs = calc.findRSperiod(float_num)
    EKE, KE, mKE = calc.calcEKE(float_num, floatid, alt_ds = alt_ds, altimetry = False, floats = True)

    fig, ax = plt.subplots(figsize = (10,5))

    EKE[rs].plot(x = "distance", cbar_kwargs={'label':'EKE ($m^{2}$ $s^{-2}$)'}, norm = mpl.colors.LogNorm(vmin=0.001, vmax=1))
    ax.invert_yaxis()
    ax.set_title('Float {}'.format(floatid))
    ax.set_ylabel('\u03C3 (kg $m^{-3}$)')
    ax.set_xlabel('distance (km)')

    if save_fig == True:
        my_path = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd','figures','velocities', 'eke')
        plt.savefig(os.path.join(my_path, 'float-%s-eke.png' %floatid), bbox_inches='tight') 

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

def runningMeanAndStd(float_num, floatid, data_on_dens, w, name, ymax = None, save_fig = False):
    rs = calc.findRSperiod(float_num)
    dens_contours = np.arange(26.9, 27.55, 0.05) 
    colors = plt.cm.rainbow(np.linspace(0,1,len(dens_contours)))

    fig, ax1 = plt.subplots(figsize = (8,5))
    fig2, ax2 = plt.subplots(figsize = (8,4))

    j = 0 
    for d in dens_contours:
        ind = stats.find_nearest(data_on_dens.potential_density, d)[0]
        da = data_on_dens[rs,ind]
        da.rolling(distance=w, center=True).mean().plot(ax = ax1, color=colors[j], alpha = 0.9)
        da.rolling(distance=w, center=True).std().plot(ax = ax2, color=colors[j], alpha = 0.9)
        j += 1

    ax2.grid()
    ax2.set_ylabel('')
    settings.tickLocations(ax2)
    # if floatid == 8490: 
        # settings.tickLocations(ax2, major = 200)  

    # set axes labels
    ax1.set_xlabel('')
    ax2.set_xlabel('distance (km)')
    ax1.set_title('rolling mean')
    ax2.set_title('rolling std')
    if ymax != None:
        ax2.set_ylim(0,ymax)
    ax2.set_xlim(0, data_on_dens.distance[rs][-1])
    ax1.text(0.01, 1.03, name, transform = ax1.transAxes, fontsize = 16)
    plt.tight_layout(h_pad = 0.7)

    # add colorbar
    cax = plt.axes([1, 0.17, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='rainbow', norm=plt.Normalize(vmin=dens_contours[0], vmax=dens_contours[-1]))
    plt.colorbar(sm, cax=cax, extend = 'both', label = 'density (kg $m^{-3})$')

    print(f'window: {w} profiles')

    if save_fig == True:
        my_path = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd','figures','T-S', 'variability')
        plt.savefig(os.path.join(my_path, 'float-{}-{}_mean_std.png'.format(floatid, name)), bbox_inches='tight') 
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def concat_on_dens(ax, data_dict, line_plot = False, d_slice = None, cbar_ax = [0.92, 0.12, 0.01, 0.75], cbar_label = None, **kwargs):
    '''Plots more than one float by total distance on a density grid
    INPUT:
    data - concatenated xr dataset on density grid (use 'concatenated_flts' function)
    flt_ind - list of indices marking the transition to a different float
    '''
    data = ct.joinFloats(data_dict, 'distance', new_dim = True)
    # # set tick labels to start at 0 km for each float 
    ticks, values, flt_dist_loc = concatTickValues(data_dict)

    # fig, ax = plt.subplots(figsize = (14,3))
    im = data.plot(x = 'distance', add_colorbar = False, **kwargs)
    ax.invert_yaxis()
    plt.xlabel('distance (km)')
    ax.set_xticks(ticks)
    ax.set_xticklabels(values)

    for i in range(0,len(flt_dist_loc)):
        ax.axvline(x = flt_dist_loc[i], linestyle = '--', color = 'k')

    cax = plt.axes(cbar_ax)
    plt.colorbar(im, cax=cax, extend = 'both', label = cbar_label)

    if line_plot == True:
        fig, ax2 = plt.subplots(figsize = (12,2))
        if d_slice != None:
            data.sel(potential_density = d_slice).mean(dim = 'potential_density').plot()
        else:
            data.mean(dim = 'potential_density').plot()
        plt.grid()
        plt.xlabel('distance (km)')
        plt.xlim(0, data.distance[-1])
        plt.ylabel(cbar_label)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(values)

        for i in range(0,len(flt_dist_loc)):
            ax2.axvline(x = flt_dist_loc[i], linestyle = '--', color = 'grey')

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def power_spectra(var, floatid, dim = 'potential_density', lst = np.arange(26.8, 27.6, 0.1), sections = None, label = None, legend = True, return_slopes = False,
                ymin = 10**-6, ymax = 10**5, bins = np.logspace(np.log10(10**-3),np.log10(10**0), 10), whiten = False, scale =10**-4):
    
    fig, ax = plt.subplots(figsize = (5,5))
    
    if sections is None:
        sections = [slice(0, len(var))]
    elif type(sections) is not list:
        sections = [sections]
        
    for section in sections:
        d = var[section]
            
        dist_interval = 2
        ds = interp.even_dist_grid(d, dist_interval)
        fs = 1/2
        S_tot = {}

        if dim == 'potential_density':
            dim = r'$\rho$'
            colors = plt.cm.rainbow(np.linspace(0,1,len(lst)))
            k = 0
            slopes = {}
            for pd in lst:
#                 signal = ds.sel(potential_density = pd, method = 'nearest')
                signal = ds.sel(potential_density = slice(pd-0.01, pd+0.01)).mean(dim = 'potential_density')
                signal = signal.interpolate_na(dim = 'distance',method="linear", fill_value="extrapolate", max_gap = 5).dropna(dim = 'distance')
                (f, S) = scipy.signal.periodogram(signal, fs, scaling='density')
                if whiten == True:
                    S = S*(4*np.pi**2*f**2)
                S_tot[pd] = S[1:-1]
                plt.plot(f, S, color = colors[k], alpha = 0.4)
                
                if bins is not None:
                    bin_values, bin_edges = scipy.stats.binned_statistic(f, S, statistic='mean', bins=bins)[0:2]
                    x, y = bin_edges[:-1][~np.isnan(bin_values)], bin_values[~np.isnan(bin_values)]
                    plt.plot(x, y, color = colors[k], label='_nolegend_') 
                    plt.scatter(x, y, color = colors[k], label='_nolegend_')
                    
                    # power law fit 
                    km100 = int(np.where(bins == 0.01)[0])
                    km10 = int(np.where(bins == 0.1)[0])
                    logx, logy = x[:km100], y[:km100]
                    m0, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log10(logx), np.log10(logy))
                    a0 = 10**intercept
                    logx, logy = x[km100:km10], y[km100:km10]
                    m1, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log10(logx), np.log10(logy))
                    a1 = 10**intercept
                    logx, logy = x[km10:], y[km10:]
                    m2, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log10(logx), np.log10(logy))
                    a2 = 10**intercept
                    print(f'density: {pd}, |{pd-0.01:.2f} : {pd+0.01:.2f}|')
                    print(f'linear fit (100-1000 km): {m0}')
                    print(f'linear fit (10-100 km): {m1}')
                    print(f'linear fit (1-10 km): {m2}')
                    slopes[pd] = np.asarray([m0,m1,m2])
                k+=1

        elif dim =='pressure':
            colors = plt.cm.rainbow(np.linspace(0,1,len(lst)))
            k = 0
            slopes = {}
            for p in lst:
#                 signal = ds.sel(pressure = p, method = 'nearest')
                signal = ds.sel(pressure = slice(p-2, p+2)).mean(dim = 'pressure')
                signal = signal.interpolate_na(dim = 'distance',method="linear", fill_value="extrapolate", max_gap = 5).dropna(dim = 'distance')
                (f, S) = scipy.signal.periodogram(signal, fs, scaling='density')
                if whiten == True:
                    S = S*(4*np.pi**2*f**2)
                S_tot[p] = S[1:-1]
                plt.plot(f, S, alpha = 0.4, color = colors[k])
                
                if bins is not None:
                    bin_values, bin_edges = scipy.stats.binned_statistic(f, S, statistic='mean', bins=bins)[0:2]
                    x, y = bin_edges[:-1][~np.isnan(bin_values)], bin_values[~np.isnan(bin_values)]
                    plt.plot(x, y, color = colors[k], label='_nolegend_') 
                    plt.scatter(x, y, color = colors[k], label='_nolegend_')
                    
                    # power law fit 
                    km100 = int(np.where(bins == 0.01)[0])
                    km10 = int(np.where(bins == 0.1)[0])
                    logx, logy = x[:km100], y[:km100]
                    m0, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log10(logx), np.log10(logy))
                    a0 = 10**intercept
                    logx, logy = x[km100:km10], y[km100:km10]
                    m1, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log10(logx), np.log10(logy))
                    a1 = 10**intercept
                    logx, logy = x[km10:], y[km10:]
                    m2, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log10(logx), np.log10(logy))
                    a2 = 10**intercept
                    print(f'pressure: {p}, |{p-2:.2f} : {p+2:.2f}|')
                    print(f'linear fit (100-1000 km): {m0}')
                    print(f'linear fit (10-100 km): {m1}')
                    print(f'linear fit (1-10 km): {m2}')
                    slopes[p] = np.asarray([m0,m1,m2])
                k+=1

    x_range = np.arange(10**-3, 10, 0.1)

    y = (x_range**-1)
    if whiten == True:
        y = y*(4*np.pi**2*x_range**2)
    k1, = ax.plot(x_range, y*(scale/10), c = 'k', linestyle = 'dotted', alpha = 0.4, linewidth = 2.5)
    
    y = (x_range**-(5/3))
    if whiten == True:
        y = y*(4*np.pi**2*x_range**2)
    k5_3, = ax.plot(x_range, y*(scale/15), c = 'k', linestyle = '-.', alpha = 0.4, linewidth = 2.5)
    
    y = (x_range**-2)
    if whiten == True:
        y = y*(4*np.pi**2*x_range**2)
    k2, = ax.plot(x_range, y*(scale/10), c = 'k', linestyle = '--', alpha = 0.4, linewidth = 2.5)

    y = (x_range**-3)
    if whiten == True:
        y = y*(4*np.pi**2*x_range**2)
    k3, = ax.plot(x_range, y*(scale/100), c = 'k',  alpha = 0.4, linewidth = 2.5)


    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('wavenumber (cpkm)')
    plt.ylabel('PSD')
    if whiten == True:
        plt.ylabel('PSD * 4$\pi^{2}k^{2}$')

    ax.text(0.02, 1.02, label, transform=ax.transAxes)
    plt.title(f'EM-{floatid}')
    plt.grid(which="both", alpha = 0.4)
    ax.set_xlim(10**-3, 1)
    ax.set_ylim(ymin, ymax)
    
    if len(lst) > 4:
        cax = plt.axes([0.92, 0.15, 0.025, 0.7])
        cmap = 'rainbow'
        norm = mpl.colors.Normalize(vmin=lst[0], vmax=lst[-1])
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=dim)
    else:
        if legend == True:
            # coloured line legend (bbox = 1.1, 1.1)
            legend1 = plt.legend(lst, bbox_to_anchor = (1, 1), title=dim, loc = 'upper right', framealpha = 1, fontsize = 11)
            ax1 = plt.gca().add_artist(legend1)
    
    # if legend == True:
    #     # grey line legend (bbox = 1.11, 0.42)
    #     legend2 = plt.legend([k1, k5_3, k2, k3], ['$k^{-1}$', '$k^{-5/3}$', '$k^{-2}$', '$k^{-3}$'], bbox_to_anchor = (1.31, 0.3), 
    #                                                                 loc = 'lower right', 
    #                                                                 framealpha = 1, fontsize = 11.5)
    #     ax2 = plt.gca().add_artist(legend2)

    plt.show()

    if return_slopes == True:
        return ax, slopes

    return ax
    
# ------------------------------------------------------------------------------------------------------------------------------------------------------








