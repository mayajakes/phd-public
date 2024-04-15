
'''Maps of surface features'''

import matplotlib.pyplot as plt 
import numpy as np
import os
import cmocean
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from shapely.geometry.polygon import LinearRing

import imp
import src.importData as imports
import src.settings as settings
import src.stats as stats
import src.calc as calc

def plotFloatTrajectory(float_num, floatid, altimetry, save_fig = False, my_path = None):
    rs = calc.findRSperiod(float_num)

    start = float_num.time[rs].values[0]
    end = float_num.time[rs].values[-1]
    start_time = str(start.astype('M8[D]'))
    end_time = str(end.astype('M8[D]'))

    mean_sea_level = altimetry.adt.sel(time = slice(start_time, end_time)).mean(dim = 'time')

    # Gridded SSH added in the background
    lon =slice(145,175)
    lat = slice(-60,-51)

    plt.rcParams['font.size'] = '14'

    fig = plt.figure(figsize=(10,4))

    mean_sea_level.sel(longitude = lon,latitude = lat).plot(alpha = 0.5, cmap = 'viridis', vmin = -1, vmax = 0.5,
                                                        cbar_kwargs=dict(label='Sea Level (m)'))

    # Plot Polar Front SSH contours - according to Sokolov & Rintoul, 2009
    PF = np.arange(1,1.3,0.1) # IMOS
    PF = np.arange(-0.5,0,0.1) # CMEMS
    CS = mean_sea_level.plot.contour(colors = 'gray', alpha = 0.7, linewidths = 1.8, linestyles = 'solid', levels = PF)
    plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')

    # Float trajectory map with points coloured in acccording to time slice
    plt.scatter(float_num.longitude, float_num.latitude, s = 20, c= 'slategrey', zorder = 2)
    plt.scatter(float_num.longitude[rs], float_num.latitude[rs], s = 20, c='w', zorder = 2)

    plt.title('Float {}'.format(floatid))
    
    if save_fig == True:
        name = 'float-%s-trajectory.png' %floatid
        settings.save_figure(fig, my_path, name)
        
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def plotMapWithDist(float_num, floatid, altimetry, dotsize = 20, panel = False, dist_list = None, save_fig = False,  my_path = None):
    plt.rcParams['font.size'] = '12'

    rs = calc.findRSperiod(float_num)
    dist = calc.cum_dist(float_num.longitude, float_num.latitude)
    
    lst = []
    #distance every 100km
    for i in range(100,1400,100):
        value = stats.find_nearest(dist[rs], i)[0]
        lst.append(value)

    start = float_num.time[rs].values[0]
    end = float_num.time[rs].values[-1] #mean altimetry during rs period

    start_time, end_time = str(start.astype('M8[D]')), str(end.astype('M8[D]'))

    lon =slice(148,160) # 160
    lat = slice(-57,-51) # -57

    # SAF, PF, SACCF = settings.frontSSH(reference = ('KimOrsi'))
    levels = np.arange(-0.7,0.4,0.1)
     
    # colors for distance markers every 100 km
    colors = ['#FFE0E0','#FFB2B2', '#F28A8A', '#E85555', '#D84747', '#C93636', '#963333', '#7F1A1A', '#691515', '#5B1313', '#440000', '#2E0000', '#040000']

    if panel == True:
        fig, axs = plt.subplots(nrows = 2, ncols=3, sharey = True, sharex = True, figsize = (16,8))
        axs = axs.flatten()

        i = 0
        for d in dist_list:
            interval = dist_list[1]-dist_list[0]
            # smaller time period around 200 km
            i_1 = stats.find_nearest(dist, d)[0]
            i_2 = stats.find_nearest(dist, d+interval)[0]
            start = float_num.time[i_1].values
            end = float_num.time[i_2].values

            start_time, end_time = str(start.astype('M8[D]')), str(end.astype('M8[D]'))

            mean_sea_level = altimetry.adt.sel(time = slice(start_time, end_time)).mean(dim = 'time')

            im = mean_sea_level.sel(longitude = lon, latitude = lat).plot(ax = axs[i], alpha = 0.4,  cmap = 'viridis', vmin = -1, vmax = 0.5,
                                                        add_colorbar = False)

            CS = mean_sea_level.plot.contour(ax = axs[i], colors = 'grey', alpha = 0.7, linewidths = 1, levels = levels)
            plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')

            axs[i].scatter(float_num.longitude, float_num.latitude, c='grey', s= dotsize,  zorder = 2)
            axs[i].scatter(float_num.longitude[rs], float_num.latitude[rs], c='w', s= dotsize, zorder = 2)
            axs[i].scatter(float_num.longitude[i_1:i_2], float_num.latitude[i_1:i_2], c = 'darkgrey', s= dotsize-4, linewidths = 0.8, edgecolors = 'k', zorder = 2)

            # sequentially colour point every 100 km distance
            if dist[rs][-1] < 900:
                axs[i].scatter(float_num.longitude[lst[0:8]], float_num.latitude[lst[0:8]], c = colors[0:8], s= dotsize-2, zorder = 3)
            elif 900 < dist[rs][-1] < 1000:
                axs[i].scatter(float_num.longitude[lst[0:9]], float_num.latitude[lst[0:9]], c = colors[0:9], s= dotsize-2, zorder = 3)
            else:
                axs[i].scatter(float_num.longitude[lst], float_num.latitude[lst], c = colors, s= dotsize-2, zorder = 3)

            axs[i].set_title('{}-{} km'.format(d, d+interval))
            axs[i].set_xlabel('')
            axs[i].set_ylabel('')
            axs[i].text(0.03, 0.03, f'{start_time}', transform = axs[i].transAxes)

            if i == 5:
                print('float {} end of rs: {}'.format(floatid, str(float_num.time[rs][-1].values.astype('M8[D]'))))
            i += 1
        
        plt.tight_layout(w_pad = 0.6)
        
        cax = plt.axes([0.99, 0.15, 0.015, 0.7])
        cbar = plt.colorbar(im, cax=cax, extend = 'both')
        cbar.set_label('ADT (m)')


    else:
        # calculate mean ssh between first and last time index for the week
        mean_sea_level = altimetry.adt.sel(time = slice(start_time, end_time)).mean(dim = 'time')
        print('Float {}: start of rs: {}, end of rs: {}'.format(floatid, start_time, end_time))

        fig = plt.figure(figsize=(6,4))

        mean_sea_level.sel(longitude = lon, latitude = lat).plot(alpha = 0.4,  cmap = 'viridis', vmin = -1, vmax = 0.5,
                                                            cbar_kwargs=dict(label='Sea Level (m)'))


        CS = mean_sea_level.plot.contour(colors = 'gray', alpha = 0.7, linewidths = 1, levels = levels)
        plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')

        plt.scatter(float_num.longitude, float_num.latitude, c='grey', s = 20, zorder = 2)
        plt.scatter(float_num.longitude[rs], float_num.latitude[rs], c='w', s = 20, zorder = 2)

        # sequentially colour point every 100 km distance
        if dist[rs][-1] < 900:
            plt.scatter(float_num.longitude[lst[0:8]], float_num.latitude[lst[0:8]], c = colors[0:8], s = 20, zorder = 3)
        elif 900 < dist[rs][-1] < 1000:
            plt.scatter(float_num.longitude[lst[0:9]], float_num.latitude[lst[0:9]], c = colors[0:9], s = 20, zorder = 3)
        else:
            plt.scatter(float_num.longitude[lst], float_num.latitude[lst], c = colors, s = 20, zorder = 3)
        

        plt.title('Float {}'.format(floatid))
    
    if save_fig == True:
        name = 'float-%s-dist-points.png' %floatid
        settings.save_figure(fig, my_path, name)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def floatDomain(floatids, bathymetry, altimetry, extent = [147,173,-60.5,-49.5], fig_size = (14, 8), pal = sns.color_palette("husl", 4), save_fig = False):
    ema = imports.importFloatData(floatids)

    # Plot domain map 
    fig = plt.figure(figsize = fig_size)

    ax = plt.axes()
    lonmin, lonmax, latmin, latmax = extent
    ax.set_xlim(lonmin,lonmax)
    ax.set_ylim(latmin,latmax)

    # Add bathymetry
    bathymetry['depth'] = bathymetry.elevation*-1
    longitude = slice(lonmin,lonmax)
    latitude = slice(latmin,latmax)

    # coarsen the data 
    new = bathymetry.coarsen(lon=10).mean().coarsen(lat=10).mean()

    new.depth.sel(lon = longitude, lat = latitude).plot(alpha = 0.5, cmap=cmocean.cm.deep, vmin=500, vmax=6000, cbar_kwargs=dict(label='depth (m)'))

    # Add SSH contours averaged during the first float deployments (8489 to 8493)
    rs = calc.findRSperiod(ema[8493])
    start = ema[8489].time.values[0]
    end = ema[8493].time[rs].values[-1] 
    start_time = str(start.astype('M8[D]'))
    end_time = str(end.astype('M8[D]'))
    mean_sea_level = altimetry.adt.sel(time = slice(start_time, end_time)).mean(dim = 'time')

    # SSH contours 
    levels = np.arange(-0.8,0.3,0.1)
    CS = mean_sea_level.plot.contour(colors = 'white', linewidths = 1.2, alpha = 0.9, levels = levels)
    plt.clabel(CS, inline=True, fontsize=12, fmt = '%1.1f')

    # Draw float trajectories
    cols = pal.as_hex()[:]
    legend = []
    i = 0
    for floatid in floatids:
        rs = calc.findRSperiod(ema[floatid])
        ax.plot(ema[floatid].longitude[rs], ema[floatid].latitude[rs],'.-', linewidth = 2.8, markersize=10, c = cols[i], markeredgecolor='k', markeredgewidth= 0.25, alpha = 0.7, zorder = 5)
        legend.append(f'EM-{floatid}')
        i+=1

    ax.legend(legend, loc = 'lower left', prop={'size': 12})

    # i = 0
    # for floatid in floatids:
    #     ax.plot(ema[floatid].longitude, ema[floatid].latitude,'-', linewidth = 2, c = cols[i], zorder = 4)
    #     ax.plot(ema[floatid].longitude, ema[floatid].latitude,'-', linewidth = 3.5, alpha = 0.4, c = 'k', zorder = 3)
    #     i+=1

    plt.ylabel(u'Latitude (\N{DEGREE SIGN}N)')
    plt.xlabel(u'Longitude (\N{DEGREE SIGN}E)')

    # Inset location relative to main plot (ax) in normalized units
    inset_x = 0.9 #0.897
    inset_y = 0.87 #0.849
    inset_size = 0.3

    # Plot inset map
    ax2 = plt.axes([0, 0, 1, 1], projection=ccrs.Orthographic(central_latitude=(latmin + latmax) / 2, 
                                                            central_longitude=(lonmin + lonmax) / 2))
    ax2.set_extent([105, 200, -85, -10])
    ax2.add_feature(cfeature.LAND)
    ax2.add_feature(cfeature.COASTLINE)
    gl = ax2.gridlines(crs=ccrs.PlateCarree(),linewidth=2, color='gray', alpha=0.5, linestyle='--', dms = True)
    gl.xlabel_style = {'size': 16, 'color': 'gray'}
    gl.ylabel_style = {'size': 16, 'color': 'gray'}

    ip = InsetPosition(ax, [inset_x - inset_size / 2,
                            inset_y - inset_size / 2,
                            inset_size,
                            inset_size])
    ax2.set_axes_locator(ip)


    # Draw red box to show domain 
    nvert = 100
    lons = np.r_[np.linspace(lonmin, lonmin, nvert),
                np.linspace(lonmin, lonmax, nvert),
                np.linspace(lonmax, lonmax, nvert)].tolist()
    lats = np.r_[np.linspace(latmin, latmax, nvert),
                np.linspace(latmax, latmax, nvert),
                np.linspace(latmax, latmin, nvert)].tolist()

    ring = LinearRing(list(zip(lons, lats)))
    ax2.add_geometries([ring], ccrs.PlateCarree(),
                    facecolor='none', edgecolor='red', linewidth=0.75)

    
    if save_fig == True:
        my_path = os.path.join(os.sep, 'Users', 'mijakes', 'checkouts', 'phd','figures')
        settings.save_figure(fig, my_path, 'domain_map.png', dpi = 300, pad = 0.2)

    return fig, ax

# -----------------------------------------------------------------------------------------------------------------------------------------------

