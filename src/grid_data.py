import scipy 
import numpy as np 
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

def bin_data(data, bins = [np.arange(-180, 185, 1), np.arange(-70, -35, 1)], min_obs = 5):
    lon_bins, lat_bins = bins
    
    lon_mid = (lon_bins + np.gradient(lon_bins)/2)[:-1]
    lat_mid = (lat_bins + np.gradient(lat_bins)/2)[:-1]
    
    #remove nans
    data_nonan = data.where(~np.isnan(data), drop = True)
    
    # grid using scipy
    data_mean = scipy.stats.binned_statistic_2d(data_nonan.lon, data_nonan.lat, data_nonan, 
                                bins = [lon_bins, lat_bins], statistic='mean')[0]
    
    data_std = scipy.stats.binned_statistic_2d(data_nonan.lon, data_nonan.lat, data_nonan, 
                                bins = [lon_bins, lat_bins], statistic='std')[0]
    
    n_obs = scipy.stats.binned_statistic_2d(data_nonan.lon, data_nonan.lat, data_nonan, 
                                bins = [lon_bins, lat_bins], statistic='count')[0]
    
    # must have at least 5 observations per bin
    mask = np.ma.masked_where(n_obs < min_obs, n_obs)
    
    # to xarray
    data_mean = xr.DataArray(data_mean, dims = ['lon', 'lat'], coords = dict(lat = (['lat'], lat_mid), 
                                                                             lon = (['lon'], lon_mid)))
    
    data_std = xr.DataArray(data_std, dims = ['lon', 'lat'], coords = dict(lat = (['lat'], lat_mid), 
                                                                             lon = (['lon'], lon_mid)))
    
    n_obs = xr.DataArray(n_obs, dims = ['lon', 'lat'], coords = dict(lat = (['lat'], lat_mid), 
                                                                     lon = (['lon'], lon_mid)))
    
    data_mean.data[mask.mask] = np.nan
    data_std.data[mask.mask] = np.nan
    n_obs.data[mask.mask] = np.nan
    
    return data_mean, n_obs


def mask_PFZ(griddata, MDT, limits = [-0.3, 0], pad = 0.1):
    PF, nSAF = limits ## ssh contours
    
    # mask the PFZ for averaging
    mdt_interp = MDT.mdt[0].interp(latitude = griddata.lat, longitude = griddata.lon)
    mask1 = np.ma.masked_where((mdt_interp < PF-pad), mdt_interp)
    mask2 = np.ma.masked_where((mdt_interp > nSAF+pad), mdt_interp)

    grid_data_copy = griddata.transpose().copy()
    grid_data_copy.data[mask1.mask] = np.nan
    grid_data_copy.data[mask2.mask] = np.nan
    
    return grid_data_copy


def plot_circumpolar(grid_data, bathy, MDT, extent = [-180, 180, -90, -36], cbar_label = '', mask = False, **kwargs):
    
    PF = -0.3
    nSAF = 0
    
    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor= None, facecolor='k')

    ax.add_feature(land)
    ax.set_extent(extent, ccrs.PlateCarree())

    gls = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.6,
                    color='gray', alpha=0.6, zorder=10, xlocs=range(-180,180,30))

    gls.xlabel_style = {'size': 10, 'color': 'k'}
    gls.ylabel_style = {'size': 0, 'color': 'k'}

    r_extent = 6521311
    r_extent *= 1.005  

    # set the plot limits
    ax.set_xlim(-r_extent, r_extent)
    ax.set_ylim(-r_extent, r_extent)

    # Prep circular boundary
    circle_path = mpath.Path.unit_circle()
    circle_path = mpath.Path(circle_path.vertices.copy() * r_extent,
                               circle_path.codes.copy())

    #set circular boundary
    ax.set_boundary(circle_path)

    # draw longitude labels
    plt.draw() 

    # Reposition the meridian tick labels
    for ea in gls._labels:
        pos = ea[2].get_position()
        if (pos[0]==150):
            ea[2].set_position([180, pos[1]])

    #plot DSC
    if mask == True:
        grid_data_copy = mask_PFZ(grid_data, MDT)
        grid_data.transpose().plot(ax = ax, transform = ccrs.PlateCarree(), alpha = 0.5, add_colorbar = False, **kwargs)

        im = grid_data_copy.plot(ax = ax, transform = ccrs.PlateCarree(),
                          cbar_kwargs = dict(label = cbar_label), **kwargs)
        
    else:
        im = grid_data.transpose().plot(ax = ax, transform = ccrs.PlateCarree(),
                                  cbar_kwargs = dict(label = cbar_label), **kwargs)

    #bathymetry
    bathy.depth.plot.contour(ax = ax, levels = [1500], colors = 'silver', linewidths = 0.8, transform = ccrs.PlateCarree())

    # plot ACC fronts
    MDT.mdt[0].plot.contour(ax = ax, levels = [PF], colors = 'k', linestyles = '-', linewidths = 1.2, alpha = 0.8,
                            transform = ccrs.PlateCarree())
    MDT.mdt[0].plot.contour(ax = ax, levels = [nSAF], colors = 'k', linestyles = '--', linewidths = 1.2, alpha = 0.8,
                            transform = ccrs.PlateCarree())

    plt.tight_layout()
    plt.title('')
    
    return fig, ax