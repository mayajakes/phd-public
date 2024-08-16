
import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt

from scipy import optimize
import cv2
import scipy
import pandas as pd


def line(x, a, b):
    y = a*x + b
    return y


class FeatureDetection:
    '''Finds contours in an image/2D array based on a given threshold value.
       Data associated wit the contoured features can then be extracted to calculate statistics e.g. height, length, temporal persistence, slope etc. 
        '''

    def __init__(self, dataArray):
        '''Input: 2D xarray dataArray'''
        self.xr_data = dataArray
        self.np_array = dataArray.data

    def resize_image(self, scale_percent = 600):
        '''Resize the image using a scale percentage of the original image. Enlarging the image will make the features smoother (reduce noise)'''
        img_copy = self.np_array.copy()
        width = int(self.np_array.shape[1] * scale_percent / 100)
        height = int(self.np_array.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(img_copy, dim)
        return resized_image

    def find_contours(self, img = None, lower_threshold = 1.5, upper_threshold = 10):
        '''Use the OpenCV Python package to find contours based on a threshold value.
           Returns a list of all contours and their data points'''
        if img is None:
            img = self.np_array

        mask = cv2.inRange(img, lower_threshold, upper_threshold)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        return contours
    
    def draw_contours(self, contours, fillvalue = 255, index = -1, filled = True, 
                      plot = True, resized = False, **kwargs):
        '''Draw contours on the image to check how well the features are detected.
           index = -1 means draw all contours. '''

        if filled: weight = -1 
        else: weight = 1

        if resized == True: img_copy = self.resize_image()
        else: img_copy = self.np_array.copy()

        cont_image = cv2.drawContours(img_copy, contours, index, fillvalue, thickness=weight, lineType=cv2.LINE_AA)

        if plot == True:
            fig, ax = plt.subplots(figsize = (10,5))
            plt.imshow(cont_image, **kwargs)

        return cont_image

    def remove_duplicates(self, contours):
        '''Open CV outputs duplicate contour coordinates. This function removes the duplicates from each chain.'''
        main_list = []
        for c in contours:
            coord_list = []
            chain = c.tolist()
            for i in range(0, len(chain)):
                coord_list.append(chain[i][0])
            
            removes_dupes = list(set(map(tuple, coord_list)))
            main_list.append(np.asarray(list(map(list, removes_dupes))))
        
        return main_list

    # def get_chains(self, contours, min_len = 6):
    #     '''Filter chains/contours to extract only those with more than a certain number of measurements given by min_len'''
    #     def filter_func(x):
    #         return len(x) >= min_len
    #     return list(filter(filter_func, self.remove_duplicates(contours)))
    

    def filter_chains(self, contours, min_len = 6):
        '''remove duplicate data points from each contour and return the indices for the contours that have 
        more than certain number of data points given by min_len'''
        no_duplicates = self.remove_duplicates(contours)

        num_points = []
        for c in no_duplicates:
            num_points.append(len(c))
        return np.where(np.asarray(num_points) >= min_len)[0]
    

    def get_chain_values(self, contours, fillvalue = 255, filled = True):
        '''Extract the data stored in the xarray DataArray associated with the contoured features.'''
        lst_values = []

        # For each list of contour points...
        for i in range(len(contours)):
            # Create a mask image that contains the contour filled in
            # zero_img = np.zeros_like(self.np_array)
            cont_image = self.draw_contours(contours, index = i, fillvalue = fillvalue,
                                            filled = filled, plot = False)
            
            # Access the image pixels and create a 1D numpy array then add to list
            z_ind, prof = np.where(cont_image == fillvalue)
            
            values = []
            dens = []
            profile = []
            for j in range(len(z_ind)):
                # extract data value and coordinates (profile and potential_density)
                values.append(self.xr_data[z_ind[j], prof[j]])
                profile.append(self.xr_data.profile[prof[j]])
                dens.append(self.xr_data.potential_density[z_ind[j]])
                
            lst_values.append(values)

        return lst_values
    

    def get_chain_lengths(self, chain_values, distance, dist_threshold = None):
        '''Find the length of the chains/contours in km. 
           distance = input 1D array of distance in km associated with each profile in the dataArray.'''
        length = []

        for i in range(len(chain_values)):
            dist = []
            for val in chain_values[i]:
                dist.append(distance[val.profile].data)

            L = np.nanmax(dist) - np.nanmin(dist)
            if dist_threshold is not None:
                if L >= dist_threshold: length.append(L)
                else: length.append(np.nan)
            else:
                length.append(L)

        return np.asarray(length)


    def get_chain_slope(self, chain_values, pressure_on_dens, dz = 'density', plot_slope = False):
        '''Find the slope of the chains/contours relative to isopycnals or isobars in the cross-stream direction.
        Use a linear least squares fit through the points of each chain to calculate the slope.'''
        
        chain_df = self.create_dataframe(chain_values, pressure_on_dens = pressure_on_dens)
        slope_stats = []

        # loop through chains
        for idx in range(len(chain_values)):
            c = chain_df[chain_df['chain'] == idx]

            x = c.distance

            if dz =='pressure':
                y = c.pressure
            elif dz =='density':
                y = c.potential_density

            gradient, intercept = optimize.curve_fit(line, xdata = x, ydata = y)[0]
            slope = gradient/1000 #km to m

            slope_stats.append([idx, slope])
            
            df = pd.DataFrame(slope_stats, columns=['chain', 'slope'])

            if plot_slope == True:
                y_fit = gradient*x + intercept

                fig, ax = plt.subplots(figsize = (5,3.5))
                plt.scatter(x, y)
                plt.plot(x, y_fit, 'r')
                ax.invert_yaxis()
    
                plt.title(f'chain index = {idx}')
                plt.xlabel('distance (km)')
                if dz == 'pressure':
                    ax.text(0.5, 0.02, f'dz/dx = {slope:.2}', transform = ax.transAxes, c = 'r')
                    ax.set_ylabel('pressure (dbar)')
                if dz == 'density':
                    ax.text(0.5, 0.02, r'$d\rho/dx$' + f' = {slope:.2}', transform = ax.transAxes, c = 'r')
                    ax.set_ylabel(r'$\rho_{\theta}$ ($kg$ $m^{-3})$')

        return df


    def get_chain_slope_stats(self, chain_values, pressure_on_dens, dz = 'density', plot_slope = False):
            '''Find the slope of the chains/contours relative to isopycnals or isobars in the cross-stream direction.
            Use a linear least squares fit through the points of each chain to calculate the slope.'''
            
            chain_df = self.create_dataframe(chain_values, pressure_on_dens = pressure_on_dens)
            slope_stats = []

            # loop through chains
            for idx in range(len(chain_values)):
                c = chain_df[chain_df['chain'] == idx]

                profiles = np.sort(c['profile'].unique())
                dist = []
                dens = []
                pres = []

                #extract distance and average denisty (if more than one point in a profile)
                for prof in profiles:
                    df_sel = c[c['profile'] == prof]

                    dist.append(df_sel['distance'].values[0]*1000)
                    dens.append(np.median(np.asarray(df_sel.potential_density.values)))

                    if pressure_on_dens is not None:
                        pres.append(np.median(np.asarray(df_sel.pressure.values)))

                if dz == 'density':
                    # slope = np.gradient(dens)/np.gradient(dist)
                    slope = np.diff(dens)/np.diff(dist)
                elif dz == 'pressure':
                    # z = gsw.z_from_p(pres, lat)
                    # slope = np.gradient(pres)/np.gradient(dist)
                    slope = np.diff(pres)/np.diff(dist)

                infinity = np.where(np.isinf(slope))[0]
                slope[infinity] = np.nan

                if plot_slope == True: 
                    x = c.distance
                    if dz =='pressure':
                        y = c.pressure
                    elif dz =='density':
                        y = c.potential_density
                    fig, ax = plt.subplots(figsize = (4,3))
                    plt.scatter(x, y)
                    plt.scatter(np.asarray(dist)/1000, pres, c = 'r')
                    ax.invert_yaxis()
                    plt.xlabel('distance (km)')
                    plt.ylabel(dz)
                    plt.title(f'chain idx: {idx}')

                    fig, ax = plt.subplots(figsize = (4,3))
                    plt.plot(np.asarray(dist)[:-1]/1000, slope, marker='o')
                    plt.axhline(y = 0, c = 'grey', linestyle = '--')
                    plt.axhspan(ymin = -0.01, ymax = 0.01, color = 'grey', alpha = 0.1)
                    plt.xlabel('distance (km)')
                    plt.ylabel('dz/dx')
                    plt.title(f'chain idx: {idx}')

                mean_abs_slope = np.nanmean(abs(slope))
                mean_slope = np.nanmean(slope)
                max_abs_slope = np.nanmax(abs(slope))
                std_slope = np.nanstd(slope)

                pc_0_01 = (len(np.where(abs(slope)>0.01)[0])/len(slope))*100

                if mean_slope < 0:
                    max_slope = max_abs_slope*-1
                else:
                    max_slope = max_abs_slope

                slope_stats.append([idx, mean_slope, max_slope, mean_abs_slope, max_abs_slope, std_slope, pc_0_01])
                
                df = pd.DataFrame(slope_stats, columns=['chain', 'mean_slope', 'max_slope', 'mean_abs_slope', 'max_abs_slope', 'slope_std', 'pc_0.01'])

            return df

    
    def create_dataframe(self, chain_values, pressure_on_dens = None, speed_on_dens = None):
        
        if len(chain_values)> 0:
            lst = []
            for i in range(len(chain_values)):
                for val in chain_values[i]:
                    if pressure_on_dens is not None:
                        pres = pressure_on_dens.sel(potential_density = val.potential_density, method = 'nearest')[val.profile]
                        lst.append([int(val.profile.data), pres.distance.data, pres.data, val.potential_density.data, i])
                        cols = ['profile', 'distance', 'pressure', 'potential_density', 'chain']
                    elif speed_on_dens is not None:
                        speed = speed_on_dens.sel(potential_density = val.potential_density, method = 'nearest')[val.profile]
                        lst.append([int(val.profile.data), speed.distance.data, speed.data, val.potential_density.data, i])
                        cols = ['profile', 'distance', 'speed', 'potential_density', 'chain']
                    else:
                        lst.append([int(val.profile.data), val.potential_density.data, i])
                        cols = ['profile', 'potential_density', 'chain']

            chain_df = pd.DataFrame(lst, columns=cols)
            return chain_df


    def get_chain_heights(self, chain_values, pressure_on_dens):
        '''Find the height of the chains in m'''
        chain_df = self.create_dataframe(chain_values, pressure_on_dens = pressure_on_dens)
        pd_interval = np.gradient(pressure_on_dens.potential_density)[0]

        chain_heights = []
        for i in range(len(chain_values)):
            c = chain_df[chain_df['chain'] == i]

            # loop through unqiue profile numbers
            profiles = np.sort(c['profile'].unique())

            for prof in profiles:
                df_sel = c[c['profile'] == prof]
                
                if len(df_sel) > 1:
                    # find the difference between maximum and minimum pressure values
                    p_min = np.nanmin(df_sel.pressure)
                    p_max = np.nanmax(df_sel.pressure)
                    H = p_max - p_min
                    # # TO DO: deal with features with multiple branches in one profile
                    # pd_grad = np.gradient(df_sel.potential_density)
                    # ind = np.where(abs(pd_grad) > pd_interval)[0]

                elif len(df_sel) == 1: 
                    # half of the pressure interval assoaicted with the local potential density
                    drho = np.diff(pressure_on_dens.potential_density)[0]
                    pdens_sel = float(df_sel.potential_density)
                    pres = pressure_on_dens[prof].sel(potential_density = slice(pdens_sel - drho, pdens_sel))
                    
                    if len(pres) < 2:
                        pres = pressure_on_dens[prof].sel(potential_density = slice(pdens_sel - drho*2, pdens_sel))
                    
                    H = np.diff(pres)[0]/2

                chain_heights.append([prof, H, i])
            
        chain_heights = pd.DataFrame(chain_heights, columns=['profile', 'height', 'chain'])

        return chain_heights
    
    def get_chain_current_speed(self, chain_values, speed_dens):
        chain_df = self.create_dataframe(chain_values, speed_on_dens = speed_dens)

        chain_speed = []
        for i in range(len(chain_values)):
            c = chain_df[chain_df['chain'] == i]
            chain_speed.append([c.speed.mean(skipna = True), c.speed.median(skipna = True), c.speed.std(skipna = True), i])
        
        chain_speed = pd.DataFrame(chain_speed, columns=['mean_speed', 'median_speed', 'std_speed', 'chain'])

        return chain_speed


    def time_change(self, chain_values, time_gps):
        '''Calculate the persistence in time of the features detected along the float tracks.'''

        if len(chain_values)> 0:
            time_diff = []
            for c in range(len(chain_values)):
                t = []
                for val in chain_values[c]:
                    t.append(time_gps[val.profile].data)
                
                # remove nats
                nats = np.where(np.isnat(t))[0]
                t = np.asarray(t).astype(float)
                t[nats] = np.nan
                t = t.astype('datetime64[ns]')
                
                dt = (np.nanmax(t) - np.nanmin(t)).astype('timedelta64[h]')

                # apply distance threshold?
                time_diff.append(dt)

            return np.asarray(time_diff)
        
    
    def print_summary_stats(self, chain_values, chain_length, chain_height, chain_time = None, chain_slope = None):
        
        print('printing overall statistics for features detected')
        print(f'no. features: {len(chain_values)}')

        cols = ['chain_idx', 'length', 'mean_height', 'mean_value']
        if chain_time is not None:
            cols = ['chain_idx', 'length', 'mean_height', 'mean_value', 'time']
        
        # Length
        print('Feature length')
        mean_L = np.nanmean(chain_length)
        print(f'mean L (km): {mean_L:.1f} +/- {scipy.stats.sem(chain_length):.1f}')

        # create a dataframe with all chains
        data = []
        mean_H = []
        mean_dsc = []
        for c in range(len(chain_values)):
            mean_value = np.nanmean(chain_values[c])
            mean_dsc.append(mean_value)

            heights1 = chain_height[chain_height['chain']==c]
            H = np.nanmean(heights1['height'])
            mean_H.append(H)

            if chain_time is not None:
                data.append([c, chain_length[c], H, mean_value, chain_time[c].astype('float')])
            else:
                data.append([c, chain_length[c], H, mean_value])

        
        print('DSC value')
        std_err = scipy.stats.sem(mean_dsc, nan_policy = 'omit')
        print(f'mean value: {np.nanmean(mean_dsc):.2f} +/- {std_err:.2f}')

        # Feature heights
        print('Height of features')
        std_err = scipy.stats.sem(mean_H, nan_policy = 'omit')
        print(f'mean H maxima: {np.nanmean(mean_H):.2f} +/- {std_err:.2f}')

        if chain_time is not None:
            # temporal persistence
            print('Time change')
            nats = np.where(np.isnat(chain_time))[0]
            dt_max = chain_time.astype(float)
            dt_max[nats] = np.nan

            std_err = scipy.stats.sem(dt_max, nan_policy = 'omit')
            print(f'mean dt: {np.nanmean(dt_max):.2f} +/- {std_err:.2f}')

        if chain_slope is not None:
            print('slope (drho/dx)')
            std_err = scipy.stats.sem(chain_slope['slope'], nan_policy = 'omit')
            mean = chain_slope['slope'].mean()
            max = chain_slope['slope'].max()
            min = chain_slope['slope'].min()
            print(f'mean avg slope: {mean:.2} +/- {std_err:.2}')
            print(f'max avg slope = {max}')
            print(f'min avg slope = {min}')

        df = pd.DataFrame(data, columns=cols)
        
        return df
