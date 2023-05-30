import datetime
import xarray as xr
import numpy as np
import gsw 
import matplotlib.pyplot as plt
import cmocean
import matplotlib

import imp 
import src.calc as calc
imp.reload(calc)

import src.importData as imports
imp.reload(imports)

class Float:
    
    def __init__(self, float_num):
        self.float_num = float_num 
        
    def getIndexList(self, week_number):
        self.calcWeeknum()
        week_list = list(self.float_num.week.values)
        return [i for i in range(len(week_list)) if week_list[i] == week_number]
        
    def calcWeeknum(self):

        t1 = self.float_num.isel(time = 0).time
        week_num = []

        for i in range(0,len(self.float_num.time)):
            t2 = self.float_num.isel(time = i).time
            diff = t2 - t1
            days = diff.values.astype('timedelta64[D]')
            days /= np.timedelta64(1, 'D')

            if days >= 0:
                week_num.append(int(days // 7))
            else:
                week_num.append(None)

        week_num = xr.DataArray(week_num, dims = 'profile')
        self.float_num['week'] = week_num
        
        return self.float_num['week']
        

    def calcTempAnomalies(self):
        '''Calculate temperature anomalies for desired week. Mean temperature is taken during the rapid sampling stage'''
        
        rapid_sample_mean = self.float_num.T.sel(profile = slice(0,260)).mean(dim = 'profile', skipna = True)
        
        distance = calc.calcDistFromStart(self.float_num)

        T_anom = self.float_num.T - rapid_sample_mean

        anomalies = xr.Dataset(data_vars = dict(T_anomaly = (["distance", "pressure"], T_anom),), 
                                    coords = dict(distance=("distance", distance), 
                                            pressure = ("pressure",self.float_num.pressure),
                                           profile = ("profile", self.float_num.profile)),)

        return anomalies   
    


    def plotTempAndSalinity(self, floatid, week_number):
        
        ind_list = self.getIndexList(week_number)
        distance = calc.calcDistFromStart(self.float_num)
            
        # created a new dataset with the temperature and salinity data with distance as the x dimension (not profile)
        newFloat = xr.Dataset({"T": (("distance", "pressure"), self.float_num.T), 
                               "S": (("distance", "pressure"), self.float_num.S),
                               "pressure" : (("pressure"), self.float_num.pressure),
                              "distance" : (("distance"), distance)})

        
        # Plot temperature and salinity sections
        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True, figsize = (12,8))

#         Temperature
        newFloat.T[ind_list].plot(ax = ax1, x='distance', y='pressure',
                                   cbar_kwargs={'label':'°C'}, 
                                   cmap = cmocean.cm.thermal)
        
#         self.float_num.T[ind_list].plot(ax = ax1, y='pressure',
#                            cbar_kwargs={'label':'°C'}, 
#                            cmap = cmocean.cm.thermal)
        
        ax1.invert_yaxis()
        ax1.set_ylabel('pressure (dbar)')
        ax1.set_xlabel('')
        ax1.set_title('Float {} - week {}'.format(floatid, week_number))

#         Salinity
        newFloat.S[ind_list].plot(ax = ax2, x='distance', y='pressure',
                                   cbar_kwargs={'label':'PSU'}, 
                                   cmap = cmocean.cm.haline)
        
#         self.float_num.S[ind_list].plot(ax = ax2, y='pressure',
#                            cbar_kwargs={'label':'PSU'}, 
#                            cmap = cmocean.cm.haline)
        ax2.invert_yaxis()
        ax2.set_ylabel('pressure (dbar)')
        ax2.set_xlabel('distance (km)')
        
    def plotFloatByWeek(self, floatid, week_number):
        '''Plot map of float trajectory highlighting the week of interest, 
        with average SSH for that week in the background. 
        Plots temperature and salinity sections for the profiles in the given week'''
        
        ind_list = self.getIndexList(week_number)
        
        start = self.float_num.time.values[ind_list][0]
        end = self.float_num.time.values[ind_list][-1]

        start_time = str(start.astype('M8[D]'))
        end_time = str(end.astype('M8[D]'))

        altimetry = imports.importSatelliteData('altimetry')

        # calculate mean ssh between first and last time index for the week
        mean_sea_level = altimetry.GSL.sel(TIME = slice(start_time, end_time)).mean(dim = 'TIME')

        # Gridded SSH added in the background
        lon =slice(145,175)
        lat = slice(-60,-51)
        levels = np.arange(-1,1,0.2)

        fig = plt.figure(figsize=(12,6))

        mean_sea_level.sel(LONGITUDE = lon,LATITUDE = lat).plot(alpha = 0.6, 
                                                            cbar_kwargs=dict(label='Sea Level (m)'))
        
        # Plot Polar Front SSH contours - according to Sokolov & Rintoul, 2009
        PF = np.arange(1,1.3,0.1)
        CS = mean_sea_level.plot.contour(colors = 'snow', linewidths = 1, levels = PF)
        plt.clabel(CS, inline=True, fontsize=10, fmt = '%1.1f')

        # Float trajectory map with points coloured in acccording to time slice
        plt.scatter(self.float_num.longitude, self.float_num.latitude)
        plt.scatter(self.float_num.longitude[ind_list], self.float_num.latitude[ind_list],c='w')
        
        plt.title('Float {} - week {}'.format(floatid, week_number))
        print('Sea Surface Height is an avergae snap shot of the week')
    
    
    def plotAnomalySection(self, floatid, week_number):
        
        ind_list = self.getIndexList(week_number)
        anomalies = self.calcTempAnomalies()
        
        fig, ax = plt.subplots(figsize = (12,4))

        anomalies.T_anomaly[ind_list].plot(x = 'distance', y='pressure',robust = True,
                                  cbar_kwargs={'label':'°C'}, vmin = -1.5, vmax = 1.5,
                                  cmap = cmocean.cm.balance)

        plt.gca().invert_yaxis()
        ax.set_ylabel('pressure (dbar)')
        ax.set_xlabel('distance (km)')
        plt.title('Float {} - week {}'.format(floatid, week_number))
        
        
    def plotEKESection(self, floatid, week_number):
        
        ind_list = self.getIndexList(week_number)
        self.float_num['EKE'] = calc.calcEKE(self.float_num.u, self.float_num.v)
        distance = calc.calcDistFromStart(self.float_num)
        
        newFloat = xr.Dataset({"EKE": (("distance", "pressure"), self.float_num.EKE), 
                       "pressure": (( "pressure"), self.float_num.pressure),
                      "distance" : (("distance"), distance)})
        
        fig, ax = plt.subplots(figsize=(12,4))
        
        newFloat.EKE[ind_list].plot(x = 'distance', y='pressure',vmin = 0.0001, vmax = 0.1, 
                 cbar_kwargs={'label':'EKE $cm^{2}$ $s{-2}$'}, 
                 norm=matplotlib.colors.LogNorm())
        
#         xr.plot.contourf(newFloat.EKE[ind_list],x = 'distance', y='pressure',vmin = 0.0001, vmax = 0.1, 
#                          cbar_kwargs={'label':'EKE $cm^{2}$ $s{-2}$'},
#                          norm=matplotlib.colors.LogNorm())
        
        ax.set_ylabel('pressure (dbar)')
        ax.set_xlabel('distance (km)')
        ax.invert_yaxis()
        plt.title('Float {} - week {}'.format(floatid, week_number))