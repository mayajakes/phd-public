
#rotation of velocities 

import numpy as np
import pyproj
from geographiclib.geodesic import Geodesic

def uvBearing(u, v):
    '''Calculates the bearing (clockwise from True North) using eastward (u) and northward (v) components of velocity'''
    theta = np.rad2deg(np.arctan2(u, v))
    theta += 360
    theta = theta % 360
    return theta


def find_stream_bearing(lons, lats):
    '''Find the bearing (clockwise from True North) between each lat and lon position.'''
    
    bearing = []
    for i in range(0, len(lats)-1):
        lat1, lat2 = lats[i], lats[i+1]
        lon1, lon2 = lons[i], lons[i+1]

        geodesic = pyproj.Geod(ellps='WGS84') # WGS84 is the reference coordinate system (Earth's centre of mass) used by GPS. 
        
        # Inverse computation to calculate the forward azimuth between two lat and lon coordinates. 
        fwd_azimuth = geodesic.inv(lon1, lat1, lon2, lat2)[0]

        # If the angle is negative (anticlockwise from N), add it to 360 to get the bearing clockwise from N.
        fwd_azimuth += 360
        fwd_azimuth = fwd_azimuth % 360

        if fwd_azimuth == 0:
            bearing.append(np.nan)
        else:
            bearing.append(fwd_azimuth)

    return np.asarray(bearing)


def rotate_vels(u, v, lons, lats, zdim = None):
    '''Rotate zonal (u) and meridional (v) velocities with respect to a stream direction calculated using lat and lon coordinates along a streamline.

    INPUT: 
    u = zonal velocity 
    v = meridional velocity 
    lons = 1D array of streamline longitudes
    lats = 1D array of streamline latituds

'''
    stream_bearing = find_stream_bearing(lons, lats)
    
    if zdim is not None: # make into 2 dimensions
        stream_bearing = np.tile(stream_bearing,(len(u[zdim]), 1)).transpose()
    
    velocity_bearing = uvBearing(u, v)
    
    # find the angle between the velocity bearing and the stream bearing
    theta = stream_bearing - velocity_bearing[:-1]
    theta = (theta + 180) % 360 - 180

    speed = np.sqrt(u**2 + v**2)

    # calculate u and v using this new angle (converting degrees to radians)
    u_rot = speed * np.cos(theta*np.pi/180)
    v_rot = speed * np.sin(theta*np.pi/180)
    
    return u_rot, v_rot
