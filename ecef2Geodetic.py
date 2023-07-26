# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 07:44:18 2020

@author: mohsen feizabadi
"""

def ecef2Geodetic(x, y, z):
    """
    Convert ECEF to geodetic coordinates.
    This function uses the Bowring algorithm from 1985.
    The accuracy is at least 11 decimals (in degrees).
    :param x,y,z: ECEF coordinates
    :rtype: tuple (lat,lon) in radians
    """
    import numpy as np

    a = 6378137 #equatorial axis of the ellipsoid of revolution
    b = 6356752.3142 #polar axis of the ellipsoid of revolution
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    e2 = (a*a - b*b) / (a*a) # first eccentricity squared
    d = (a*a - b*b) / b
    
    p2 = np.square(x) + np.square(y)
    p = np.sqrt(p2)
    r = np.sqrt(p2 + z*z)
    tu = b*z*(1 + d/r)/(a*p)
    tu2 = tu*tu
    cu3 = (1/np.sqrt(1 + tu2))**3
    su3 = cu3*tu2*tu
    tp = (z + d*su3)/(p - e2*a*cu3)
    lat = np.arctan(tp)
    lon = np.arctan2(y,x)
    h = p/np.cos(lat) - a/np.sqrt(1-(e2*np.sin(lat)**2))
    
    return lat, lon, h

