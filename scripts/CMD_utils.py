#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:22:19 2022

@author: Ani
"""

import numpy as np
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ugali.analysis.isochrone import factory as isochrone_factory
from dustmaps.sfd import SFDQuery

def deReddenPS(tab, rakey = 'raMean', deckey = 'decMean'):
    
    coords = SkyCoord(ra = tab['raMean'], dec = tab['decMean'], unit=(u.deg, u.deg), frame='icrs')
    sfd = SFDQuery()
    ebmv = sfd(coords)
    
    #De-redden following Tonry et al. 2012
    tab['gMean_dered'] = tab['gMeanPSFMag'] - ((3.613 -0.0972*(tab['gMeanPSFMag'] - \
                         tab['iMeanPSFMag']) + 0.0100*(tab['gMeanPSFMag'] - tab['iMeanPSFMag'])**2)*0.88*ebmv)
    tab['rMean_dered'] = tab['rMeanPSFMag'] - (2.585 - 0.0315*((tab['gMeanPSFMag'] - \
                                                tab['iMeanPSFMag'])))*0.88*ebmv
    tab['iMean_dered'] = tab['iMeanPSFMag'] - (1.908 - 0.0152*(tab['gMeanPSFMag'] - \
                                                tab['iMeanPSFMag']))*0.88*ebmv
    return tab
    

def load_isochrone(z, age, dist, alphafe = '+0.2'):
    '''
    load an isochrone, LF model for a given metallicity, age, distance
    '''
    iso = isochrone_factory('Dotter', survey='ps1', age=age, \
                            distance_modulus=5*np.log10(dist*1000) - 5, z=z, \
                            band_1 = 'g', band_2 = 'r')
        
    iso.afe = alphafe
    
    initial_mass,mass_pdf,actual_mass,mag_1,mag_2 = iso.sample(mass_steps=1e2)
    mag_1 = mag_1 +iso.distance_modulus
    mag_2 = mag_2 +iso.distance_modulus
    
    mmag_1 = interp1d(initial_mass, mag_1, fill_value = 'extrapolate')
    mmag_2 = interp1d(initial_mass, mag_2, fill_value = 'extrapolate')
    mmass_pdf = interp1d(initial_mass, mass_pdf, fill_value = 'extrapolate')
        
    return iso, initial_mass, mass_pdf, actual_mass, mag_1, mag_2, mmag_1, mmag_2, \
                mmass_pdf
                
def plot_isochrone(iso, mag1, mag2, mass_pdf):
    
    x_bins = np.arange(-0.5, 1.0, 0.04)
    y_bins = np.arange(15, 23, 0.2)
        
    H, xedges, yedges = np.histogram2d(mag1 - mag2, \
                                       mag1, bins = [x_bins, y_bins], \
                                       weights = mass_pdf)
        
    H_counts, xedges, yedges = np.histogram2d(mag1 - mag2, \
                                              mag1, bins = [x_bins, y_bins]) 
    H = H/H_counts
        
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    col = ax0.scatter(mag1-mag2, mag1,\
                c=mass_pdf,marker='o',facecolor='none', \
                norm=LogNorm(vmin=0.000001, vmax=0.001))
    plt.colorbar(col, label = 'Density')
    ax0.invert_yaxis()
    ax0.set_xlabel('%s - %s'%(iso.band_1,iso.band_2)); ax0.set_ylabel(iso.band_1)
    ax0.set_ylim(23,15); ax0.set_xlim(-0.5,1.0)
    
    ax1.imshow(H.T, origin='lower', extent=[x_bins[0], x_bins[-1], \
                                                        y_bins[0], y_bins[-1]], \
               aspect='auto', norm=LogNorm(vmin=0.000001, vmax=0.001))
    ax1.set_xlabel('g-i')
    ax1.set_ylabel('g')
    ax1.invert_yaxis()
    plt.show()
    
    return x_bins, y_bins, H

def simpleSln(tab, mag1, mag2, masses, tolerance, maxmag, mass_thresh = 0.843):
    
    ind = (masses < mass_thresh)
    mag1 = mag1[ind]
    mag2 = mag2[ind]
    
    iso_low = interp1d(mag1, mag1 - mag2 - tolerance, \
                       fill_value = 'extrapolate')
    iso_high = interp1d(mag1, mag1 - mag2 + tolerance, \
                        fill_value = 'extrapolate')
    iso_model = interp1d(mag1, mag1 - mag2, fill_value = 'extrapolate')
    sel = (iso_low(tab['gMean_dered']) < ((tab['gMean_dered'] - tab['rMean_dered']) + \
                                         2*np.sqrt(tab['gMeanPSFMagErr']**2 + tab['rMeanPSFMagErr']**2))) & \
                            (iso_high(tab['gMean_dered']) > (tab['gMean_dered'] - tab['rMean_dered']) - \
                                         2*np.sqrt(tab['gMeanPSFMagErr']**2 + tab['rMeanPSFMagErr']**2)) & \
                            (tab['gMean_dered'] > maxmag)

    return sel, iso_model, iso_low, iso_high