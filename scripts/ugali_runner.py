#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 21:17:44 2022

@author: Ani
"""

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import CMD_utils as cu

age = 10 #Age in Gyr
dist = 23 #distance in kpc
feh = -1.5 #metallicity
tolerance = 0.2 #Tolerance in simple CMD selection

#Convert feh to z
Y_p     = 0.245            # Primordial He abundance (WMAP, 2003)
c       = 1.54             # He enrichment ratio 
ZX_solar = 0.0229
z = (1 - Y_p)/( (1 + c) + (1/ZX_solar) * 10**(-feh))
print(z)

#Load data, quick quality flags pre-processing (Pal 5 is the example)
tab = Table.read('PS1DR2-Pal5_xm.fits')
qual_flags = (tab['gMeanPSFMagErr'] < 0.1) & (tab['rMeanPSFMagErr'] < 0.1) & \
             (tab['iMeanPSFMagErr'] < 0.1) & \
             (tab['gMeanPSFMagErr'] > 0.0) & (tab['rMeanPSFMagErr'] > 0.0) & \
             (tab['iMeanPSFMagErr'] > 0.0) & \
            (abs(tab['gMeanPSFMag']) < 25) & (abs(tab['rMeanPSFMag']) < 25) & \
            (abs(tab['iMeanPSFMag']) < 25)
tab = tab[qual_flags]


#De-redden the pan-Starrs photometry, star/galaxy separation with PanStarrs
#star/galaxy separation not yet implemented
tab = cu.deReddenPS(tab)


#Isochrone selection
iso, initial_mass, mass_pdf, actual_mass, mag_1, mag_2, mmag_1, mmag_2, \
                mmass_pdf = cu.load_isochrone(z, age, dist, alphafe = '+0.2')
masses = np.arange(min(initial_mass), max(initial_mass), 0.00001)
mag1_interp = mmag_1(masses)
mag2_interp = mmag_2(masses)
masspdf_interp = mmass_pdf(masses)

cu.plot_isochrone(iso, mag1_interp, mag2_interp, masspdf_interp)

#select all stars within some tolerance (~0.1 mag in color space) of the isochrone
sel, iso_model, iso_low, iso_high = \
            cu.simpleSln(tab, mag1_interp, mag2_interp, masses, tolerance, min(mag_1))

#Spatial density plot
yaxis = np.arange(0, 26, 0.01)
gr_obs = tab['gMean_dered'] - tab['rMean_dered']

fig, axarr = plt.subplots(1, 2, figsize = (12, 4))
axarr[0].plot(gr_obs, tab['gMean_dered'], '.k')
axarr[0].plot(gr_obs[sel], tab['gMean_dered'][sel], '.r')
axarr[0].plot(iso_model(yaxis), yaxis, '--b')
axarr[0].plot(iso_low(yaxis), yaxis, '--y')
axarr[0].plot(iso_high(yaxis), yaxis, '--y')
axarr[0].set_xlabel('g-r')
axarr[0].set_ylabel('g')
axarr[0].set_ylim([15, 22])
axarr[0].set_xlim([-1, 1.5])
axarr[0].invert_yaxis()

axarr[1].hist2d(tab['raMean'][sel], tab['decMean'][sel], bins = (100, 100), \
           vmin = 130, vmax = 250)
axarr[1].set_xlabel('RA (deg)')
axarr[1].set_ylabel('DEC (deg)')
plt.show()

