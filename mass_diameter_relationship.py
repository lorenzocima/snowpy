#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:31:07 2017

@author: cima
"""
from scipy.optimize import curve_fit

###To put in the main file###

###Relation Mass-Diameter (with fit)


fig1=plt.figure()

h=plt.hist2d(GM_data_set['Dmax'], GM_data_set['mass'], bins=80, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, 1.5)]),cmap=my_cmap)
plt.colorbar()
plt.axis([xmin, xmax, ymin, 1.5])
plt.xlabel('Dmax')
plt.ylabel('mass')
plt.title('M-D Relationship (GM)')
#plt.grid(color='gray')

def line(x, a, b):
    return a*(x**b)

parameter, pcovariance = curve_fit(line,GM_data_set['Dmax'], GM_data_set['mass'])
xfine = np.linspace(0, 5, 100)
plt.plot(xfine, line(xfine, parameter[0], parameter[1]), 'k--')
a=parameter[0]
b=parameter[1]
plt.text(1,1.4, 'M=aD**b')
plt.text(0.4,1.2, 'a=')
plt.text(0.4,1.1, 'b=')
plt.text(0.6, 1.2, a)
plt.text(0.6, 1.1, b)
