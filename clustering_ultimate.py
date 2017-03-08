#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:17:57 2017

@author: cima
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import matplotlib


from sklearn.cluster import KMeans
from sklearn import mixture
#from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

from particle_classes import data_psd

##Data import from .txt file to data frame

filename='/home/cima/Documents/Helsinki Work/Data_BAECC2014_txt/Snow_particles_20141106_1107.txt'

column_names=['Date_Time', 'Ddeq', 'V', 'A', 
'Dmin', 'Dmaj', 'Dmax', 'mass', 'W', 'H']

data=pd.read_table(filename, delim_whitespace=True, parse_dates=[['time','Ddeq']])

data.columns=column_names

##Area Ratio calculation and join with data frame 

area_ratio=data['A']/((math.pi/4.0)*(data['Dmax'])**2)
data['Aratio']=area_ratio
    
##Aspect Ratio calculation and join with data frame 

aspect_ratio=data['Dmin']/data['Dmax'] 
data['ASPratio']=aspect_ratio

'''    
##Ratio of rectangular volume

volume_ratio=data['H']/data['W']
data['VOLratio']=volume_ratio

'''    

##Selection of period of interest 

start_time='2014-11-06 22:40:00'
end_time='2014-11-06 22:44:00'

data=data[(data['Date_Time']>=start_time) & (data['Date_Time']<=end_time)]
          
          
##Selection of number of clustering

number_cluster=2


##Plot something of the data 

##data.plot(x='Ddeq', y='V', kind='scatter',color='Black', title='Title')

##Clustering

######################################################################################
##K-Means algorithm (Preprocessing is essential for kmeans algorithm (mean=0 std=1))##
######################################################################################

normalize=StandardScaler().fit(data[['Ddeq','V','Aratio','ASPratio']])
normalized_data= normalize.transform(data[['Ddeq','V','Aratio','ASPratio']])


km = KMeans(n_clusters=number_cluster,n_init=40, init='k-means++')
km.fit(normalized_data)
labels=data['KM_ID']=km.predict(normalized_data)
km.get_params()

#data.plot(kind='scatter', x='Ddeq',y='V', c=labels, colormap='Set1')
#plt.axis([0, 5, 0, 3])

#Divide the cluster (different type of snow)

KM_cluster0=data[data['KM_ID']==0]
KM_cluster1=data[data['KM_ID']==1]
KM_cluster2=data[data['KM_ID']==2]


#########################################################
##Gaussian Mixture (no necessity of preprocessing data)##
#########################################################

gmix=mixture.GaussianMixture(n_components=number_cluster)
gmix.fit(data[['Ddeq','V','Aratio','ASPratio']])
data['GM_ID']=gmix.predict(data[['Ddeq','V','Aratio','ASPratio']])

#Divide the cluster (different type of snow)

GM_cluster0=data[data['GM_ID']==0]
GM_cluster1=data[data['GM_ID']==1]
GM_cluster2=data[data['GM_ID']==2]



#Calculate mass and number of snow particle

KM_w0=KM_cluster0['mass'].sum()
KM_w1=KM_cluster1['mass'].sum()
KM_w2=KM_cluster2['mass'].sum()

KM_n0=len(KM_cluster0.index)
KM_n1=len(KM_cluster1.index)
KM_n2=len(KM_cluster2.index)

GM_w0=GM_cluster0['mass'].sum()
GM_w1=GM_cluster1['mass'].sum()
GM_w2=GM_cluster2['mass'].sum()

GM_n0=len(GM_cluster0.index)
GM_n1=len(GM_cluster1.index)
GM_n2=len(GM_cluster2.index)


########
##Plot##
########

##Option for plotting (select x/y variable -> 'Ddeq','V','Aratio' )

x_variable='Ddeq'
y_variable='V'

if y_variable=='Aratio':
    xmin=0
    xmax=5
    ymin=0
    ymax=1.25
if y_variable=='V':
    xmin=0
    xmax=5
    ymin=0
    ymax=3
        
##Option for plotting (select the colormap)

my_cmap=copy.copy(matplotlib.cm.jet)
my_cmap.set_under('w')

#my_cmap = LinearSegmentedColormap.from_list('mycmap', ['white', 'blue', 'cyan', 'green', 'yellow','red'])

##Option for plotting (select dataset -> data, cluster0, cluster1, cluster2)

KM_data_set=KM_cluster1

GM_data_set=GM_cluster1

    
fig=plt.figure()
h=plt.hist2d(KM_data_set[x_variable],KM_data_set[y_variable], bins=80, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
plt.colorbar()
plt.axis([xmin, xmax, ymin, ymax])
plt.xlabel(x_variable)
plt.ylabel(y_variable)
plt.title('K-Means')
#plt.grid(color='gray')
plt.show()



fig1=plt.figure()
h=plt.hist2d(GM_data_set[x_variable], GM_data_set[y_variable], bins=80, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
plt.colorbar()
plt.axis([xmin, xmax, ymin, ymax])
plt.xlabel(x_variable)
plt.ylabel(y_variable)
plt.title('Gaussian Mixture')
#plt.grid(color='gray')
plt.show()


###PSD analysis 
              
range_min=0
range_max=15
delta=0.25

fig2=plt.figure()
plt.hist(GM_cluster0['Ddeq'], range=(range_min,range_max), bins=range_max/delta, color='blue')
count0, division0 = np.histogram(GM_cluster0['Ddeq'],range=(range_min,range_max), bins=range_max/delta)
plt.xlabel('Ddeq')
plt.ylabel('Number of particles')


plt.hist(GM_cluster1['Ddeq'], range=(range_min,range_max), bins=range_max/delta, color='red')
count1, division1 = np.histogram(GM_cluster1['Ddeq'],range=(range_min,range_max), bins=range_max/delta)
plt.xlabel('Ddeq')
plt.ylabel('Number of particles')



plt.hist(GM_cluster2['Ddeq'], range=(range_min,range_max), bins=range_max/delta, color='yellow')
count2, division2 = np.histogram(GM_cluster2['Ddeq'],range=(range_min,range_max), bins=range_max/delta)
plt.xlabel('Ddeq')
plt.ylabel('Number of particles')
plt.show()



'''
plt.hist(data['Ddeq'], range=(range_min,range_max), bins=range_max/delta, color='black')
count2, division2 = np.histogram(data['Ddeq'],range=(range_min,range_max), bins=range_max/delta)
plt.xlabel('Ddeq')
plt.ylabel('Number of particles')
plt.show()
'''

division0=np.delete(division0,-1)
division1=np.delete(division1,-1)
division2=np.delete(division2,-1)

fig3=plt.figure()
plt.plot(division0, count0, color='blue')
plt.plot(division1, count1, color='red')
plt.plot(division2, count2, color='yellow')
plt.xlim(0,6)
plt.show()

new_data_psd=data_psd.mean()
new_data_psd=new_data_psd[3:]#eliminate D0,N0 and Lambda to have only data for each class



fig4=plt.figure()
plt.plot(division0/2, (count0+count1), color='black')
plt.plot(new_data_psd,'k-o',label='Nt')
plt.xlim(0,6)
plt.show()


'''
###Relation Mass-Diameter (with fit)

KM_data_set=KM_cluster1
GM_data_set=GM_cluster1
xmin=0
xmax=8
ymin=0
ymax=2

fig1=plt.figure()

#h=plt.hist2d(GM_data_set['Dmax'], GM_data_set['mass'], bins=80, normed=True,
#             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, 1.5)]),cmap=my_cmap)
plt.plot(GM_data_set['Dmax'], GM_data_set['mass'],'ko')
#plt.colorbar()
plt.axis([xmin, xmax, ymin, ymax])
plt.xlabel('Dmax')
plt.ylabel('mass')
plt.title('M-D Relationship (GM)')
#plt.grid(color='gray')

def line(x, a, b, c):
    return a*(x**b)

parameter, pcovariance = curve_fit(line,GM_data_set['Dmax'], GM_data_set['mass'])
xfine = np.linspace(0, 5, 100)
plt.plot(xfine, line(xfine, parameter[0], parameter[1],parameter[2]),'r--', linewidth=4)
a=parameter[0]
b=parameter[1]
plt.text(1,1.4, 'M=aD**b')
plt.text(0.4,1.2, 'a=')
plt.text(0.4,1.1, 'b=')
plt.text(0.6, 1.2, a)
plt.text(0.6, 1.1, b)

'''

