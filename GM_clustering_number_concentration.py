#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:13:45 2017

@author: cima
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import statistics

from sklearn import mixture
#from matplotlib.colors import LinearSegmentedColormap
from particle_classes import nt_needles,nt_needles_aggr,nt_graupeln


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

##Selection of number of clustering

number_cluster=2

GM_n_needles=[]
GM_n_needles_aggr=[]
GM_n_graupeln=[]
GM_total=[]
GM_time=[]
i=0

##Selection of period of interest 

start_time='2014-11-06 22:40:00'
end_time='2014-11-06 23:45:00'

step_time=datetime.datetime.strptime(start_time,'%Y-%m-%d %H:%M:%S')+datetime.timedelta(minutes=5)
step_time=step_time.strftime('%Y-%m-%d %H:%M:%S')

while(step_time<=end_time):
    new_data=data[(data['Date_Time']>=start_time) & (data['Date_Time']<=step_time)]
    
    #########################################################
    ##Gaussian Mixture (no necessity of preprocessing data)##
    #########################################################

    gmix=mixture.GaussianMixture(n_components=number_cluster)
    gmix.fit(new_data[['Ddeq','V','Aratio','ASPratio']])
    new_data['GM_ID']=gmix.predict(new_data[['Ddeq','V','Aratio','ASPratio']])

    #Divide the cluster (different type of snow)

    GM_cluster0=new_data[new_data['GM_ID']==0]
    GM_cluster1=new_data[new_data['GM_ID']==1]
    GM_cluster2=new_data[new_data['GM_ID']==2]


    #GM_n0.append(len(GM_cluster0.index))
    #GM_n1.append(len(GM_cluster1.index))
    #GM_n2.append(len(GM_cluster2.index))

    GM_time.append(datetime.datetime.strptime(step_time, '%Y-%m-%d %H:%M:%S'))
        
    start_time=step_time
    step_time=datetime.datetime.strptime(step_time,'%Y-%m-%d %H:%M:%S')+datetime.timedelta(minutes=5) 
    step_time=step_time.strftime('%Y-%m-%d %H:%M:%S')
    
    i=i+1
    print(step_time)
    print(i)
    
    ##Calculate centroid position to assign 
    ##the correct cluster at the correct snow particle
     
    Ddeq_centroid=[]
    Aratio_centroid=[] 
    V_centroid=[]
   
    Ddeqc0=np.sum(GM_cluster0[['Ddeq']])/len(GM_cluster0)
    Ddeqc1=np.sum(GM_cluster1[['Ddeq']])/len(GM_cluster1)
    Ddeqc2=np.sum(GM_cluster2[['Ddeq']])/len(GM_cluster2)
     
     
    Aratioc0=np.sum(GM_cluster0[['Aratio']])/len(GM_cluster0)
    Aratioc1=np.sum(GM_cluster1[['Aratio']])/len(GM_cluster1)
    Aratioc2=np.sum(GM_cluster2[['Aratio']])/len(GM_cluster2)
     
    Vc0=np.sum(GM_cluster0[['V']])/len(GM_cluster0)
    Vc1=np.sum(GM_cluster1[['V']])/len(GM_cluster1)
    Vc2=np.sum(GM_cluster2[['V']])/len(GM_cluster2)
            
    Ddeq_centroid=(Ddeqc0[0],Ddeqc1[0],Ddeqc2[0])
    Aratio_centroid=(Aratioc0[0],Aratioc1[0],Aratioc2[0])
    V_centroid=(Vc0[0],Vc1[0],Vc2[0])
     
    ##Combination of condition on V, Aratio, Ddeq to classify the particle 
            
    if(min(V_centroid)<1.5):
        needles=V_centroid.index(min(V_centroid))
    if(max(Aratio_centroid)>0.5):
        graupeln=Aratio_centroid.index(max(Aratio_centroid))
        if(needles==graupeln):
            graupeln=Ddeq_centroid.index(statistics.median(Ddeq_centroid))
    if(min(Aratio_centroid)<0.7):
        needlesaggr=Aratio_centroid.index(min(Aratio_centroid))
        if(needles==needlesaggr):
            needlesaggr=Ddeq_centroid.index(max(Ddeq_centroid))
        if(graupeln==needlesaggr):#aggr has more dispersion (rule to separate from graupeln)
            if(needlesaggr==0):
                if(max(GM_cluster0['Ddeq'])-min(GM_cluster0['Ddeq'])>2.5):
                    if(max(Aratio_centroid)>0.55):
                        needlesaggr=50
                    elif(max(GM_cluster0['V'])-min(GM_cluster0['V'])>2):
                        graupeln=50
            if(needlesaggr==1):
                if(max(GM_cluster1['Ddeq'])-min(GM_cluster1['Ddeq'])>2.5):
                    if(max(Aratio_centroid)>0.55):
                        needlesaggr=50
                    elif(max(GM_cluster1['V'])-min(GM_cluster1['V'])>2):
                        graupeln=50             
            if(needlesaggr==2):
                if(max(GM_cluster2['Ddeq'])-min(GM_cluster2['Ddeq'])>2.5):
                    if(max(Aratio_centroid)>0.55):
                        needlesaggr=50
                    elif(max(GM_cluster2['V'])-min(GM_cluster2['V'])>2):
                        graupeln=50
                         
    if(needles==0):
       GM_n_needles.append(len(GM_cluster0))
    if(needles==1):
       GM_n_needles.append(len(GM_cluster1))
    if(needles==2):
       GM_n_needles.append(len(GM_cluster2))
    elif(needles==50 or needles==100):
       GM_n_needles.append(0)
       
    if(needlesaggr==0):
       GM_n_needles_aggr.append(len(GM_cluster0))
    if(needlesaggr==1):
       GM_n_needles_aggr.append(len(GM_cluster1))
    if(needlesaggr==2):
       GM_n_needles_aggr.append(len(GM_cluster2))
    elif(needlesaggr==50):
       GM_n_needles_aggr.append(0)
       
    if(graupeln==0):
       GM_n_graupeln.append(len(GM_cluster0))
    if(graupeln==1):
       GM_n_graupeln.append(len(GM_cluster1))
    if(graupeln==2):
       GM_n_graupeln.append(len(GM_cluster2))
    elif(graupeln==50):
       GM_n_graupeln.append(0)
 

       
GM_total=[sum(x) for x in zip(GM_n_needles,GM_n_needles_aggr,GM_n_graupeln)]
    


'''
    ########
    ##Plot##
    ########

    ##Option for plotting (select x/y variable -> 'Ddeq','V','Aratio' )

    x_variable='Aratio'
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

    ##Option for plotting (select dataset -> data, cluster0, cluster1, cluster2)
    GM_data_set=GM_cluster0
    
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
'''



###    
###Use of PSD to calculate the concentration number of the particle
###
    
filename_psd='/home/cima/Documents/Helsinki Work/Data_BAECC2014_txt/SnowPSD_20141106_1107.txt'

column_names=['Date_Time', 'D0','N0','Lambda', 
              '0.125','0.375','0.625','0.875','1.125','1.375','1.625','1.875',
              '2.125','2.375','2.625','2.875','3.125','3.375','3.625','3.875',
              '4.125','4.375','4.625','4.875','5.125','5.375','5.625','5.875',
              '6.125','6.375','6.625','6.875','7.125','7.375','7.625','7.875',
              '8.125','8.375','8.625','8.875','9.125','9.375','9.625','9.875',
              '10.125','10.375','10.625','10.875','11.125','11.375','11.625',
              '11.875','12.125','12.375','12.625','12.875','13.125','13.375',
              '13.625','13.875','14.125','14.375','14.625','14.875','15.125',
              '15.375','15.625','15.875','16.125','16.375','16.625','16.875',
              '17.125','17.375','17.625','17.875','18.125','18.375','18.625',
              '18.875','19.125','19.375','19.625','19.875','20.125','20.375',
              '20.625','20.875','21.125','21.375','21.625','21.875','22.125',
              '22.375','22.625','22.875','23.125','23.375','23.625','23.875',
              '24.125','24.375','24.625','24.875','25.125','25.375','25.625',
              '25.875','26']

    
data_psd=pd.read_table(filename_psd, delim_whitespace=True, parse_dates=[['time','D0']])
data_psd=data_psd.drop('26',1)
data_psd.columns=column_names

start_time=datetime.datetime.strptime(step_time,'%Y-%m-%d %H:%M:%S')-datetime.timedelta(minutes=5*i)
start_time=start_time.strftime('%Y-%m-%d %H:%M:%S')


data_psd=data_psd[(data_psd['Date_Time']>=start_time) & (data_psd['Date_Time']<=end_time)]

                 
GM_nt=[]
GM_vobs=[]
GM_nt_needles=[]
GM_nt_needles_aggr=[]
GM_nt_graupeln=[]


for i in range (0, len(data_psd)):
    GM_nt.append((data_psd['N0'].iloc[i])*(data_psd['Lambda'].iloc[i])**-1)
    GM_vobs.append(GM_total[i]/GM_nt[i])
        
    GM_nt_needles.append(GM_n_needles[i]/(GM_vobs[i]))
    GM_nt_needles_aggr.append(GM_n_needles_aggr[i]/(GM_vobs[i]))
    GM_nt_graupeln.append(GM_n_graupeln[i]/(GM_vobs[i]))


fig1=plt.figure()
plt.plot(GM_time,GM_nt_needles,'b--o',label='GM_needles')
plt.plot(GM_time,GM_nt_needles_aggr,'r--o',label='GM_needlesaggr')
plt.plot(GM_time,GM_nt_graupeln,'k--o',label='GM_graupel')
plt.xlabel('Time')
plt.ylabel('Number concentration [m**-3]')            
plt.legend(loc='upper right', ncol=3)
plt.yscale('log') 
    
    

fig2=plt.figure()
plt.plot(GM_time,GM_nt_needles,'b--o',label='GM_needles')
plt.plot(GM_time,GM_nt_needles_aggr,'r--o',label='GM_needlesaggr')
plt.plot(GM_time,GM_nt_graupeln,'k--o',label='GM_graupel')
plt.plot(GM_time,nt_needles,'b-o',label='Class_needles')
plt.plot(GM_time,nt_needles_aggr,'r-o',label='Class_needlesaggr')
plt.plot(GM_time,nt_graupeln,'k-o',label='Class_graupel')
plt.xlabel('Time')
plt.ylabel('Number concentration [m**-3]')            
plt.legend(loc='upper right', ncol=3)
#plt.yscale('log')   





