#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:12:04 2017

@author: lorenzo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename_psd='/home/cima/Documents/Helsinki Work/Data_BAECC2014_txt/SnowPSD_20140215_0216.txt'

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

##Selection of period of interest 

#start_time=tstart
#end_time=tend
#data_psd=data_psd[(data_psd['Date_Time']>=start_time) & (data_psd['Date_Time']<=end_time)]

number_t=[]


for i in range (0, len(data_psd)):
    if(data_psd['N0'].iloc[i]==0 or data_psd['Lambda'].iloc[i]==0):
        number_t.append(0)
    else:   
        number_t.append((data_psd['N0'].iloc[i])*(data_psd['Lambda'].iloc[i])**-1)

fig1=plt.figure()
plt.plot(data_psd['Date_Time'],number_t,'k-',label='Number_t')
plt.xlabel('Time')
plt.ylabel('Number concentration log10[m**-3]')            
plt.legend(loc='upper left')
plt.yscale('log') 









'''
###Make average to confront with analysis of separate clusters

data_psd=data_psd.mean()
data_psd=data_psd[3:]#eliminate D0,N0 and Lambda to have only data for each class


###PSD analysis 

range_min=0
range_max=15
delta=0.25

fig3=plt.figure()
plt.plot(data_psd, color='black')
plt.xlim(0,6)
plt.xlabel('Ddeq')
plt.ylabel('Number of particles')
plt.title(start_time)



'''

'''
###Test to find only the common date between psd e classification
common_date=[]
   
for ii in range(0,len(data_psd)):
    k=0
    for jj in range (0, len(time)):
        if(datetime.datetime.strftime(data_psd['Date_Time'].iloc[ii],'%Y-%m-%d %H:%M')==datetime.datetime.strftime(time[jj],'%Y-%m-%d %H:%M')):
            #k=1 
            common_date.append(data_psd['Date_Time'].iloc[ii])
            
    #if (k==0):
        #print(data_psd['Date_Time'].iloc[ii])
'''

