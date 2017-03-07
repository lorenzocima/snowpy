#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:07:37 2017

@author: cima
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:20:08 2017

@author: lorenzo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import matplotlib
import matplotlib.patches as mpatches
import datetime
import statistics
import matplotlib.dates as mdates

from sklearn.cluster import KMeans,DBSCAN,MeanShift,estimate_bandwidth
from sklearn import mixture
from sklearn.metrics import silhouette_score
#from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist,pdist
from scipy import stats

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

start_time='2014-11-06 18:00:00'
end_time='2014-11-07 17:00:00'

data=data[(data['Date_Time']>=start_time) & (data['Date_Time']<=end_time)]
              
##Selection of piece of dataset with dimension of 1000 measurements

yes_needles=[]
yes_graupeln=[] 
yes_needlesaggr=[]        
total_n_cluster=[]
time=[]

start=0
step=6000 #how many data use for each iteration step
end=step
while start<len(data):
     data1=data.iloc[start:end]
     if(len(data1)<(step//10)):
         break

     normalize=StandardScaler().fit(data1[['Ddeq','V','Aratio','ASPratio']])
     normalized_data= normalize.transform(data1[['Ddeq','V','Aratio','ASPratio']])

     ###Silhouette Method for number of cluster 
     ###(to use only with period of time very short)->expensive for RAM
    
     ##Necessary restart the variables before the new cycle
     ##with high values to not go into if condition if isn't necessary
     needles=50
     needlesaggr=50
     graupeln=50    
    
     ###Using K-Means
     silhouette_value_KM=[]
     silhouette_value_KM.append(0)
    
     silhouette_KM_max=0
     n_cluster_GM=0
        
     for i in range(2,4):
         clusterer = KMeans(n_clusters=i)
         cluster_labels = clusterer.fit_predict(normalized_data)
         silhouette_avg = silhouette_score(normalized_data, labels=cluster_labels)
         silhouette_value_KM.append(silhouette_avg)
         #print("For n_clusters =", i,
         #     "The average silhouette_score is :", silhouette_avg)
         if (silhouette_avg > silhouette_KM_max):
             silhouette_KM_max=silhouette_avg
             n_cluster_GM=i
            
             print('Estimated number of cluster with K-Means ---> ', n_cluster_GM)
             
             #it's necessary a treshold to find 1 cluster
             #treshold means that the silhouette coefficient isn't
             #sufficient elevate to ensure the accuracy of result of number
             #of cluster find with the algorithm
             if (silhouette_KM_max < 0.29):
                 n_cluster_GM=1
                 print('Correction number of cluster --->', n_cluster_GM)
                 
         if(i==3):       
             if(silhouette_KM_max-silhouette_avg<0.1 and silhouette_avg>0.3
                and silhouette_KM_max<0.4):
                 n_cluster_GM=i
                 print('Another correction number of cluster --->', n_cluster_GM)        
     
     ##use number of cluster found with k-means algorithm to run 
     ##gaussian mixture (more accurate clusterization)
     
     silhouette_value_GM=silhouette_value_KM
     
     
     '''           
     x_variable='Ddeq'
     y_variable='Aratio'

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
     
     my_cmap=copy.copy(matplotlib.cm.jet)
     my_cmap.set_under('w')    
     
     fig3=plt.figure()
     plt.subplot(3, 2, 3)
     h=plt.hist2d(data1[x_variable],data1[y_variable], bins=120, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
     plt.colorbar()
     plt.axis([xmin, xmax, ymin, ymax])
     plt.xlabel(x_variable)
     plt.ylabel(y_variable)
     plt.title('Gaussian Mixture All Data')
     #plt.grid(color='gray')
   
     plt.subplot(3, 1, 1)
     plt.plot(range(1,4), silhouette_value_GM,'ro-',label='K-Means')
     plt.axis([0,10,0,0.5])
     plt.xlabel('k')
     plt.ylabel('Average Silhouette')
     plt.title('Selecting k with the Average Silhouette Method')
     plt.text(1,0.4,n_cluster_GM)
     plt.text(0.4,0.4, 'cluster=')
     plt.show()
     
    '''
     
     gmix=mixture.GaussianMixture(n_components=n_cluster_GM)
     gmix.fit(data1[['Ddeq','V','Aratio','ASPratio']])
     data1['GM_ID']=gmix.predict(data1[['Ddeq','V','Aratio','ASPratio']])
     
     GM_cluster0=data1[data1['GM_ID']==0]
     GM_cluster1=data1[data1['GM_ID']==1]
     GM_cluster2=data1[data1['GM_ID']==2]
          
         
 
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
                     if(max(Aratio_centroid)>0.65):
                         needlesaggr=50
                     elif(max(GM_cluster0['V'])-min(GM_cluster0['V'])>2):
                         graupeln=50
             if(needlesaggr==1):
                 if(max(GM_cluster1['Ddeq'])-min(GM_cluster1['Ddeq'])>2.5):
                     if(max(Aratio_centroid)>0.65):
                         needlesaggr=50
                     elif(max(GM_cluster1['V'])-min(GM_cluster1['V'])>2):
                         graupeln=50             
             if(needlesaggr==2):
                 if(max(GM_cluster2['Ddeq'])-min(GM_cluster2['Ddeq'])>2.5):
                     if(max(Aratio_centroid)>0.65):
                         needlesaggr=50
                     elif(max(GM_cluster2['V'])-min(GM_cluster2['V'])>2):
                         graupeln=50
                       
     '''                     
     plt.subplot(3, 2, 4)
     h=plt.hist2d(GM_cluster0[x_variable],GM_cluster0[y_variable], bins=120, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
     plt.colorbar()
     plt.axis([xmin, xmax, ymin, ymax])
     plt.xlabel(x_variable)
     plt.ylabel(y_variable)
     if(needles==0):
         plt.title('GM Needles')
         yes_needles.append(100)
     if(needlesaggr==0):
         plt.title('GM Needles Aggregate')
         yes_needlesaggr.append(100)
     if(graupeln==0):
         plt.title('GM Graupeln')
         yes_graupeln.append(100)
     if(needles==50 and needlesaggr==50 and graupeln==50):
         plt.text(1, 0.5, 'Clustering Error')
         
     plt.subplot(3, 2, 5)
     h=plt.hist2d(GM_cluster1[x_variable],GM_cluster1[y_variable], bins=120, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
     plt.colorbar()
     plt.axis([xmin, xmax, ymin, ymax])
     plt.xlabel(x_variable)
     plt.ylabel(y_variable)
     if(needles==1):
         plt.title('GM Needles')
         yes_needles.append(100)
     if(needlesaggr==1):
         plt.title('GM Needles Aggregate')
         yes_needlesaggr.append(100)
     if(graupeln==1):
         plt.title('GM Graupeln')
         yes_graupeln.append(100)
     if(needles==50 and needlesaggr==50 and graupeln==50):
         plt.text(1,0.5,'Clustering Error')
         
     plt.subplot(3, 2, 6)
     h=plt.hist2d(GM_cluster2[x_variable],GM_cluster2[y_variable], bins=120, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
     plt.colorbar()
     plt.axis([xmin, xmax, ymin, ymax])
     plt.xlabel(x_variable)
     plt.ylabel(y_variable)
     if(needles==2):
         plt.title('GM Needles')
         yes_needles.append(100)
     if(needlesaggr==2):
         plt.title('GM Needles Aggregate')
         yes_needlesaggr.append(100)
     if(graupeln==2):
         plt.title('GM Graupeln')
         yes_graupeln.append(100)
     if(needles==50 and needlesaggr==50 and graupeln==50):
         plt.text(1,0.5,'Clustering Error')
         
     print(needles,needlesaggr,graupeln)
     
     ##If the values not change it means that this particle is not present         
     if(needles==50 and (needlesaggr!=50 or graupeln!=50)):
         yes_needles.append(0)
     if(needlesaggr==50 and (needles!=50 or graupeln!=50)):
         yes_needlesaggr.append(0)
     if(graupeln==50 and (needles!=50 or needlesaggr!=50)):
         yes_graupeln.append(0)
     if(needles==50 and needlesaggr==50 and graupeln==50):
         yes_needles.append(np.nan)
         yes_needlesaggr.append(np.nan)
         yes_graupeln.append(np.nan)
     '''
     
     total_n_cluster.append(n_cluster_GM)
     time.append(data1['Date_Time'].iloc[len(data1)//2])    
     start=end+1
     end=end+step+1

#add 30 seconds for removing time-step sovrapposition
for t in range(1,len(time)):
    if (time[t]==time[t-1]):
        time[t]=time[t]+datetime.timedelta(seconds=30)
        
#Plot number of cluster over time
fig4=plt.figure()
ax1=plt.subplot(4, 1, 1)
plt.plot(time, total_n_cluster, 'ko-')
plt.ylim(0,5) 
plt.xlim(datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')-datetime.timedelta(minutes=2),
         datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')+datetime.timedelta(minutes=2))    
plt.xlabel('Time')
plt.ylabel('Number of cluster')
fig4.autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
'''
#Plot type of cluster over time
plt.subplot(4, 1, 2,sharex=ax1)
plt.plot(time,yes_needles,'ro-')
plt.axhline(y=0,c="black",linewidth=0.8,zorder=0)
plt.axhline(y=100,c="black",linewidth=0.8,zorder=0)
plt.ylim(-10,110) 
plt.xlabel('Time')
plt.ylabel('Needles')
fig4.autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

plt.subplot(4, 1, 3, sharex=ax1)
plt.plot(time,yes_needlesaggr, 'go-')
plt.axhline(y=0,c="black",linewidth=0.8,zorder=0)
plt.axhline(y=100,c="black",linewidth=0.8,zorder=0)
plt.ylim(-10,110) 
plt.ylabel('Needles Aggregate')
fig4.autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

plt.subplot(4, 1, 4, sharex=ax1)
plt.plot(time,yes_graupeln,'bo-')
plt.axhline(y=0,c="black",linewidth=0.8,zorder=0)
plt.axhline(y=100,c="black",linewidth=0.8,zorder=0)
plt.ylim(-10,110) 
plt.ylabel('Graupeln')
fig4.autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
'''

