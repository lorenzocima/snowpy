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

from sklearn.cluster import KMeans,DBSCAN,MeanShift,estimate_bandwidth
from sklearn import mixture
from sklearn.metrics import silhouette_score
#from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist,pdist
from scipy import stats

##Data import from .txt file to data frame

filename='/home/cima/Documents/Helsinki Work/Data_BAECC2014_txt/Snow_particles_20140215_0216.txt'

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

start_time='2014-02-16 00:40:00'
end_time='2014-02-16 00:46:00'

data=data[(data['Date_Time']>=start_time) & (data['Date_Time']<=end_time)]
              
##Selection of piece of dataset with dimension of 1000 measurements
         
total_n_cluster=[]
time=[]

start=0
end=1000
while start<len(data):
     data1=data.iloc[start:end]
     


     normalize=StandardScaler().fit(data1[['Ddeq','V','Aratio','ASPratio']])
     normalized_data= normalize.transform(data1[['Ddeq','V','Aratio','ASPratio']])

     ###Silhouette Method for number of cluster 
     ###(to use only with period of time very short)->expensive for RAM
    
    
    
    
     ###Using K-Means
     silhouette_value_KM=[]
     silhouette_value_KM.append(0)
    
     silhouette_KM_max=0
     n_cluster_KM=0
        
     for i in range(2,9):
         clusterer = KMeans(n_clusters=i)
         cluster_labels = clusterer.fit_predict(normalized_data)
         silhouette_avg = silhouette_score(normalized_data, labels=cluster_labels)
         silhouette_value_KM.append(silhouette_avg)
         print("For n_clusters =", i,
              "The average silhouette_score is :", silhouette_avg)
         if (silhouette_avg > silhouette_KM_max):
             silhouette_KM_max=silhouette_avg
             n_cluster_KM=i
            
             print('Estimated number of cluster with K-Means ---> ', n_cluster_KM)
             
             #it's necessary a treshold to find 1 cluster
             if (silhouette_avg < 0.3):
                 n_cluster_KM=1
                 print('Correction number of cluster --->', n_cluster_KM)
             
         
     
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
     h=plt.hist2d(data1[x_variable],data1[y_variable], bins=80, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
     plt.colorbar()
     plt.axis([xmin, xmax, ymin, ymax])
     plt.xlabel(x_variable)
     plt.ylabel(y_variable)
     plt.title('K-Means All Data')
     #plt.grid(color='gray')
   
     plt.subplot(3, 1, 1)
     plt.plot(range(1,9), silhouette_value_KM,'ro-',label='K-Means')
     plt.axis([0,10,0,0.5])
     plt.xlabel('k')
     plt.ylabel('Average Silhouette')
     plt.title('Selecting k with the Average Silhouette Method')
     plt.text(1,0.4,n_cluster_KM)
     plt.text(0.4,0.4, 'cluster=')
     plt.show()
     
     
     km = KMeans(n_clusters=n_cluster_KM, init='k-means++',n_init=40, random_state=200)
     km.fit(normalized_data)
     labels=data1['KM_ID']=km.predict(normalized_data)
     KM_cluster0=data1[data1['KM_ID']==0]
     KM_cluster1=data1[data1['KM_ID']==1]
     KM_cluster2=data1[data1['KM_ID']==2]

     plt.subplot(3, 2, 4)
     h=plt.hist2d(KM_cluster0[x_variable],KM_cluster0[y_variable], bins=80, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
     plt.colorbar()
     plt.axis([xmin, xmax, ymin, ymax])
     plt.xlabel(x_variable)
     plt.ylabel(y_variable)
     plt.title('K-Means Cluster 0')
     
     plt.subplot(3, 2, 5)
     h=plt.hist2d(KM_cluster1[x_variable],KM_cluster1[y_variable], bins=80, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
     plt.colorbar()
     plt.axis([xmin, xmax, ymin, ymax])
     plt.xlabel(x_variable)
     plt.ylabel(y_variable)
     plt.title('K-Means Cluster 1')
     
     plt.subplot(3, 2, 6)
     h=plt.hist2d(KM_cluster2[x_variable],KM_cluster2[y_variable], bins=80, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
     plt.colorbar()
     plt.axis([xmin, xmax, ymin, ymax])
     plt.xlabel(x_variable)
     plt.ylabel(y_variable)
     plt.title('K-Means Cluster 2')
         
     total_n_cluster.append(n_cluster_KM)
     time.append(data1['Date_Time'].iloc[len(data1)//2])
     start=end+1
     end=end+1001

#add 30 seconds for removing time-step sovrapposition
for t in range(1,len(time)):
    if (time[t]==time[t-1]):
        time[t]=time[t]+datetime.timedelta(seconds=30)
        
       
fig4=plt.figure()
plt.plot(time, total_n_cluster, 'ko')
plt.ylim(0,5)     
plt.xlabel('Time')
plt.ylabel('Number of cluster')




     
'''               
     ###Using Gaussian Mixture

     silhouette_value_GM=[]
     silhouette_value_GM.append(0)
    
     silhouette_GM_max=0
     n_cluster_GM=0

     for l in range(2,9):
         clusterer = mixture.GaussianMixture(n_components=l)
         cluster_labels_1 = clusterer.fit(data1[['Ddeq','V','Aratio','ASPratio']])
         cluster_labels = clusterer.predict(data1[['Ddeq','V','Aratio','ASPratio']])
         silhouette_avg = silhouette_score(data1[['Ddeq','V','Aratio','ASPratio']], labels=cluster_labels)
         silhouette_value_GM.append(silhouette_avg)
         #print("For n_clusters =", l,
         #     "The average silhouette_score is :", silhouette_value_GM)
         if (silhouette_avg > silhouette_GM_max):
             silhouette_GM_max=silhouette_avg
             n_cluster_GM=l
        
         print('Estimated number of cluster with Gaussian Mixture ---> ', n_cluster_GM)
    

     plt.plot(range(1,9), silhouette_value_GM,'bx-', label='Gaussian Mixture')
     plt.axis([0,10,0,0.5])
     plt.xlabel('k')
     plt.ylabel('Average Silhouette')
     plt.title('Selecting k with the Average Silhouette Method')
     plt.legend()
     plt.show()

     
     
'''             
          



'''
variable='V'

figx=plt.figure()
GM_cluster0[variable].plot.density(color='red')
GM_cluster1[variable].plot.density(color='blue')
GM_cluster2[variable].plot.density(color='green')

#bandwidth=estimate_bandwidth(normalized_data, quantile=0.045)

#ms=MeanShift(bandwidth=bandwidth,bin_seeding=True)
#ms.fit(data[['Ddeq','V','Aratio','ASPratio']])

labels1=km.labels_
cluster_centers=km.cluster_centers_
labels_unique=np.unique(labels1)
n_clusters_=len(labels_unique)
print('Number of estimated clusters:', n_clusters_)

#data.plot(kind='scatter', x='Ddeq',y='Aratio', c=labels1, colormap='Set1')


##Find centroid(a,b) of cluster 

a=np.sum(KM_data_set[['Ddeq']])/len(KM_cluster2)
b=np.sum(KM_data_set[['Aratio']])/len(KM_cluster2)
plt.scatter(a,b)
'''

'''
#Add fit with plot
x = np.linspace(0.2,5,100) # 100 linearly spaced numbers
y = 0.73*(x)**0.06 # function
plt.plot(x,y, color='black')
'''


'''

###Elbow method to understand the correct number of cluster (no precise)

meandistorsion =[]
for k in range(1,11):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(normalized_data)
    meandistorsion.append(sum(np.min(cdist(normalized_data,
    kmeans.cluster_centers_,'euclidean'),axis=1))/normalized_data.shape[0])
    if (k>1):
        a=meandistorsion[k-1]-meandistorsion[k-2]
        print('Slope of adiacent point', a)
            
        if (meandistorsion[k-1]-meandistorsion[k-2]>-0.0999):
                print('The number of cluster is', k-2)
                          
    
    
fig2=plt.figure()
plt.plot(range(1,11),meandistorsion,'bx-')
plt.axis([0,12,0.8,2])
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()
'''


'''

###DBSCAN to find number of cluster (too expensive for RAM)

eps_grid=np.linspace(0.3,1.2,num=10)
silhouette_scores=[]
eps_best=eps_grid[0]
silhouette_score_max=-1
model_best=None
labels_best=None

for eps in eps_grid:
    model=DBSCAN(eps=eps, min_samples=5).fit(data[['Ddeq','V','Aratio','ASPratio']])
    labels=model.labels_
    silhouette_scores.append(silhouette_score)
    print('Epsilon:',eps, '---> silhouette score:',silhouette_score)
    if silhouette_score>silhouette_score_max:
        silhouette_score_max=silhouette_score
        eps_best=eps
        model_best=model
        labels_best=labels
        

print('Best epsilon=', eps_best)

model=model_best
labels=labels_best

offset=0
if -1 in labels:
    offset=-1
    
num_cluster=len(set(labels))-offset

print('Estimated number of cluster=',num_cluster)


'''