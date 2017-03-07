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

start_time='2014-02-15 21:00:00'
end_time='2014-02-16 02:00:00'

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
     if(len(data1)<(step//2)):
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
     n_cluster_KM=0
        
     for i in range(2,4):
         clusterer = KMeans(n_clusters=i)
         cluster_labels = clusterer.fit_predict(normalized_data)
         silhouette_avg = silhouette_score(normalized_data, labels=cluster_labels)
         silhouette_value_KM.append(silhouette_avg)
         #print("For n_clusters =", i,
         #     "The average silhouette_score is :", silhouette_avg)
         if (silhouette_avg > silhouette_KM_max):
             silhouette_KM_max=silhouette_avg
             n_cluster_KM=i
            
             print('Estimated number of cluster with K-Means ---> ', n_cluster_KM)
             
             #it's necessary a treshold to find 1 cluster
             #treshold means that the silhouette coefficient isn't
             #sufficient elevate to ensure the accuracy of result of number
             #of cluster find with the algorithm
             if (silhouette_KM_max < 0.29):
                 n_cluster_KM=1
                 print('Correction number of cluster --->', n_cluster_KM)
                 
         if(i==3):       
             if(silhouette_KM_max-silhouette_avg<0.05 and silhouette_avg>0.3
                and silhouette_KM_max<0.4):
                 n_cluster_KM=i
                 print('Another correction number of cluster --->', n_cluster_KM)        
         
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
     plt.title('K-Means All Data')
     #plt.grid(color='gray')
   
     plt.subplot(3, 1, 1)
     plt.plot(range(1,4), silhouette_value_KM,'ro-',label='K-Means')
     plt.axis([0,10,0,0.5])
     plt.xlabel('k')
     plt.ylabel('Average Silhouette')
     plt.title('Selecting k with the Average Silhouette Method')
     plt.text(1,0.4,n_cluster_KM)
     plt.text(0.4,0.4, 'cluster=')
     plt.show()
     
     '''
     km = KMeans(n_clusters=n_cluster_KM, init='k-means++',n_init=40)
     km.fit(normalized_data)
     labels=data1['KM_ID']=km.predict(normalized_data)
     
     KM_cluster0=data1[data1['KM_ID']==0]
     KM_cluster1=data1[data1['KM_ID']==1]
     KM_cluster2=data1[data1['KM_ID']==2]
      
 
     ##Calculate centroid position to assign 
     ##the correct cluster at the correct snow particle
     
     Ddeq_centroid=[]
     Aratio_centroid=[] 
     V_centroid=[]
   
     Ddeqc0=np.sum(KM_cluster0[['Ddeq']])/len(KM_cluster0)
     Ddeqc1=np.sum(KM_cluster1[['Ddeq']])/len(KM_cluster1)
     Ddeqc2=np.sum(KM_cluster2[['Ddeq']])/len(KM_cluster2)
     
     
     Aratioc0=np.sum(KM_cluster0[['Aratio']])/len(KM_cluster0)
     Aratioc1=np.sum(KM_cluster1[['Aratio']])/len(KM_cluster1)
     Aratioc2=np.sum(KM_cluster2[['Aratio']])/len(KM_cluster2)
     
     Vc0=np.sum(KM_cluster0[['V']])/len(KM_cluster0)
     Vc1=np.sum(KM_cluster1[['V']])/len(KM_cluster1)
     Vc2=np.sum(KM_cluster2[['V']])/len(KM_cluster2)
            
     Ddeq_centroid=(Ddeqc0[0],Ddeqc1[0],Ddeqc2[0])
     Aratio_centroid=(Aratioc0[0],Aratioc1[0],Aratioc2[0])
     V_centroid=(Vc0[0],Vc1[0],Vc2[0])
     
     ##Combination of condition on V, Aratio, Ddeq to classify the particle 
     '''
     if(n_cluster_KM==3):
         
         if(Aratio_centroid.index(min(Aratio_centroid))==V_centroid.index(min(V_centroid))
             or Aratio_centroid.index(min(Aratio_centroid))==V_centroid.index(statistics.median(V_centroid))):
             needles=Aratio_centroid.index(min(Aratio_centroid))
         if(Aratio_centroid.index(max(Aratio_centroid))==Ddeq_centroid.index(statistics.median(Ddeq_centroid))
             or Aratio_centroid.index(max(Aratio_centroid))==Ddeq_centroid.index(min(Ddeq_centroid))):
             graupeln=Aratio_centroid.index(max(Aratio_centroid))
         if(Aratio_centroid.index(statistics.median(Aratio_centroid))==Ddeq_centroid.index(max(Ddeq_centroid))):
             needlesaggr=Aratio_centroid.index(statistics.median(Aratio_centroid))
     '''
     
     if(n_cluster_KM==2 or n_cluster_KM==1 or n_cluster_KM==3):
         
         if(Aratio_centroid.index(min(Aratio_centroid))==V_centroid.index(min(V_centroid)) 
             or Aratio_centroid.index(min(Aratio_centroid))==V_centroid.index(statistics.median(V_centroid)) and
            min(Aratio_centroid)<0.7 and min(V_centroid)<1.2):
             needles=Aratio_centroid.index(min(Aratio_centroid))
         if(Aratio_centroid.index(max(Aratio_centroid))==Ddeq_centroid.index(statistics.median(Ddeq_centroid)) 
             or Aratio_centroid.index(max(Aratio_centroid))==Ddeq_centroid.index(min(Ddeq_centroid)) and
            max(Aratio_centroid)>0.7):
             graupeln=Aratio_centroid.index(max(Aratio_centroid))
         if(Aratio_centroid.index(statistics.median(Aratio_centroid))==Ddeq_centroid.index(max(Ddeq_centroid)) and 
            statistics.median(Aratio_centroid)<0.7 and max(Ddeq_centroid)>1.5):
             needlesaggr=Aratio_centroid.index(statistics.median(Aratio_centroid))
         
     '''    
     plt.subplot(3, 2, 4)
     h=plt.hist2d(KM_cluster0[x_variable],KM_cluster0[y_variable], bins=120, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
     plt.colorbar()
     plt.axis([xmin, xmax, ymin, ymax])
     plt.xlabel(x_variable)
     plt.ylabel(y_variable)
     if(needles==0):
         plt.title('K-Means Needles')
         yes_needles.append(100)
     if(needlesaggr==0):
         plt.title('K-Means Needles Aggregate')
         yes_needlesaggr.append(100)
     if(graupeln==0):
         plt.title('K-Means Graupeln')
         yes_graupeln.append(100)
     if(needles==50 and needlesaggr==50 and graupeln==50):
         plt.text(1, 0.5, 'Clustering Error')
         
     plt.subplot(3, 2, 5)
     h=plt.hist2d(KM_cluster1[x_variable],KM_cluster1[y_variable], bins=120, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
     plt.colorbar()
     plt.axis([xmin, xmax, ymin, ymax])
     plt.xlabel(x_variable)
     plt.ylabel(y_variable)
     if(needles==1):
         plt.title('K-Means Needles')
         yes_needles.append(100)
     if(needlesaggr==1):
         plt.title('K-Means Needles Aggregate')
         yes_needlesaggr.append(100)
     if(graupeln==1):
         plt.title('K-Means Graupeln')
         yes_graupeln.append(100)
     if(needles==50 and needlesaggr==50 and graupeln==50):
         plt.text(1,0.5,'Clustering Error')
         
     plt.subplot(3, 2, 6)
     h=plt.hist2d(KM_cluster2[x_variable],KM_cluster2[y_variable], bins=120, normed=True,
             vmin=10**-60, vmax=1, range=np.array([(xmin, xmax), (ymin, ymax)]),cmap=my_cmap)
     plt.colorbar()
     plt.axis([xmin, xmax, ymin, ymax])
     plt.xlabel(x_variable)
     plt.ylabel(y_variable)
     if(needles==2):
         plt.title('K-Means Needles')
         yes_needles.append(100)
     if(needlesaggr==2):
         plt.title('K-Means Needles Aggregate')
         yes_needlesaggr.append(100)
     if(graupeln==2):
         plt.title('K-Means Graupeln')
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
     
     total_n_cluster.append(n_cluster_KM)
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
plt.plot(time, total_n_cluster, 'ko--')
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