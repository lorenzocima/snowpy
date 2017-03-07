# coding: utf-8
'''
 n = needles
 na = needles aggregate
 da = dense aggregate
 g = graupeln
'''
##Necessary put video every 5 minutes (start with file x0 or x5) 
##to compare directly with psd file

import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import sys

home = os.path.expanduser('~')

#filename=os.path.join(home ,'Documents/Helsinki Work/IC-PCA/Particle/classes/classes.csv')
filename=os.path.join(home ,'DATA/snow/classes.csv')


data=pd.read_csv(filename)

#Remove from data the bad crop particle
data=data[data['flags']!='bad_crop;rectangular']
data=data[data['flags']!='bad_crop']

small=data.loc[data['5nn'] == 's']
needles=data.loc[data['5nn'] == 'n']
needles_aggr=data.loc[data['5nn'] == 'na']
dense_aggr=data.loc[data['5nn'] == 'da']
graupeln=data.loc[data['5nn'] == 'g']
unclass=data.loc[data['5nn'] == 'un']


def move_particles():
    particle_folder = os.path.join(home, 'Documents/Helsinki Work/IC-PCA/Particle')
    
    small_folder=os.path.join(particle_folder, 'Small_final')
    unclass_folder=os.path.join(particle_folder, 'Unclass_final')
    needles_folder=os.path.join(particle_folder, 'Needles_final')
    needles_aggr_folder=os.path.join(particle_folder, 'Needlesaggr_final')
    dense_aggr_folder=os.path.join(particle_folder, 'Denseaggr_final')
    graupeln_folder=os.path.join(particle_folder, 'Graupeln_final')
    
    folders=[needles_folder,needles_aggr_folder,dense_aggr_folder,graupeln_folder,small_folder,unclass_folder]
    
    ###Exit from script if there are not image to analyze in Picture folder
    if not any (fname.endswith('.png') for fname in os.listdir(particle_folder)):
        sys.exit('Error message!\n There are no images to analyze in Picture folder\n Probably is sufficient to execute the plot')
           
    ###Eliminate all the image in all folder classification to subscribe the new 
    for ii in folders:      
    
        for image in os.listdir(ii):
            os.remove(os.path.join(ii,image))
    
            
    shape=[needles,needles_aggr,dense_aggr,graupeln,small,unclass]        
    
    ###Move all the particle in the correct folder    
        
    for particle_filename in os.listdir(particle_folder):
        print('Moving process...')
        
        a=-1
        for j in shape:
            a=a+1
            for i in range(0,len(j)):
                if(particle_filename==(j['name'].iloc[i]+'.png')):
                    shutil.move(os.path.join(particle_folder,particle_filename),folders[a])
                    if(j['flags'].iloc[i]=='bad_crop;rectangular' or j['flags'].iloc[i]=='bad_crop'):
                        os.remove(os.path.join(particle_folder,folders[a],particle_filename))
                    
    print('End of moving process, all the particle are (probably) in the correct folder')           


#########################################
################PLOT#####################
#########################################

###
###Condition for the plot -> all the particle or without small and unclass
###

all_particle = True  ### True for all the particle 
                     ### False only particle of interest

percentage = False  ### True for the percentage
                    ### False for the number of particle  


n_needles = []
n_needles_aggr = []
n_graupeln = []
n_small = []
n_unclass = []
total = []
number = [n_needles, n_needles_aggr,n_graupeln,n_small,n_unclass]
time = []
period = 0


for k in range(1, len(data)):
    
    current_time=data['name'].iloc[k][0:data['name'].iloc[k].index('-')]
    previous_time=data['name'].iloc[k-1][0:data['name'].iloc[k-1].index('-')]                  
    
    if(k==1):
        tstart=current_time                 
    
    if (current_time!=previous_time):
        time.append(datetime.datetime.strptime(previous_time, '%Y%m%d%H%M'))
        
        if all_particle:
            n_total=len(needles.loc[period:k-1])+len(needles_aggr.loc[period:k-1])+len(graupeln.loc[period:k-1])+len(dense_aggr.loc[period:k-1])+len(small.loc[period:k-1])+len(unclass.loc[period:k-1])
        else:
            n_total=len(needles.loc[period:k-1])+len(needles_aggr.loc[period:k-1])+len(graupeln.loc[period:k-1])+len(dense_aggr.loc[period:k-1])            
        

        n_needles.append(len(needles.loc[period:k-1]))
        n_needles_aggr.append(len(needles_aggr.loc[period:k-1]))
        n_graupeln.append((len(graupeln.loc[period:k-1])+len(dense_aggr.loc[period:k-1])))
        n_small.append(len(small.loc[period:k-1]))
        n_unclass.append(len(unclass.loc[period:k-1]))
        total.append(int(n_total))
        period=k
        
    ###Last part of the data
    if (k==len(data)-1):
        time.append(datetime.datetime.strptime(current_time, '%Y%m%d%H%M'))
        
        if(all_particle=='yes'):
            n_total=len(needles.loc[period::])+len(needles_aggr.loc[period::])+len(graupeln.loc[period::])+len(dense_aggr.loc[period::])+len(small.loc[period::])+len(unclass.loc[period::])
        elif(all_particle=='no'):
            n_total=len(needles.loc[period::])+len(needles_aggr.loc[period::])+len(graupeln.loc[period::])+len(dense_aggr.loc[period::])
       
        n_needles.append(len(needles.loc[period::]))
        n_needles_aggr.append(len(needles_aggr.loc[period::]))
        n_graupeln.append((len(graupeln.loc[period::])+len(dense_aggr.loc[period::])))
        n_small.append(len(small.loc[period::]))
        n_unclass.append(len(unclass.loc[period::]))
        total.append(int(n_total))

        tend=current_time


###Percentage or number of particle      
        
if percentage:
    total=[x/100 for x in total]

    n_needles=[x/y for x,y in zip(n_needles,total)]
    n_needles_aggr=[x/y for x,y in zip(n_needles_aggr,total)]
    n_graupeln=[x/y for x,y in zip(n_graupeln,total)]
    n_small=[x/y for x,y in zip(n_small,total)]
    n_unclass=[x/y for x,y in zip(n_unclass,total)]

    ylimit=100 #limit of percentage
    ylab='Percentage [%]'           
else:
    ylimit=len(data)/len(time) #limit to see the line of the plot well
    ylab='Number of particle'        
        
        
fig=plt.figure()
plt.plot(time,n_needles,'b-o',label='Needles')
plt.plot(time,n_needles_aggr,'r-o', label='Needles aggregate')
plt.plot(time,n_graupeln,'k-o', label='Graupel')   
if all_particle:
    plt.plot(time,n_small,'y-o', label='Small')
    plt.plot(time,n_unclass,'g-o', label='Unclass')
plt.ylim(0, ylimit)
plt.xlim(datetime.datetime.strptime(tstart, '%Y%m%d%H%M')-datetime.timedelta(minutes=2),
         datetime.datetime.strptime(tend, '%Y%m%d%H%M')+datetime.timedelta(minutes=2))       
plt.xlabel('Time')
plt.ylabel(ylab)
fig.autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.legend(loc='upper left', ncol=3)
plt.show()

###    
###Use of PSD to calculate the concentration number of the particle
###

def plot_psd():
    ###To use is necessary to set 5 minute step in data frame video
    filename_psd = os.path.join(home, 'Documents/Helsinki Work/Data_BAECC2014_txt/SnowPSD_20141106_1107.txt')
    
    column_names = ['Date_Time', 'D0','N0','Lambda', 
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
    
    data_psd=data_psd[(data_psd['Date_Time']>=tstart) & (data_psd['Date_Time']<=tend)]
    
    nt=[]
    vobs=[]
    nt_needles=[]
    nt_needles_aggr=[]
    nt_graupeln=[]
    
    for i in range (0, len(data_psd)):
        nt.append((data_psd['N0'].iloc[i])*(data_psd['Lambda'].iloc[i])**-1)
        vobs.append(total[i]/nt[i])
            
        nt_needles.append(n_needles[i]/(vobs[i]))
        nt_needles_aggr.append(n_needles_aggr[i]/(vobs[i]))
        nt_graupeln.append(n_graupeln[i]/(vobs[i]))
    
    plt.figure()
    plt.plot(time,nt_needles,'b-o',label='Nt_n')
    plt.plot(time,nt_needles_aggr,'r-o',label='Nt_na')
    plt.plot(time,nt_graupeln,'k-o',label='Nt_g')
    plt.xlabel('Time')
    plt.ylabel('Number concentration [m**-3]')            
    plt.legend(loc='upper left', ncol=3)
    #plt.yscale('log')

#Plot nt_n+nt_na and plot only n_t
'''
fig2=plt.figure()
nt_nna=[sum(x) for x in zip(nt_needles, nt_needles_aggr)]
plt.plot(time,nt_nna,'m-o',label='Nt_n+Nt_na')
plt.plot(time,nt_graupeln,'k-o',label='Nt_g')
plt.xlabel('Time')
plt.ylabel('Number concentration [m**-3]')            
plt.legend(loc='upper left', ncol=3)
#plt.yscale('log')   

fig3=plt.figure()
plt.plot(time,nt,'y-o',label='Nt')
plt.xlabel('Time')
plt.ylabel('Number concentration [m**-3]')            
plt.legend(loc='upper left', ncol=3)
#plt.yscale('log')    
'''    
