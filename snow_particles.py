# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

filename='Snow_particles_20141106_1107.txt'

column_names=['Date_Time', 'Ddeq(mm)', 'V(m/s)', 'A(mm^2)', 
'Dmin(mm)', 'Dmaj(mm)', 'Dmax(mm)', 'mass(mg)', 'W(mm)', 'H(mm)']

data=pd.read_table(filename, delim_whitespace=True, parse_dates=[['time','Ddeq']]).head(1000)

data.columns=column_names

data.plot(x='Ddeq(mm)', y='V(m/s)', kind='scatter',color='Black', title='Title')

#data.plot(x='Date_Time', y='mass(mg)', kind='line', color='Blue')


