#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:50:10 2017

@author: cima
"""
FFMPEG_BIN = "ffmpeg"
import subprocess as sp
import numpy
import matplotlib.pyplot as plt

command = [ FFMPEG_BIN,
            '-i', '/home/cima/Documents/Helsinki Work/PIP avi/February1516/004201402152300_q.avi',
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

# read 420*360*3 bytes (= 1 frame)
raw_image = pipe.stdout.read(720 * 576)
# transform the byte read into a numpy array
image =  numpy.fromstring(raw_image, dtype='uint8')
# throw away the data in the pipe's buffer.
pipe.stdout.flush()
plt.imshow( image )