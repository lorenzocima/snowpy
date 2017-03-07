#!/bin/bash
# My first script

echo "###################################################################"
echo "###################################################################"
echo "Processing Matlab script"
echo "Video process -> Frame extrapolation -> Snow particle extrapolation"
echo "###################################################################"
echo "###################################################################"
sleep 5

matlab -nodesktop -nosplash -r "run('/home/cima/Documents/Helsinki Work/IC-PCA/extraction_snowflake.m');quit"

echo "##############################################################################"
echo "##############################################################################"
echo "Move snow particle in correct folder -> Plot snow particle evolution with time"
echo "##############################################################################"
echo "##############################################################################"
sleep 5

cd /home/cima/Documents/Helsinki Work/Python/
python ./particle_classes.py
