#!/bin/bash

#SBATCH --nodes=1

#SBATCH --partition=small

#SBATCH --gres=gpu:1

#SBATCH --ntasks-per-node=1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=cedric.pixel7@gmail.com

source ~/.bashrc

conda activate anemoi

python3 ex3.py makani non-linear ../data/earth_gt.nc ../data_earth_test.nc 10 4 "air_temperature,upward_air_velocity,geopotential_height,x_wind,y_wind"
