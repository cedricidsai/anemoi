#!/bin/bash

#SBATCH --nodes=1

#SBATCH --partition=small

#SBATCH --gres=gpu:1

#SBATCH --ntasks-per-node=1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=cedric.pixel7@gmail.com

source ~/.bashrc

conda activate anemoi

python3 ex3.py non-linear ../data/earth_train.nc ../data_earth_test.nc 8 512 5 10 "air_temperature,upward_air_velocity,geopotential_height,x_wind,y_wind"
