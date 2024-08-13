#!/bin/bash

#SBATCH --nodes=1

#SBATCH --partition=small

#SBATCH --gres=gpu:1

#SBATCH --ntasks-per-node=1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=cedric.pixel7@gmail.com

source ~/.bashrc

conda activate anemoi

python3 anemoi.py non-linear ../data/earth_train.nc ../data/earth_test.nc 8 512 5 10 "temperature,geopotential,vertical_velocity,u_component_of_wind,v_component_of_wind"
