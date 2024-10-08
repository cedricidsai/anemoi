import xarray as xr
wb = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')

# gc_era5_ex = xr.load_dataset('../data/datasets/dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc')
# from datetime import datetime

# now = datetime.now()
# day = now.strftime("%Y-%m-%d")
# print(day)

import numpy as np

day = '2015-01-31'

degrees = 1

ground_truth = wb.sel(time=slice('2014-01-01', day))
ground_truth = ground_truth.interp(longitude=np.arange(0,360,degrees), latitude=np.arange(-90,91,degrees), kwargs={"fill_value": "extrapolate"})
ground_truth[[
 'geopotential_at_surface',
 'land_sea_mask',
 'level',
 'time',
 '2m_temperature',
 'mean_sea_level_pressure',
 '10m_v_component_of_wind',
 '10m_u_component_of_wind',
 'total_precipitation_6hr',
 'temperature',
 'geopotential',
 'u_component_of_wind',
 'v_component_of_wind',
 'vertical_velocity',
 'specific_humidity']].to_netcdf('../data/earth_gt_%i.nc'%degrees)
