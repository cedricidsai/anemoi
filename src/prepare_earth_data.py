import xarray as xr
import numpy as np


start = '2014-01-01'

end = '2014-12-31'

gt = xr.open_dataset('../data/earth_gt.nc')

bench = gt.sel(time=slice(start, end))

bench = bench.swap_dims({'lat':'latitude'})
bench = bench.swap_dims({'lon':'longitude'})

bench = bench.rename_dims({'level':'pressure'})
bench = bench.rename_vars({'level':'pressure'})

bench = bench.drop_vars([ 'geopotential_at_surface',
 'land_sea_mask',
 'time',
 '2m_temperature',
 'mean_sea_level_pressure',
 '10m_v_component_of_wind',
 '10m_u_component_of_wind',
 'specific_humidity'
 ])

bench.to_netcdf('../data/earth_train.nc')

print(bench)

start = '2015-01-01'

end = '2015-01-31'

bench = gt.sel(time=slice(start, end))
bench = bench.swap_dims({'lat':'latitude'})
bench = bench.swap_dims({'lon':'longitude'})

bench = bench.rename_dims({'level':'pressure'})
bench = bench.rename_vars({'level':'pressure'})

bench = bench.drop_vars([ 'geopotential_at_surface',
 'land_sea_mask',
 'time',
 '2m_temperature',
 'mean_sea_level_pressure',
 '10m_v_component_of_wind',
 '10m_u_component_of_wind',
 'specific_humidity'
 ])

bench.to_netcdf('../data/earth_test.nc')


# toa = xr.open_mfdataset('../data/toa/toa_solar_radiation_2014.nc')

# toa = toa.sel(time=slice(start, end))

# toa = toa.interp(longitude=bench.coords['lon'], latitude=bench.coords['lat'], kwargs={"fill_value": "extrapolate"})

# # geo = bench['geopotential_at_surface']
# # land_sea = bench['land_sea_mask']

# bench['toa_incident_solar_radiation'] = toa['tisr']
# del toa
