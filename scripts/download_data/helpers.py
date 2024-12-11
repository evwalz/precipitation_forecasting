import numpy as np
import xarray as xr
import pandas as pd
from cdo import Cdo
import os
import subprocess

def sel_grid(data):
    data_lat = data.sel(latitude = np.arange(18.5))
    data_lat_lon = data_lat.sel(longitude = np.arange(-25, 35.5))
    return data_lat_lon

def con_grid(input_file, output_file, target_grid_file):
    # Initialize CDO
    cdo = Cdo()
    #target_grid_file
    cdo.remapcon(target_grid_file, input=input_file, output=output_file)


def get_shear(data_dir):
    u925_file = xr.open_dataset(data_dir + 'era5_u_component_of_wind_925.nc')
    u600_file = xr.open_dataset(data_dir + 'era5_u_component_of_wind_600.nc')
    v925_file = xr.open_dataset(data_dir + 'era5_v_component_of_wind_925.nc')
    v600_file = xr.open_dataset(data_dir + 'era5_v_component_of_wind_600.nc')
    u925 = sel_grid(u925_file)
    u600 = sel_grid(u600_file)
    v925 = sel_grid(v925_file)
    v600 = sel_grid(v600_file)
    z = 925 - 600
    diff_u = u925.squeeze().u - u600.squeeze().u
    diff_v = v925.squeeze().v - v600.squeeze().v
    diff_u2 = diff_u ** 2
    diff_v2 = diff_v ** 2
    diff_mag = (diff_u2 + diff_v2)**(0.5)
    shear = diff_mag / z
    # Create a new DataArray for shear
    shear_da = xr.DataArray(
        data=shear, 
        dims=u925.squeeze().dims,  # Use the same dimensions as diff_mag
        coords=u925.squeeze().coords,  # Copy coordinates from diff_mag
        name="shear"  # Give it a name
    )

    return shear_da


def get_pressure_tendency(data_dir):
    data = xr.open_dataset(data_dir + 'era5_surface_pressure.nc')
    data = sel_grid(data)
    time_var_all = data.valid_time.values
    data_original = data.sel(valid_time = time_var_all[1:])
    data_prev = data.sel(valid_time = time_var_all[0:-1])
    data_prev = data_prev.assign({'valid_time': time_var_all[1:]})
    pressure_tendency = data_original - data_prev
    return pressure_tendency


def sel_grid_stream(data):
    x1 = np.arange(335, 360, 1)
    x2 = np.arange(35.5)
    x = np.concatenate((x1, x2))
    data_lat = data.sel(lat = np.arange(18.5))
    data_lat_lon = data_lat.sel(lon = x)
    data_lat_lon = data_lat_lon.assign({'lon': np.arange(-25, 35.5)})
    data_lat_lon = data_lat_lon.squeeze().drop('pressure_level')
    return data_lat_lon


# streamline, shear and pressure tendency
# cdo commands for streamline:
# cdo -b 32 -setname,svo relvorticirty-700.nc
#  cdo -merge relvor700_svo.nc zerodiv_sd.nc svosd.nc
# cdo -b 32 -selvar,stream -sp2gp -dv2ps -gp2sp -remapcon,t479grid svosd.nc stream_remapcon.nc
# cdo -b 32 -remapcon,grid.txt stream_remapcon.nc stream_remapcon_1_1.nc


#cdo -b 32 -remapcon,grid.txt stream_remapcon.nc stream_remapcon_1_1.nc
#cdo -b 32 -selvar,stream -sp2gp -dv2ps -gp2sp -remapcon,t479grid svosd.nc stream_remapcon.nc
#cdo -merge relvor700_svo.nc zerodiv_sd.nc svosd.nc
#cdo -b 32 -setname,svo adaptor.mars.internal-1678110728.1012454-21015-15-5d4c5c22-b210-444d-9f3a-ec2b954466ac.nc relvor700_svo.nc

def get_stream(data_dir):
    os.chdir(data_dir)
    # Define the CDO commands as a list of strings
    commands = [
        "cdo -b 32 -setname,svo era5_vorticity_700.nc era5_vorticity_700_svo.nc",
        "cdo -L -b 32 -chname,svo,sd -mulc,0 era5_vorticity_700_svo.nc era5_zerodiv.nc",
        "cdo -merge era5_vorticity_700_svo.nc era5_zerodiv.nc era5_svosd.nc",
        "cdo -L -b 32 -remapbil,stream_grid.txt -selvar,stream -sp2gp -dv2ps -gp2sp -remapbil,t511grid era5_svosd.nc era5_stream.nc"
        ]
    # Execute each command
    for cmd in commands:
        try:
            print(f"Running command: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            print("Command completed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command: {cmd}")
            print(e)
            break
    data = xr.open_dataset(data_dir + 'era5_stream.nc')
    data = sel_grid_stream(data)
    return data

def accum_vimd(data_dir):
    variable = 'vertically_integrated_moisture_divergence'
    data = xr.open_dataset(data_dir + 'era5_'+variable+'_regrid.nc')
    daily_sum = data.groupby(data.valid_time.dt.date).sum()
    all_time_values = data.valid_time.values
    
    daily_time = all_time_values[np.arange(24, len(all_time_values), 24)]
    last_day = daily_time[-1] + np.timedelta64(1, "D")
    all_daily_time = np.append(daily_time, last_day)

    ds_time = daily_sum.rename({'date':'valid_time'})
    ds_time = ds_time.assign({'valid_time': all_daily_time})
    return ds_time


def get_name(variable):
    if variable == 'total_column_water_vapour':
        name = 'tcwv'
    elif variable == 'convective_available_potential_energy':
        name = 'cape'
    elif variable == 'vertically_integrated_moisture_divergence':
        name = 'vimd'
    elif variable == 'total_column_cloud_liquid_water':
        name = 'cloudwater'
    elif variable == 'total_cloud_cover':
        name = 'cloudcover'
    elif variable == 'convective_inhibition':
        name = 'cin'
    elif variable == 'k_index':
        name = 'kindx'
    elif variable == '2m_temperature':
        name = 't2m'
    elif variable == '2m_dewpoint_temperature':
        name = 'd2m'
    elif variable == 'pressure_tendency':
        name = 'pressure_tendency'
    elif variable == 'temperature_850':
        name = 'temp_850'
    elif variable == 'temperature_500':
        name = 'temp_500'
    elif variable == 'specific_humidity_925':
        name = 'spec_humid925'
    elif variable == 'relative_humidity_300':
        name = 'rh300'
    elif variable == 'relative_humidity_500':
        name = 'rh500'
    elif variable == 'specific_humidity_600':
        name = 'spec_humid600'
    elif variable == 'specific_humidity_700':
        name = 'spec_humid700'
    elif variable == 'specific_humidity_500':
        name = 'spec_humid500'
    elif variable == 'shear':
        name = 'shear925_600'
    elif variable == 'stream':
        name = 'stream'
    else:
        raise ValueError('name does not exist')
    return name


