# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:12:41 2023

@author: u300737
"""
import sys
from metpy.calc import wind_components,wind_direction

from metpy.units import units
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def get_lambert_rotation(do_west=True):
    if do_west:
        region="west"
    else:
        region="east"
    wd_lon=np.array(xr.open_dataset(region+"_param_165.nc")["longitude"])
    print(wd_lon[0,0]) # check for lon >330 (east)
    wd_lat=np.array(xr.open_dataset(region+"_param_165.nc")["latitude"])
    wd_lam_u=np.array(xr.open_dataset(region+"_param_165.nc")["u10"])
    wd_lam_v=np.array(xr.open_dataset(region+"_param_166.nc")["v10"])
    # get lambert direction
    wd_lam_wdir  =np.array(wind_direction(wd_lam_u*units['m/s'],wd_lam_v*units['m/s']))
    wd_lam_wspeed=np.array(np.sqrt(wd_lam_u**2+wd_lam_v**2))
    
    # north south oriented
    wd_ne_wdir=np.array(xr.open_dataset(region+"_param_260260.nc")["wdir10"])
    wd_ne_wspeed=np.array(xr.open_dataset(region+"_param_207.nc")["si10"])
    wd_ne_u,wd_ne_v=wind_components(wd_ne_wspeed*units["m/s"],
                                    wd_ne_wdir*units["deg"])
    wd_ne_u=np.array(wd_ne_u)
    wd_ne_v=np.array(wd_ne_v)
    projection=ccrs.AzimuthalEquidistant(central_longitude=-5.0,
                                                 central_latitude=70)
    quiver_fig,axs=plt.subplots(1,2,figsize=(16,9),
                            subplot_kw={'projection': projection})
    # get Quivervalues
    step=100
    wd_quiver_lon=np.array(wd_lon[::step,::step])
    wd_quiver_lat=np.array(wd_lat[::step,::step])
    quiver_lam_u=wd_lam_u[::step,::step]
    quiver_lam_v=wd_lam_v[::step,::step]
    quiver_ne_u=wd_ne_u[::step,::step]
    quiver_ne_v=wd_ne_v[::step,::step]
    axs[0].coastlines(resolution="50m")
    axs[1].coastlines(resolution="50m")
    
    quiver_lam=axs[0].quiver(wd_quiver_lon,
                             wd_quiver_lat,
                         quiver_lam_u,quiver_lam_v,color="white",
                         edgecolor="k",lw=1,
                         scale_units="inches",
                         pivot="mid",width=0.01,
                         transform=ccrs.PlateCarree())
    
    quiver_ne=axs[1].quiver(wd_quiver_lon,
                            wd_quiver_lat,
                         quiver_ne_u,quiver_ne_v,color="white",
                         edgecolor="k",lw=1,
                         scale_units="inches",
                         pivot="mid",width=0.01,
                         transform=ccrs.PlateCarree())
    
    # calculate difference between both wdir
    diff_wd_wdir_ne_lam=wd_ne_wdir-wd_lam_wdir
    diff_wd_wdir_ne_lam_df=pd.DataFrame(diff_wd_wdir_ne_lam)
    new_wd_lam_wdir=wd_lam_wdir+diff_wd_wdir_ne_lam
    
    new_wd_lam_u,new_wd_lam_v=wind_components(wd_lam_wspeed*units["m/s"],
                                              new_wd_lam_wdir*units["deg"])
    new_wd_lam_u=np.array(new_wd_lam_u)
        #quiver_fig[0,0].set_title("Lambert \n CARRA")
        #quiver_fig[0,1].set_title("North \n East")
        
        #wd_u_v_wdir=np.atan2(#wd_wind
    file_name="Wdir_lam_to_ne_correction_"
    file_name+=region+".csv"
    diff_wd_wdir_ne_lam_df.to_csv(file_name)
    print("Lambert rotation angle saved as:",file_name)
##---------------------------------------------------------------------------##
# Out of CARRA (lambert rotated)
do_west=False
#if do_west:
get_lambert_rotation(do_west=do_west)