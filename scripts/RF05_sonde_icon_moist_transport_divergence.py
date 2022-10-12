# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 10:41:54 2022

@author: u300737
"""

import os
import sys

import pandas as pd
import numpy as np
import xarray as xr

import scipy.interpolate as scint

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

processing_subf="hamp_processing_python/"
current_path=os.getcwd()
major_path=current_path+"/../../../Work/GIT_Repository/"
script_path=current_path+"/../scripts/"
halo_ac3_path=current_path+"/../../"+processing_subf
device_path=halo_ac3_path+"/src/"
plot_device_path=current_path+"/../../"+processing_subf+"/plotting/"
config_path=current_path+"/../config/"
sys.path.insert(1,script_path)
sys.path.insert(2,halo_ac3_path)
sys.path.insert(3,device_path)
sys.path.insert(4,config_path)
# campaigns_path packages
import flightcampaign
import moisturebudget as Budgets

# processing_path packages
import data_config

import measurement_instruments_ql as Measurement_Instruments_QL
#import halodataplot
import quicklook_dicts as Quicklook_Dicts
#sys.exit()
flight="RF05"
date="20220315"

campaign="HALO_AC3"
device_data_path=major_path+"hamp_processing_py/"+processing_subf

campaign_path=major_path+"/"+campaign+"/"

radar_dict={}
bahamas_dict={}  
#plot_path=current_path+"/Plots/"
#if not os.path.exists(plot_path):
#    os.makedirs(plot_path)
# Plot LDR and melting layer
font_size=26
matplotlib.rcParams.update({"font.size":font_size})    

#######################################################################
#######################################################################
### Processed radar
# Radar reflectivity
cfg_dict=Quicklook_Dicts.get_prcs_cfg_dict(flight,date,campaign,device_data_path)
# Data Handling 
#datasets_dict, data_reader_dict=Quicklook_Dicts.get_data_handling_attr_dicts()

# Get Plotting Handling
#plot_handler_dict, plot_cls_args_dict,plot_fct_args_dict=\
#                            Quicklook_Dicts.get_plotting_handling_attrs_dict()


HALO_Devices_cls=Measurement_Instruments_QL.HALO_Devices(cfg_dict)
HALO_Devices_cls.update_major_data_path(device_data_path+"Flight_Data/"+\
                                        campaign+"")
Bahamas_cls=Measurement_Instruments_QL.BAHAMAS(HALO_Devices_cls)
#Radar_cls=Measurement_Instruments_QL.RADAR(HALO_Devices_cls)
#Radar_cls.open_attitude_corrected_data()
#Bahamas_cls.open_bahamas_data()
aircraft_df=pd.read_csv(campaign_path+"data/bahamas/"+"HALO_Aircraft_RF05.csv")
aircraft_df=pd.DataFrame(data=np.nan,columns=["roll","hdg","sfc","lat","lon"],
                  index=pd.DatetimeIndex(\
                         np.array(aircraft_df["time"])))
#aircraft_df["roll"]=np.array(bahamas_dict["IRS_PHI"])                             
#aircraft_df=aircraft_df.resample("1s").mean()
       
#aircraft_df["roll"]=np.array(bahamas_dict[flight]["IRS_PHI"])                             
        
Sondes_cls=Measurement_Instruments_QL.Dropsondes(HALO_Devices_cls)
Sondes_cls.calc_integral_variables(integral_var_list=["IWV","IVT"])
sonde_data=Sondes_cls.sonde_dict


relevant_warm_sector_sondes=[0,1,2,9,10,11,12]
#relevant_cold_sector_sondes=[3,5,11,12] # -----> to be filled
relevant_times=[*sonde_data["reference_time"].keys()]
#sys.exit()

# warm internal legs:
slight_time_shift=pd.Timedelta(9.5,"min")
warm_start=pd.Timestamp(relevant_times[0])-slight_time_shift
warm_end=pd.Timestamp(relevant_times[3])+slight_time_shift
#radar_dbz=pd.DataFrame(data=np.array(Radar_cls.attcorr_radar_ds["dBZg"][:]).T,
#                       index=pd.DatetimeIndex(np.array(Radar_cls.attcorr_radar_ds.time[:])),
#                       columns=np.array(Radar_cls.attcorr_radar_ds.height[:]))

# Warm radar region
#warm_radar=radar_dbz.loc[warm_start:warm_end,:]
warm_aircraft_df=aircraft_df.loc[warm_start:warm_end]
#warm_aircraft_df=warm_aircraft_df.loc[warm_radar.index]
#warm_radar=warm_radar[abs(warm_aircraft_df["roll"])<5]

#BAHAMAS.bahamas_ds
    
# Quicklook Plotter
#Plotter=Data_Plotter.Quicklook_Plotter(cfg_dict)
#Radar_Plotter=Data_Plotter.Radar_Quicklook(cfg_dict) 
#Radar_Plotter.plot_single_radar_cfad(cold_radar,
#                                     raw_measurements=False,
#                                     is_calibrated=False)   
#Radar_Plotter.plot_single_radar_cfad(warm_radar,
#                                     raw_measurements=False,
#                                     is_calibrated=False)   

###############################################################################
relevant_sectors={}
relevant_sectors["warm"]=relevant_warm_sector_sondes

## --> add relevant sondes with "cold" and "warm" key.
    
#bahamas_dict[flight]=Bahamas_cls.bahamas_ds
#radar_dict[flight]=Radar_cls.attcorr_radar_ds
inflow=False

# Load config file
config_file=data_config.load_config_file(os.getcwd(),"data_config_file")

#%% Rain 
#%% Divergence domain    
### Prepare the pattern for regression method
cmpgn_cls=flightcampaign.HALO_AC3(
                             is_flight_campaign=True,
                             major_path=config_file["Data_Paths"]["campaign_path"],
                             aircraft="HALO",interested_flights=[flight],
                             instruments=["radar","radiometer","sonde"])               


Moisture_CONV=Budgets.Moisture_Convergence(cmpgn_cls,flight,config_file,
                 grid_name="Real_Sondes",do_instantan=False)
#Budget_plots=Budgets.Moisture_Budget_Plots(cmpgn_cls,flight,config_file,
#                 grid_name="Real_Sondes",do_instantan=False)
 


#%% warm sector

warm_relevant_times=[relevant_times[warm_time] \
                     for warm_time in relevant_warm_sector_sondes]
warm_sondes_lat=[sonde_data["reference_lat"][time].data[0] \
                 for time in warm_relevant_times]
warm_sondes_lon=[sonde_data["reference_lon"][time].data[0] \
                 for time in warm_relevant_times]

sondes_pos_all=pd.DataFrame(data=np.nan,columns=["Halo_Lat","Halo_Lon"],
                            index=pd.DatetimeIndex(warm_relevant_times))
sondes_pos_all["Halo_Lat"][:]=warm_sondes_lat
sondes_pos_all["Halo_Lon"][:]=warm_sondes_lon

warm_domain_values={}
warm_domain_values={}

# Positions relevant for divergence calculations
sondes_pos_all=Moisture_CONV.get_xy_coords_for_domain(sondes_pos_all)

uninterp_vars={}
interp_vars={}
time_list=[str(pd.Timestamp(time)) for time in warm_relevant_times]
zmax_grid=[float(sonde_data["alt"][time][:].max()) for time in warm_relevant_times]
zmax_grid=pd.Series(zmax_grid).min()//10*10+10
Z_grid=np.arange(0,zmax_grid,step=10)

warm_div_vars={}
cold_div_vars={}
for met_var in ["u","v","wind","q","transport"]:
    interp_vars=pd.DataFrame(data=np.nan,index=Z_grid,columns=time_list)
    t=0    
    for time in warm_relevant_times:
        uninterp_vars["u"]=pd.Series(data=np.array(sonde_data["u_wind"][time][:]),
                            index=np.array(sonde_data["alt"][time][:]))
        uninterp_vars["v"]=pd.Series(data=np.array(sonde_data["v_wind"][time][:]),
                            index=np.array(sonde_data["alt"][time][:]))
        
        uninterp_vars["wind"]=pd.Series(data=np.array(sonde_data["wspd"][time][:]),
                            index=np.array(sonde_data["alt"][time][:]))
        uninterp_vars["q"]=pd.Series(data=np.array(sonde_data["q"][pd.Timestamp(time)][:]),
                            index=np.array(sonde_data["alt"][time][:]))
        uninterp_vars["transport"]=uninterp_vars["wind"]*uninterp_vars["q"]
    
        not_nan_index=uninterp_vars[met_var].index.dropna()
    
        interp_func=scint.interp1d(not_nan_index,
                                   uninterp_vars[met_var].loc[not_nan_index],
                                   kind="nearest",bounds_error=False,
                                   fill_value=np.nan)
        interp_vars[time_list[t]]=pd.Series(data=interp_func(Z_grid),
                                    index=Z_grid)
        t+=1 
    warm_div_vars[met_var]=interp_vars    

print("Warm Sector Preparations Done")        
warm_mean_qv,warm_dx_qv,warm_dy_qv=\
        Moisture_CONV.run_haloac3_sondes_regression(sondes_pos_all,
                                                    warm_div_vars,
                                                    "transport")
warm_mean_q,warm_dx_q_calc,warm_dy_q_calc=\
    Moisture_CONV.run_haloac3_sondes_regression(sondes_pos_all,
                                                warm_div_vars,"q")
            
warm_mean_scalar_wind,warm_dx_scalar_wind,warm_dy_scalar_wind=\
    Moisture_CONV.run_haloac3_sondes_regression(sondes_pos_all,
                                                warm_div_vars,
                                                "wind")

warm_div_qv=(warm_dx_qv+warm_dy_qv)*1000
warm_div_scalar_wind = (warm_dx_scalar_wind+warm_dy_scalar_wind)
warm_div_q_calc      = (warm_dx_q_calc+warm_dy_q_calc)
#div_mass=div_wind*domain_values["q"].mean(axis=0).values*1000
intersect_index=warm_div_qv.index.intersection(warm_div_scalar_wind.index)
intersect_index=intersect_index.intersection(warm_div_q_calc.index)
warm_div_scalar_mass=warm_div_scalar_wind.loc[intersect_index]*\
                warm_div_vars["q"].loc[intersect_index].mean(axis=1).values*1000
warm_adv_q_calc=warm_div_q_calc.loc[intersect_index]*\
            warm_div_vars["wind"].loc[intersect_index].mean(axis=1).values*1000

#%% Plotting            
divergence_plot=plt.figure(figsize=(16,9))
ax1=divergence_plot.add_subplot(121)
ax2=divergence_plot.add_subplot(122)
ax1.plot(warm_div_scalar_mass.values,warm_div_scalar_mass.index.values/1000,
         color="darkorange",lw=3,label="warm sector")
ax1.set_xlim([-3e-4,3e-4])
ax1.set_xticks([-3e-4,-1.5e-4,0,1.5e-4,3e-4])
ax1.set_xticklabels(["-3e-4","-1.5e-4","0","1.5e-4","3e-4"])
ax1.set_xlabel("Mass Divergence ($\mathrm{gkg}^{-1}\mathrm{s}^{-1}$)")
ax1.set_ylabel("Height (km)")
ax1.axvline(x=0,ls="--",lw=3,color="grey")
ax1.set_ylim([0,10])
for axis in ['bottom','left']:
    ax1.spines[axis].set_linewidth(3)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.yaxis.set_tick_params(width=2,length=6)
ax1.xaxis.set_tick_params(width=2,length=6)
ax1.legend(loc="upper right",fontsize=22,bbox_to_anchor=[1.15,1.0])
ax2.plot(warm_adv_q_calc.values,warm_adv_q_calc.index.values/1000,
         color="darkorange",lw=3)
#ax2.plot(cold_adv_q_calc.values,cold_adv_q_calc.index.values/1000,
#         color="purple",lw=3)

ax2.set_ylim([0,10])
ax2.set_xlim([-3e-4,3e-4])
ax2.set_xticks([-3e-4,-1.5e-4,0,1.5e-4,3e-4])
ax2.set_xticklabels(["-3e-4","-1.5e-4","0","1.5e-4","3e-4"])
ax2.axvline(x=0,ls="--",lw=3,color="grey")
ax2.set_xlabel("Moisture Advection ($\mathrm{gkg}^{-1}\mathrm{s}^{-1}$)")
for axis in ['bottom','left']:
    ax2.spines[axis].set_linewidth(3)
ax2.set_yticklabels("")
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.yaxis.set_tick_params(width=2,length=6)
ax2.xaxis.set_tick_params(width=2,length=6)

sns.despine(offset=10)
plt.subplots_adjust(wspace=0.3)
fig_name=flight+"_Sector_IVT_convergence.png"
#divergence_plot.savefig(plot_path+fig_name,
#                        dpi=300,bbox_inches="tight")
#print("Figure saved as:",plot_path+fig_name)
#sys.exit()
# Plot the divergence results
#Budget_plots.plot_single_flight_and_sector_regression_divergence(
#                                                    "warm_sector",4,
#                                                    warm_div_qv,warm_div_scalar_mass,
#                                                    warm_adv_q_calc,warm_adv_q_scalar)
            
#            # Save the data
#            budget_regression_profile_df=pd.DataFrame(data=np.nan,index=div_qv.index,
#                                                      columns=["CONV","ADV","TRANSP"])
#            budget_regression_profile_df["CONV"]=div_scalar_mass.values
#            budget_regression_profile_df["ADV"]=adv_q_calc.values
#            budget_regression_profile_df["TRANSP"]=div_qv.values
#            
#                
#            budget_file_name=flight+"_AR_"+sector+"_"+grid_name+"_regr_sonde_no_"+\
#                str(number_of_sondes)+".csv"
#            budget_regression_profile_df.to_csv(path_or_buf=budget_data_path+budget_file_name)    
sys.exit()

#mean_u,dx_u,dy_u=Moisture_CONV.run_regression(sondes_pos_all,domain_values,"u")
#mean_v,dx_v,dy_v=Moisture_CONV.run_regression(sondes_pos_all,domain_values,"v")
            
