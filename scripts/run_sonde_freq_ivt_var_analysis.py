# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:04:01 2022

@author: u300737
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Change path to working script directory
current_path=os.getcwd()
print(current_path)
major_path = os.path.abspath("../../../")
base_working_path=major_path+"/my_GIT/Synthetic_Airborne_Arctic_ARs"
aircraft_base_path=major_path+"/Work/GIT_Repository/"
working_path  = base_working_path+"/src/"
config_path   = base_working_path+"/config/"
plotting_path = base_working_path+"/plotting/"

sys.path.insert(1, os.path.join(sys.path[0], working_path))
sys.path.insert(2, os.path.join(sys.path[0], config_path))
sys.path.insert(3, os.path.join(sys.path[0], plotting_path))

print(working_path)
os.chdir(working_path)

import flightcampaign as Campaign
import flightmapping# as RFmaps
import flight_track_creator
import data_config

import flight_track_creator

import atmospheric_rivers
import gridonhalo
from reanalysis import ERA5,CARRA
import ivtvariability as IVT_handler
from ivtvariability import IVT_variability

def calc_ivt_stats_all_RFs(flight_dates,flight_to_analyse=None,
                           campaign_to_analyse=None,return_single_flight=False):
    
    ivt_stats=pd.DataFrame(index=range(18),
                           columns=["flight","TIVT","IVT_max","IVT_std"])
    i=0
    for campaign in flight_dates.keys():
        for flight in flight_dates[campaign]:
            date=flight_dates[campaign][flight]
            if campaign=="Second_Synthetic_Study":
                cmpgn_cls=Campaign.Second_Synthetic_Study(
                             is_flight_campaign=True,
                             major_path=config_file["Data_Paths"]["campaign_path"],
                             aircraft="HALO",interested_flights=[flight],
                             instruments=["radar","radiometer","sonde"])               
            else:
                cmpgn_cls=Campaign.North_Atlantic_February_Run(is_flight_campaign=True,
                             major_path=config_file["Data_Paths"]["campaign_path"],
                             aircraft="HALO",interested_flights=[flight],
                             instruments=["radar","radiometer","sonde"])
            # Allocate classes for campaign
            cmpgn_cls.specify_flights_of_interest(flight[0])
            #na_run.create_directory(directory_types=["data"])
            #cmpgn_cls=na_run
        
            ###################################################################
            # for flight
            Flight_Tracker=flight_track_creator.Flighttracker(cmpgn_cls,flight,
                                ar_of_day,track_type="internal",
                                shifted_lat=0,shifted_lon=0,
                                load_save_instantan=False)
            halo_df,campaign_path=Flight_Tracker.get_synthetic_flight_track(
                as_dict=False)
            ################################################################################
            # for reanalysis (CARRA)
            carra_lvls_path=cmpgn_cls.campaign_path+"/data/CARRA/"
            #print(carra_lvls_path)    
            carra=CARRA(for_flight_campaign=True,
                campaign=campaign,research_flights=None,
                carra_path=carra_lvls_path) 
            CARRA_on_HALO=gridonhalo.CARRA_on_HALO(halo_df,carra_lvls_path,
                True,campaign,config_file["Data_Paths"]["campaign_path"],
                [flight],flight_dates[campaign][flight],config_file,ar_of_day=ar_of_day,
                synthetic_flight=True,do_instantaneous=False)
            ###############################################################################
            # Load HMPs and IVT
            CARRA_on_HALO.load_or_calc_interpolated_hmp_data()
            # Load flight track interpolated data
            halo_carra_hmp=CARRA_on_HALO.halo_carra_hmp 
            halo_carra_hmp.index=pd.DatetimeIndex(halo_carra_hmp.index)
            halo_carra_hmp.name="CARRA"
            halo_carra_hmp=cmpgn_cls.calc_distance_to_IVT_max(halo_df,
                                                        halo_carra_hmp)
            
            halo_carra_hmp["highres_Interp_IWV"]=\
                halo_carra_hmp["Interp_IWV_clc"].values
            halo_carra_hmp["highres_Interp_IVT"]=\
                halo_carra_hmp["Interp_IVT"].values
            halo_carra_inflow = \
                halo_carra_hmp.loc[halo_df[halo_df["leg_type"]=="inflow"].index]
            halo_carra_outflow = \
                halo_carra_hmp.loc[halo_df[halo_df["leg_type"]=="outflow"].index]
            
            halo_carra_inflow.name="CARRA"
            halo_carra_outflow.name="CARRA"
            if return_single_flight:
                if campaign==campaign_to_analyse:
                    if flight==flight_to_analyse:
                        halo_df=IVT_handler.calc_halo_delta_distances(halo_df)
                        analyzed_flight_df=halo_df[halo_df["leg_type"]==leg_type].copy()
                        if leg_type=="inflow":
                            hmp_flow=halo_carra_inflow.copy()
                        else:
                            hmp_flow=halo_carra_outflow.copy()
                        hmp_flow.name="CARRA"
                        cmpgn_cls_to_analyse=cmpgn_cls
        
            TIVT=atmospheric_rivers.Atmospheric_Rivers.calc_TIVT_of_cross_sections_in_AR_sector(
                    halo_carra_inflow,halo_carra_outflow,halo_carra_hmp.name)
            ivt_stats["flight"].iloc[i*2]   = date+"_inflow"
            ivt_stats["TIVT"].iloc[i*2]     = TIVT["inflow"]/10e6
            ivt_stats["IVT_max"].iloc[i*2]  = \
                    halo_carra_inflow["highres_Interp_IVT"].max()
            ivt_stats["IVT_std"].iloc[i*2]  = \
                    halo_carra_inflow["highres_Interp_IVT"].std()
            ivt_stats["flight"].iloc[i*2+1]  = date+"_outflow"
            ivt_stats["TIVT"].iloc[i*2+1]    = TIVT["inflow"]/10e6
            ivt_stats["IVT_max"].iloc[i*2+1] =\
                halo_carra_outflow["highres_Interp_IVT"].max()
            ivt_stats["IVT_std"].iloc[i*2+1] =\
                halo_carra_outflow["highres_Interp_IVT"].std()
            i+=1
    ivt_stats.to_csv(path_or_buf=cmpgn_cls_to_analyse.campaign_data_path+\
                 "/IVT_Stats_Overall_all_RFs.csv",index=True)
    print("IVT Statistics saved as:",
      cmpgn_cls_to_analyse.campaign_data_path+\
          "/IVT_Stats_Overall_all_RFs.csv")
    if return_single_flight:
        return hmp_flow,analyzed_flight_df,cmpgn_cls_to_analyse
    else:
        return None
def sonde_freq_ivtvar_analysis(hmp_flow,analyzed_flight_df,cmpgn_cls_to_analyse):
    log_file_name="logging_ivt_variability_carra.log"
    ivt_logger=IVT_handler.ICON_IVT_Logger(log_file_path=aircraft_base_path,
                                        file_name=log_file_name)
    ivt_logger.create_plot_logging_file()
    
    #IVT_var_Plotter.plot_TIVT_error_study()
    # Define Sondes
    sonde_dict={}
    sonde_dict["Pres"]=pd.DataFrame()
    sonde_dict["q"]=pd.DataFrame()
    sonde_dict["Wspeed"]=pd.DataFrame()
    sonde_dict["IVT"]=pd.DataFrame()
    only_model_sounding_profiles=True 
    sounding_frequency=7
    ivt_handler_cls=IVT_handler.IVT_variability(hmp_flow,
                                    None,sonde_dict,
                                    only_model_sounding_profiles,
                                    sounding_frequency,
                                    analyzed_flight_df,
                                    cmpgn_cls_to_analyse.plot_path,"SAR_internal",
                                    flight_to_analyse,ivt_logger)
    # Plot routine
    ivt_var_plot_cls=IVT_handler.IVT_Variability_Plotter(hmp_flow,None,sonde_dict,
                                    only_model_sounding_profiles,sounding_frequency,
                                    analyzed_flight_df,
                                    cmpgn_cls_to_analyse.plot_path,ar_of_day,
                                    flight_to_analyse,ivt_logger)
    ############ Definitions for frequency analysis
    tivt_res_csv_file=cmpgn_cls_to_analyse.campaign_data_path+ar_of_day+"_"+\
                    ivt_var_plot_cls.flight+"_"+leg_type+"_"+\
                            "TIVT_Sonde_Resolution.csv"
    print(tivt_res_csv_file)
    # Resolutions to analyse
    resolutions=["300s","360s","480s","600s","720s","900s","1200s","1500s"]
    resolutions_int=["300","360","480","600","720","900","1200","1500"]

    #Reference TIVT using the total grid representation
    ivt_handler_cls.grid_ivt=ivt_handler_cls.grid_dict["highres_Interp_IVT"]
    ivt_handler_cls.calc_single_tivt_from_ivt()
    reference_tivt=ivt_handler_cls.tivt_grid
    # Preconfigurations for subsampled tivt
    resampled_tivt=pd.DataFrame(columns=["Resolution","TIVT_Mean",
                                                 "TIVT_Std","Resampled_Distance"])
    number_of_shifts=10
    df_resampled_tivt=pd.DataFrame(columns=resolutions,
                                               index=np.arange(0,number_of_shifts))
    df_resampled_tivt.index.name="Index_Shift_Factor"

    df_resampled_tivt.name=reference_tivt

    # Define continuous halo and ivt to reindex from
    continuous_ivt=ivt_handler_cls.grid_ivt.copy()
    continuous_halo=ivt_handler_cls.halo.copy()
    print(continuous_ivt)
    i=0
    for res in resolutions:
        # the sondes are slightly shifted in 10 steps 
        # to better analyse the sensitivity to positioning.
        # the shifts goes up to one half of the resolution
        #resampled_grid_ivt=ivt_handler_cls.grid_ivt.asfreq(res)
           # Shift index to assess the sensitivity to spatial location of soundings.
  
        shift_factor=int(float(resolutions_int[i])/(2*number_of_shifts))
        #series of tivt based on given resolution showing values with shift_factors given in seconds.
        res_case_tivt=pd.Series(index=np.arange(0,number_of_shifts)*shift_factor)  
        for shift in range(number_of_shifts):
            # Resample aircraft and grid ivt data to specific soundings
            resampled_grid_ivt=continuous_ivt.iloc[int(shift_factor*shift):]
            resampled_grid_ivt=resampled_grid_ivt.iloc[::int(resolutions_int[i])]
            # regardless the shift, the initial/end values of the resampled grid ivt need to be the first and
            # last value of the cross-section 
            resampled_grid_ivt.iloc[0]=continuous_ivt.iloc[0]
            resampled_grid_ivt.iloc[-1]=continuous_ivt.iloc[-1]
            resampled_index_list=resampled_grid_ivt.index.tolist()
            resampled_index_list[0]=ivt_handler_cls.grid_ivt.index[0]
            resampled_index_list[-1]=ivt_handler_cls.grid_ivt.index[-1]
            resampled_grid_ivt.index=resampled_index_list
            resampled_halo_df=continuous_halo.reindex(resampled_grid_ivt.index)
            #print(resampled_grid_ivt)
            # Recalculate the distances as the data is now resampled with larger time gaps.
            time_frequency=resampled_grid_ivt.index.to_series().diff()
            resampled_halo_df["time_Difference"]=time_frequency
            resampled_halo_df["time_Difference"]=resampled_halo_df["time_Difference"].dt.total_seconds()
            resampled_halo_df["delta_distance"]=resampled_halo_df["groundspeed"]*resampled_halo_df["time_Difference"]
            resampled_halo_df["delta_distance"].iloc[0]=0.0
            resampled_halo_df["cumsum_distance"]=resampled_halo_df["delta_distance"].cumsum()
            # assign the resampled 
            ivt_handler_cls.grid_ivt=resampled_grid_ivt
            ivt_handler_cls.halo=resampled_halo_df
            ivt_handler_cls.calc_single_tivt_from_ivt()
            res_case_tivt.iloc[shift]=ivt_handler_cls.tivt_grid
        df_resampled_tivt.loc[:,res]=res_case_tivt.values    
        i+=1
    
    df_resampled_tivt["REAL-CARRA-TIVT"]=float(df_resampled_tivt.name)
    df_resampled_tivt["cumsum_distance"]=float(resampled_halo_df["cumsum_distance"].values[-1])
    df_resampled_tivt.to_csv(path_or_buf=tivt_res_csv_file,index=True)
    print(df_resampled_tivt)
    print("Saved TIVT Sonde Resolution CSV File under:"+tivt_res_csv_file)
    return df_resampled_tivt
# Config File
config_file=data_config.load_config_file(aircraft_base_path,"data_config_file")
analyse_all_flights=False
flight_dates={"North_Atlantic_Run":
              {"SRF02":"20180224",
               "SRF04":"20190319",#},
               "SRF07":"20200416",#},
               "SRF08":"20200419"
              },
              "Second_Synthetic_Study":
              {"SRF02":"20110317",
               "SRF03":"20110423",
               "SRF08":"20150314",
               "SRF09":"20160311",
               "SRF12":"20180225"
               }}
flight_tracks_dict={}
campaigns=[*flight_dates.keys()]
shifted_lat=0
shifted_lon=0
ar_of_day="SAR_internal"

leg_type="outflow"
flight_to_analyse="SRF08"
campaign_to_analyse="Second_Synthetic_Study"#"North_Atlantic_Run"#"Second_Synthetic_Study"
# Loop over cross-sections from all flights
hmp_flow,analyzed_flight_df,cmpgn_cls_to_analyse=calc_ivt_stats_all_RFs(
                        flight_dates,flight_to_analyse=flight_to_analyse,
                        campaign_to_analyse=campaign_to_analyse,
                        return_single_flight=True)
sonde_freq_ivtvar_analysis(hmp_flow,analyzed_flight_df,cmpgn_cls_to_analyse)