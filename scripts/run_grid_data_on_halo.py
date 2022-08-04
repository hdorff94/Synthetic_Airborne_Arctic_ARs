#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:35:51 2020

@author: u300737
"""
import os
import sys
import data_config

import numpy as np
import pandas as pd
import xarray as xr

###############################################################################
import Flight_Campaign
from Flight_Mapping import FlightMaps

###############################################################################
#Grid Data
from Reanalysis import ERA5,CARRA 
from ICON import ICON_NWP as ICON
import Grid_on_HALO
###############################################################################
# Plot scripts
import matplotlib.pyplot as plt

try:
    from typhon.plots import styles
except:
    print("Typhon module cannot be imported")

import Interp_Data_Plotting 
#-----------------------------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore")
#-----------------------------------------------------------------------------#        
"""
###############################################################################
    Main Script for running interpolation of griddata on flight path
###############################################################################
"""    
def main(config_file_path=os.getcwd(),
         campaign="NA_February_Run",hmp_plotting_desired=False,
         hmc_plotting_desired=True,
         plot_data=True,ar_of_day="AR_internal",
         flight=["SRF04"],
         era_is_desired=False,carra_is_desired=True,
         icon_is_desired=False,synthetic_flight=False,
         synthetic_campaign=True,
         ######################################################################
         # USEFUL values
         synthetic_icon_lat=0,
         synthetic_icon_lon=0,
         #synthetic_icon_lat=-7,
         #synthetic_icon_lon=-12,
         #synthetic_icon_lat=1.8,
         ######################################################################
         #synthetic_icon_lon=6.2,
         #synthetic_icon_lat=-5.5,
         #synthetic_icon_lon=5.0,
         #synthetic_icon_lat=-5.75,
         #synthetic_icon_lon=-5.0,
         track_type="internal",
         merge_all_legs=False,
         pick_legs=["inflow","internal","outflow"],
         do_instantaneous=False):
    if campaign=="NAWDEX":
        years={"RF01":"2016","RF02":"2016","RF03":"2016","RF04":"2016",
               "RF05":"2016","RF06":"2016","RF07":"2016","RF08":"2016",
               "RF09":"2016","RF10":"2016","RF11":"2016","RF12":"2016"}
        months={"RF01":"09","RF02":"09","RF03":"09","RF04":"09","RF05":"09",
                "RF06":"10","RF07":"10","RF08":"10","RF09":"10","RF10":"10",
                "RF11":"10","RF12":"10"}
        days={"RF01":"17","RF02":"21","RF03":"23","RF04":"26","RF05":"27",
           "RF06":"06","RF07":"09","RF08":"09","RF09":"10","RF10":"13",
           "RF11":"14","RF12":"15"}
    
    elif campaign=="NA_February_Run":
        flights=["SRF01","SRF02","SRF03","SRF04","SRF05","SRF06","SRF07"]
        days={"SRF01":"30","SRF02":"24","SRF03":"26",
              "SRF04":"19","SRF05":"20","SRF06":"23","SRF07":"16","SRF08":"19"}
        months={"SRF01":"05","SRF02":"02","SRF03":"02",
                "SRF04":"03","SRF05":"04","SRF06":"04","SRF07":"04",
                "SRF08":"04"}
        years={"SRF01":"2017","SRF02":"2018","SRF03":"2018",
               "SRF04":"2019","SRF05":"2019","SRF06":"2016",
               "SRF07":"2020","SRF08":"2020"}
        icon_is_desired=False
    
    elif campaign=="Second_Synthetic_Study":
        flights=["SRF02","SRF03","SRF06","SRF07","SRF08","SRF09","SRF12"]
        
        days={"SRF01":"15","SRF02":"17","SRF03":"23",
                             "SRF04":"03","SRF05":"25","SRF06":"25",
                             "SRF07":"07","SRF08":"14","SRF09":"11",
                             "SRF10":"12","SRF11":"28","SRF12":"25",
                             "SRF13":"13"}
            
        months={"SRF01":"03","SRF02":"03","SRF03":"04",
                               "SRF04":"03","SRF05":"04","SRF06":"03",
                               "SRF07":"03","SRF08":"03","SRF09":"03",
                               "SRF10":"03","SRF11":"04","SRF12":"02",
                               "SRF13":"04"}
            
        years={"SRF01":"2011","SRF02":"2011","SRF03":"2011",
                        "SRF04":"2012","SRF05":"2012","SRF06":"2014",
                        "SRF07":"2015","SRF08":"2015","SRF09":"2016",
                        "SRF10":"2016","SRF11":"2016","SRF12":"2018",
                        "SRF13":"2020"}
            
    else:
        raise Exception("This campaign is not defined and will be ignored.")
    hours_time=['00:00', '01:00', '02:00','03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00','09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00','15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00','21:00', '22:00', '23:00',]
    
    analysing_campaign=True
    
    # Load config file
    config_file=data_config.load_config_file(config_file_path,
                                             "data_config_file")
    
    
    print("Analyse given flight: ",flight[0])
    
    
    date=years[flight[0]]+months[flight[0]]+days[flight[0]]
    plot_cfad=True
    if synthetic_campaign:
        plot_cfad=False
    #-------------------------------------------------------------------------#
    # Boolean Definition of Task to do in Analysis
    # Define the hydrometeor parameters to analyze and to plot         
    if synthetic_campaign:
        synthetic_flight=True
    synthetic_icon=False        # Default is False
    include_retrieval=False
    do_orographic_masking=False
    do_moisture_budget=False
    #-------------------------------------------------------------------------#
    if flight[0]=="RF12":
        do_orographic_masking=True
    if synthetic_flight:
        ar_of_day="S"+ar_of_day
        if synthetic_icon_lat!=0:
            ar_of_day=ar_of_day+"_"+str(synthetic_icon_lat)
    #if flight[0]=="SRF06":
    #    carra_is_desired=False
    
    # Load Halo Dataset
    if not synthetic_campaign:
        if campaign=="NAWDEX":
            # Load the class
            nawdex=Flight_Campaign.NAWDEX(is_flight_campaign=True,
                major_path=config_file["Data_Paths"]["campaign_path"],
                aircraft="HALO",instruments=["radar","radiometer","sonde"])
        
            nawdex.specify_flights_of_interest(flight[0])
            nawdex.create_directory(directory_types=["data"])
            if flight[0]=="RF10" or flight[0]=="RF03":
                icon_is_desired=True
            
            if not synthetic_flight:
                halo_df,campaign_path=nawdex.load_aircraft_position("NAWDEX")
            else:
                import flight_track_creator
                Flight_Tracker=flight_track_creator.Flighttracker(
                                            nawdex,flight[0],ar_of_day,
                                            track_type=track_type,
                                            shifted_lat=synthetic_icon_lat,
                                            shifted_lon=synthetic_icon_lon)
            
                halo_df,campaign_path=Flight_Tracker.run_flight_track_creator()
                # If halo df is a dict, then this might arise from the 
                # internal leg created flight track. So merge it to a 
                # single dataframe, or loc current leg from it
            
                if isinstance(halo_df,dict):
                    halo_dict=halo_df.copy()
                    halo_df,time_legs_df=Flight_Tracker.concat_track_dict_to_df(
                                                        halo_df,merge_all=False,
                                                        pick_legs=pick_legs)
                    print("Synthetic flight track loaded")
                
            halo_df["Closest_Lat"]=np.nan
            halo_df["Closest_Lon"]=np.nan
            halo_df["Hour"]=pd.DatetimeIndex(halo_df.index).hour
            halo_df["Minutes"]=pd.DatetimeIndex(halo_df.index).minute
            halo_df["Minutesofday"]=halo_df["Hour"]*60+halo_df["Minutes"]
            halo_df.index=pd.DatetimeIndex(halo_df.index)
            cmpgn_cls=nawdex
        
    else:
         # Flight Campaign is Synthetic
         if campaign=="NA_February_Run":
             na_run=Flight_Campaign.North_Atlantic_February_Run(
                                    is_flight_campaign=True,
                                    major_path=config_file["Data_Paths"]\
                                                ["campaign_path"],aircraft="HALO",
                                    interested_flights=flight,
                                    instruments=["radar","radiometer","sonde"])
         elif campaign=="Second_Synthetic_Study":
             cpgn_cls_name="Second_Synthetic_Study"
             na_run=Flight_Campaign.Second_Synthetic_Study(
                             is_flight_campaign=True,
                             major_path=config_file["Data_Paths"]["campaign_path"],
                             aircraft="HALO",interested_flights=flights,
                             instruments=["radar","radiometer","sonde"])               
         else:
             raise Exception("Wrong campaign name given.")
         
         na_run.specify_flights_of_interest(flight[0])
         na_run.create_directory(directory_types=["data"])
         cmpgn_cls=na_run
         
         import flight_track_creator
         Flight_Tracker=flight_track_creator.Flighttracker(
                                            na_run,flight[0],ar_of_day,
                                            track_type=track_type,
                                            shifted_lat=synthetic_icon_lat,
                                            shifted_lon=synthetic_icon_lon,
                                            load_save_instantan=do_instantaneous)
            
         #flight_track_exists=Flight_Tracker.check_if_flight_track_exists()
         #if flight_track_exists:
         halo_df,campaign_path=Flight_Tracker.get_synthetic_flight_track()
         print("Synthetic flight track loaded")
         # If halo df is a dict, then this may arise from the 
         # internal leg created flight track. So merge it to a 
         # single dataframe, or loc current leg from it
         if isinstance(halo_df,dict):
             aircraft_dict=halo_df.copy()
             halo_df,leg_dicts=Flight_Tracker.concat_track_dict_to_df()
         else:
             aircraft_dict=Flight_Tracker.make_dict_from_aircraft_df()
             
                    
         halo_df["Hour"]=pd.DatetimeIndex(halo_df.index).hour
                            #pd.DatetimeIndex(halo_df.index).hour[0]
         halo_df["Minutes"]=pd.DatetimeIndex(halo_df.index).minute
         halo_df["Minutesofday"]=halo_df["Hour"]*60+halo_df["Minutes"]
         halo_df["Minutesofday"]=halo_df["Minutesofday"]#-\
                         #halo_df["Minutesofday"].iloc[0]
         if "distance" in halo_df.columns:
             del halo_df["distance"]
    #Define the file names of hydrometeor data and paths
    flight_name=flight[0]
    if do_instantaneous:
        flight_name=flight_name+"_instantan"
    
    if ar_of_day:
        interpolated_hmp_file=flight_name+"_"+ar_of_day+\
                                "_HMP_ERA_HALO_"+date+".csv"
    else:
        interpolated_hmp_file="HMP_ERA_HALO_"+date+".csv"
    
    hydrometeor_lvls_path=cmpgn_cls.campaign_path+"/data/ERA-5/"
    hydrometeor_lvls_file="hydrometeors_pressure_levels_"+date+".nc"
        
    if synthetic_flight:
        interpolated_hmp_file="Synthetic_"+interpolated_hmp_file
        # Until now ERA5 is not desired
    if ar_of_day is not None:
        interpolated_iwc_file=flight_name+"_"+ar_of_day+"_IWC_"+date+".csv"        
    else:
        interpolated_iwc_file=flight_name+"_IWC_"+date+".csv"
    if synthetic_flight:
        interpolated_iwc_file="Synthetic_"+interpolated_iwc_file
    if icon_is_desired:
        if synthetic_icon:
            hydrometeor_lvls_path=hydrometeor_lvls_path+"Latitude_"+\
            str(synthetic_icon_lat)+"/"
       
    else:
        print("This none dataset of the flight campaign.")
        print("No airborne datasets will be integrated.")
    #-------------------------------------------------------------------------#
    #%% ERA5 class & ERA5 on HALO Class
    era5=ERA5(for_flight_campaign=True,campaign=campaign,research_flights=None,
                     era_path=hydrometeor_lvls_path)

    ERA5_on_HALO=Grid_on_HALO.ERA_on_HALO(
                                halo_df,hydrometeor_lvls_path,
                                hydrometeor_lvls_file,interpolated_iwc_file,
                                analysing_campaign,campaign,
                                config_file["Data_Paths"]["campaign_path"],
                                flight,date,config_file,ar_of_day=ar_of_day,
                                synthetic_flight=synthetic_flight,
                                do_instantaneous=do_instantaneous)
    #%% CARRA class & CARRA on HALO class
    if carra_is_desired:
        interpolated_carra_file=""
        carra_lvls_path=cmpgn_cls.campaign_path+"/data/CARRA/"
    
        carra=CARRA(for_flight_campaign=True,
                    campaign=campaign,research_flights=None,
                    carra_path=carra_lvls_path) 
        
        CARRA_on_HALO=Grid_on_HALO.CARRA_on_HALO(
                                halo_df,carra_lvls_path,
                                analysing_campaign,campaign,
                                config_file["Data_Paths"]["campaign_path"],
                                flight,date,config_file,ar_of_day=ar_of_day,
                                synthetic_flight=synthetic_flight,
                                do_instantaneous=do_instantaneous)
   
    #%% HALO Aircraft Data    
    if not synthetic_flight:
        # Load HAMP radar
        radar=cmpgn_cls.load_hamp_data(campaign,flight,instrument="radar",
                                flag_data=True,bahamas_desired=True) 
        # Load HAMP Microwave Radiometer
        mwr=cmpgn_cls.load_hamp_data(campaign,flight,instrument="radiometer")
    
        # Cut dataset to AR core cross-section
        if ar_of_day:
            #radar
            halo_df,radar,ar_of_day=ERA5_on_HALO.cut_halo_to_AR_crossing(
                                                ar_of_day, flight[0], 
                                                halo_df,radar,
                                                device="radar")
            #radiometer
            halo_df,mwr,ar_of_day=ERA5_on_HALO.cut_halo_to_AR_crossing(
                                                ar_of_day, flight[0], 
                                                halo_df,mwr,
                                                device="radiometer")
        
            # Update halo_df in ERA5_on_HALO class with cutted dataset
            ERA5_on_HALO.update_halo_df(halo_df,change_last_index=True)
            CARRA_on_HALO.update_halo_df(halo_df,change_last_index=True)
            radar["Position"].to_csv(path_or_buf=cmpgn_cls.data_path+\
                             "BAHAMAS/HALO_Aircraft_"+flight[0]+".csv")
        
        # Load Dropsonde datasets
    
        if not flight[0]=="RF08":
            Dropsondes,Upsampled_Dropsondes=cmpgn_cls.load_ar_processed_dropsondes(
                                                    ERA5_on_HALO,date,
                                                    radar,halo_df,flight,
                                                    with_upsampling=True,
                                                    ar_of_day=ar_of_day)
        else:
            Dropsondes={}
    else:
        Dropsondes={}
        radar={}
        mwr={}
    
    last_index=len(halo_df.index)
    lat_changed=False
    
    #%% Gridded data (Simulations and Reanalysis)
    icon_major_path=cmpgn_cls.campaign_path+"/data/ICON_LEM_2KM/"
    hydrometeor_icon_path=cmpgn_cls.campaign_path+"/data/ICON_LEM_2KM/"
    if synthetic_icon:
        if not synthetic_icon_lat==None:
            hydrometeor_icon_path=hydrometeor_icon_path+"Latitude_"+\
                str(synthetic_icon_lat)+"/"
            lat_changed=True   
    if not os.path.exists(hydrometeor_icon_path):
        os.mkdir(hydrometeor_icon_path)
        
    icon_resolution=2000 # units m
    if icon_is_desired:
        icon_var_list=ICON.lookup_ICON_AR_period_data(flight,ar_of_day,
                                                 icon_resolution,
                                                 hydrometeor_icon_path,
                                                 synthetic=synthetic_flight)
        interp_icon_hmp_file=flight[0]+"_"+ar_of_day+"_"+"interpolated_HMP.csv"
        if synthetic_flight:
            interp_icon_hmp_file="Synthetic_"+interp_icon_hmp_file
    
        ICON_on_HALO=Grid_on_HALO.ICON_on_HALO(
                            cmpgn_cls,icon_var_list,halo_df,flight,date,
                            interpolated_hmp_file=interp_icon_hmp_file,
                            interpolated_hmc_file=None,ar_of_day=ar_of_day,
                            synthetic_icon=synthetic_icon,
                            synthetic_flight=synthetic_flight)
    
        ICON_on_HALO.update_ICON_hydrometeor_data_path(hydrometeor_icon_path)
    
    else:
        icon_var_list=[]
               
    #%% Processing, Interpolation onto Flight Path
    # If interpolated data does not exist, load ERA-5 Dataset
    
    # Create HALO interpolated total column data  if not existent, 
    # if HMPs already interpolated onto HALO for given flight, load csv-file.
    if hmp_plotting_desired:
        #if synthetic_icon:
        #    halo_df["latitude"]=halo_df["latitude"]+synthetic_icon_lat
        #    print("Changed Lat of HALO Aircraft for Synthetic Observations")
        #    lat_changed=True
        if not os.path.exists(hydrometeor_lvls_path):
                os.makedirs(hydrometeor_lvls_path)
        print("Path to open: ", hydrometeor_lvls_path)
        print("open hydrometeor_levels")
        
        #----------------- ERA-5 ---------------------------------------------#
        ERA5_on_HALO.update_interpolated_hmp_file(interpolated_hmp_file)
        halo_era5=ERA5_on_HALO.load_hmp(cmpgn_cls)
        halo_era5=halo_era5.groupby(level=0).first()#drop_duplicates(keep="first")
        halo_df=halo_df.groupby(level=0).first()
        if "Interp_IVT" in halo_era5.columns:
            if not "groundspeed" in halo_df.columns:
                if radar!={}:
                    halo_df.index=pd.DatetimeIndex(halo_df.index)
                    halo_df["groundspeed"]=radar["Position"]["groundspeed"].\
                                            loc[halo_df.index]
            halo_era5=cmpgn_cls.calc_distance_to_IVT_max(
                        halo_df,
                        halo_era5)
        #---------------------------------------------------------------------#
        if carra_is_desired:
            CARRA_on_HALO.load_or_calc_interpolated_hmp_data()
            #update_interpolated_hmp_file(self,interpolated_hmp_file):
            high_res_hmp=CARRA_on_HALO.halo_carra_hmp.copy()
        #----------------- ICON ----------------------------------------------#
        if icon_is_desired:
            ICON_on_HALO.update_ICON_hydrometeor_data_path(hydrometeor_icon_path)
            halo_icon_hmp=ICON_on_HALO.load_interpolated_hmp()
            #high_res_hmp=halo_icon_hmp.copy()
       #----------------------------------------------------------------------# 
    #%%
    if hmc_plotting_desired:
        #if synthetic_icon:
            #if synthetic_icon_lat!=0:
       #     hydrometeor_lvls_path=hydrometeor_lvls_path+"Latitude_"+\
           # str(synthetic_icon_lat)+"/"
       #----------------------------------------------------------------------#
       # ERA5 
       if era_is_desired:
           halo_era5_hmc=ERA5_on_HALO.load_hwc()
       #----------------------------------------------------------------------#
       # CARRA
       if not carra_is_desired:
           pass
       else:
           CARRA_on_HALO.load_or_calc_interpolated_hmc_data()
           halo_carra_hmc=CARRA_on_HALO.carra_halo_hmc
       #
       # ICON                                                               
       if not icon_is_desired:
            pass
       else:
           #Get vertical profiles of moisture/ hydrometeors
           halo_icon_hmc=ICON_on_HALO.load_hwc(with_hydrometeors=True)
       #----------------------------------------------------------------------#    
       # Retrieval
       if flight==["RF10"]:
            if include_retrieval:
                retrieval_dict=cmpgn_cls.load_radiometer_retrieval(
                                       campaign,
                                       variables=["T","rho_v"],
                                       calculate_spec_hum=True,
                                       sonde_p=Upsampled_Dropsondes["Pres"])
                retrieval_dict=cmpgn_cls.vertical_integral_retrieval(
                                        retrieval_dict,Upsampled_Dropsondes)
                
    #%%
    """
        Plotting of datasets from HALO_ERA_Plotting class   
    """
    levels=np.linspace(0,50,50)
    
    #-------------- Plot Path Specifications ---------------------------------#
    plot_path=cmpgn_cls.campaign_path+"/plots/"+flight[0]+"/"
    if not os.path.exists(plot_path):
                        os.mkdir(plot_path)

    #Check if plot path exists, if not create it.
    if synthetic_icon:
        plot_path=plot_path+"synthetic_measurements/"
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
    
        if synthetic_icon_lat is not None:
            plot_path=plot_path+"Latitude_"+str(synthetic_icon_lat)+"/"
            if not os.path.exists(plot_path):
                        os.mkdir(plot_path)

    if synthetic_flight:
        plot_path=plot_path+"Synthetic_Measurements/"
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)            

    if ar_of_day!=None:
        plot_path=plot_path+ar_of_day+"/"
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
    #-------------------------------------------------------------------------#
    # Major Plotter Function Classes
    ERA_HALO_Plotting   = Interp_Data_Plotting.ERA_HALO_Plotting(
                                        flight,ar_of_day=ar_of_day,
                                        plot_path=plot_path,
                                        synthetic_campaign=synthetic_flight)    
    CARRA_HALO_Plotting = Interp_Data_Plotting.CARRA_HALO_Plotting(
                                        plot_path=plot_path,
                                        flight=flight,ar_of_day=ar_of_day,
                                        synthetic_campaign=synthetic_flight)
    # Mainly for ICON Plotting, but IVT cross-section plotter using ERA-5
    # and/or ICON is included in ICON_HALO_Plotting
    if icon_is_desired:
        icon_plot_path=plot_path+"ICON_2km/"
        if not os.path.exists(icon_plot_path):
                    os.mkdir(icon_plot_path)
    else:
        icon_plot_path=plot_path # If ICON is not included, 
                                 # plot_path remains due to IVT-plot                        
    
    ICON_HALO_Plotting  = Interp_Data_Plotting.ICON_HALO_Plotting(cmpgn_cls,
                                plot_path=icon_plot_path,
                                flight=flight,ar_of_day=ar_of_day,
                                synthetic_campaign=synthetic_flight)
    
    #-------------------------------------------------------------------------#
    #Load Flight map class
    if synthetic_campaign:
        if track_type=="internal":
            track_dict=aircraft_dict.copy()
    else:
        track_dict=None
        
    Flightmap=FlightMaps(cmpgn_cls.major_path,cmpgn_cls.campaign_path,
                         cmpgn_cls.aircraft,cmpgn_cls.instruments,
                         cmpgn_cls.interested_flights,plot_path=plot_path,
                         flight=flight[0],ar_of_day=ar_of_day,
                         synthetic_campaign=synthetic_flight,
                         synthetic_icon_lat=synthetic_icon_lat,
                         synthetic_icon_lon=synthetic_icon_lon,
                         track_type=track_type,pick_legs=pick_legs,
                         track_dict=track_dict)
    
    AR_IWV_section_combined_mapping=\
                            Flightmap.plot_flight_combined_IWV_map_AR_crossing
    AR_flight_section_mapping=Flightmap.plot_flight_map_AR_crossing
    #-------------------------------------------------------------------------#
    set_font=ERA_HALO_Plotting.specify_plotting()
    style_name="typhon"
    if plot_cfad:
        try:
            cfad_hist=cmpgn_cls.calculate_cfad_radar_reflectivity(
                                radar["Reflectivity"])
        
            with plt.style.context(styles(style_name)):
                print("Plots created with Typhon")
                new_cfad=cmpgn_cls.plot_cfad_2d_hist(
                    cfad_hist,plot_path=plot_path,
                    flagged_data=True,
                    ar_of_day=ar_of_day)
        except:
            pass
    ###########################################################################
    ## HMP visualization
    plt.close()
    if plot_data:
        if hmp_plotting_desired:
            icon_plot_path=plot_path+"ICON_2km/"
            #ERA_HALO_Plotting.plot_radar_era5_time_series(
            #                        radar,halo_era5,Dropsondes,flight,
            #                        plot_path,save_figure=True)
            ICON_HALO_Plotting.plot_IVT_icon_era5_sondes(halo_era5,
                                          Dropsondes,last_index,date,
                                          with_ICON=False,
                                          with_CARRA=carra_is_desired)
            ###################################################################            
            # Map the AR flight intersection
            last_hour=pd.DatetimeIndex(halo_df.index).hour[-1]#
            if not synthetic_flight:
                radar["Position"]=radar["Position"].loc[halo_df.index[0]:\
                                                    halo_df.index[-1]]
                
                try:
                    cmpgn_cls.plot_radar_AR_quicklook(radar,ar_of_day,
                                                   flight[0],plot_path)
                except:
                    pass
                # Moisture and Moisture Budget
            AR_flight_section_mapping(ERA5_on_HALO,radar,Dropsondes,cmpgn_cls)
            Flightmap.plot_AR_moisture_components_map(ERA5_on_HALO,radar,
                                                  Dropsondes,cmpgn_cls)
            
            if not flight[0].endswith("instantan"):
                Flightmap.plot_moisture_budget(ERA5_on_HALO,radar,
                                       Dropsondes,cmpgn_cls)
            
            
            if icon_is_desired:
                ### Add radar latitude in here
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                
                # Map the AR flight intersection
                try:
                    with plt.style.context(styles(style_name)):
                        print("Plots created with Typhon")
                        ICON_HALO_Plotting.plot_IVT_icon_era5_sondes(halo_era5,
                                          Dropsondes,last_index,date,
                                          with_ICON=True,
                                          with_dropsondes=True,
                                          synthetic_icon_lat=synthetic_icon_lat)
                except:
                    pass
                try:    
                    ICON_HALO_Plotting.plot_hmp_icon_era5_sondes(radar,
                                        halo_icon_hmp,halo_era5,Dropsondes,
                                        last_index,flight,ar_of_day,
                                        synthetic_icon_lat=synthetic_icon_lat)
                except:
                    pass
            
                try:
                    AR_IWV_section_combined_mapping(radar,Dropsondes,cmpgn_cls,
                                                last_hour,
                                                opt_plot_path=plot_path)
                except:
                    pass
            
                try:
                    Flightmap.plot_flight_map_Hydrometeorpaths_AR_crossing(radar,
                                        Dropsondes,cmpgn_cls,last_hour,
                                        with_ICON=True)
                except: 
                    pass
            
            else:
                if not synthetic_flight:
                    try:
                        cmpgn_cls.plot_radar_AR_quicklook(radar,ar_of_day,
                                                   flight[0],plot_path)
                    except:
                        pass

        #if not synthetic_icon:
            Flightmap.plot_flight_map_Hydrometeorpaths_AR_crossing(radar,
                                        Dropsondes,cmpgn_cls,last_hour,
                                        with_ICON=False)
        ###########################################################################
        ## HMC visualization
        if hmc_plotting_desired:
            # Sometimes typhon works, sometimes not, this is why 
            # try except commands are used.
            with plt.style.context(styles(style_name)):
                print("Plots created with Typhon")
                #try:
                #    ERA_HALO_Plotting.plot_radar_era5_combined_hwc(radar,
                #                                halo_era5_hmc,date,
                #                                'temperature',
                #                                save_figure=True)
        
                #except:
                    #    pass
                #try:
                    #    ICON_HALO_Plotting.plot_radar_icon_hwc(
                    #                    radar,halo_icon_hmc,
                    #                    "temperature",save_figure=True)
                #except:
                    #pass
                #
                if not synthetic_flight:
                    try:
                        cmpgn_cls.plot_hamp_brightness_temperatures(mwr["T_b"],
                                flight,date,halo_era5_hmc["IWC"].index[0],
                                halo_era5_hmc["IWC"].index[-1],
                                ar_of_day=ar_of_day,
                                plot_path=plot_path)
                    except:
                        pass
                
                    try:
                        cmpgn_cls.plot_radar_AR_quicklook(radar,ar_of_day,
                                                   flight[0],plot_path)
                    except:
                        pass
                
                try:
                    
                    cmpgn_cls.plot_AR_sonde_thermodynamics(Upsampled_Dropsondes,
                                    radar,date,flight[0],
                                    os.getcwd()+"/"+flight[0]+"/",
                                    Upsampled_Dropsondes["AirT"].index[0],
                                    Upsampled_Dropsondes["AirT"].index[-1],
                                    plot_path=plot_path,save_figure=True,
                                    low_level=True,ar_of_day=ar_of_day)
                except:
                    print("No dropsonde plot created")
                if track_type=="internal":
                    ERA_HALO_Plotting.internal_leg_representativeness(cmpgn_cls,
                                                                      ERA5_on_HALO,
                                                                      flight[0],
                                                                      halo_df,
                                                                      halo_era5_hmc)
                    #if icon_is_desired:
                        #try:
                        #    ICON_HALO_Plotting.mean_internal_leg_representativeness(
                        #                cmpgn_cls,ICON,ICON_on_HALO,
                        #                flight[0],halo_df,halo_icon_hmc)
                        #except:
                        #    pass
                        #try:    
                        #    Flightmap.plot_ar_section_internal_leg_ICON(cmpgn_cls)
                        #except:
                        #    pass
                        #try:
                        #    ICON_HALO_Plotting.internal_leg_representativeness(
                        #                                cmpgn_cls,ICON,ICON_on_HALO,
                        #                                flight[0],halo_df,
                        #                                halo_icon_hmc)
                        #except:
                        #    pass
                        #        sys.exit()
                    
                    # can also be done when having synthetic observations
                    #Low level plot
                try:
                    ERA_HALO_Plotting.two_H_plot_radar_era5_combined_hwc(
                                ERA5_on_HALO,
                                radar,halo_era5_hmc,date,
                                'temperature',
                                halo_era5_hmc["IWC"].index[0],
                                halo_era5_hmc["IWC"].index[-1],
                                save_figure=True,
                                do_masking=do_orographic_masking,
                                low_level=True)
                except:
                    pass
                try:
                    ERA_HALO_Plotting.two_H_plot_radar_era5_combined_hwc(
                                        ERA5_on_HALO,
                                        radar,halo_era5_hmc,
                                        date,'temperature',
                                        halo_era5_hmc["IWC"].index[0],
                                        halo_era5_hmc["IWC"].index[-1],
                                        save_figure=True,low_level=False)
                except:
                    pass
            
                # All levels
                try: 
                    ERA_HALO_Plotting.plot_HALO_AR_ERA_thermodynamics(
                                    ERA5_on_HALO,radar,halo_era5_hmc,date,
                                    halo_era5_hmc["IWC"].index[0],
                                    halo_era5_hmc["IWC"].index[-1],
                                    do_masking=do_orographic_masking,
                                    save_figure=True,low_level=False)
                except:
                    pass
                # Low level
                try:
                    ERA_HALO_Plotting.plot_HALO_AR_ERA_thermodynamics(
                                    ERA5_on_HALO,
                                    radar,halo_era5_hmc,date,
                                    halo_era5_hmc["IWC"].index[0],
                                    halo_era5_hmc["IWC"].index[-1],
                                    do_masking=do_orographic_masking,
                                    low_level=True)
                    if carra_is_desired:
                        CARRA_HALO_Plotting.plot_specific_humidity_profile(
                            halo_carra_hmc,halo_df,Dropsondes,radar,date,
                            halo_carra_hmc["u"].index[0],
                            halo_carra_hmc["u"].index[-1],
                            do_masking=do_orographic_masking,
                            low_level=True,AR_sector="all")
                except:
                    pass
            
                if icon_is_desired:                    
                    if ar_of_day:
                        print("Load calc ERA z total column interpolated data")
                        halo_era5=pd.read_csv(hydrometeor_lvls_path+\
                                          interpolated_hmp_file)
                        halo_era5.index=pd.DatetimeIndex(halo_era5.iloc[:,0])
                    
                        ICON_HALO_Plotting.plot_HALO_AR_ICON_thermodynamics(
                                            halo_icon_hmc,halo_era5,
                                            Dropsondes,radar,date,
                                            os.getcwd()+"/"+flight[0]+"/",
                                            halo_icon_hmc["u"].index[0],
                                            halo_icon_hmc["u"].index[-1],
                                            hydrometeor_icon_path,
                                            with_ivt=True,do_masking=False,
                                            save_figure=True, low_level=True)
                        try:
                            ICON_HALO_Plotting.plot_AR_q_lat(halo_icon_hmc,halo_df,
                                            Dropsondes,radar,date,
                                            os.getcwd()+"/"+flight[0]+"/",
                                            halo_icon_hmc["u"].index[0],
                                            halo_icon_hmc["u"].index[-1],
                                            hydrometeor_icon_path,
                                            with_ivt=True,do_masking=False,
                                            save_figure=True, low_level=True)
                        except:
                            pass
                        if flight[0]=="RF10":
                            if include_retrieval:
                                cmpgn_cls.retrieval_humidity_plotting(
                                    halo_icon_hmc,retrieval_dict,Dropsondes,
                                    Upsampled_Dropsondes,date,
                                    flight[0],os.getcwd()+"/"+\
                                        flight[0]+"/",
                                        halo_icon_hmc["u"].index[0],
                                        halo_icon_hmc["u"].index[-1],
                                    hydrometeor_icon_path,
                                    plot_path=plot_path,
                                    with_ivt=True,
                                    do_masking=False,
                                    save_figure=True,
                                    low_level=True, 
                                    ar_of_day=ar_of_day)
    #%% Return of data
    if hmp_plotting_desired:
        halo_era5.name="ERA5"
        if not flight[0]=="SRF06":
            if carra_is_desired or icon_is_desired:
                high_res_hmp=high_res_hmp.groupby(level=0).first()
                try:
                    halo_era5["highres_Interp_IWV"]=high_res_hmp["Interp_IWV_clc"].values
                except:
                    halo_era5["highres_Interp_IWV"]=high_res_hmp["Interp_IWV"]
                halo_era5["highres_Interp_IVT"]=high_res_hmp["Interp_IVT"].values
                if carra_is_desired and not icon_is_desired:
                    halo_era5.name="CARRA"
                elif icon_is_desired:
                    halo_era5.name="ICON"
        if "aircraft_dict" in locals().keys():
            return halo_era5,radar,aircraft_dict
        else:
            return halo_era5,radar
    else:
        halo_grid_hmc=halo_era5_hmc.copy()
        halo_grid_hmc["name"]="ERA5"
        if carra_is_desired and not icon_is_desired:
            halo_grid_hmc=halo_carra_hmc.copy()
            halo_grid_hmc["name"]="CARRA"
        #elif icon_is_desired:
        #    halo_era5_hmc["name"]="ICON"
        if "aircraft_dict" in locals().keys():
            return halo_grid_hmc,radar,aircraft_dict        
        else:
            return halo_grid_hmc,radar
            
if __name__=="__main__":
    main(campaign="NA_February_Run",
         flight=["SRF07"],plot_data=True,synthetic_campaign=True)
