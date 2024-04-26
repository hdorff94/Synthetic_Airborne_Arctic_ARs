# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 14:09:33 2021

@author: u300737
"""

import os
import glob
import sys

def main(campaign="North_Atlantic_Run",flights=["RF10","SRF02",
                                                "SRF03","SRF04",
                                                "SRF05","SRF06"],
         ar_of_days=["AR_internal"],do_daily_plots=True,calc_hmp=True,calc_hmc=True,
         era_is_desired=True,carra_is_desired=False,icon_is_desired=False,
         do_instantaneous=False, include_hydrometeors=False):
    #%% Predefining all paths to take scripts and data from and where to store
    actual_working_path=os.getcwd()+"/../"
    os.chdir(actual_working_path+"/config/")

    import init_paths
    import data_config
    working_path=init_paths.main()
    
    airborne_data_importer_path=working_path+"/Work/GIT_Repository/"
    airborne_script_module_path=actual_working_path+"/scripts/"
    airborne_processing_module_path=actual_working_path+"/src/"
    airborne_plotting_module_path=actual_working_path+"/plotting/"
    os.chdir(airborne_processing_module_path)
    sys.path.insert(1,airborne_script_module_path)
    sys.path.insert(2,airborne_processing_module_path)
    sys.path.insert(3,airborne_plotting_module_path)
    sys.path.insert(4,airborne_data_importer_path)
    # %% Load relevant modules
    
    import flightcampaign
    import run_grid_data_on_halo
    # Load config file
    config_file=data_config.load_config_file(airborne_data_importer_path,
                                             "data_config_file")
    
    do_plots=do_daily_plots
    if (campaign=="North_Atlantic_Run") or (campaign=="Second_Synthetic_Study"):
        synthetic_campaign=True
        synthetic_flight=True
    else:
        synthetic_campaign=False
        synthetic_flight=False

    
    if campaign=="NAWDEX":
        cpgn_cls_name="NAWDEX"
        NAWDEX=flightcampaign.NAWDEX(is_flight_campaign=True,
                    major_path=config_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",instruments=["radar","radiometer","sonde"])
        NAWDEX.dropsonde_path=NAWDEX.major_path+NAWDEX.name+"/data/HALO-Dropsonden/"
        cmpgn_cls=NAWDEX
    elif campaign=="North_Atlantic_Run":
        cpgn_cls_name="NA_February_Run"
        na_run=flightcampaign.North_Atlantic_February_Run(
                                        is_flight_campaign=True,
                                        major_path=config_file["Data_Paths"]\
                                                    ["campaign_path"],
                                        aircraft="HALO",
                                        interested_flights=flights,
                                        instruments=["radar","radiometer","sonde"])
        cmpgn_cls=na_run
    elif campaign=="Second_Synthetic_Study":
        cpgn_cls_name="Second_Synthetic_Study"
        na_run=flightcampaign.Second_Synthetic_Study(
            is_flight_campaign=True,major_path=config_file["Data_Paths"]["campaign_path"],
            aircraft="HALO",interested_flights=flights,
            instruments=["radar","radiometer","sonde"])
        cmpgn_cls=na_run               
    elif campaign=="HALO_AC3":
        cpgn_cls_name="HALO_AC3"
        if config_file["Data_Paths"]["campaign_path"]!=airborne_data_importer_path:
            config_file["Data_Paths"]["campaign_path"]=airborne_data_importer_path
        ac3_run=flightcampaign.HALO_AC3(is_flight_campaign=True,
            major_path=config_file["Data_Paths"]["campaign_path"],
            aircraft="HALO",interested_flights=flights,
            instruments=["radar","radiometer","sonde"])
        cmpgn_cls=ac3_run
    HMCs={}
    HMPs={}
    HALO_dict_dict={}
    i=0
    for flight in flights:
        
        HMCs[flight]={}
        HMPs[flight]={}
    
        for ar_of_day in [ar_of_days]:
            ar_of_day=ar_of_day[0]
            print(ar_of_day)
            #sys.exit()
            if not flight.startswith("S"):
                synthetic_campaign=False
            else:
                if cpgn_cls_name=="NAWDEX":
                    print("Campaign name has to be changed")
                    cpgn_cls_name="NA_February_Run"
                if synthetic_campaign==False:
                    synthetic_campaign=True
            if calc_hmp:
                #try:
                    HMPs[flight][ar_of_day],ar_rf_radar,HALO_dict_dict[flight]=\
                        run_grid_data_on_halo.main(
                            config_file_path=airborne_data_importer_path,
                            campaign=cpgn_cls_name,
                            hmp_plotting_desired=calc_hmp,
                            hmc_plotting_desired=calc_hmc,
                            plot_data=do_plots,
                            ar_of_day=ar_of_day,flight=[flight],
                            era_is_desired=era_is_desired,
                            carra_is_desired=carra_is_desired,
                            icon_is_desired=icon_is_desired,
                            synthetic_campaign=synthetic_campaign,
                            synthetic_flight=synthetic_flight,
                            do_instantaneous=do_instantaneous)
                
            if calc_hmc:
                    HMCs[flight][ar_of_day],ar_rf_radar,HALO_dict_dict[flight]=\
                        run_grid_data_on_halo.main(
                            config_file_path=airborne_data_importer_path,        
                            campaign=cpgn_cls_name,
                            hmp_plotting_desired=False,
                            hmc_plotting_desired=calc_hmc,
                            plot_data=do_plots,
                            ar_of_day=ar_of_day,flight=[flight],
                            era_is_desired=era_is_desired,
                            carra_is_desired=carra_is_desired,
                            icon_is_desired=icon_is_desired,
                            synthetic_campaign=synthetic_campaign,
                            synthetic_flight=synthetic_flight,
                            do_instantaneous=do_instantaneous,
                            include_hydrometeors=include_hydrometeors)
                
            #if 'Reflectivity' in ar_rf_radar.keys():
            #    if i==0:
            #        AR_radar=ar_rf_radar["Reflectivity"]
            #    else:
            #        AR_radar=pd.concat([AR_radar,ar_rf_radar["Reflectivity"]],
            #                       ignore_index=True)
            i+=1
    if calc_hmp:
        print("DATASET NAME:",HMPs[flight][ar_of_day].name)
        return HMPs,HALO_dict_dict,cmpgn_cls
    if calc_hmc:
        print("DATASET NAME:",HMCs[flight][ar_of_day]["name"])
        return HMCs,HALO_dict_dict,cmpgn_cls

###############################################################################
#%% Main data and plot creator
if __name__=="__main__":
    # This part runs the main funciton meaning all the stuff 
    # from run_grid_on_haloas well as plots and 
    # then (!) additionally is used for testing of IVT variability handling 
    # that will be runned in the ipnyb later on in order to not confuse people
    # too much. 
    
    # Relevant specifications for running , those are default values
    calc_hmp=False
    calc_hmc=True
    do_plotting=False
    
    ar_of_day=["AR_internal"]#["AR_entire_1"]#"AR_internal"]#"AR_entire_2"]#["AR3"]#"AR_entire"#"#internal"#"AR_entire"
    campaign_name="Second_Synthetic_Study"#"Second_Synthetic_Study"#"North_Atlantic_Run"#"HALO_AC3"#"
    if not campaign_name=="HALO_AC3":
        synthetic_campaign=True
    else:
        synthetic_campaign=False
    if synthetic_campaign:
        flights_to_analyse={#"SRF02":"20180224",#,#,
                        #"SRF04":"20190319",#}#,#,m,
                        #"SRF07":"20200416",#}#,#,#}#,#}#,
                        #"SRF08":"20200419"#,}
        #Second Synthetic Study
        
        #"SRF02":"20110317",
        #"SRF03":"20110423",#,
        #"SRF08":"20150314",#,
        "SRF09":"20160311",#,
        "SRF12":"20180225"
        }
    else:
        flights_to_analyse={#"RF02":"20220312",
                            #"RF03":"20220313",
                            #"RF04":"20220314",
                            #"RF05":"20220315",
                            "RF06":"20220316",
                            #"RF07":"20220320"
                            
                            #"RF10":"20161013"
                            }        
    use_era=True
    use_carra=True
    use_icon=False
    flights=[*flights_to_analyse.keys()]
    do_instantaneous=True
    include_hydrometeors=False
    Hydrometeors,HALO_Dict,cmpgn_cls=main(campaign=campaign_name,flights=flights,
                                          ar_of_days=ar_of_day,
                                          era_is_desired=use_era, 
                                          icon_is_desired=use_icon,
                                          carra_is_desired=use_carra,
                                          do_daily_plots=do_plotting,
                                          calc_hmp=calc_hmp,calc_hmc=calc_hmc,
                                          do_instantaneous=do_instantaneous,
                                          include_hydrometeors=include_hydrometeors)
    if do_instantaneous:
        import sys
        sys.exit()
    #%%
    ### IVT climatology
    
    #run_plot_IVT_long_term_stats(cmpgn_cls, Hydrometeors,flights_to_analyse)    
    
    ###
    """
    #%% IVT Variability
    flight_leg_to_take="inflow" # this can be either "inflow" or "outflow" or "both"
    single_flight_to_use=[*flights_to_analyse.keys()][0]
    analysed_flight=single_flight_to_use
    analysed_flight_df=HALO_Dict[analysed_flight][flight_leg_to_take]
    major_data_pah=config_file["Data_Paths"]\
                 ["campaign_path"]
    # sounding analysis
    # Define number of sondes per cross-section
    sonde_no=6
    # Logging
    import IVT_Variability_handler as IVT_handler
    
    
    log_file_name="logging_ivt_variability_icon.log"
    ivt_logger=IVT_handler.ICON_IVT_Logger(log_file_path=os.getcwd(),
                                            file_name=log_file_name)
    ivt_logger.create_plot_logging_file()
    sounding_frequency="standard"    
    # Dropsondes
    if sounding_frequency=="standard":
        sounding_name="Sounding"
        ivt_logger.icon_ivt_logger.info(
                        "Consider Standard Sounding")
    elif sounding_frequency=="Upsampled":
        sounding_name="Upsampled_Sounding"
        ivt_logger.icon_ivt_logger.info(
                    "Consider Upsampled Sounding")
    else: 
        sounding_name="Sounding"
        ivt_logger.icon_ivt_logger.info(
                    "Consider Standard Sounding")
    print(Hydrometeors)
    if not calc_hmc:
        grid_dict_hmc=None
        grid_dict_hmp=Hydrometeors[single_flight_to_use]["AR_internal"]
    if not calc_hmp:
        grid_dict_hmp=None
        grid_dict_hmc=Hydrometeors[single_flight_to_use]["AR_internal"]
    
    import IVT_Variability_handler as IVT_handler
    
    # Get sondes
    # Flight tracks where no sondes have been launched use synthetic soundings
    # So far, they are spaced equidistantly along cross-section
    sonde_dict={}
    sonde_dict["Pres"]=pd.DataFrame()
    sonde_dict["q"]=pd.DataFrame()
    sonde_dict["Wspeed"]=pd.DataFrame()
    sonde_dict["IVT"]=pd.DataFrame()
    
    ### here a new function should be added that creates 
    ### synthetic soundings at certain intervals along 
    ### HALO AR cross-sections
    #sonde_dict["IVT"]
    if analysed_flight.startswith("S"):
        only_model_sounding_profiles=True
        ar_of_day="AR_internal"
    else:
        only_model_sounding_profiles=False

    
    ## apparently need to be defined 
    # plot_path, ar_of_day,flight,
    if calc_hmc:
        if grid_dict_hmc["name"]=="ERA5":
            grid_dict_hmc["p"]=pd.DataFrame(data=np.tile(
                np.array(grid_dict_hmc["IWC"].columns[:].astype(float)),
                (grid_dict_hmc["IWC"].shape[0],1)),
                columns=[grid_dict_hmc["IWC"].columns[:]],
                index=grid_dict_hmc["IWC"].index)
        elif grid_dict_hmc["name"]=="CARRA":
            grid_dict_hmc["p"]=pd.DataFrame(data=np.tile(
                np.array(grid_dict_hmc["u"].columns[:].astype(float)),
                (grid_dict_hmc["u"].shape[0],1)),
                columns=[grid_dict_hmc["u"].columns[:]],
                index=grid_dict_hmc["u"].index)
    
    sonde_aircraft_df,sonde_dict=IVT_handler.create_synthetic_sondes(
                                        analysed_flight_df,
                            Hydrometeors[single_flight_to_use]["AR_internal"],
                            sonde_dict,hmps_used=calc_hmp,no_of_sondes=sonde_no)
    
    # Open classes
    # IVT Processing
    IVT_handler_cls=IVT_handler.IVT_variability(grid_dict_hmp,
                                    grid_dict_hmc,sonde_dict,
                                    only_model_sounding_profiles,
                                    sounding_frequency,
                                    HALO_Dict[single_flight_to_use]["inflow"],
                                    cmpgn_cls.plot_path,ar_of_day,
                                    analysed_flight,ivt_logger)
    # Plot routine
    IVT_var_Plotter=IVT_handler.IVT_Variability_Plotter(grid_dict_hmp,
                                    grid_dict_hmc,sonde_dict,
                                    only_model_sounding_profiles,sounding_frequency,
                                    HALO_Dict[single_flight_to_use]["inflow"],
                                    cmpgn_cls.plot_path,ar_of_day,
                                    analysed_flight,ivt_logger)

        #%% IVT Analysis
    if calc_hmc:
#        IVT_handler_cls.calc_vertical_quantiles(do_all_preps=True)
#        IVT_handler_cls.calc_vertical_quantiles(use_grid=False,do_all_preps=True)
        # Vertical moisture transport variability
        IVT_var_Plotter.plot_IVT_vertical_variability()    
    
    if calc_hmp:
        if analysed_flight.startswith("S"):
            synthetic_flight=True
        else:
            synthetic_flight=False
        
        IVT_var_Plotter.plot_model_sounding_frequency_comparison(
                            name_of_grid_data=grid_dict_hmp.name)
        IVT_var_Plotter.plot_distance_based_IVT(False,
                            synthetic_flight=synthetic_flight,delete_sondes=None,
                            name_of_grid_data=grid_dict_hmp.name)
        IVT_var_Plotter.plot_distance_based_IVT(False,
                            synthetic_flight=synthetic_flight,delete_sondes=None,
                            name_of_grid_data=grid_dict_hmp.name,show_sondes=False)
        IVT_handler_cls.study_TIVT_sondes_grid_frequency_dependency()
        
        if use_era:
            IVT_handler_cls.TIVT["grid_name"]="ERA5"
        if use_carra:
            IVT_handler_cls.TIVT["grid_name"]="CARRA"
        if use_icon:
            IVT_handler_cls.TIVT["grid_name"]="ICON_2km"
        IVT_var_Plotter.TIVT=IVT_handler_cls.TIVT
        IVT_var_Plotter.plot_TIVT_error_study()
        
        #if calc_hmp:
        #    plot_ivt_in_outflow(cmpgn_cls,Hydrometeors,Halo_dict,config_file,high_res=True)
    

        #    cfad_ar_df=NAWDEX.calculate_cfad_radar_reflectivity(AR_radar)
        #    NAWDEX.plot_cfad_2d_hist(cfad_ar_df,os.getcwd()+"/NAWDEX/plots/",True)
"""