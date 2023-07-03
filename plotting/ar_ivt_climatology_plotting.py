# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:58:33 2022

@author: u300737
"""
def main(first_flights=["SRF02","SRF04","SRF07","SRF08"],
         snd_flights=["SRF02","SRF03","SRF08","SRF09","SRF12"]):
    """
    This creates the AR climatology comparison (Fig02 of the manuscript)
    by running run_plot_combined_campaign_IVT_long_term_stats from
    ivtclimatology module

    Parameters
    ----------
    first_flights : TYPE, optional
        DESCRIPTION. The default is ["SRF02","SRF04","SRF07","SRF08"].
    snd_flights : TYPE, optional
        DESCRIPTION. The default is ["SRF02","SRF03","SRF08","SRF09","SRF12"].

    Returns
    -------
    None.

    """
    
    import os
    import sys
    #%%
    ### Define all paths and import the relevant modules
    actual_working_path=os.getcwd()+"/../"
    os.chdir(actual_working_path+"/config/")
    
    import init_paths
    import data_config
    working_path=init_paths.main()
        
    airborne_data_importer_path=working_path+"/Work/GIT_Repository/"
    airborne_processing_module_path=actual_working_path+"/src/"
    airborne_plotting_module_path=actual_working_path+"/plotting/"
    os.chdir(airborne_processing_module_path)
    sys.path.insert(1,airborne_processing_module_path)
    sys.path.insert(2,airborne_plotting_module_path)
    sys.path.insert(3,airborne_data_importer_path)
    
    import flightcampaign
    import ivtclimatology
    #%% Predefine one campaign class to store the plot in
    config_file_path=airborne_data_importer_path
    config_file=data_config.load_config_file(config_file_path,
                                             "data_config_file")
    cmpgn_cls_st=flightcampaign.North_Atlantic_February_Run(
            is_flight_campaign=True,
            major_path=config_file["Data_Paths"]\
                ["campaign_path"],aircraft="HALO",
            interested_flights=first_flights,instruments=[])
    cmpgn_cls_snd=flightcampaign.Second_Synthetic_Study(
            is_flight_campaign=True,
            major_path=config_file["Data_Paths"]\
                ["campaign_path"],aircraft="HALO",
            interested_flights=snd_flights,
            instruments=[])

    #%% Create Figure
    ivtclimatology.run_plot_combined_campaign_IVT_long_term_stats([cmpgn_cls_st,
                                                               cmpgn_cls_snd],
                                 upper_lat=90,lower_lat=55,
                                 western_lon=-30,eastern_lon=90,
                                 add_single_flight=None)

if __name__ == "__main__":
    main()    