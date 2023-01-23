# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:05:50 2022

@author: u300737
"""
import os
import sys
import warnings
    
# Principal Tools
#import numpy as np
#import pandas as pd
def importer():
    warnings.filterwarnings("ignore")
    #import seaborn as sns
    # Change path to working script directory
    paths_dict={}
    paths_dict["current_path"]          = os.getcwd()
    paths_dict["major_path"]            = os.path.abspath("../../../")
    paths_dict["base_working_path"]     = paths_dict["major_path"]+ \
                                        "/my_GIT/Synthetic_Airborne_Arctic_ARs"
    paths_dict["aircraft_base_path"]    = paths_dict["major_path"]+\
                                            "/Work/GIT_Repository/"
    paths_dict["config_path"]           = paths_dict["base_working_path"]+\
                                            "/config/"
    paths_dict["working_path"]          = paths_dict["base_working_path"]+\
                                            "/src/"
    paths_dict["script_path"]           = paths_dict["base_working_path"]+\
                                            "/scripts/"
    paths_dict["major_script_path"]     = paths_dict["base_working_path"]+\
                                            "/major_scripts/"
    #plotting_path = base_working_path+"/plotting/"
    
    sys.path.insert(1, os.path.join(sys.path[0], paths_dict["working_path"]))
    sys.path.insert(2, os.path.join(sys.path[0], paths_dict["config_path"]))
    sys.path.insert(3, os.path.join(sys.path[0], paths_dict["script_path"]))
    sys.path.insert(4, os.path.join(sys.path[0], paths_dict["major_script_path"]))
    print(paths_dict["working_path"])
    os.chdir(paths_dict["working_path"])
    return paths_dict

def main(flight_dates,do_plotting=False,instantan=False,sector_sonde_no=3,
         calc_hmp=False,calc_hmc=True,use_era=True,use_carra=False,
         use_icon=False,grid_name="ERA5",do_supplements=True,
         flight_locations=False):    
    paths_dict=importer()
    # Relevant created classes and modules
    import flightcampaign
    #from atmospheric_rivers import Atmospheric_Rivers

    #Grid Data
    # Run routines
    #import run_grid_data_on_halo # to run single days
    #import campaignAR_plotter # to run analysis for sequence of single days and create combined plots
    import data_config
    # IVT variability

    #from ivtvariability import IVT_Variability_Plotter
    #------------------------------------------------------------------------------#

    #import interpdata_plotting
    import moisturebudget as Budgets
    # Config File
    config_file=data_config.load_config_file(paths_dict["aircraft_base_path"],
                                             "data_config_file")

    # Major configurations
    #campaign="Second_Synthetic_Study"#"North_Atlantic_Run"#"Second_Synthetic_Study"
    #"North_Atlantic_Run"#"Second_Synthetic_Study"#"North_Atlantic_Run"### 
    init_flight="SRF02"
    if use_icon:
        grid_name="ICON_2km"
    elif use_carra:
        grid_name="CARRA"

    #flights=[*flight_dates[campaign].keys()]
    init_cmpgn_cls=flightcampaign.North_Atlantic_February_Run(
                is_flight_campaign=True,
                major_path=config_file["Data_Paths"]["campaign_path"],
                aircraft="HALO",interested_flights=init_flight,
                instruments=["radar","radiometer","sonde"])
    
    budget_data_path=init_cmpgn_cls.campaign_data_path+"budget/"
    if not os.path.exists(budget_data_path):
        os.makedirs(budget_data_path)
            
    budget_plot_path=init_cmpgn_cls.plot_path+"budget/"
    if not os.path.exists(budget_plot_path):
        os.makedirs(budget_plot_path)

    Moisture_CONV=Budgets.Moisture_Convergence(init_cmpgn_cls,init_flight,
                                           config_file,grid_name=grid_name,
                                           flight_dates=flight_dates,
                                           do_instantan=instantan,
                                           sonde_no=sector_sonde_no)

    Budget_plots=Budgets.Moisture_Budget_Plots(init_cmpgn_cls,init_flight,
                                           config_file,grid_name=grid_name,
                                           do_instantan=instantan)

    Moisture_CONV.calc_moisture_convergence_from_regression_method(
            config_file_path=paths_dict["aircraft_base_path"],
            do_plotting=do_plotting,
            calc_hmp=calc_hmp,calc_hmc=calc_hmc,
            use_era=use_era,use_carra=use_carra,
            use_icon=use_icon,do_supplements=do_supplements,
            use_flight_sonde_locations=flight_locations)
    return None

if __name__=="__main__":
    #%% Get data from all flights
    #
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

    flight_locations=True # has to be True for instantan comparison if sondes on 
                            # same positions should be compared
    main(flight_dates,grid_name="CARRA",instantan=True,do_supplements=False,
         flight_locations=flight_locations)
