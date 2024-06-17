# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:56:17 2022

@author: u300737
"""
#Basics
import os
import sys

# Calc Tools
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
#%% Path definitions for importing modules
def importer():
    major_path = os.path.abspath("../../../")
    base_working_path=major_path+"/my_GIT/Synthetic_Airborne_Arctic_ARs/"
    aircraft_base_path=major_path+"/Work/GIT_Repository/"
    working_path  = base_working_path+"/src/"
    config_path   = base_working_path+"/config/"
    scripts_path  = base_working_path+"/major_scripts/"
    plotting_path = base_working_path+"/plotting/"
    
    plot_figures_path = aircraft_base_path+\
                            "/../Synthetic_AR_Paper/Manuscript/Paper_Plots/"
                            
    sys.path.insert(1, os.path.join(sys.path[0], working_path))
    sys.path.insert(2, os.path.join(sys.path[0], config_path))
    sys.path.insert(3, os.path.join(sys.path[0], plotting_path))
    sys.path.insert(4,os.path.join(sys.path[0],  scripts_path))

    paths_dict={}
    paths_dict["aircraft_base_path"] = aircraft_base_path
    paths_dict["working_path"]       = working_path
    paths_dict["plotting_path"]      = plotting_path
    paths_dict["plot_figures_path"]  = plot_figures_path
    return paths_dict    

def main(figure_to_create="fig13", include_haloac3=False):
    
    paths_dict=importer()
    # Import relevant modules    
    import flight_track_creator
    import data_config
    # Config File
    config_file=data_config.load_config_file(paths_dict["aircraft_base_path"],
                                             "data_config_file")

    # Relevant created classes and modules
    import flightcampaign
    import moisturebudget as Budgets
    #####################################################################
    #%% Specifications
    # Major configurations
    campaign="Second_Synthetic_Study"#"North_Atlantic_Run"

    init_flight="SRF08"
    grid_name="CARRA"#"CARRA"#"ERA5"
    sonde_no="3"
    do_instantan=False
    do_plotting=True
    save_for_manuscript=True
    scalar_based_div=False
    if do_instantan:
        flight=init_flight+"_instantan"
    else:
        flight=init_flight

    flight_dates={"North_Atlantic_Run":
              {"SRF02":"20180224",
               "SRF04":"20190319",
               "SRF07":"20200416",
               "SRF08":"20200419"
              },
              "Second_Synthetic_Study":
              {"SRF02":"20110317",
               "SRF03":"20110423",
               "SRF08":"20150314",
               "SRF09":"20160311",
               "SRF12":"20180225"
              }
        }
    # Access classes
    flights=[*flight_dates[campaign].keys()]

    if campaign=="North_Atlantic_Run":
        cmpgn_cls=flightcampaign.North_Atlantic_February_Run(
                    is_flight_campaign=True,
                    major_path=config_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=flight,
                    instruments=["radar","radiometer","sonde"])
    elif campaign=="Second_Synthetic_Study":
        cmpgn_cls=flightcampaign.Second_Synthetic_Study(
                    is_flight_campaign=True,
                    major_path=config_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=flight,
                    instruments=["radar","radiometer","sonde"])
    else:
        pass

    #---------------------------------------------------------------------#
    # HALO-(AC)3 contribution
    if include_haloac3:
        haloac3_div   = pd.DataFrame(data=np.nan,index=range(8),
                             columns=["values","sector"])
        era5_div      = pd.DataFrame(data=np.nan,index=range(8),
                             columns=["values","sector"])
        inst_era5_div = pd.DataFrame(data=np.nan,index=range(8),
                             columns=["values","sector"])
        
        haloac3_div.iloc[0:4,0]    = [0.497,0.5396,-0.006,-0.035]
        #haloac3_div.iloc[0:4,0]    = [0.320,0.369,-0.016,-0.034]
        haloac3_div.iloc[0:4,1]    = [1,1,1,1] #"Warm\nCONV"]
        haloac3_div.iloc[4:8,0]    = [0.27,0.5539,-0.5935,-0.278]
        #haloac3_div.iloc[4:8,0]    = [0.162,0.309,-0.432,-0.199]
        haloac3_div.iloc[4:8,1]    =  [2,2,2,2] #"Warm\nADV"]
        # Continuous ERA5 for instantaneous intercomparison
        era5_div.iloc[0:4,0]       = [0.606904,0.539302,
                                      0.014326,-0.004315] #warm conv
        era5_div.iloc[0:4,1]       = [1,1,1,1]
        era5_div.iloc[4:8,0]       = [-0.000405,0.500588,
                                      -0.563,-0.43281] # ADV
        era5_div.iloc[4:8,1]       = [2,2,2,2]
        
        inst_era5_div.iloc[0:4,0]  = [-0.637,1.074779,-0.406,0.247] # Warm CONV
        inst_era5_div.iloc[0:4,1]  = [1,1,1,1]
        inst_era5_div.iloc[4:8,0]  = [-0.773947,-0.798314,
                                      -0.648433,-1.340648] # Warm ADV
        inst_era5_div.iloc[4:8,1]  = [2,2,2,2]
        
        #inst_haloac3_div
    else:
        haloac3_div   = pd.DataFrame()
        era5_div      = pd.DataFrame()
        inst_era5_div = pd.DataFrame()
    #---------------------------------------------------------------------#
    # Moisture Classes
    Moisture_CONV=\
        Budgets.Moisture_Convergence(cmpgn_cls,flight,config_file,
                 flight_dates=flight_dates,grid_name=grid_name,
                 do_instantan=do_instantan,sonde_no=sonde_no,
                 calc_from_scalar_values=scalar_based_div)
    Budget_plots=Budgets.Moisture_Budget_Plots(cmpgn_cls,flight,config_file,
                 grid_name=grid_name,do_instantan=do_instantan,
                 sonde_no=sonde_no,scalar_based_div=scalar_based_div,
                 include_halo_ac3_components=haloac3_div,
                 include_era5_components=era5_div,
                 include_inst_components=inst_era5_div,
                 hours_to_use=1)
    Inst_Budget_plots=Budgets.Moisture_Budget_Plots(cmpgn_cls,flight,
                config_file,grid_name=grid_name,do_instantan=True,
                sonde_no=sonde_no,scalar_based_div=scalar_based_div,
                include_halo_ac3_components=haloac3_div,
                include_era5_components=era5_div,
                include_inst_components=inst_era5_div,
                hours_to_use=1)
    on_flight_tracks=True
    if not haloac3_div.shape[0]==0:
        save_for_manuscript=False
        
    if figure_to_create.startswith("fig12"):
        # Create sonde profiles ADV,CONV exemplary case
        Sectors,Ideal_Sectors,cmpgn_cls=\
            Moisture_CONV.load_moisture_convergence_single_case()
        if do_plotting:
            Budget_plots.plot_single_case(Sectors,Ideal_Sectors,
                                save_as_manuscript_figure=save_for_manuscript)
    elif figure_to_create.startswith("fig13"):
        # Budget components (continuous, sonde-based)
        Campaign_Budgets,Campaign_Ideal_Budgets=\
            Moisture_CONV.get_overall_budgets()
        if do_plotting:
            Budget_plots.moisture_convergence_cases_overview(
                            Campaign_Budgets,Campaign_Ideal_Budgets,
                            save_as_manuscript_figure=save_for_manuscript,
                            with_mean_error=True)
    
    elif figure_to_create.startswith("fig14"):
        # Budget components (continuous flight, continuous instantaneous)
        # Here we take the continuous representation
        # so reset the sonde no
        # Instantan Budgets
        Inst_Moisture_CONV=Budgets.Moisture_Convergence(cmpgn_cls,
                flight+"_instantan",config_file,flight_dates=flight_dates,
                grid_name=grid_name,do_instantan=True,
                calc_from_scalar_values=scalar_based_div)
        #Inst_Moisture_CONV.sonde_no="100"
        Inst_Budgets,Inst_Ideal_Budgets=Inst_Moisture_CONV.get_overall_budgets(
                        use_flight_tracks=on_flight_tracks)
        # Airborne Budgets
        Campaign_Budgets,Campaign_Ideal_Budgets=\
            Moisture_CONV.get_overall_budgets(use_flight_tracks=on_flight_tracks)
        
        #---------------------------------------------------------------------#
        # OLD
        #Inst_Budget_plots.moisture_convergence_cases_overview(
        #                    Campaign_Budgets=Campaign_Budgets,
        #                    Campaign_Ideal_Budgets=Campaign_Ideal_Budgets,
        #                    Campaign_Inst_Budgets={},
        #                    Campaign_Inst_Ideal_Budgets=Inst_Ideal_Budgets,
        #                    instantan_comparison=True,
        #                    save_as_manuscript_figure=False)
        #---------------------------------------------------------------------#
        Inst_Budget_plots.moisture_convergence_time_instantan_comparison(
                Campaign_Budgets=Campaign_Budgets,
                Campaign_Ideal_Budgets=Campaign_Ideal_Budgets,
                Campaign_Inst_Budgets={},
                Campaign_Inst_Ideal_Budgets=Inst_Ideal_Budgets,
                save_as_manuscript_figure=save_for_manuscript,
                use_flight_tracks=on_flight_tracks,
                plot_mean_error=True)
        
    elif figure_to_create.startswith("fig15"):
        # This figure compares continuous instantan divergence representation
        # with the time-propagating sonde based values
        
        Campaign_Budgets,Campaign_Ideal_Budgets=\
            Moisture_CONV.get_overall_budgets()
        Inst_Moisture_CONV=Budgets.Moisture_Convergence(cmpgn_cls,
                                    flight+"_instantan",config_file,
                                    flight_dates=flight_dates,
                                    grid_name=grid_name,do_instantan=True,
                                    calc_from_scalar_values=scalar_based_div)
        Inst_Budgets,Inst_Ideal_Budgets=Inst_Moisture_CONV.get_overall_budgets(
            use_flight_tracks=on_flight_tracks)
       
        if do_plotting:
            Inst_Budget_plots.moisture_convergence_cases_overview(
                            Campaign_Budgets=Campaign_Budgets,
                            Campaign_Ideal_Budgets=Campaign_Ideal_Budgets,
                            Campaign_Inst_Budgets={},
                            Campaign_Inst_Ideal_Budgets=Inst_Ideal_Budgets,
                            instantan_comparison=True,
                            save_as_manuscript_figure=False)
            Inst_Budget_plots.plot_rmse_instantan_sonde(
                save_as_manuscript_figure=True)
            #-----------------------------------------------------------------#
            # OLD to be deleted
            #            #Inst_Budget_plots.sonde_divergence_error_bar(
            #    save_as_manuscript_figure=True)            
            #        Flight_Moisture_CONV=Moist_Convergence(
            #                        cmpgn_cls,flight,self.cfg_file,
            #                        grid_name=self.grid_name,do_instantan=False)    
            #            Flight_Sectors,Flight_Ideal_Sectors,cmpgn_cls=\
            #      Flight_Moisture_CONV.load_moisture_convergence_single_case()            
            #-----------------------------------------------------------------#
    elif figure_to_create.startswith("fig_supplements"):
        import interpdata_plotting
        #---------------------------------------------------------------------#
        #%% Vertical distribution of moisture & wind speed cross-section contour
        calc_hmp=False
        calc_hmc=True
        do_plotting=False
        synthetic_campaign=True
        ar_of_day=["AR_internal"]
 
        NA_flights_to_analyse={"SRF02":"20180224",#,#,
                               "SRF04":"20190319",#}#,#,
                               "SRF07":"20200416",#}#,#,#}#,#}#,
                               "SRF08":"20200419"
                               }
        #Second Synthetic Study
        SND_flights_to_analyse={"SRF02":"20110317",
                                "SRF03":"20110423",#,
                                "SRF08":"20150314",#,
                                "SRF09":"20160311",#,
                                "SRF12":"20180225"
                                }
        use_era=False
        use_carra=True
        use_icon=False
        
        
        na_flights=[*NA_flights_to_analyse.keys()]
        snd_flights=[*SND_flights_to_analyse.keys()]
        
        do_instantaneous=True        
        
        #---------------------------------------------------------------------#
        # Instantan Budgets
        Inst_Moisture_CONV=Budgets.Moisture_Convergence(cmpgn_cls,
                                    flight+"_instantan",config_file,
                                    flight_dates=flight_dates,
                                    grid_name=grid_name,do_instantan=True)
        
        Inst_Budgets,Inst_Ideal_Budgets=Inst_Moisture_CONV.get_overall_budgets(
            use_flight_tracks=on_flight_tracks)
        # Airborne Budgets
        flight_Budgets,flight_Ideal_Budgets=Moisture_CONV.get_overall_budgets(
            use_flight_tracks=on_flight_tracks)
        
        #######################################################################
        # Sonde position comparison/verification
        # read sonde positions
        Inst_Budget_plots.compare_inst_sonde_pos(flight_dates,
                                    Campaign_Budgets=flight_Budgets,
                                    Campaign_Inst_Budgets=Inst_Budgets,
                                    save_as_manuscript_figure=False,
                                    use_flight_tracks=on_flight_tracks)
        #######################################################################
        # Flight-specific mean error in convergence frontal sector
        #######################################################################
        Inst_Budget_plots.mean_errors_per_flight(flight_dates,
                                                 flight_Ideal_Budgets,
                                                 Inst_Ideal_Budgets,
                                                 save_as_manuscript_figure=True)
        
if __name__=="__main__":
    # Figures to create choices:
    #figure_to_create="fig12_single_case_sector_profiles"
    #figure_to_create="fig13_campaign_divergence_overviews"
    figure_to_create="fig14_divergence_instantan_errorbars"
    #figure_to_create="fig15_campaign_divergence_overview_instantan_comparison"
    
    #figure_to_create="fig_supplements_sonde_pos_comparison"
    #figure_to_create="fig12_campaign_divergence_overviews"
    main(figure_to_create=figure_to_create,
         include_haloac3=True)
