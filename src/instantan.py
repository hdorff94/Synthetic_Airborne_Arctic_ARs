# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:24:54 2022

@author: u300737
"""
import gridonhalo as GridHalo
class Instationarity(GridHalo.ERA_on_HALO,GridHalo.CARRA_on_HALO,):
    
    def __init__(self):
        self.na_campaign_name  = "North_Atlantic_Run"
        self.snd_campaign_name = "Second_Synthetic_Study"
        self.flight_dates      = {"North_Atlantic_Run":
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
                                       "SRF12":"20180225"}
                                     }
        self.na_flights  = [*self.flight_dates[self.na_campaign_name.keys()]]
        self.snd_flights = [*self.flight_dates[self.snd_campaign_name.keys()]]
        self.use_era     = True
        self.use_carra   = True
        if self.use_carra:
            self.ivt_arg="highres_Interp_IVT"
        else:
            self.ivt_arg="Interp_IVT"
        self.path_declarer()
    
    def path_declarer(self):
        import os
        import sys
        # Allocate path dictionary with all relevant variables
        path_dict={}
        # Change path to working script directory
        path_dict["current_path"]=os.getcwd()
        print(path_dict["current_path"])
        
        path_dict["major_path"]        = os.path.abspath("../../../")
        path_dict["base_working_path"] = path_dict["major_path"]+\
                                        "/my_GIT/Synthetic_Airborne_Arctic_ARs"
        path_dict["aircraft_base_path"]= path_dict["major_path"]+\
                                            "/Work/GIT_Repository/"
        path_dict["working_path"]      = path_dict["base_working_path"]+\
                                            "/src/"
        path_dict["script_path"]       = path_dict["base_working_path"]+\
                                            "/scripts/"
        path_dict["major_script_path"] = path_dict["base_working_path"]+\
                                            "/major_scripts/"
        
        path_dict["config_path"]       = path_dict["base_working_path"]+\
                                            "/config/"
        path_dict["plotting_path"]     = path_dict["base_working_path"]+\
                                            "/plotting/"
        
        sys.path.insert(1, os.path.join(sys.path[0],
                                path_dict["working_path"]))
        sys.path.insert(2,os.path.join(sys.path[0],
                                path_dict["script_path"]))
        sys.path.insert(3,os.path.join(sys.path[0],
                                path_dict["major_script_path"]))
        sys.path.insert(4, os.path.join(sys.path[0],
                                path_dict["config_path"]))
        sys.path.insert(5, os.path.join(sys.path[0],
                                path_dict["plotting_path"]))
        self.path_dict=path_dict
        
    def campaign_data_declarer(self):
        self.path_declarer()
        import data_config
        #-----------------------------------------------------------------------------#
        # Config File
        self.analyse_all_flights=True
        ## Configurations
        self.synthetic_campaign=True
        self.synthetic_flight=True
        self.cfg_file=data_config.load_config_file(
                        self.path_dict["aircraft_base_path"],
                        "data_config_file")
        
    def load_hmp_flights(self,flight_dates):
        import campaignAR_plotter
        
        # First campaign
        NA_Hydrometeors,NA_HALO_Dict,na_cls=campaignAR_plotter.main(
                    campaign=self.na_campaign_name,flights=self.na_flights,
                    era_is_desired=self.use_era, icon_is_desired=False,
                    carra_is_desired=self.use_carra, do_daily_plots=False,
                    calc_hmp=True,calc_hmc=False, do_instantaneous=False)
                                                           
        NA_Hydrometeors_inst,NA_HALO_Dict_inst,na_cls=campaignAR_plotter.main(
                    campaign=self.na_campaign_name,flights=self.na_flights,
                    era_is_desired=self.use_era,icon_is_desired=False,
                    carra_is_desired=self.use_carra,do_daily_plots=False,
                    calc_hmp=True,calc_hmc=False, do_instantaneous=True)
        
        # Second campaign
        SND_Hydrometeors,SND_HALO_Dict,SND_cls=campaignAR_plotter.main(
                    campaign=self.snd_campaign_name,flights=self.snd_flights,
                    era_is_desired=self.use_era, icon_is_desired=False,
                    carra_is_desired=self.use_carra, do_daily_plots=False,
                    calc_hmp=True,calc_hmc=False, do_instantaneous=False)
                                                           
        SND_Hydrometeors_inst,SND_HALO_Dict_inst,SND_cls=campaignAR_plotter.main(
                    campaign=self.snd_campaign_name,flights=self.snd_flights,
                    era_is_desired=self.use_erav,icon_is_desired=False,
                    carra_is_desired=self.use_carra,do_daily_plots=False,
                    calc_hmp=True,calc_hmc=False,
                    do_instantaneous=True)                                                        
    