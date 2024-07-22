# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:32:51 2022

@author: u300737
"""
#import data_config
import os
import sys

import numpy as np
import pandas as pd

import metpy.calc as mpcalc
from metpy.units import units
    
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
from ivtvariability import IVT_Variability_Plotter

if not "flightcampaign" in sys.modules:
    import flightcampaign as Flight_Campaign

class Moisture_Budgets():
    def __init__(self):
        pass
    
class Moisture_Convergence(Moisture_Budgets):
    
    def __init__(self,cmpgn_cls,flight,config_file,flight_dates={},
                 sonde_no=3,sector_types=["warm","core","cold"],
                 ar_of_day="AR",grid_name="ERA5",do_instantan=False,
                 calc_from_scalar_values=True):
        
        self.cmpgn_cls=cmpgn_cls
        self.grid_name=grid_name
        self.do_instantan=do_instantan
        self.flight=flight
        self.config_file=config_file
        self.ar_of_day=ar_of_day
        self.scalar_based_div=calc_from_scalar_values
        if flight_dates=={}:
            
            self.flight_dates={"North_Atlantic_Run":
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
        self.flight_dates=flight_dates
        self.sonde_no=sonde_no
        self.sector_types=sector_types
        self.sector_colors={"warm_sector":"orange",
                            "core":"darkgreen",
                            "cold_sector":"darkblue"}            
    
    def new_vertically_integrated_divergence(self):
        scalar_based_div=self.scalar_based_div
        integrated_divergence={}
        
        for sector in self.sector_types:
            ###################################################################
            # get mean pressure values
            p_grid=self.sector_sonde_values[sector]["pres"].mean(axis=1)
            pres_index=pd.Series(p_grid*100)
            #print(pres_index)
            try:
                pres_index=pres_index.loc[self.div_scalar_mass[sector].index]
            except:
                pres_index=pres_index.loc[self.div_vector_mass[sector].index]
            g=9.82
            ###################################################################
            # 
            integrated_divergence[sector]={}
            # Values are vector based
            if isinstance(self.div_vector_mass,pd.Series):
                    integrated_divergence[sector]["mass_div"]= 1/(g*997)*np.trapz(\
                        self.div_vector_mass[sector].values/1000,axis=0,
                    x=pres_index)*1000*3600
            else:
                integrated_divergence[sector]["mass_div"]= 1/(g*997)*np.trapz(\
                    self.div_vector_mass[sector]["val"].values/1000,axis=0,
                    x=pres_index)*1000*3600
                #integrated_divergence[sector]["mass_div_min"]=1/(g*997)*np.trapz(\
                #    (self.div_vector_mass[sector]["val"].values-
                #     self.div_vector_mass[sector]["unc"].values)*\
                #        pres_index[::-1])*1000*3600
                #integrated_divergence[sector]["mass_div_max"]=1/(g*997)*np.trapz(\
                #    (self.div_vector_mass[sector]["val"].values+
                #     self.div_vector_mass[sector]["unc"].values)*\
                #        pres_index[::-1])*1000*3600
                
            if isinstance(self.adv_q_vector,pd.Series):
                integrated_divergence[sector]["q_ADV"]=1/(g*997)*np.trapz(\
                    self.adv_q_vector[sector].values/1000,
                    axis=0,x=pres_index)*1000*3600
            else:
                integrated_divergence[sector]["q_ADV"]=1/(g*997)*np.trapz(\
                    self.adv_q_vector[sector]["val"].values/1000,
                    axis=0,x=pres_index)*1000*3600
                    
                #integrated_divergence[sector]["q_ADV_min"]=1/(g*997)*np.trapz(\
                #    (self.adv_q_vector[sector]["val"].values-
                #     self.adv_q_vector[sector]["unc"].values)*\
                #        pres_index)/1000*3600
                #integrated_divergence[sector]["q_ADV_max"]=1/(g*997)*np.trapz(\
                #    (self.adv_q_vector[sector]["val"].values+
                #     self.adv_q_vector[sector]["unc"].values)*\
                #pres_index)/1000*3600
            #for term in ["ADV","CONV","TRANSP"]:
            #        if term=="ADV":
            #            series_term=term+"_calc"
            #        else:
            #            series_term=term
            #        core_budgets[term].at[campaign_id+flight+\
            #                              "_sonde_"+str(self.sonde_no)+term]=\
            #        1/g*np.trapz(core[series_term][::-1],axis=0,x=pres_index[::-1])
            #        warm_budgets[term].at[campaign_id+flight+\
            #                              "_sonde_"+str(self.sonde_no)+term]=\
            #        1/g*np.trapz(warm[series_term][::-1],
            #                     axis=0,x=pres_index[::-1])
        self.integrated_divergence=integrated_divergence
    def vertically_integrated_divergence(self):
        scalar_based_div=self.scalar_based_div
        integrated_divergence={}
        
        for sector in self.sector_types:
            ###################################################################
            # get mean pressure values
            p_grid=self.sector_sonde_values[sector]["pres"].mean(axis=1)
            pres_index=pd.Series(p_grid*100)
            #print(pres_index)
            try:
                pres_index=pres_index.loc[self.div_scalar_mass[sector].index]
            except:
                pres_index=pres_index.loc[self.div_vector_mass[sector].index]
            g=9.82
            ###################################################################
            # 
            integrated_divergence[sector]={}
            if scalar_based_div:
                if isinstance(self.div_scalar_mass,pd.Series):
                    integrated_divergence[sector]["mass_div"]= 1/(g*997)*np.trapz(\
                        self.div_scalar_mass[sector].values[::-1]*pres_index[::-1])/\
                        1000*3600
                else:
                    integrated_divergence[sector]["mass_div"]= 1/(g*997)*np.trapz(\
                        self.div_scalar_mass[sector]["val"].values[::-1]*\
                            pres_index[::-1])/1000*3600
                    integrated_divergence[sector]["mass_div_min"]=1/(g*997)*np.trapz(\
                        (self.div_scalar_mass[sector]["val"].values[::-1]-
                         self.div_scalar_mass[sector]["unc"].values[::-1])*\
                            pres_index[::-1])/1000*3600
                    integrated_divergence[sector]["mass_div_max"]=1/(g*997)*np.trapz(\
                        (self.div_scalar_mass[sector]["val"].values[::-1]+
                         self.div_scalar_mass[sector]["unc"].values[::-1])*\
                            pres_index[::-1])/1000*3600
                    
                if isinstance(self.adv_q_calc,pd.Series):
                    integrated_divergence[sector]["q_ADV"]=1/(g*997)*np.trapz(\
                        self.adv_q_calc[sector].values[::-1]*pres_index[::-1])/\
                        1000*3600
                else:
                    integrated_divergence[sector]["q_ADV"]=1/(g*997)*np.trapz(\
                        self.adv_q_calc[sector]["val"].values[::-1]*\
                            pres_index[::-1])/1000*3600
                    integrated_divergence[sector]["q_ADV_min"]=1/(g*997)*np.trapz(\
                        (self.adv_q_calc[sector]["val"].values[::-1]-
                         self.adv_q_calc[sector]["unc"].values[::-1])*\
                            pres_index[::-1])/1000*3600
                    integrated_divergence[sector]["q_ADV_max"]=1/(g*997)*np.trapz(\
                        (self.adv_q_calc[sector]["val"].values[::-1]+
                         self.adv_q_calc[sector]["unc"].values[::-1])*\
                    pres_index[::-1])/1000*3600
            else:
                # Values are vector based
                if isinstance(self.div_vector_mass,pd.Series):
                    integrated_divergence[sector]["mass_div"]= 1/(g*997)*np.trapz(\
                        self.div_vector_mass[sector].values*pres_index)/\
                        1000*3600
                else:
                    integrated_divergence[sector]["mass_div"]= 1/(g*997)*np.trapz(\
                        self.div_vector_mass[sector]["val"].values*\
                            pres_index)/1000*3600
                    integrated_divergence[sector]["mass_div_min"]=1/(g*997)*np.trapz(\
                        (self.div_vector_mass[sector]["val"].values-
                         self.div_vector_mass[sector]["unc"].values)*\
                            pres_index[::-1])/1000*3600
                    integrated_divergence[sector]["mass_div_max"]=1/(g*997)*np.trapz(\
                        (self.div_vector_mass[sector]["val"].values+
                         self.div_vector_mass[sector]["unc"].values)*\
                            pres_index[::-1])/1000*3600
                    
                if isinstance(self.adv_q_vector,pd.Series):
                    integrated_divergence[sector]["q_ADV"]=1/(g*997)*np.trapz(\
                        self.adv_q_vector[sector].values*pres_index)/\
                        1000*3600
                else:
                    integrated_divergence[sector]["q_ADV"]=1/(g*997)*np.trapz(\
                        self.adv_q_vector[sector]["val"].values*\
                            pres_index)/1000*3600
                    integrated_divergence[sector]["q_ADV_min"]=1/(g*997)*np.trapz(\
                        (self.adv_q_vector[sector]["val"].values-
                         self.adv_q_vector[sector]["unc"].values)*\
                            pres_index)/1000*3600
                    integrated_divergence[sector]["q_ADV_max"]=1/(g*997)*np.trapz(\
                        (self.adv_q_vector[sector]["val"].values+
                         self.adv_q_vector[sector]["unc"].values)*\
                    pres_index)/1000*3600
            #for term in ["ADV","CONV","TRANSP"]:
            #        if term=="ADV":
            #            series_term=term+"_calc"
            #        else:
            #            series_term=term
            #        core_budgets[term].at[campaign_id+flight+\
            #                              "_sonde_"+str(self.sonde_no)+term]=\
            #        1/g*np.trapz(core[series_term][::-1],axis=0,x=pres_index[::-1])
            #        warm_budgets[term].at[campaign_id+flight+\
            #                              "_sonde_"+str(self.sonde_no)+term]=\
            #        1/g*np.trapz(warm[series_term][::-1],
            #                     axis=0,x=pres_index[::-1])
        self.integrated_divergence=integrated_divergence
                
            #integrated_divergence[sector]["q_ADV"]=
            #integrated_divergence[sector]["mass_div"]=
    
    def run_rough_budget_closure(self,wind_field,q_field,moisture_transport,
                                 wind_sector_inflow,wind_sector_outflow,
                                 q_sector_inflow,q_sector_outflow,
                                 moist_transport_sector_inflow,
                                 moist_transport_sector_outflow,pressure,
                                 AR_inflow,AR_outflow,sector="core",
                                 do_supplements=False):
        
        #mean q_core
        mean_trpz_wind=wind_field.mean()
        mean_trpz_q=q_field.mean()
        mean_trpz_moist_transport=moisture_transport.mean()

        mean_sector_trpz_wind=pd.concat([wind_sector_inflow,wind_sector_outflow]).mean()
        mean_sector_trpz_q=pd.concat([q_sector_inflow,q_sector_outflow]).mean()

        mean_sector_trpz_moist_transport=pd.concat([moist_transport_sector_inflow,
                                                moist_transport_sector_outflow]).mean()

        print("Inflow leg distance: ",str((AR_inflow["AR_inflow_"+sector]\
                ["IVT_max_distance"][-1]-AR_inflow["AR_inflow_"+sector]\
                    ["IVT_max_distance"][0])/1000)+" km")
        print("Outflow leg distance: ",str((AR_outflow["AR_outflow_"+sector]\
                ["IVT_max_distance"][-1]-AR_outflow["AR_outflow_"+sector]\
                    ["IVT_max_distance"][0])/1000)+" km")
    
        inflow_lat_lon=(AR_inflow["AR_inflow_"+sector]["Halo_Lat"].mean(),
                        AR_inflow["AR_inflow_"+sector]["Halo_Lat"].mean())
        outflow_lat_lon=(AR_outflow["AR_outflow_"+sector]["Halo_Lat"].mean(),
                     AR_outflow["AR_outflow_"+sector]["Halo_Lat"].mean())
        
        import gridonhalo as Grid_on_HALO
        
        mean_distance=Grid_on_HALO.harvesine_distance(inflow_lat_lon,
                                                      outflow_lat_lon)
        Budget_plots=Moisture_Budget_Plots(self.cmpgn_cls,
                                           self.flight,
                                           self.config_file,
                                           grid_name=self.grid_name)
        if do_supplements:
            Budget_plots.plot_comparison_sector_leg_wind_q_transport(
            mean_trpz_wind,mean_trpz_q,mean_trpz_moist_transport,
            mean_sector_trpz_q,mean_sector_trpz_wind,
            mean_sector_trpz_moist_transport,
            pressure,sector)
    
            budget_profile_df=Budget_plots.plot_moisture_budget_divergence_components(
            q_sector_inflow,q_sector_outflow,wind_sector_inflow,wind_sector_outflow,
            moist_transport_sector_inflow,moist_transport_sector_outflow,
            mean_sector_trpz_q,mean_sector_trpz_wind,mean_sector_trpz_moist_transport,
            pressure,mean_distance,sector)
    
    def add_synthetic_sondes(self,sector_to_plot="warm",
            additional_sondes=pd.DataFrame()):
        if additional_sondes.isempty():
            print("DataFrame with additional sondes is empty.",
                  "Nothing is done.")
        else:    
            new_sondes_pos_all=self.sondes_pos_all.copy()
            del new_sondes_pos_all[sector_to_plot]["dx"]
            del new_sondes_pos_all[sector_to_plot]["dy"]
            sector_sonde_values={}
            sector_relevant_times={}
    
            new_sondes_pos_all[sector_to_plot]=\
                new_sondes_pos_all[sector_to_plot].append(additional_sondes)
            # Positions relevant for divergence calculations
            self.sondes_pos_all[sector_to_plot]=self.get_xy_coords_for_domain(
               new_sondes_pos_all[sector_to_plot])

    def load_moisture_convergence_single_case(self,campaign="same"):
        """
    

        Parameters
        ----------
        campaign : str, Default is same
        flight campaign to be analysed. If campaign=="same" cmpgn_cls.name
        serves as campaign str argument
        flight : str
            flight from campaign to be analysed, check correct connection between
            flight number SRF* and chosen campaign.
            
            ---> this was deprecated for the class
            
        sonde_no : str, optional
            Sonde no used for divergence calculations. The default is "2", 
            representing aircraft-based synthetic profiles.
            The other possible str argument is "100" which means 100 sondes per sector
            which is approximately a continuous derivation of moisture convergence
            in this sector. However, this ideal case is considered any way.
    
        grid_name : str
        grid dataset in which divergence is calculated, so far ERA5 and CARRA 
        is possible, if it was calculated before. 
        
        Raises
        ------
        Exception
            if a wrong campaign name is used.

        Returns
        -------
        None.

        """
        if not campaign=="same":
           # if not "Flight_Campaign" in sys.modules():
            import flightcampaign
            if campaign=="North_Atlantic_Run":
                cmpgn_cls=flightcampaign.North_Atlantic_February_Run(
                                    is_flight_campaign=True,
                                    major_path=self.config_file["Data_Paths"]\
                                                ["campaign_path"],aircraft="HALO",
                                    interested_flights=self.flight,
                                    instruments=["radar","radiometer","sonde"])

            elif campaign=="Second_Synthetic_Study":
                #cpgn_cls_name="Second_Synthetic_Study"
                cmpgn_cls=flightcampaign.Second_Synthetic_Study(
                             is_flight_campaign=True,
                             major_path=self.config_file["Data_Paths"]["campaign_path"],
                             aircraft="HALO",interested_flights=self.flight,
                             instruments=["radar","radiometer","sonde"])               
         
            elif campaign=="HALO_AC3":
                print("so far not included but planned")
                sys.exit()
            else:
                raise Exception("Wrong campaign.", campaign, 
                        "is not defined to be analysed")    
        else:
            campaign=self.cmpgn_cls.name
        
                        
        budget_data_path=self.cmpgn_cls.campaign_data_path+"/budget/"
    
        #---------------------------------------------------------------------#
        Sectors={}
        Ideal_Sectors={}

        ## Access sector-based divergence values   
        sectors=["core","warm_sector","cold_sector"]
        #s
        file_end=".csv"
        for sector in sectors:
            #if not self.do_instantan:
            budget_file=self.flight+"_AR_"+sector+"_"+self.grid_name+\
                            "_regr_sonde_no_"+str(self.sonde_no)
            budget_ideal_file=self.flight+"_AR_"+sector+"_"+self.grid_name+\
                                "_regr_sonde_no_100"
            if not self.scalar_based_div:
                budget_file+="_vectorised"
                budget_ideal_file+="_vectorised"
            budget_file       += file_end
            budget_ideal_file += file_end
            if sector=="core":
                print("Read budget file",budget_file)
            
            #    budget_file=self.flight+"_instantan_AR_"+sector+"_"+self.grid_name+\
            #    "_regr_sonde_no_"+sonde_no+".csv"
            #    budget_ideal_file=self.flight+"_instantan_AR_"+sector+"_"+\
            #                        self.grid_name+\
            #                            "_regr_sonde_no_100"+".csv"
            if (not self.flight.startswith("SRF12")):
                sector_values=pd.read_csv(budget_data_path+budget_file)
                sector_values.index=sector_values["Unnamed: 0"]
                del sector_values["Unnamed: 0"]
                Sectors[sector]=sector_values
        
                sector_values_ideal=pd.read_csv(budget_data_path+\
                                                budget_ideal_file)
                sector_values_ideal.index=sector_values_ideal["Unnamed: 0"]
                Ideal_Sectors[sector]=sector_values_ideal
            else:
                if sector!="cold_sector":
                    sector_values=pd.read_csv(budget_data_path+budget_file)
                    sector_values.index=sector_values["Unnamed: 0"]
                    del sector_values["Unnamed: 0"]
                    Sectors[sector]=sector_values
                    
                    sector_values_ideal=pd.read_csv(budget_data_path+\
                                                    budget_ideal_file)
                    sector_values_ideal.index=sector_values_ideal["Unnamed: 0"]
                    Ideal_Sectors[sector]=sector_values_ideal
                else:
                    Sectors[sector]=pd.DataFrame()
                    Ideal_Sectors[sector]=pd.DataFrame()
        return Sectors,Ideal_Sectors, self.cmpgn_cls
    
    # get overall divergences
    def get_overall_budgets(self,use_flight_tracks=False):
        # For sondes
        Summary_Budgets={}
        Summary_Ideal_Budgets={}
        core_budgets={}
        warm_budgets={}
        cold_budgets={}
        core_budgets["ADV"]=pd.Series()
        core_budgets["CONV"]=pd.Series()
        core_budgets["TRANSP"]=pd.Series()
        warm_budgets["ADV"]=pd.Series()
        warm_budgets["CONV"]=pd.Series()
        warm_budgets["TRANSP"]=pd.Series()
        cold_budgets["ADV"]=pd.Series()
        cold_budgets["CONV"]=pd.Series()
        cold_budgets["TRANSP"]=pd.Series()
        # Ideal
        core_ideal_budgets={}
        warm_ideal_budgets={}
        cold_ideal_budgets={}
        
        core_ideal_budgets["ADV"]=pd.Series()
        core_ideal_budgets["CONV"]=pd.Series()
        core_ideal_budgets["TRANSP"]=pd.Series()
        warm_ideal_budgets["ADV"]=pd.Series()
        warm_ideal_budgets["CONV"]=pd.Series()
        warm_ideal_budgets["TRANSP"]=pd.Series()
        cold_ideal_budgets["ADV"]=pd.Series()
        cold_ideal_budgets["CONV"]=pd.Series()
        cold_ideal_budgets["TRANSP"]=pd.Series()
        
            
        for campaign in self.flight_dates.keys():
            if campaign=="North_Atlantic_Run":
                campaign_id="NA"
                budget_data_path=self.cmpgn_cls.major_path+"NA_February_Run/data/budget/"
            else:
                campaign_id="Snd"
                budget_data_path=self.cmpgn_cls.major_path+campaign+"/data/budget/"
    
            for flight in self.flight_dates[campaign].keys():
                name_arg=""
                
                if self.do_instantan:
                    flight=flight+"_instantan"
                    if use_flight_tracks:
                        name_arg="_on_flight"
                #Core
                file_end=".csv"
                
                core_file=flight+"_AR_core_"+self.grid_name+\
                "_regr_sonde_no_"+str(self.sonde_no)+name_arg
                
                core_ideal_file=flight+"_AR_core_"+self.grid_name+\
                "_regr_sonde_no_100"+name_arg
                if not self.scalar_based_div:
                    core_file+="_vectorised"
                    core_ideal_file+="_vectorised"
                core_file+=file_end
                core_ideal_file+=file_end
                
                core=pd.read_csv(budget_data_path+core_file)
                core_ideal=pd.read_csv(budget_data_path+core_ideal_file)
                if not "level" in core.columns:
                    core.index=core["Unnamed: 0"]
                    core_ideal.index=core["Unnamed: 0"]
                    del core["Unnamed: 0"], core_ideal["Unnamed: 0"]
                else:
                    core.index=core["level"]
                    core_ideal.index=core_ideal["level"]
                    del core["level"],core_ideal["level"]            
        
                # Warm sector
                warm_file=flight+"_AR_warm_sector_"+self.grid_name+\
                    "_regr_sonde_no_"+str(self.sonde_no)+name_arg
                warm_ideal_file=flight+"_AR_warm_sector_"+self.grid_name+\
                    "_regr_sonde_no_100"+name_arg
                
                if not self.scalar_based_div:
                    warm_file+="_vectorised"
                    warm_ideal_file+="_vectorised"
                warm_file+=file_end
                warm_ideal_file+=file_end
                
                warm=pd.read_csv(budget_data_path+warm_file)
                warm_ideal=pd.read_csv(budget_data_path+warm_ideal_file)
                if not "level" in warm.columns:
                    warm.index=warm["Unnamed: 0"]
                    warm_ideal.index=warm_ideal["Unnamed: 0"]
                    del warm["Unnamed: 0"], warm_ideal["Unnamed: 0"]
                else:
                    warm.index=warm["level"]
                    warm_ideal.index=warm_ideal["level"]
                    del warm["level"], warm_ideal["level"]
                
                # Cold sector
                if not flight.startswith("SRF12"):
                    cold_file=flight+"_AR_cold_sector_"+self.grid_name+\
                        "_regr_sonde_no_"+str(self.sonde_no)+name_arg
                    cold_ideal_file=flight+"_AR_cold_sector_"+self.grid_name+\
                    "_regr_sonde_no_100"+name_arg
                    if not self.scalar_based_div:
                        cold_file+="_vectorised"
                        cold_ideal_file+="_vectorised"
                    cold_file+=file_end
                    cold_ideal_file+=file_end
                    cold=pd.read_csv(budget_data_path+cold_file)
                    cold_ideal=pd.read_csv(budget_data_path+cold_ideal_file)
                    if not "level" in cold.columns:
                        cold.index=cold["Unnamed: 0"]
                        cold_ideal.index=cold_ideal["Unnamed: 0"]
                        del cold["Unnamed: 0"], cold_ideal["Unnamed: 0"]
                    else:
                        cold.index=cold["level"]
                        cold_ideal.index=cold_ideal["level"]
                        del cold["level"],cold_ideal["level"]
        
                pres_index=pd.Series(core.index*100)
                g=1 #---> division by g happens later
                for term in ["ADV","CONV","TRANSP"]:
                    if term=="ADV":
                        series_term=term+"_calc"
                    else:
                        series_term=term
                    core_budgets[term].at[campaign_id+flight+\
                                          "_sonde_"+str(self.sonde_no)+term]=\
                    1/g*np.trapz(core[series_term][::-1],axis=0,x=pres_index[::-1])
                    warm_budgets[term].at[campaign_id+flight+\
                                          "_sonde_"+str(self.sonde_no)+term]=\
                    1/g*np.trapz(warm[series_term][::-1],
                                 axis=0,x=pres_index[::-1])
            
                    core_ideal_budgets[term].at[campaign_id+flight+\
                                          "_sonde_100"+term]=\
                    1/g*np.trapz(core_ideal[series_term][::-1],
                                 axis=0,x=pres_index[::-1])
                    warm_ideal_budgets[term].at[campaign_id+flight+\
                                          "_sonde_100"+term]=\
                    1/g*np.trapz(warm_ideal[series_term][::-1],
                                 axis=0,x=pres_index[::-1])
                    if not flight=="SRF12":
                        cold_budgets[term].at[campaign_id+flight+\
                                          "_sonde_"+str(self.sonde_no)+term]=\
                        1/g*np.trapz(cold[series_term][::-1],
                                     axis=0,x=pres_index[::-1])
            
                        cold_ideal_budgets[term].at[campaign_id+flight+\
                                          "_sonde_100"+term]=\
                        1/g*np.trapz(cold_ideal[series_term][::-1],
                                     axis=0,x=pres_index[::-1])
                    else:
                        cold_budgets[term].at[campaign_id+flight+\
                                          "_sonde_"+self.sonde_no+term]=\
                        np.nan
            
                        cold_ideal_budgets[term].at[campaign_id+flight+\
                                                    "_sonde_100"+term]=\
                        np.nan
            
        Summary_Budgets["core"]               = core_budgets
        Summary_Budgets["warm_sector"]        = warm_budgets
        Summary_Budgets["cold_sector"]        = cold_budgets

        Summary_Ideal_Budgets["core"]         = core_ideal_budgets
        Summary_Ideal_Budgets["warm_sector"]  = warm_ideal_budgets
        Summary_Ideal_Budgets["cold_sector"]  = cold_ideal_budgets
            
        return Summary_Budgets, Summary_Ideal_Budgets
    
    def calc_moisture_convergence_from_regression_method(self,
            config_file_path="",do_plotting=False,
            calc_hmp=True,calc_hmc=False,use_era=True,use_carra=False,
            use_icon=False,do_supplements=False,
            use_flight_sonde_locations=False):
        
        import data_config
        import flightcampaign
        
        if config_file_path=="":
            aircraft_base_path=os.getcwd()
        else:
            aircraft_base_path=config_file_path
        #if not "data_config" in sys.modules:
        if not "campaignAR_plotter" in sys.modules:
            import campaignAR_plotter
        if not "atmospheric_rivers" in sys.modules:
            import atmospheric_rivers
        #if not "flightcampaign" in sys.modules:
        # Config File
        self.config_file=data_config.load_config_file(aircraft_base_path,
                                                      "data_config_file")


        if self.grid_name=="CARRA":
            use_carra=True
        for campaign in self.flight_dates:
            init_flight="SRF02" # this is just for initiating the class
            if campaign=="North_Atlantic_Run":
                self.cmpgn_cls=flightcampaign.North_Atlantic_February_Run(
                    is_flight_campaign=True,
                    major_path=self.config_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=init_flight,
                    instruments=["radar","radiometer","sonde"])
            elif campaign=="Second_Synthetic_Study":
                self.cmpgn_cls=flightcampaign.Second_Synthetic_Study(
                    is_flight_campaign=True,
                    major_path=self.config_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=init_flight,
                    instruments=["radar","radiometer","sonde"])
                
            else:
                raise Exception("Wrong campaign assigned.")
            flights=[*self.flight_dates[campaign].keys()]
            default_Hydrometeors,default_HALO_Dict,cmpgn_cls=\
                        campaignAR_plotter.main(
                                        campaign=campaign,flights=flights,
                                        era_is_desired=use_era, 
                                        icon_is_desired=use_icon,
                                        carra_is_desired=use_carra,
                                        do_daily_plots=do_plotting,
                                        calc_hmp=True,calc_hmc=False,
                                        do_instantaneous=self.do_instantan)

            HMCs,default_HALO_Dict,cmpgn_cls=campaignAR_plotter.main(
                                        campaign=campaign,flights=flights,
                                        era_is_desired=use_era, 
                                        icon_is_desired=use_icon,
                                        carra_is_desired=use_carra,
                                        do_daily_plots=do_plotting,
                                        calc_hmp=False,calc_hmc=True,
                                        do_instantaneous=self.do_instantan)

            budget_data_path=cmpgn_cls.campaign_data_path+"budget/"
            if not os.path.exists(budget_data_path):
                os.makedirs(budget_data_path)            
            budget_plot_path=cmpgn_cls.plot_path+"budget/"
            if not os.path.exists(budget_plot_path):
                os.makedirs(budget_plot_path)
            
        
            
            for flight in flights:
                # Init Plot class
                Budget_plots=Moisture_Budget_Plots(cmpgn_cls, flight, 
                                self.config_file,grid_name=self.grid_name,
                                do_instantan=self.do_instantan)
                        
                if self.do_instantan:
                    flight=flight+"_instantan"
                    analysed_flight=flight.split("_")[0]
                else:
                    analysed_flight=flight        
                ar_of_day="SAR_internal"
                grid_name=HMCs[analysed_flight]\
                                        ["AR_internal"]["name"]
                if not use_flight_sonde_locations:
                    Hydrometeors=default_Hydrometeors
                    HALO_Dict=default_HALO_Dict
                else: 
                    Hydrometeors,HALO_Dict,cmpgn_cls=\
                        campaignAR_plotter.main(campaign=campaign,
                            flights=flights,era_is_desired=use_era, 
                            icon_is_desired=use_icon,
                        carra_is_desired=use_carra,
                        do_daily_plots=do_plotting,
                        calc_hmp=True,calc_hmc=False,
                        do_instantaneous=False)

                AR_inflow, AR_outflow=atmospheric_rivers.Atmospheric_Rivers.\
                                                locate_AR_cross_section_sectors(
                                                    HALO_Dict,Hydrometeors,
                                                    analysed_flight)
                TIVT_inflow,TIVT_outflow=atmospheric_rivers.Atmospheric_Rivers.\
                                                    calc_TIVT_of_sectors(
                                                        AR_inflow,AR_outflow,
                                                        grid_name)
                                   
                for sector in ["cold_sector","core","warm_sector"]:
                    if flight.startswith("SRF12"):
                        if sector=="cold_sector":
                            continue
                    for number_of_sondes in [self.sonde_no,100]:    
                        print(flight)
                        #-----------------------------------------------------#
                        # Sonde number
                        sondes_selection={}
                        sondes_selection["inflow_"+sector]=np.linspace(
                                0,AR_inflow["AR_inflow_"+sector].shape[0]-1,
                                num=number_of_sondes).astype(int)
                        sondes_selection["outflow_"+sector]=np.linspace(
                                0,AR_outflow["AR_outflow_"+sector].shape[0]-1,
                                num=number_of_sondes).astype(int)
                        #-- Loc and locate sondes for regression method ------#
                        inflow_sondes_times=\
                                AR_inflow["AR_inflow_"+sector].index[\
                                    sondes_selection["inflow_"+sector]]
                        outflow_sondes_times=\
                                AR_outflow["AR_outflow_"+sector].index[\
                                        sondes_selection["outflow_"+sector]]
                        if use_flight_sonde_locations:
                            inst_HALO=default_HALO_Dict[analysed_flight].copy()
                            if not "old_index" in inst_HALO["inflow"].columns:
                                inst_HALO["inflow"]["old_index"]=\
                                    inst_HALO["inflow"]["Unnamed: 0"]
                                inst_HALO["outflow"]["old_index"]=\
                                    inst_HALO["outflow"]["Unnamed: 0"]    
                            
                            new_inflow_sondes_times=[]
                            new_outflow_sondes_times=[]
                            for time in range(inflow_sondes_times.shape[0]):
                                # Inflow
                                inst_time_in=inst_HALO["inflow"][\
                                    inst_HALO["inflow"]["old_index"]==\
                                        str(inflow_sondes_times[time])].index.values[0]
                                new_inflow_sondes_times.append(inst_time_in)
                                # Outflow
                                inst_time_out=inst_HALO["outflow"][\
                                    inst_HALO["outflow"]["old_index"]==\
                                        str(outflow_sondes_times[time])].index.values[0]
                                new_outflow_sondes_times.append(inst_time_out)
                        
                        sondes_pos_inflow=\
                                AR_inflow["AR_inflow_"+sector][\
                                    ["Halo_Lat","Halo_Lon"]].loc[\
                                                        inflow_sondes_times]
                        sondes_pos_outflow=\
                                AR_outflow["AR_outflow_"+sector][\
                                    ["Halo_Lat","Halo_Lon"]].loc[\
                                        outflow_sondes_times]
                        sondes_pos_all=pd.concat(
                                [sondes_pos_inflow,sondes_pos_outflow])
                #-------------------------------------------------------------#
                        if not "q" in HMCs[analysed_flight]["AR_internal"].keys():
                            HMCs[analysed_flight]["AR_internal"]["q"]=\
                                    HMCs[analysed_flight]["AR_internal"]\
                                        ["specific_humidity"].copy()
                        if not use_flight_sonde_locations:
                            inflow_times  =   inflow_sondes_times
                            outflow_times =   outflow_sondes_times
                        else:
                            inflow_times  =   new_inflow_sondes_times
                            outflow_times =   new_outflow_sondes_times
                            
                        q_inflow_sondes=\
                                HMCs[analysed_flight]["AR_internal"]["q"].loc[\
                                                        inflow_times]
                        q_outflow_sondes=\
                            HMCs[analysed_flight]["AR_internal"]["q"].loc[\
                                                        outflow_times]
                
                        u_inflow_sondes=\
                            HMCs[analysed_flight]["AR_internal"]["u"].loc[\
                                                        inflow_times]
                        u_outflow_sondes=\
                            HMCs[analysed_flight]["AR_internal"]["u"].loc[\
                                                        outflow_times]
                
                        v_inflow_sondes=\
                            HMCs[analysed_flight]["AR_internal"]["v"].loc[\
                                                        inflow_times]
                        v_outflow_sondes=\
                            HMCs[analysed_flight]["AR_internal"]["v"].loc[\
                                                        outflow_times]
                
                        wind_inflow_sondes=np.sqrt(u_inflow_sondes**2+\
                                           v_inflow_sondes**2)
                
                        moist_transport_inflow=\
                            q_inflow_sondes*wind_inflow_sondes
                
                        wind_outflow_sondes=np.sqrt(u_outflow_sondes**2+\
                                            v_outflow_sondes**2)
                
                        moist_transport_outflow=q_outflow_sondes*\
                            wind_outflow_sondes
                
                        #######################################################
                        ar_inflow=AR_inflow["AR_inflow"]
                        ar_outflow=AR_outflow["AR_outflow"]
                        
                        if number_of_sondes<10:
                            Budget_plots.plot_AR_TIVT_cumsum_quicklook(
                                ar_inflow,ar_outflow)
                
                            IVT_Variability_Plotter.plot_inflow_outflow_IVT_sectors(
                                                cmpgn_cls,AR_inflow,AR_outflow,
                                                TIVT_inflow,TIVT_outflow,
                                                grid_name,flight)
                        HMCs[analysed_flight]["AR_internal"]["wind"]=np.sqrt(
                            HMCs[analysed_flight]["AR_internal"]["u"]**2+\
                                HMCs[analysed_flight]["AR_internal"]["v"]**2)
    
                        if do_supplements:
                            q_field=HMCs[analysed_flight]\
                                        ["AR_internal"]["q"].copy()
                            wind_field=HMCs[analysed_flight]["AR_internal"]\
                                        ["wind"].copy()
    
                            moisture_transport=q_field*wind_field
                        
                            q_sector_inflow=\
                                q_field.loc[AR_inflow[\
                                                "AR_inflow_"+sector].index]
                            q_sector_outflow=q_field.loc[AR_outflow[\
                                                "AR_outflow_"+sector].index]
    
                            wind_sector_inflow=wind_field.loc[AR_inflow[\
                                                "AR_inflow_"+sector].index]
                            wind_sector_outflow=wind_field.loc[AR_outflow[\
                                                "AR_outflow_"+sector].index]
    
                            moist_transport_sector_inflow=moisture_transport.loc[\
                                                AR_inflow["AR_inflow_"+sector].index]
                            moist_transport_sector_outflow=moisture_transport.loc[\
                                                AR_outflow["AR_outflow_"+sector].index]
    
                            pressure=q_field.columns.astype(float)
                        
                            self.run_rough_budget_closure(
                                    wind_field,q_field,moisture_transport,
                                    wind_sector_inflow,wind_sector_outflow,
                                    q_sector_inflow,q_sector_outflow,
                                    moist_transport_sector_inflow,
                                    moist_transport_sector_outflow,pressure,
                                    AR_inflow,AR_outflow,sector=sector)
                        #######################################################
                        #-----------------------------------------------------#
                        ### Prepare the pattern for regression method
                
                        sondes_pos_all=self.get_xy_coords_for_domain(
                                                sondes_pos_all)
                
                        domain_values={}
                        moist_transport_inflow=moist_transport_inflow.groupby(level=0).last()
                        moist_transport_outflow=moist_transport_outflow.groupby(level=0).last()
                        u_inflow_sondes=u_inflow_sondes.groupby(level=0).last()
                        u_outflow_sondes=u_outflow_sondes.groupby(level=0).last()
                        v_inflow_sondes=v_inflow_sondes.groupby(level=0).last()
                        v_outflow_sondes=v_outflow_sondes.groupby(level=0).last()
                        q_inflow_sondes=q_inflow_sondes.groupby(level=0).last()
                        q_outflow_sondes=q_outflow_sondes.groupby(level=0).last()
                        wind_inflow_sondes=wind_inflow_sondes.groupby(level=0).last()
                        wind_outflow_sondes=wind_outflow_sondes.groupby(level=0).last()
                        domain_values["transport"]=pd.concat([moist_transport_inflow,
                                                              moist_transport_outflow])
                        domain_values["u"]=pd.concat([u_inflow_sondes,
                                                          u_outflow_sondes])
                        
                        domain_values["v"]=pd.concat([v_inflow_sondes,
                                                          v_outflow_sondes])
                        domain_values["q"]=pd.concat([q_inflow_sondes,
                                                          q_outflow_sondes])
                        domain_values["wind"]=pd.concat([wind_inflow_sondes,
                                                      wind_outflow_sondes])
                
                        
                        mean_qv,dx_qv,dy_qv=self.run_regression(sondes_pos_all,
                                                        domain_values,"transport")
                
                        mean_q,dx_q_calc,dy_q_calc=self.run_regression(sondes_pos_all,
                                                               domain_values,"q")
                
                        mean_scalar_wind,dx_scalar_wind,dy_scalar_wind=self.run_regression(
                                        sondes_pos_all,domain_values,"wind")
                        
                        if not self.scalar_based_div:
                            # Then we also need the wind components
                            mean_u,dx_u,dy_u=self.run_regression(sondes_pos_all,
                                                         domain_values,"u")
                            mean_v,dx_v,dy_v=self.run_regression(sondes_pos_all,
                                                         domain_values,"v")
                    
                        div_qv=(dx_qv+dy_qv)*1000
                        div_scalar_wind=(dx_scalar_wind+dy_scalar_wind)
                        #div_scalar_wind=(dx_scalar_wind+dy_scalar_wind)
                        div_scalar_mass=div_scalar_wind*\
                        domain_values["q"].mean(axis=0).values*1000
                        adv_q_scalar=div_qv-div_scalar_mass
                        
                        if self.scalar_based_div:
                            # Adv term based on divergence of q from run_regression
                            adv_q_calc=(dx_q_calc+dy_q_calc)*\
                                domain_values["wind"].mean(axis=0).values*1000
                            # Simply the difference of Moisture transport 
                            # divergence and the scalar based mass divergence
                            # redundat
                            #div_mass=div_wind*domain_values["q"].mean(axis=0).values*1000
                        else:
                            """
                            sector_div_vector_mass["val"]=\
                                 (sector_dx_u_wind[0].loc[intersect_index]+\
                                  sector_dy_v_wind[0].loc[intersect_index])*\
                                     self.sector_sonde_values[sector]["q"].loc[\
                                            intersect_index].mean(axis=1).values*1000 #for g/kg from kg/kg
                            sector_div_vector_mass["unc"]=\
                                np.sqrt(sector_dx_u_wind["unc"].loc[intersect_index]**2+\
                                        sector_dy_v_wind["unc"].loc[intersect_index]**2)*\
                                self.sector_sonde_values[sector]["q"].loc[\
                                        intersect_index].mean(axis=1).values*1000
                            
                            #Moisture Advection (v* nabla_q)
                            sector_adv_q_vector=pd.DataFrame()
                            sector_adv_q_vector["val"]=\
                                (sector_mean_u.loc[intersect_index]*\
                                 sector_dx_q_vector[0].loc[intersect_index]+
                                 sector_mean_v.loc[intersect_index]*\
                                 sector_dy_q_vector[0].loc[intersect_index])*1000
                            
                            """
                            div_mass=domain_values["q"].mean(axis=0).values*1000*\
                                (dx_u+dy_v)
                                
                            adv_q=(domain_values["u"].mean(axis=0).values*dx_q_calc+\
                                   domain_values["v"].mean(axis=0).values*dy_q_calc)*\
                                    1000
                            
                        if do_supplements:
                            Budget_plots.\
                            plot_single_flight_and_sector_regression_divergence(
                            sector,self.sonde_no,div_qv,div_scalar_mass,
                            adv_q_calc,adv_q_scalar)
                            #Budget_plots.plot_sector_based_comparison()
                            #---> to be added
                            # Sector-based comparison of values
                            #fig=plt.figure(figsize=(9,12))
                            #ax1=fig.add_subplot(111)
                            #ax1.plot(div_qv.values,div_qv.index,label="div: transp")
                            #ax1.axvline(x=0,ls="--",color="grey",lw=2)
                            
                            #ax1.plot(div_scalar_mass.values,
                            #         div_scalar_mass.index,
                            #         label="div: scalar mass")
                            
                            #ax1.plot(adv_q_calc,adv_q_calc.index,
                            #         label="adv_calc:q",c="darkgreen")
                            
                            #ax1.plot(adv_q_scalar,adv_q_scalar.index,
                            # label="adv_scalar:q",c="green",ls="--")
                    
                            #ax1.invert_yaxis()
                            #ax1.set_xlim([-2e-4,1e-4])
                            #ax1.set_xticks([-2e-4,0,2e-4])
                            #ax1.set_ylim([1000,300])
                            #ax1.legend()
                            #budget_plot_file_name=flight+"_"+grid_name+\
                            #    "_AR_"+sector+"_regr_sonde_no_"+\
                            #        str(number_of_sondes)+".png"
                            #fig.savefig(budget_plot_path+"/supplementary/"+
                            #            budget_plot_file_name,
                            #            dpi=300,bbox_inches="tight")
                            #print("Figure saved as:",budget_plot_path+\
                            #      "/supplements/"+budget_plot_file_name)
                        
                        if number_of_sondes<10:
                            # create mean values plots
                            #mean_profile_fig=plt.figure(figsize=(16,9))
                            #ax1_mean=mean_profile_fig.add_subplot(131)
                            #ax2_mean=mean_profile_fig.add_subplot(132)
                            #ax3_mean=mean_profile_fig.add_subplot(133)

                            #ax1_mean.plot(domain_values["wind"].mean()*\
                            #  (q_outflow_sondes.mean()-q_inflow_sondes.mean()),
                            #  q_inflow_sondes.columns.astype(float),
                            #  color=Budget_plots.sector_colors[sector])
                            #ax1_mean.text(-0.01,150,"ADV")
                            #ax2_mean.plot(domain_values["q"].mean()*\
                            #  (wind_outflow_sondes.mean()-\
                            #   wind_inflow_sondes.mean()),
                            #  wind_outflow_sondes.columns.astype(float),
                            #  color=Budget_plots.sector_colors[sector])
                            #ax2_mean.text(-0.01,150,"Mass Div")
                            #ax3_mean.plot(moist_transport_outflow.mean()-\
                            #  moist_transport_inflow.mean(),
                            #  moist_transport_outflow.columns.astype(float))
                            #ax3_mean.text(-0.01,150,s="Transp Div")
                            pass
                        #if do_supplements:
                        #    ax1_mean.invert_yaxis()
                        #    ax2_mean.invert_yaxis()
                        #    ax3_mean.invert_yaxis() 
                        #    ax1_mean.set_xlim([-0.02,0.02])
                        #    ax2_mean.set_xlim([-0.02,0.02])
                        #    ax3_mean.set_xlim([-0.02,0.02])  
                        #    file_name=flight+"_simplified_divergence_sonde_no_"+\
                        #                str(number_of_sondes)+".png"
                        #    mean_profile_fig.savefig(budget_data_path+file_name)
                        #    print("Figure saved as:",budget_data_path+file_name)
                        # Save sonde budget components as dataframe
                        budget_regression_profile_df=pd.DataFrame(data=np.nan,
                                        index=div_qv.index,
                                        columns=["CONV","ADV_calc","ADV_diff",
                                                 "TRANSP"])
                        if self.scalar_based_div:
                            budget_regression_profile_df["CONV"]=\
                                div_scalar_mass.values
                            budget_regression_profile_df["ADV_calc"]=\
                                adv_q_calc.values
                            budget_regression_profile_df["ADV_diff"]=\
                                adv_q_scalar.values
                            budget_regression_profile_df["TRANSP"]=\
                                div_qv.values
                        else:
                            budget_regression_profile_df["CONV"]=\
                                div_mass.values
                            budget_regression_profile_df["ADV_calc"]=\
                                adv_q.values
                            
                            budget_regression_profile_df["ADV_diff"]=\
                                    adv_q_scalar.values
                            budget_regression_profile_df["TRANSP"]=\
                                    div_qv.values
      
                        # Save budget values
                        name_arg=""
                        file_end=".csv"
                        if use_flight_sonde_locations:
                            name_arg="_on_flight"+name_arg
                        budget_file_name=flight+"_AR_"+sector+"_"+\
                                    grid_name+"_regr_sonde_no_"+str(number_of_sondes)+\
                                        name_arg
                        if not self.scalar_based_div:
                            budget_file_name+="_vectorised"
                        budget_file_name+=file_end
                        
                        budget_regression_profile_df.to_csv(
                            path_or_buf=budget_data_path+budget_file_name)    
                        print("Convergence components saved as: ",
                              budget_data_path+budget_file_name)
                        # Save sonde positions
                        #if number_of_sondes<10:
                        sonde_pos_fname=flight+"_Sonde_Location_"+sector+"_"+\
                                        grid_name+"_regr_sonde_no_"+str(number_of_sondes)
                        if use_flight_sonde_locations:
                                sonde_pos_fname=sonde_pos_fname+"_on_flight"
                                        
                        sonde_pos_fname=sonde_pos_fname+".csv"
                        sondes_pos_all.to_csv(path_or_buf=budget_data_path+\
                                                 sonde_pos_fname)
                        print("Sonde position saved as:",
                              budget_data_path+sonde_pos_fname)
                
            
    @staticmethod
    def get_xy_coords_for_domain(domain):
        x_coor=[]
        y_coor=[]
        for i in range(domain.shape[0]):

            x_coor.append(domain["Halo_Lon"].iloc[i] * 111.320 * \
                    np.cos(np.radians(domain["Halo_Lat"].iloc[i])) * 1000)
            y_coor.append(domain["Halo_Lat"].iloc[i] * 110.54 * 1000)
        
        xc = np.mean(x_coor, axis=0)
        yc = np.mean(y_coor, axis=0)

        delta_x = x_coor - xc  # *111*1000 # difference of sonde long from mean long
        delta_y = y_coor - yc  # *111*1000 # difference of sonde lat from mean lat
        domain["dx"] = delta_x
        domain["dy"] = delta_y
    
        print("domain ready for regression")
        return domain 
    @staticmethod
    def run_haloac3_icon_sonde_regression(geo_domain,domain_values,parameter):
        # similar to run_regression but with inverted values
        # for pressure values
        from sklearn import linear_model

        regr = linear_model.LinearRegression()
        ## Gridded Sonde values can contain nans, drop them
        nonnan_values=domain_values[parameter].dropna(axis=1,how='any').T
        mean_parameter = pd.Series(data=np.nan,
                        index=nonnan_values.index.astype(float)) 
        dx_parameter = pd.Series(data=np.nan,
                        index=nonnan_values.index.astype(float)) 
        dy_parameter = pd.Series(data=np.nan,
                        index=nonnan_values.index.astype(float)) 
        
        # number of sondes available for regression
        #Ns = pd.Series(data=np.nan,index=domain_values[parameter].index.astype(float)) 

        for k in range(mean_parameter.shape[0]):
            #Ns[k] = id_[:, k].sum()
            X_dx = geo_domain["dx"].values
            X_dy = geo_domain["dy"].values
            
            X = list(zip(X_dx, X_dy))
            
            Y_parameter = nonnan_values.iloc[k,:].values
            regr.fit(X, Y_parameter)

            mean_parameter.iloc[k] = regr.intercept_
            dx_parameter.iloc[k], dy_parameter.iloc[k] = regr.coef_

        
        return mean_parameter, dx_parameter, dy_parameter
    @staticmethod
    def unify_vertical_dropsonde_grid(sonde_data,Z_grid,time_list,
                        relevant_times=[],sector_type="warm",
                        met_vars=["u","v","wind","q","pres","transport"]):

        import scipy.interpolate as scint

        uninterp_vars={}
        interp_vars={}
        sector_div_vars={}
        for met_var in met_vars:
            interp_vars=pd.DataFrame(data=np.nan,
                                     index=Z_grid,columns=time_list)
            t=0    
            for time in relevant_times:
                uninterp_vars["u"]=pd.Series(data=np.array(
                    sonde_data["u_wind"][time][:]),
                    index=np.array(sonde_data["alt"][time][:]))
                uninterp_vars["v"]=pd.Series(data=np.array(sonde_data["v_wind"][time][:]),
                            index=np.array(sonde_data["alt"][time][:]))
        
                uninterp_vars["wind"]=pd.Series(data=\
                                np.array(sonde_data["wspd"][time][:]),
                                index=np.array(sonde_data["alt"][time][:]))
                uninterp_vars["q"]=pd.Series(data=\
                                np.array(sonde_data["q"][pd.Timestamp(time)][:]),
                                index=np.array(sonde_data["alt"][time][:]))
                uninterp_vars["transport"]=\
                    uninterp_vars["wind"]*uninterp_vars["q"]
                uninterp_vars["pres"]=pd.Series(data=np.array(
                                        sonde_data["pres"][time][:]),
                                        index=np.array(sonde_data["alt"][time][:]))
                not_nan_index=uninterp_vars[met_var].index.dropna()

                # common scipy method with met_var as a function of z
                interp_func=scint.interp1d(not_nan_index,
                                   uninterp_vars[met_var].loc[not_nan_index],
                                   kind="nearest",bounds_error=False,
                                   fill_value=np.nan)
                interp_vars[time_list[t]]=pd.Series(data=interp_func(Z_grid),
                                    index=Z_grid)
                t+=1 
            sector_div_vars[met_var]=interp_vars    
        sector_div_vars["sector_name"]=sector_type
        return sector_div_vars
    @staticmethod
    def run_haloac3_sondes_regression(geo_domain,domain_values,parameter,
                                      with_uncertainty=False,
                                      regression_method="least-squared"):
        
        print("Perform divergence via regression method:",regression_method)
        if with_uncertainty==True:
           import statsmodels.api as sm 
        # similar to run_regression but with inverted values
        # for pressure values
        from sklearn import linear_model

        regr = linear_model.LinearRegression()
        ## Gridded Sonde values can contain nans, drop them
        nonnan_values=domain_values[parameter].dropna(axis=0,how='any')
        mean_parameter = pd.Series(data=np.nan,
                        index=nonnan_values.index.astype(float)) 
        dx_parameter = pd.Series(data=np.nan,
                        index=nonnan_values.index.astype(float)) 
        dy_parameter = pd.Series(data=np.nan,
                        index=nonnan_values.index.astype(float)) 
        if with_uncertainty:
            unc_dx_parameter=dx_parameter.copy()
            unc_dy_parameter=dy_parameter.copy()
        # number of sondes available for regression
        #Ns = pd.Series(data=np.nan,index=domain_values[parameter].index.astype(float)) 
        
        for k in range(mean_parameter.shape[0]):
            #Ns[k] = id_[:, k].sum()
            X_dx = geo_domain["dx"].values
            X_dy = geo_domain["dy"].values
            
            X = list(zip(X_dx, X_dy))
            
            Y_parameter = nonnan_values.iloc[k,:].values
            #-----------------------------------------------------------------#
            if with_uncertainty:
            ### for uncertainty estimates
                N = len(X)
                p= 2+1  # plus one because LinearRegression adds an intercept term
                X_with_intercept = np.empty(shape=(N, p), dtype=np.float)
                X_with_intercept[:, 0] = 1
                X_with_intercept[:, 1] = X_dx[:]
                X_with_intercept[:, 2] = X_dy[:]
                ols = sm.OLS(Y_parameter,X_with_intercept)
                ols_result = ols.fit()
                ols_result.summary()
                unc_dx_parameter.iloc[k]= ols_result.bse[1]
                unc_dy_parameter.iloc[k]= ols_result.bse[2]

            #-----------------------------------------------------------------#
            
            regr.fit(X, Y_parameter)

            mean_parameter.iloc[k] = regr.intercept_
            dx_parameter.iloc[k], dy_parameter.iloc[k] = regr.coef_

        if with_uncertainty:
            dx_parameter=dx_parameter.to_frame()
            dx_parameter["unc"]=unc_dx_parameter.values
            dy_parameter=dy_parameter.to_frame()
            dy_parameter["unc"]=unc_dy_parameter.values
        return mean_parameter, dx_parameter, dy_parameter
    @staticmethod
    def run_regression(geo_domain,domain_values, parameter):
        """
        Input :
        circle : xarray dataset
                 dataset with sondes of a circle, with dx and dy calculated
                
        parameter : string
                    the parameter on which regression is to be carried out
    
        Output :

        mean_parameter : mean of parameter (intercept)
        m_parameter, c_parameter    : coefficients of regression

        """
        from sklearn import linear_model

        regr = linear_model.LinearRegression()
        """
        id_u = ~np.isnan(domain_values["u"].values)
        id_v = ~np.isnan(domain_values["v"].values)
        id_q = ~np.isnan(domain_values["q"].values)
        id_x = ~np.isnan(geo_domain["dx"].values)
        id_y = ~np.isnan(geo_domain["dy"].values)
        id_wind = ~np.isnan(domain_values["wind"].values)
        id_transport = ~np.isnan(domain_values["transport"].values)
        
        #id_quxv = np.logical_and(np.logical_and(id_q, id_u), np.logical_and(id_x, id_v))
        #id_ = np.logical_and(np.logical_and(id_wind, id_transport), id_quxv)
        """
        # for pressure values
        mean_parameter = pd.Series(data=np.nan,
                        index=domain_values["u"].columns.astype(float)) 
        dx_parameter = pd.Series(data=np.nan,
                        index=domain_values["u"].columns.astype(float)) 
        dy_parameter = pd.Series(data=np.nan,
                        index=domain_values["u"].columns.astype(float)) 
        
        # number of sondes available for regression
        Ns = pd.Series(data=np.nan,index=domain_values["u"].columns.astype(float)) 

        for k in range(mean_parameter.shape[0]):
            #Ns[k] = id_[:, k].sum()
            X_dx = geo_domain["dx"].values
            X_dy = geo_domain["dy"].values
            
            X = list(zip(X_dx, X_dy))
            
            Y_parameter = domain_values[parameter].iloc[:,k].values
            regr.fit(X, Y_parameter)

            mean_parameter.iloc[k] = regr.intercept_
            dx_parameter.iloc[k], dy_parameter.iloc[k] = regr.coef_

        
        return mean_parameter, dx_parameter, dy_parameter
    #@staticmethod
    def get_sector_sonde_values(self,Dropsondes,relevant_sector_sondes):
        relevant_times=[*Dropsondes["reference_time"].keys()]
        print(relevant_times)
        
        sondes_pos_all={}
        sector_sonde_values={}
        sector_relevant_times={}
            
        # Geolocate the sondes once again to make
        # it compatible for divergence calculation
        self.sondes_pos_all={}
        for sector in self.sector_types:
            sector_relevant_times[sector]=[relevant_times[sector_time] \
                        for sector_time in relevant_sector_sondes[sector]]
            sector_sondes_lat=[Dropsondes["reference_lat"][time].data \
                        for time in sector_relevant_times[sector]]
            sector_sondes_lon=[Dropsondes["reference_lon"][time].data \
                        for time in sector_relevant_times[sector]]
            
            # Merge location in dataframe
            sondes_pos_all[sector]=pd.DataFrame(
                                data=np.nan,columns=["Halo_Lat","Halo_Lon"],
                                index=pd.DatetimeIndex(
                                    sector_relevant_times[sector]))
            sondes_pos_all[sector]["Halo_Lat"][:]=sector_sondes_lat
            sondes_pos_all[sector]["Halo_Lon"][:]=sector_sondes_lon
            sector_domain_values={}
            sector_domain_values={}
            # Positions relevant for divergence calculations
            self.sondes_pos_all[sector]=self.get_xy_coords_for_domain(
                sondes_pos_all[sector])
            
            # heights of sonde measurements are on 
            # irregular grid depending on fall. 
            # We interpolate onto a 10-m grid.
            time_list=[str(pd.Timestamp(time)) \
                       for time in sector_relevant_times[sector]]
            # the regular grid requires a regular vertical grid with 
            # maximum height based on all considered sondes
            zmax_grid=[float(Dropsondes["alt"][time][:].max()) \
                       for time in sector_relevant_times[sector]]
            zmax_grid=pd.Series(zmax_grid).min()//10*10+10
            Z_grid=np.arange(0,zmax_grid,step=10)
        
                # Get relevant sonde values
            sector_sonde_values[sector]=\
                self.unify_vertical_dropsonde_grid(Dropsondes,Z_grid,time_list,
                    relevant_times=sector_relevant_times[sector],
                    sector_type=sector)#
        return sector_sonde_values
    def perform_entire_sonde_ac3_divergence_vector_calcs(self,
        Dropsondes,relevant_sector_sondes,with_uncertainty=False):
        if not hasattr(self,"sector_sonde_values"):
            self.sector_sonde_values=self.get_sector_sonde_values(
                        Dropsondes,relevant_sector_sondes)
        self.div_vector_mass={}
        self.adv_q_vector={}
        for sector in self.sector_types:
            sector_mean_qv,sector_dx_qv,sector_dy_qv=\
                self.run_haloac3_sondes_regression(self.sondes_pos_all[sector],
                                self.sector_sonde_values[sector],
                                "transport",with_uncertainty=with_uncertainty)

            sector_q,sector_dx_q_vector,sector_dy_q_vector=\
                self.run_haloac3_sondes_regression(self.sondes_pos_all[sector],
                                        self.sector_sonde_values[sector],"q",
                                        with_uncertainty=with_uncertainty)          

            sector_mean_scalar_wind,sector_dx_scalar_wind,sector_dy_scalar_wind=\
                self.run_haloac3_sondes_regression(self.sondes_pos_all[sector],
                                    self.sector_sonde_values[sector],"wind",
                                    with_uncertainty=with_uncertainty)
            sector_mean_u,sector_dx_u_wind,sector_dy_u_wind=\
                self.run_haloac3_sondes_regression(self.sondes_pos_all[sector],
                                        self.sector_sonde_values[sector],"u",
                                        with_uncertainty=with_uncertainty)
            sector_mean_v,sector_dx_v_wind,sector_dy_v_wind=\
                self.run_haloac3_sondes_regression(self.sondes_pos_all[sector],
                                        self.sector_sonde_values[sector],"v",
                                        with_uncertainty=with_uncertainty)
            sector_div_qv=(sector_dx_qv+sector_dy_qv)*1000
            
            # combine specific humidity
            if isinstance(sector_dx_q_vector,pd.Series):
                sector_div_q_vector       = (sector_dx_q_vector+\
                                             sector_dy_q_vector)
            elif isinstance(sector_dx_q_vector,pd.DataFrame):
                sector_div_q_calc       = pd.DataFrame()
                sector_div_q_calc["val"]= sector_dx_q_vector[0]+\
                                            sector_dy_q_vector[0]
                sector_div_q_calc["unc"]=\
                   np.sqrt(sector_dx_q_vector["unc"]**2+\
                           sector_dy_q_vector["unc"]**2) 
            else:
                Exception("Something went completely wrong in the regression")
            # combine wind
            if isinstance(sector_dx_scalar_wind,pd.Series):
                sector_div_scalar_wind = (sector_dx_scalar_wind+\
                                      sector_dy_scalar_wind)
            elif isinstance(sector_dx_scalar_wind,pd.DataFrame):
                sector_div_scalar_wind=pd.DataFrame()
                sector_div_scalar_wind["val"] = (sector_dx_scalar_wind[0]+\
                                      sector_dy_scalar_wind[0])
                ##Gaussian uncertainty
                #sector_div_scalar_wind["unc"]=\
                #    np.sqrt(sector_dx_scalar_wind["unc"]**2+\
                #            sector_dy_scalar_wind["unc"]**2)
            else:
                Exception("Something went completely wrong in the regression")
            
            # Intersection checks for products needed
            intersect_index=sector_div_qv.index.intersection(
                                    sector_div_scalar_wind.index)
            intersect_index=intersect_index.intersection(
                            sector_div_q_calc.index)
            #-----------------------------------------------------------------#
            # Both Divergence terms
            
            # Mass Divergence (q * nabla_v)
            sector_div_vector_mass=pd.DataFrame()
            sector_div_vector_mass["val"]=\
                 (sector_dx_u_wind[0].loc[intersect_index]+\
                  sector_dy_v_wind[0].loc[intersect_index])*\
                     self.sector_sonde_values[sector]["q"].loc[\
                            intersect_index].mean(axis=1).values*1000 #for g/kg from kg/kg
            sector_div_vector_mass["unc"]=\
                np.sqrt(sector_dx_u_wind["unc"].loc[intersect_index]**2+\
                        sector_dy_v_wind["unc"].loc[intersect_index]**2)*\
                self.sector_sonde_values[sector]["q"].loc[\
                        intersect_index].mean(axis=1).values*1000
            
            #Moisture Advection (v* nabla_q)
            sector_adv_q_vector=pd.DataFrame()
            sector_adv_q_vector["val"]=\
                (sector_mean_u.loc[intersect_index]*\
                 sector_dx_q_vector[0].loc[intersect_index]+
                 sector_mean_v.loc[intersect_index]*\
                 sector_dy_q_vector[0].loc[intersect_index])*1000
            sector_adv_q_vector["unc"]=\
                (sector_dx_q_vector["unc"].loc[intersect_index]*\
                 sector_mean_u.loc[intersect_index]+\
                 sector_dy_q_vector["unc"].loc[intersect_index]*\
                 sector_mean_v.loc[intersect_index])*1000
                
            self.div_vector_mass[sector] = sector_div_vector_mass
            
            self.adv_q_vector[sector]    = sector_adv_q_vector
            
            self.q_stat                  = {}
            self.q_stat[sector]          = pd.DataFrame()
            self.q_stat[sector]["mean"]  = \
                                    self.sector_sonde_values[sector]["q"].mean(axis=1)
            self.q_stat[sector]["std"]   = \
                                    self.sector_sonde_values[sector]["q"].std(axis=1)
            
            self.wsp_stat                = {}
            self.wsp_stat[sector]        = pd.DataFrame()
            self.wsp_stat[sector]["mean"]= \
                self.sector_sonde_values[sector]["wind"].mean(axis=1)
            self.wsp_stat[sector]["std"] = \
                self.sector_sonde_values[sector]["wind"].std(axis=1)
            
            self.save_moisture_transport_divergence(sector)
            
    def perform_entire_sonde_ac3_divergence_scalar_calcs(self,
        Dropsondes,relevant_sector_sondes,with_uncertainty=False):
        
        if not hasattr(self,"sector_sonde_values"):
            self.sector_sonde_values=self.get_sector_sonde_values(
                        Dropsondes,relevant_sector_sondes)
        
        self.div_scalar_mass={}
        self.adv_q_calc={}
        for sector in self.sector_types:
            sector_mean_qv,sector_dx_qv,sector_dy_qv=\
                self.run_haloac3_sondes_regression(self.sondes_pos_all[sector],
                                self.sector_sonde_values[sector],
                                "transport",with_uncertainty=with_uncertainty)

            sector_q,sector_dx_q_calc,sector_dy_q_calc=\
                self.run_haloac3_sondes_regression(self.sondes_pos_all[sector],
                                        self.sector_sonde_values[sector],"q",
                                        with_uncertainty=with_uncertainty)          

            sector_mean_scalar_wind,sector_dx_scalar_wind,sector_dy_scalar_wind=\
                self.run_haloac3_sondes_regression(self.sondes_pos_all[sector],
                                    self.sector_sonde_values[sector],"wind",
                                    with_uncertainty=with_uncertainty)
            
            sector_div_qv=(sector_dx_qv+sector_dy_qv)*1000
            
            # combine specific humidity
            if isinstance(sector_dx_q_calc,pd.Series):
                sector_div_q_calc       = (sector_dx_q_calc+sector_dy_q_calc)
            elif isinstance(sector_dx_q_calc,pd.DataFrame):
                sector_div_q_calc       = pd.DataFrame()
                sector_div_q_calc["val"]= sector_dx_q_calc[0]+sector_dy_q_calc[0]
                sector_div_q_calc["unc"]=\
                   np.sqrt(sector_dx_q_calc["unc"]**2+\
                           sector_dy_q_calc["unc"]**2) 
            else:
                Exception("Something went completely wrong in the regression")
            # combine wind
            if isinstance(sector_dx_scalar_wind,pd.Series):
                sector_div_scalar_wind = (sector_dx_scalar_wind+\
                                      sector_dy_scalar_wind)
            elif isinstance(sector_dx_scalar_wind,pd.DataFrame):
                sector_div_scalar_wind=pd.DataFrame()
                sector_div_scalar_wind["val"] = (sector_dx_scalar_wind[0]+\
                                      sector_dy_scalar_wind[0])
                #Gaussian uncertainty
                sector_div_scalar_wind["unc"]=\
                    np.sqrt(sector_dx_scalar_wind["unc"]**2+\
                            sector_dy_scalar_wind["unc"]**2)
            else:
                Exception("Something went completely wrong in the regression")
            
            # Intersection checks for products needed
            intersect_index=sector_div_qv.index.intersection(
                                    sector_div_scalar_wind.index)
            intersect_index=intersect_index.intersection(
                            sector_div_q_calc.index)
            #-----------------------------------------------------------------#
            # Both Divergence terms
            
            # Mass Divergence (q * nabla_v)
            if isinstance(sector_div_scalar_wind,pd.Series):
                sector_div_scalar_mass=\
                    sector_div_scalar_wind.loc[intersect_index]*\
                        self.sector_sonde_values[sector]["q"].loc[\
                            intersect_index].mean(axis=1).values*1000
            elif isinstance(sector_div_scalar_wind,pd.DataFrame):
                sector_div_scalar_mass=pd.DataFrame()
                sector_div_scalar_mass["val"]=\
                    sector_div_scalar_wind["val"].loc[intersect_index]*\
                        self.sector_sonde_values[sector]["q"].loc[\
                            intersect_index].mean(axis=1).values*1000
                sector_div_scalar_mass["unc"]=\
                    sector_div_scalar_wind["unc"].loc[intersect_index]*\
                        self.sector_sonde_values[sector]["q"].loc[\
                            intersect_index].mean(axis=1).values*1000
                    
            else:
                Exception("Something went completely wrong")
            
            #Moisture Advection (v* nabla_q)
            if isinstance(sector_div_q_calc,pd.Series):
                sector_adv_q_calc=sector_div_q_calc.loc[intersect_index]*\
                self.sector_sonde_values[sector]["wind"].loc[intersect_index].\
                    mean(axis=1).values*1000
            
            elif isinstance(sector_div_q_calc,pd.DataFrame):
                sector_adv_q_calc=pd.DataFrame()
                sector_adv_q_calc["val"]=\
                    sector_div_q_calc["val"].loc[intersect_index]*\
                        self.sector_sonde_values[sector]["wind"].loc[\
                            intersect_index].mean(axis=1).values*1000
                sector_adv_q_calc["unc"]=\
                    sector_div_q_calc["unc"].loc[intersect_index]*\
                        self.sector_sonde_values[sector]["wind"].loc[intersect_index].\
                            mean(axis=1).values*1000
                
            self.div_scalar_mass[sector] = sector_div_scalar_mass
            self.adv_q_calc[sector]      = sector_adv_q_calc
            self.save_moisture_transport_divergence(sector)
    
    def save_moisture_transport_divergence(self,sector):
        #print("Save mass convergence")
        print(self.ar_of_day)
        save_data_path=self.cmpgn_cls.campaign_data_path+"/data/budgets/"
        if not os.path.exists(save_data_path):
            os.mkdir(save_data_path)
        mass_conv_file_name         = self.flight[0]+"_"+self.ar_of_day+"_"+\
                                        sector+"_"+self.grid_name+\
                                            "_mass_convergence.csv"
        adv_q_file_name             = self.flight[0]+"_"+self.ar_of_day+"_"+\
                                        sector+"_"+self.grid_name+"_adv_q.csv"
        q_stat_file_name            = self.flight[0]+"_"+self.ar_of_day+"_"+\
                                        sector+"_"+self.grid_name+\
                                            "_q_stat.csv"
        
        vector_mass_conv_file_name  = self.flight[0]+"_"+self.ar_of_day+"_"+\
                                        sector+"_"+self.grid_name+\
                                            "_vector_mass_convergence.csv"
        vector_adv_q_file_name      = self.flight[0]+"_"+self.ar_of_day+"_"+\
                                        sector+"_"+self.grid_name+\
                                            "_vector_adv_q.csv"
        wsp_stat_file_name          = self.flight[0]+"_"+self.ar_of_day+"_"+\
                                        sector+"_"+self.grid_name+\
                                            "_wsp_stat.csv"
        
        if hasattr(self,"div_scalar_mass"):
            self.div_scalar_mass[sector].to_csv(
                save_data_path+mass_conv_file_name)    
            print("mass convergence saved as: ",
                              save_data_path+mass_conv_file_name)
        if hasattr(self,"adv_q_calc"):                
            self.adv_q_calc[sector].to_csv(save_data_path+adv_q_file_name)
            print("moisture advection saved as: ",
              save_data_path+adv_q_file_name)
        if hasattr(self,"adv_q_vector"):
            self.adv_q_vector[sector].to_csv(
                save_data_path+vector_adv_q_file_name)
            print("moisture advection saved as: ",
              save_data_path+vector_adv_q_file_name)
        if hasattr(self,"div_vector_mass"):
            self.div_vector_mass[sector].to_csv(
                save_data_path+vector_mass_conv_file_name)
            print("vector mass convergence saved as: ",
                              save_data_path+vector_mass_conv_file_name)
        # Statistics of quantities
        if hasattr(self,"q_stat"):
            self.q_stat[sector].to_csv(save_data_path+q_stat_file_name)
            print("q statistics saved as: ",save_data_path+q_stat_file_name)
        if hasattr(self,"wsp_stat"):
            self.wsp_stat[sector].to_csv(save_data_path+wsp_stat_file_name)
            print("wsp statistics saved as: ",save_data_path+wsp_stat_file_name)
            
#-----------------------------------------------------------------------------#
class Surface_Evaporation(Moisture_Budgets):
    def __init__(self,cmpgn_name,flight,major_work_path,flight_dates={},
                 sector_types=["warm","core","cold"],
                 ar_of_day="AR",grid_name="ERA5",do_instantan=False):
        
        self.cmpgn_name="HALO_AC3"
        self.grid_name=grid_name
        self.do_instantan=do_instantan
        self.flight=flight
        self.major_work_path=major_work_path
        self.ar_of_day=ar_of_day
        self.relevant_sondes_dict={}
        self.internal_sondes_dict={}
        """ So far not possible
        if self.flight[0]=="RF05":
            if self.ar_of_day=="AR_entire_1":
                self.relevant_warm_sector_sondes=[0,1,2,3,9,10,11,12]
                self.relevant_cold_sector_sondes=[4,5,6]
                self.relevant_warm_internal_sondes=[7,8,13,14]
                self.relevant_sondes_dict["warm_sector"]        = {}
                self.relevant_sondes_dict["warm_sector"]["in"]  = \
                    sonde_times_series.iloc[relevant_warm_sector_sondes[0:4]]
            relevant_sondes_dict["warm_sector"]["out"] = \ 
            sonde_times_series.iloc[relevant_warm_sector_sondes[4::]]
            relevant_sondes_dict["cold_sector"]        = {}
            relevant_sondes_dict["cold_sector"]["in"]  = \
                sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
            #relevant_sondes_dict["cold_sector"]["out"] = \
                sonde_times_series.iloc[relevant_cold_sector_sondes[3::]]
            synthetic_sonde_times_series=pd.Series(data=["7synth","8synth","9synth"],
                                     index=pd.DatetimeIndex(["2022-03-15 12:55","2022-03-15 13:05","2022-03-15 13:15"]))
            
            relevant_sondes_dict["cold_sector"]["out"] = synthetic_sonde_times_series
            internal_sondes_dict["warm"]               = sonde_times_series.iloc[relevant_warm_internal_sondes]
            internal_sondes_dict["cold"]               = ["2022-03-15 11:30:00","2022-03-15 13:35"]   
        elif ar_of_day=="AR_entire_2":
            relevant_warm_sector_sondes=[9,10,11,12,15,16,17,18]
            relevant_cold_sector_sondes=[19,20,21]
            relevant_warm_internal_sondes=[13,14,22,23]
            relevant_sondes_dict["warm_sector"]        = {}
            relevant_sondes_dict["warm_sector"]["in"]  = sonde_times_series.iloc[relevant_warm_sector_sondes[4::]]
            relevant_sondes_dict["warm_sector"]["out"] = sonde_times_series.iloc[relevant_warm_sector_sondes[0:4]]
            relevant_sondes_dict["cold_sector"]        = {}
            relevant_sondes_dict["cold_sector"]["in"]  = pd.Series()#sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
            relevant_sondes_dict["cold_sector"]["out"] = sonde_times_series.iloc[relevant_cold_sector_sondes]
            internal_sondes_dict["warm"]               = sonde_times_series.iloc[relevant_warm_internal_sondes]
            elif flight[0]=="RF06":
                if ar_of_day=="AR_entire_1":
            relevant_warm_sector_sondes=[0,1,2,8,9,10]
            relevant_cold_sector_sondes=[3,4,5,11,12]
            relevant_warm_internal_sondes=[7,22]
            relevant_sondes_dict["warm_sector"]        = {}
            relevant_sondes_dict["warm_sector"]["in"]  = sonde_times_series.iloc[relevant_warm_sector_sondes[0:3]]
            relevant_sondes_dict["warm_sector"]["out"] = sonde_times_series.iloc[relevant_warm_sector_sondes[3:]]
            relevant_sondes_dict["cold_sector"]        = {}
            relevant_sondes_dict["cold_sector"]["in"]  = sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
            relevant_sondes_dict["cold_sector"]["out"] = sonde_times_series.iloc[relevant_cold_sector_sondes[3:]]
            internal_sondes_dict["warm"]               = sonde_times_series.iloc[relevant_warm_internal_sondes]
            elif ar_of_day=="AR_entire_2":
                relevant_warm_sector_sondes=[8,9,16,17]
            relevant_cold_sector_sondes=[10,11,12,18,19]
            relevant_warm_internal_sondes=[14,15,21,22]
            relevant_sondes_dict["warm_sector"]        = {}
            relevant_sondes_dict["warm_sector"]["in"]  = sonde_times_series.iloc[relevant_warm_sector_sondes[0:2]]
            relevant_sondes_dict["warm_sector"]["out"] = sonde_times_series.iloc[relevant_warm_sector_sondes[2::]]
            relevant_sondes_dict["cold_sector"]        = {}
            relevant_sondes_dict["cold_sector"]["in"]  = sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
            relevant_sondes_dict["cold_sector"]["out"] = sonde_times_series.iloc[relevant_cold_sector_sondes[3::]]
            internal_sondes_dict["warm"]               = sonde_times_series.iloc[relevant_warm_internal_sondes]
        """
    def load_sonde_data(self):
        """
        ######################################################################
        From Main Script for running interpolation of griddata on flight path
        #######################################################################
        """
        from simplified_flight_leg_handling import simplified_run_grid_main
        
        self.halo_era5,self.halo_df,self.cmpgn_cls,ERA5_on_HALO,radar,Dropsondes=\
        simplified_run_grid_main(flight=self.flight,
                config_file_path=self.major_work_path,
                ar_of_day=self.ar_of_day)
        if not "Lat" in [*Dropsondes.keys()]:
            sondes_lon=[[*Dropsondes["reference_lon"].values()][sonde].data[0] \
                    for sonde in range(Dropsondes["IWV"].shape[0])]
                    
            sondes_lat=[[*Dropsondes["reference_lat"].values()][sonde].data[0]\
                    for sonde in range(Dropsondes["IWV"].shape[0])]
            Dropsondes["Lat"]=pd.Series(data=np.array(sondes_lat),
                                                index=Dropsondes["IWV"].index)
            Dropsondes["Lon"]=pd.Series(data=np.array(sondes_lon),
                                                index=Dropsondes["IWV"].index)
            self.sonde_times_series=pd.Series(
                index=Dropsondes["IWV"].index.values,
                data=range(Dropsondes["IWV"].shape[0]))
            self.Dropsondes=Dropsondes
    
    def select_surface_data(self):
        temp_sonde = self.Dropsondes["tdry"].copy()
        #temp_min   = temp_sonde-temp_sonde_unc
        #temp_max   = temp_sonde+temp_sonde_unc
        wspd_sonde = self.Dropsondes["wspd"].copy()
        shum_sonde = self.Dropsondes["q"].copy()
        pres_sonde = self.Dropsondes["pres"].copy()
        sst        = self.halo_era5["Interp_SST"]
        temp_sonde_unc =0.5
        sst_unc    = 0.5
        shum_unc   = 0.0002
        #sst_min    = sst-sst_unc
        #sst_max    = sst+sst_unc
        #shum_min   = shum_sonde-shum_unc
        #shum_max   = shum_sonde+shum_unc
        
        self.surface_data=pd.DataFrame(data=np.nan, 
            columns=["Temp","Shum","Wind","Pres","ERA5_SST"],
            index=pd.DatetimeIndex([*temp_sonde.keys()]))

        for n,sonde in enumerate([*temp_sonde.keys()]):
            max_alt=float(temp_sonde[sonde].isel({"time":slice(0,10)})["gpsalt"].max())
            # Add threshold if max alt is too high, i.e. sondes did not reach 
            # the ground. 
            if max_alt>150:
                print("Sonde is to high")
                continue                
            self.surface_data["Temp"].loc[sonde]=\
                temp_sonde[sonde].isel({"time":slice(0,10)}).mean()
            self.surface_data["Wind"].loc[sonde]=\
                wspd_sonde[sonde].isel({"time":slice(0,10)}).mean()
            self.surface_data["Shum"].loc[sonde]=\
                shum_sonde[pd.Timestamp(sonde)].iloc[0:10].mean()
            self.surface_data["Pres"].loc[sonde]=\
                pres_sonde[sonde].isel({"time":slice(0,10)}).mean()
            try:
                self.surface_data["ERA5_SST"].loc[pd.Timestamp(sonde)]=\
                    sst.loc[pd.Timestamp(sonde)]
                
                print(pd.Timestamp(sonde))
            except:
                print("Out of flight pattern")
            self.surface_data["ERA5_SST_min"]=self.surface_data["ERA5_SST"]-sst_unc
            self.surface_data["ERA5_SST_max"]=self.surface_data["ERA5_SST"]+sst_unc
            self.surface_data["Shum_min"]    =self.surface_data["Shum"]-shum_unc
            self.surface_data["Shum_max"]    =self.surface_data["Shum"]+shum_unc
            self.surface_data["Temp_min"]    =self.surface_data["Temp"]-temp_sonde_unc
            self.surface_data["Temp_max"]    =self.surface_data["Temp"]+temp_sonde_unc
            
    def calc_sat_q(self):
        # Get saturated specific humidity for given pressure and SST
        
        calc_mrsat=mpcalc.saturation_mixing_ratio
        calc_q_from_mr=mpcalc.specific_humidity_from_mixing_ratio
        saturation_mr=calc_mrsat(
            self.surface_data["Pres"].values*units.hPa,
            self.surface_data["ERA5_SST"].values*units.K)
        saturation_mr_max=calc_mrsat(
            self.surface_data["Pres"].values*units.hPa,
            self.surface_data["ERA5_SST_max"].values*units.K)
        saturation_mr_min=calc_mrsat(
            self.surface_data["Pres"].values*units.hPa,
            self.surface_data["ERA5_SST_min"].values*units.K)                                                        
        self.surface_data["Qsat"]=np.array(calc_q_from_mr(saturation_mr))
        self.surface_data["Qsat_min"]=np.array(calc_q_from_mr(saturation_mr_min))
        self.surface_data["Qsat_max"]=np.array(calc_q_from_mr(saturation_mr_max))
    def calc_rho_air(self):
        calc_mr_from_q=mpcalc.mixing_ratio_from_specific_humidity
        #MR
        self.surface_data["Mixr"]=\
            calc_mr_from_q(self.surface_data["Shum"].values)
        self.surface_data["Mixr_min"]=calc_mr_from_q(
            self.surface_data["Shum_min"].values)
        self.surface_data["Mixr_max"]=calc_mr_from_q(
            self.surface_data["Shum_max"].values)
        #Tvir
        self.surface_data["Tvir"]=(self.surface_data["Temp"]+273.15)*\
            (1+0.61*self.surface_data["Mixr"])
        self.surface_data["Tvir_min"]=(self.surface_data["Temp_min"]+273.15)*\
            (1+0.61*self.surface_data["Mixr_min"])
        self.surface_data["Tvir_max"]=(self.surface_data["Temp_max"]+273.15)*\
            (1+0.61*self.surface_data["Mixr_max"])
        #Rho Air    
        self.surface_data["RhoA"]=self.surface_data["Pres"]*100/\
            (287.05*self.surface_data["Tvir"])
        self.surface_data["RhoA_min"]=self.surface_data["Pres"]*100/\
            (287.05*self.surface_data["Tvir_max"])
        self.surface_data["RhoA_max"]=self.surface_data["Pres"]*100/\
            (287.05*self.surface_data["Tvir_min"])

    def prepare_dropsonde_data(self):     
        self.load_sonde_data()
        self.select_surface_data()
    def define_uncertainties(self):
        self.surface_data["Wind_unc"]=0.15 #
        self.surface_data
    def bulk_evap(self):
        
        self.surface_data["Drag"]=1.4e-3
        self.surface_data["Drag"].loc[self.surface_data["Wind"]>=13]=1.6e-3
        self.surface_data["Evap"]=self.surface_data["Drag"]*self.surface_data["RhoA"]*\
            (self.surface_data["Qsat"]-self.surface_data["Shum"])*self.surface_data["Wind"] # units kg/s
    def calc_evaporation(self,add_uncertainty=True):
        if not hasattr(self,"surface_data"):
            self.prepare_dropsonde_data()
        else:
            self.calc_sat_q()
            self.calc_rho_air()
        
        self.surface_data["Drag"]=1.4e-3
        self.surface_data["Drag"].loc[self.surface_data["Wind"]>=13]=1.6e-3
        self.surface_data["Evap"]=self.surface_data["Drag"]*self.surface_data["RhoA"]*\
            (self.surface_data["Qsat"]-self.surface_data["Shum"])*\
                self.surface_data["Wind"] # units kg/s
        if add_uncertainty:
            #-----------------------------------------------------------------#
            # Get uncertainties:
            # Using min and max values
            self.surface_data["RhoA_unc"]=np.sqrt((self.surface_data["RhoA_max"]-\
                                              self.surface_data["RhoA_min"])**2)
            self.surface_data["Shum_unc"]=np.sqrt((self.surface_data["Shum_max"]-\
                                              self.surface_data["Shum_min"])**2)
            self.surface_data["Qsat_unc"]=np.sqrt((self.surface_data["Qsat_max"]-
                                            self.surface_data["Qsat_min"])**2)
            self.surface_data["Wind_unc"]=np.sqrt(2)/10# for component-wise unc 0.1
            print("Rho_unc",self.surface_data["RhoA_unc"].mean())
            print("Shum_unc",self.surface_data["Shum_unc"].mean())
            print("Wind_unc",self.surface_data["Wind_unc"].mean())
            print("Qsat_unc",self.surface_data["Qsat_unc"].mean())
            #-----------------------------------------------------------------#
            # Gaussian error propagation
            #evap=C_drag*x1*(x2-x3)*x4 ### d=(x2-x3)
            #evap_unc/evap=C_drag*sqrt((u1/x1)**(2)*(u4/x4)**2\
            #    *(sqrt([u2+u3]**2)/d**2)
            #-----------------------------------------------------------------#
            normed_Evap_unc=\
                np.sqrt((self.surface_data["RhoA_unc"]/\
                         self.surface_data["RhoA"])**2+\
                (self.surface_data["Wind_unc"]/self.surface_data["Wind"])**2+\
                (self.surface_data["Qsat_unc"]**2+self.surface_data["Shum_unc"]**2)/\
                (self.surface_data["Qsat"]-self.surface_data["Shum"])**2)*\
                1#*self.surface_data["Drag"]
            #-----------------------------------------------------------------#
            # Now it needs to be multiplied with "Evap_unc"
            self.surface_data["Evap_unc"]=normed_Evap_unc*self.surface_data["Evap"]
class Surface_Precipitation():
    
    def __init__(self,cmpgn_name,cmpgn_cls,flight,
                 major_work_path,flight_dates={},
                 sector_types=["warm","core","cold"],
                 ar_of_day="AR",grid_name="ERA5",do_instantan=False):
            
            self.cmpgn_cls=cmpgn_cls
            self.cmpgn_name="HALO_AC3"
            self.grid_name=grid_name
            self.do_instantan=do_instantan
            self.flight=flight
            self.major_work_path=major_work_path
            self.ar_of_day=ar_of_day
            self.precip_rate_path=self.cmpgn_cls.campaign_data_path+\
                "/data/precip_rates/"

    
    def get_relevant_sondes_dict(self,Dropsondes):
        if not "Lat" in [*Dropsondes.keys()]:
           sondes_lon=[[*Dropsondes["reference_lon"].values()][sonde].data[0] \
                    for sonde in range(Dropsondes["IWV"].shape[0])]
                    
           sondes_lat=[[*Dropsondes["reference_lat"].values()][sonde].data[0]\
                    for sonde in range(Dropsondes["IWV"].shape[0])]
           Dropsondes["Lat"]=pd.Series(data=np.array(sondes_lat),
                                                index=Dropsondes["IWV"].index)
           Dropsondes["Lon"]=pd.Series(data=np.array(sondes_lon),
                                                index=Dropsondes["IWV"].index)

        sonde_times_series=pd.Series(index=Dropsondes["IWV"].index.values,
                                     data=range(Dropsondes["IWV"].shape[0]))
        relevant_sondes_dict={}

        if self.flight[0]=="RF05":
            if self.ar_of_day=="AR_entire_1":
                relevant_warm_sector_sondes=[0,1,2,3,9,10,11,12]
                relevant_cold_sector_sondes=[4,5,6]
                relevant_internal_sondes=[7,8,13,14]
                relevant_sondes_dict["warm_sector"]        = {}
                relevant_sondes_dict["warm_sector"]["in"]  = \
                    sonde_times_series.iloc[relevant_warm_sector_sondes[0:4]]
                relevant_sondes_dict["warm_sector"]["out"] = \
                    sonde_times_series.iloc[relevant_warm_sector_sondes[4::]]
                relevant_sondes_dict["cold_sector"]        = {}
                relevant_sondes_dict["cold_sector"]["in"]  = \
                    sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
                relevant_sondes_dict["cold_sector"]["out"] = \
                    sonde_times_series.iloc[relevant_cold_sector_sondes[3::]]
                relevant_sondes_dict["internal"]           = \
                    sonde_times_series.iloc[relevant_internal_sondes]
            elif self.ar_of_day=="AR_entire_2":
                relevant_warm_sector_sondes=[9,10,11,12,15,16,17,18]
                relevant_cold_sector_sondes=[19,20,21]
                relevant_sondes_dict["warm_sector"]        = {}
                relevant_sondes_dict["warm_sector"]["in"]  = \
                    sonde_times_series.iloc[relevant_warm_sector_sondes[4::]]
                relevant_sondes_dict["warm_sector"]["out"] = \
                    sonde_times_series.iloc[relevant_warm_sector_sondes[0:4]]
                relevant_sondes_dict["cold_sector"]        = {}
                relevant_sondes_dict["cold_sector"]["in"]  = pd.Series()#sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
                relevant_sondes_dict["cold_sector"]["out"] = \
                    sonde_times_series.iloc[relevant_cold_sector_sondes]
    
        if self.flight[0]=="RF06":
            if self.ar_of_day=="AR_entire_1":
                relevant_warm_sector_sondes=[0,1,2,8,9,10]
                relevant_cold_sector_sondes=[3,4,5,10,11,12]
                relevant_warm_internal_sondes=[7,22]
                relevant_sondes_dict["warm_sector"]        = {}
                relevant_sondes_dict["warm_sector"]["in"]  = \
                    sonde_times_series.iloc[relevant_warm_sector_sondes[0:3]]
                relevant_sondes_dict["warm_sector"]["out"] = \
                    sonde_times_series.iloc[relevant_warm_sector_sondes[3:]]
                relevant_sondes_dict["cold_sector"]        = {}
                relevant_sondes_dict["cold_sector"]["in"]  = \
                    sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
                relevant_sondes_dict["cold_sector"]["out"] = \
                    sonde_times_series.iloc[relevant_cold_sector_sondes[3:]]
                relevant_sondes_dict["internal"]           = \
                    sonde_times_series.iloc[relevant_warm_internal_sondes]
            elif self.ar_of_day=="AR_entire_2":
                relevant_warm_sector_sondes=[8,9,16,17]
                relevant_cold_sector_sondes=[10,11,12,18,19]
                relevant_warm_internal_sondes=[]
                relevant_sondes_dict["warm_sector"]        = {}
                relevant_sondes_dict["warm_sector"]["in"]  = \
                    sonde_times_series.iloc[relevant_warm_sector_sondes[0:2]]
                relevant_sondes_dict["warm_sector"]["out"] = \
                    sonde_times_series.iloc[relevant_warm_sector_sondes[2::]]
                relevant_sondes_dict["cold_sector"]        = {}
                relevant_sondes_dict["cold_sector"]["in"]  = \
                    sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
                relevant_sondes_dict["cold_sector"]["out"] = \
                    sonde_times_series.iloc[relevant_cold_sector_sondes[3::]]  
        self.relevant_sondes_dict=relevant_sondes_dict
        
    def get_warm_sector_geoloc_boundaries(self):
        # radar sector
        if self.flight[0]=="RF05":
            if self.ar_of_day=="AR_entire_1":
                # old
                #warm_radar_rain=radar_precip_rate.loc[radar_precip_rate["lat"].between(71.412415,76.510696)]
                #warm_radar_rain=warm_radar_rain.loc[warm_radar_rain["lon"].between(-6.153736,8.080936)]
                # new
                max_warm_lat=76.510696
                max_warm_lon=10.080936
                min_warm_lat=70.412415
                min_warm_lon=-6.153736
                #cold_radar_rain=radar_precip_rate.loc[radar_precip_rate["lat"].between(72.492348,77.000370)]
                #cold_radar_rain=radar_precip_rate.loc[radar_precip_rate["lon"].between(-15.085639,-4.658210)]
            elif self.ar_of_day=="AR_entire_2":
                max_warm_lat=76.510696
                max_warm_lon=8.080936
                min_warm_lat=72.850830
                min_warm_lon=-4.379189
            else:
                Exception("no other AR sector defined")# apply sector to icon

        elif self.flight[0]=="RF06":
            if self.ar_of_day=="AR_entire_1":
                max_warm_lat=73.430244
                max_warm_lon=21.382162
                min_warm_lat=71.144676
                min_warm_lon=11.011759
        
            elif self.ar_of_day=="AR_entire_2":
                max_warm_lat=75.811005
                max_warm_lon=25.683155
                min_warm_lat=72.973465
                min_warm_lon=18.015015
        
        self.max_warm_lat=max_warm_lat
        self.max_warm_lon=max_warm_lon
        self.min_warm_lat=min_warm_lat
        self.min_warm_lon=min_warm_lon
    
    def select_warm_precip(self,radar_precip_rate,halo_icon_hmp,include_icon=True):
        """
        Caution: this routine uses the lat/lon thresholds to access ALL leg
        parts that are included in the warm AR sector domain

        Parameters
        ----------
        radar_precip_rate : pd.DataFrame
            DESCRIPTION.
        halo_icon_hmp : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        warm_radar_rain : pd.DataFrame
            DESCRIPTION.
        warm_icon_rain : pd.DataFrame
            DESCRIPTION.

        """
        self.get_warm_sector_geoloc_boundaries()
        warm_radar_rain=radar_precip_rate.loc[radar_precip_rate["lat"].between(
            self.min_warm_lat,self.max_warm_lat)]
        warm_radar_rain=warm_radar_rain.loc[warm_radar_rain["lon"].between(
            self.min_warm_lon,self.max_warm_lon)]        
        if include_icon:
            warm_icon_rain=halo_icon_hmp.loc[warm_radar_rain.index]
            warm_icon_rain["rate"]=warm_icon_rain["Interp_Precip"]
        else: 
            warm_icon_rain=pd.DataFrame()
        #try:
        #    cold_icon_rain=halo_icon_hmp.loc[cold_radar_rain.index]
        #    cold_icon_rain["rate"]=cold_icon_rain["Interp_Precip"]
        #except:
        #    print("No cold sector available")
        return warm_radar_rain,warm_icon_rain
    def save_precip_rates_series(self,radar_rain_df,sector="all"):
        precip_file_name=sector+"_precip_"+\
            self.flight[0]+"_"+self.ar_of_day+".csv"
        radar_rain_df.to_csv(self.precip_rate_path+precip_file_name)
        print(sector+" precipitation saved as:",
              self.precip_rate_path+precip_file_name)

class IWV_tendency(Moisture_Budgets):
    def __init__(self,cmpgn_name,flight,Dropsondes,IWV_retrieval,
                 major_work_path,flight_dates={},
                 sector_types=["warm","core","cold"],
                 ar_of_day="AR",grid_name="ERA5",do_instantan=False):
        
        self.flight=flight
        self.Dropsondes= Dropsondes
        self.major_work_path=major_work_path
        self.ar_of_day=ar_of_day
        self.relevant_sondes_dict={}
        self.internal_sondes_dict={}
        self.cmpgn_name="HALO_AC3"
        self.grid_name=grid_name
        self.do_instantan=do_instantan
        if self.flight[0]=="RF05":
            if self.ar_of_day=="AR_entire_1":
                self.inflow_times=["2022-03-15 10:11","2022-03-15 11:13"]
                self.internal_times=["2022-03-15 11:18","2022-03-15 12:14"]
                self.outflow_times=["2022-03-15 12:20","2022-03-15 13:15"]
            elif self.ar_of_day=="AR_entire_2":
                self.inflow_times=["2022-03-15 14:30","2022-03-15 15:25"]
                self.internal_times=["2022-03-15 13:20","2022-03-15 14:25"]
                self.outflow_times=["2022-03-15 12:20","2022-03-15 13:15"]
        if self.flight[0]=="RF06":
            if self.ar_of_day=="AR_entire_1":
                self.inflow_times=["2022-03-16 10:45","2022-03-16 11:21"]
                self.internal_times=["2022-03-16 11:25","2022-03-16 12:10"]
                self.outflow_times=["2022-03-16 12:15","2022-03-16 12:50"]
            elif self.ar_of_day=="AR_entire_2":
                self.inflow_times=["2022-03-16 12:12","2022-03-16 12:55"]
                self.internal_times=["2022-03-16 12:58","2022-03-16 13:40"]
                self.outflow_times=["2022-03-16 13:45","2022-03-16 14:18"]
        
    def get_relevant_sondes_dict(self):
        Dropsondes=self.Dropsondes
        if not "Lat" in [*Dropsondes.keys()]:
           sondes_lon=[[*Dropsondes["reference_lon"].values()][sonde].data[0] \
                    for sonde in range(Dropsondes["IWV"].shape[0])]
                    
           sondes_lat=[[*Dropsondes["reference_lat"].values()][sonde].data[0]\
                    for sonde in range(Dropsondes["IWV"].shape[0])]
           Dropsondes["Lat"]=pd.Series(data=np.array(sondes_lat),
                                                index=Dropsondes["IWV"].index)
           Dropsondes["Lon"]=pd.Series(data=np.array(sondes_lon),
                                                index=Dropsondes["IWV"].index)

        self.sonde_times_series=pd.Series(index=Dropsondes["IWV"].index.values,
                                     data=range(Dropsondes["IWV"].shape[0]))
        relevant_sondes_dict={}
        internal_sondes_dict={}
        #---------------------------------------------------------------------#
        # Define/Select relevant sondes
        if self.flight[0]=="RF05":
            if self.ar_of_day=="AR_entire_1":
                relevant_warm_sector_sondes=[0,1,2,3,9,10,11,12]
                relevant_cold_sector_sondes=[4,5,6]
                relevant_warm_internal_sondes=[7,13] #13
                relevant_sondes_dict["warm_sector"]        = \
                    {}
                relevant_sondes_dict["warm_sector"]["in"]  = \
                    self.sonde_times_series.iloc[relevant_warm_sector_sondes[0:4]]
                relevant_sondes_dict["warm_sector"]["out"] = \
                    self.sonde_times_series.iloc[relevant_warm_sector_sondes[4::]]
                relevant_sondes_dict["cold_sector"]        = \
                    {}
                relevant_sondes_dict["cold_sector"]["in"]  = \
                    self.sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
                #relevant_sondes_dict["cold_sector"]["out"] = \
                #    sonde_times_series.iloc[relevant_cold_sector_sondes[3::]]
                synthetic_sonde_times_series=\
                    pd.Series(data=["7synth","8synth","9synth"],
                        index=pd.DatetimeIndex(
                            ["2022-03-15 12:55","2022-03-15 13:05",
                             "2022-03-15 13:15"]))
                relevant_sondes_dict["cold_sector"]["out"] = \
                    synthetic_sonde_times_series
                internal_sondes_dict["warm"]               = \
                    self.sonde_times_series.iloc[relevant_warm_internal_sondes]
                internal_sondes_dict["cold"]               = \
                    ["2022-03-15 11:30:00","2022-03-15 13:35"]   
            elif self.ar_of_day=="AR_entire_2":
                relevant_warm_sector_sondes=[9,10,11,12,15,16,17,18]
                relevant_cold_sector_sondes=[19,20,21]
                relevant_warm_internal_sondes=[13,22]
                relevant_sondes_dict["warm_sector"]        = {}
                relevant_sondes_dict["warm_sector"]["in"]  = \
                    self.sonde_times_series.iloc[relevant_warm_sector_sondes[4::]]
                relevant_sondes_dict["warm_sector"]["out"] = \
                    self.sonde_times_series.iloc[relevant_warm_sector_sondes[0:4]]
                relevant_sondes_dict["cold_sector"]        = {}
                relevant_sondes_dict["cold_sector"]["in"]  = pd.Series()#sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
                relevant_sondes_dict["cold_sector"]["out"] = \
                    self.sonde_times_series.iloc[relevant_cold_sector_sondes]
                internal_sondes_dict["warm"]               = \
                    self.sonde_times_series.iloc[relevant_warm_internal_sondes]
        elif self.flight[0]=="RF06":
            if self.ar_of_day=="AR_entire_1":
                relevant_warm_sector_sondes=[0,1,2,8,9,10]
                relevant_cold_sector_sondes=[3,4,5,11,12]
                relevant_warm_internal_sondes=[7,22]
                relevant_sondes_dict["warm_sector"]        = {}
                relevant_sondes_dict["warm_sector"]["in"]  = \
                    self.sonde_times_series.iloc[relevant_warm_sector_sondes[0:3]]
                relevant_sondes_dict["warm_sector"]["out"] = \
                    self.sonde_times_series.iloc[relevant_warm_sector_sondes[3:]]
                relevant_sondes_dict["cold_sector"]        = {}
                relevant_sondes_dict["cold_sector"]["in"]  = \
                    self.sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
                relevant_sondes_dict["cold_sector"]["out"] = \
                    self.sonde_times_series.iloc[relevant_cold_sector_sondes[3:]]
                internal_sondes_dict["warm"]               = \
                    self.sonde_times_series.iloc[relevant_warm_internal_sondes]
            elif self.ar_of_day=="AR_entire_2":
                relevant_warm_sector_sondes=[8,9,16,17]
                relevant_cold_sector_sondes=[10,11,12,18,19]
                relevant_warm_internal_sondes=[14,21]
                relevant_sondes_dict["warm_sector"]        = {}
                relevant_sondes_dict["warm_sector"]["in"]  = \
                    self.sonde_times_series.iloc[relevant_warm_sector_sondes[0:2]]
                relevant_sondes_dict["warm_sector"]["out"] = \
                    self.sonde_times_series.iloc[relevant_warm_sector_sondes[2::]]
                relevant_sondes_dict["cold_sector"]        = {}
                relevant_sondes_dict["cold_sector"]["in"]  = \
                    self.sonde_times_series.iloc[relevant_cold_sector_sondes[0:3]]
                relevant_sondes_dict["cold_sector"]["out"] = \
                    self.sonde_times_series.iloc[relevant_cold_sector_sondes[3::]]
                internal_sondes_dict["warm"]               = \
                    self.sonde_times_series.iloc[relevant_warm_internal_sondes]
                internal_sondes_dict["warm"].iloc[-1]      = 100
                internal_sondes_dict["warm"]=internal_sondes_dict["warm"].rename(
                index={internal_sondes_dict["warm"].index[1]:pd.Timestamp("2022-03-16 17:05")})
                #-------------------------------------------------------------#
        self.internal_sondes_dict=internal_sondes_dict


class Moisture_Budget_Plots(Moisture_Convergence):
    
    def __init__(self,cmpgn_cls,flight,config_file,grid_name="ERA5",
                 do_instantan=False,sonde_no=3, scalar_based_div=True,
                 include_halo_ac3_components=pd.DataFrame(),
                 include_era5_components=pd.DataFrame(),
                 include_inst_components=pd.DataFrame(),
                 hours_to_use=24):
        
        super().__init__(cmpgn_cls,flight,config_file,
                         grid_name,do_instantan)
        self.plot_path=self.cmpgn_cls.plot_path+"/budget/" # ----> to be filled
        self.grid_name=grid_name
        self.sonde_no=sonde_no
        self.scalar_based_div=scalar_based_div
        if not include_halo_ac3_components.shape[0]==0:
            self.haloac3_div=include_halo_ac3_components
        if not include_era5_components.shape[0]==0:
            self.era5_div=include_era5_components
        if not include_inst_components.shape[0]==0:
            self.inst_div=include_inst_components
        self.hours_to_use=hours_to_use
        if self.hours_to_use==24:
            self.unit="$\mathrm{mmd\,}^{-1}$"
        elif self.hours_to_use==1:
            self.unit="$\mathrm{mm\,h}^{-1}$"
        
    #-------------------------------------------------------------------------#
    # Preprocessing for plots
    def allocate_budgets(self,Campaign_Budgets={},
                         Campaign_Ideal_Budgets={},
                         Campaign_Inst_Budgets={},
                         Campaign_Inst_Ideal_Budgets={},
                         already_in_mm_h=True):
        
        if Campaign_Budgets!={}:
            self.Campaign_Budgets=Campaign_Budgets
        if Campaign_Ideal_Budgets!={}:
            self.Campaign_Ideal_Budgets=Campaign_Ideal_Budgets
        if Campaign_Inst_Budgets!={}:
            self.Campaign_Inst_Budgets=Campaign_Inst_Budgets
        if Campaign_Inst_Ideal_Budgets!={}:
            self.Campaign_Inst_Ideal_Budgets=Campaign_Inst_Ideal_Budgets
        
        self.Campaign_Budgets["in_mm_h"]             = already_in_mm_h
        self.Campaign_Ideal_Budgets["in_mm_h"]       = already_in_mm_h
        
        # Instantaneous attributes are not necessarily given.
        if hasattr(self,"Campaign_Inst_Budgets"):
            self.Campaign_Inst_Budgets["in_mm_h"]        = already_in_mm_h
        if hasattr(self,"Campaign_Inst_Budgets"):
            self.Campaign_Inst_Ideal_Budgets["in_mm_h"]  = already_in_mm_h
        
    def calc_budgets_in_mm_h(self):
        """
        
        This function calculates the budget contributions of moisture transport
        divergence from units per second to mm per hour for both representations
        sonde based or continuous
        
        """
        if not self.Campaign_Budgets["in_mm_h"]:
            gravit_norm=1/9.82
            time_factor=3600/1000
        else:
            gravit_norm=1
            time_factor=1
            print("Values are already in mm/h")
        if hasattr(self,"Campaign_Budgets"):
            warm_budgets=self.Campaign_Budgets["warm_sector"]
            core_budgets=self.Campaign_Budgets["core"]
            cold_budgets=self.Campaign_Budgets["cold_sector"]
            self.budget_regions=pd.DataFrame()
            
            self.budget_regions["Warm\nADV"]=gravit_norm*\
                            warm_budgets["ADV"].values*time_factor
            self.budget_regions["Warm\nCONV"]=gravit_norm*\
                            warm_budgets["CONV"].values*time_factor
            self.budget_regions["Core\nADV"]=gravit_norm*\
                            core_budgets["ADV"].values*time_factor
            self.budget_regions["Core\nCONV"]=gravit_norm*\
                            core_budgets["CONV"].values*time_factor
            self.budget_regions["Cold\nADV"]=gravit_norm*\
                            cold_budgets["ADV"].values*time_factor
            self.budget_regions["Cold\nCONV"]=gravit_norm*\
                            cold_budgets["CONV"].values*time_factor
        
        if hasattr(self,"Campaign_Ideal_Budgets"):
            warm_ideal_budgets=self.Campaign_Ideal_Budgets["warm_sector"]
            core_ideal_budgets=self.Campaign_Ideal_Budgets["core"]
            cold_ideal_budgets=self.Campaign_Ideal_Budgets["cold_sector"]
            
            self.budget_ideal_regions=pd.DataFrame()
            
            self.budget_ideal_regions["Warm\nADV"]=gravit_norm*\
                            warm_ideal_budgets["ADV"].values*time_factor
            self.budget_ideal_regions["Warm\nCONV"]=gravit_norm*\
                            warm_ideal_budgets["CONV"].values*time_factor
            self.budget_ideal_regions["Core\nADV"]=gravit_norm*\
                            core_ideal_budgets["ADV"].values*time_factor
            self.budget_ideal_regions["Core\nCONV"]=gravit_norm*\
                            core_ideal_budgets["CONV"].values*time_factor
            self.budget_ideal_regions["Cold\nADV"]=gravit_norm*\
                            cold_ideal_budgets["ADV"].values*time_factor
            self.budget_ideal_regions["Cold\nCONV"]=gravit_norm*\
                            cold_ideal_budgets["CONV"].values*time_factor
        
        if hasattr(self,"Campaign_Inst_Budgets"):
            warm_inst_budgets=self.Campaign_Inst_Budgets["warm_sector"]
            core_inst_budgets=self.Campaign_Inst_Budgets["core"]
            cold_inst_budgets=self.Campaign_Inst_Budgets["cold_sector"]
            
            self.budget_inst_regions=pd.DataFrame()
            self.budget_inst_regions["Warm\nADV"]=gravit_norm*\
                            warm_inst_budgets["ADV"].values*time_factor
            self.budget_inst_regions["Warm\nCONV"]=gravit_norm*\
                            warm_inst_budgets["CONV"].values*time_factor
            self.budget_inst_regions["Core\nADV"]=gravit_norm*\
                            core_inst_budgets["ADV"].values*time_factor
            self.budget_inst_regions["Core\nCONV"]=gravit_norm*\
                            core_inst_budgets["CONV"].values*time_factor
            self.budget_inst_regions["Cold\nADV"]=gravit_norm*\
                            cold_inst_budgets["ADV"].values*time_factor
            self.budget_inst_regions["Cold\nCONV"]=gravit_norm*\
                            cold_inst_budgets["CONV"].values*time_factor
        
        if hasattr(self,"Campaign_Inst_Ideal_Budgets"):
            warm_inst_ideal_budgets=\
                self.Campaign_Inst_Ideal_Budgets["warm_sector"]
            core_inst_ideal_budgets=\
                self.Campaign_Inst_Ideal_Budgets["core"]
            cold_inst_ideal_budgets=\
                self.Campaign_Inst_Ideal_Budgets["cold_sector"]
            
            self.budget_inst_ideal_regions=pd.DataFrame()
            
            self.budget_inst_ideal_regions["Warm\nADV"]=gravit_norm*\
                                        warm_inst_ideal_budgets["ADV"].values*\
                                                        time_factor
            self.budget_inst_ideal_regions["Warm\nCONV"]=gravit_norm*\
                                        warm_inst_ideal_budgets["CONV"].values*\
                                                        time_factor
            self.budget_inst_ideal_regions["Core\nADV"]=gravit_norm*\
                core_inst_ideal_budgets["ADV"].values*time_factor
            self.budget_inst_ideal_regions["Core\nCONV"]=gravit_norm*\
                core_inst_ideal_budgets["CONV"].values*time_factor
            self.budget_inst_ideal_regions["Cold\nADV"]=gravit_norm*\
                cold_inst_ideal_budgets["ADV"].values*time_factor
            self.budget_inst_ideal_regions["Cold\nCONV"]=gravit_norm*\
                        cold_inst_ideal_budgets["CONV"].values*time_factor
        
    
    ###############################################################################
    # Plot Major functions
    # Figure 12
    def plot_single_case(self,Sectors,Ideal_Sectors,
                         do_log_scale=True,
                         save_as_manuscript_figure=False):
        """
    
        This function plots vertical profiles of moisture ADV and mass CONV and 
        moisture transport convergence. (Fig.12 in manuscript)
        Parameters
        ----------
        Sectors : dictionary 
            containing all three variables in a DataFrame with keys of sectors
            using the defined sonde_no.
        
        Ideal_Sectors : dictionary
            containing all three variabels in DataFrame with key of sectors
            using many sondes as 

        Returns
        -------
        None.
        
        """
        from matplotlib.ticker import NullFormatter
        #---------- Get sector-based divergence from dict --------------------#
        core=Sectors["core"]
        warm_sector=Sectors["warm_sector"]
        cold_sector=Sectors["cold_sector"]
        
        core_ideal=Ideal_Sectors["core"]
        warm_sector_ideal=Ideal_Sectors["warm_sector"]
        cold_sector_ideal=Ideal_Sectors["cold_sector"]
        #---------------------------------------------------------------------#
    
        matplotlib.rcParams.update({"font.size":24})
        profile=plt.figure(figsize=(12,9))
        #######################################################################
        #Moisture advection
        ax1=profile.add_subplot(131)
        ax2=profile.add_subplot(132)
        ax3=profile.add_subplot(133)
        
        if do_log_scale:
            ax1.set_yscale("log")
            ax2.set_yscale("log")
            ax3.set_yscale("log")
        
        ax1.plot(core["ADV_calc"],
             core["ADV_calc"].index.astype(int),lw=1,
             color="green",ls="--")
    
        ax1.plot(warm_sector["ADV_calc"],
             warm_sector["ADV_calc"].index.astype(int),lw=1,
             color="orange",ls="--")
    
        ax1.plot(cold_sector["ADV_calc"],
             cold_sector["ADV_calc"].index.astype(int),lw=1,
             color="blue",ls="--")
        # ideal sector
        ax1.plot(core_ideal["ADV_calc"],
             core_ideal["ADV_calc"].index.astype(int),lw=2,label="core",
             color="darkgreen")
    
        ax1.plot(warm_sector_ideal["ADV_calc"],
             warm_sector_ideal["ADV_calc"].index.astype(int),lw=2,
             label="warm sector",color="darkorange")
    
        ax1.plot(cold_sector_ideal["ADV_calc"],
             cold_sector_ideal["ADV_calc"].index.astype(int),lw=2,
             label="cold sector",color="darkblue")
    
        ax1.axvline(0,ls="--",lw=2,color="k")
        ax1.fill_betweenx(y=core.index.astype(float),
                          x1=core["ADV_calc"],
                          x2=core_ideal["ADV_calc"], 
                          color="green",alpha=0.3)
        ax1.fill_betweenx(y=warm_sector.index.astype(float),
                          x1=warm_sector["ADV_calc"],
                          x2=warm_sector_ideal["ADV_calc"], 
                          color="orange",alpha=0.3)
        ax1.fill_betweenx(y=core.index.astype(float),
                          x1=cold_sector["ADV_calc"],
                          x2=cold_sector_ideal["ADV_calc"], 
                          color="blue",alpha=0.3)
    
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(3)
    
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.yaxis.set_tick_params(width=3,length=6)
        
        ax1.legend(loc="upper left",fontsize=14)
        ax1.set_title("\nADV",fontsize=20)
        ax1.set_xlim([-3e-4,3e-4])
        ax1.set_xticks([-3e-4,0,3e-4])
        ax1.set_xticklabels(["-3e-4","0","3e-4"])
        ax1.xaxis.set_tick_params(width=3,length=6)
        
        sns.despine(ax=ax1,offset=10)
        ax1.yaxis.set_minor_formatter(NullFormatter())
        ax1.set_ylim([200,1000])
        ax1.invert_yaxis()
        ax1.set_yticks([])
        ax1.set_yticklabels([],which="both")
        #######################################################################
        ax2.axvline(0,ls="--",lw=2,color="k")
        ax2.plot(core["CONV"],core["CONV"].index.astype(int),
                 label="core",color="green",ls="--")
        ax2.plot(warm_sector["CONV"],warm_sector["CONV"].index.astype(int),
             label="warm sector",color="orange",ls="--")
        ax2.plot(cold_sector["CONV"],cold_sector["CONV"].index.astype(int),
             label="core",color="blue",ls="--")
        # ideal
        ax2.plot(core_ideal["CONV"],core_ideal["CONV"].index.astype(int),
                 label="core",color="darkgreen",lw=2)
        ax2.plot(warm_sector_ideal["CONV"],
                 warm_sector_ideal["CONV"].index.astype(int),
                 label="warm sector",color="darkorange",lw=2)
        ax2.plot(cold_sector_ideal["CONV"],
                 cold_sector_ideal["CONV"].index.astype(int),
                 label="core",color="darkblue",lw=2)
        
        ax2.set_ylim([200,1000])
        
        ax2.fill_betweenx(y=core.index.astype(float),
                          x1=core["CONV"],
                          x2=core_ideal["CONV"], 
                          color="green",alpha=0.3)
        ax2.fill_betweenx(y=warm_sector.index.astype(float),
                          x1=warm_sector["CONV"],
                          x2=warm_sector_ideal["CONV"], 
                          color="orange",alpha=0.3)
        ax2.fill_betweenx(y=core.index.astype(float),
                          x1=cold_sector["CONV"],
                          x2=cold_sector_ideal["CONV"], 
                          color="blue",alpha=0.3)
    
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(3)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_title("mass DIV",fontsize=20)
        ax2.set_xlim([-3e-4,3e-4])
        ax2.set_xticks([-3e-4,0,3e-4])
        ax2.set_xticklabels(["-3e-4","0","3e-4"])
        ax2.xaxis.set_tick_params(width=3,length=6)
        ax2.yaxis.set_tick_params(width=3,length=6)
    
        ax2.set_xlabel("Flux divergence in $\mathrm{gkg}^{-1}\mathrm{s}^{-1}$")
        
        ax2.invert_yaxis()
        ax2.yaxis.set_minor_formatter(NullFormatter())
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        sns.despine(offset=10)
        #######################################################################
        core["sum_trans"]=core["CONV"]+core["ADV_calc"]
        warm_sector["sum_trans"]=warm_sector["CONV"]+warm_sector["ADV_calc"]
        cold_sector["sum_trans"]=cold_sector["CONV"]+cold_sector["ADV_calc"]
        
        
        core_ideal["sum_trans"]=core_ideal["CONV"]+core_ideal["ADV_calc"]
        warm_sector_ideal["sum_trans"]=warm_sector_ideal["CONV"]+\
                                        warm_sector_ideal["ADV_calc"]
        cold_sector_ideal["sum_trans"]=cold_sector_ideal["CONV"]+\
                                            cold_sector_ideal["ADV_calc"]
        
        ax3.plot(core["CONV"]+core["ADV_calc"],
             core["TRANSP"].index.astype(int),label="core",color="green",ls="--")
        ax3.plot(warm_sector["CONV"]+warm_sector["ADV_calc"],
             warm_sector["CONV"].index.astype(int),
             label="warm sector",color="orange",ls="--")
        ax3.plot(cold_sector["CONV"]+cold_sector["ADV_calc"],
             cold_sector["TRANSP"].index.astype(int),
             label="cold sector",color="blue",ls="--")
        # ideal
        ax3.plot(core_ideal["CONV"]+core_ideal["ADV_calc"],
             core_ideal["TRANSP"].index.astype(int),
             label="core",color="darkgreen",lw=2)
        ax3.plot(warm_sector_ideal["CONV"]+warm_sector_ideal["ADV_calc"],
             warm_sector_ideal["CONV"].index.astype(int),
             label="warm sector",color="darkorange",lw=2)
        ax3.plot(cold_sector_ideal["CONV"]+cold_sector_ideal["ADV_calc"],
             cold_sector_ideal["TRANSP"].index.astype(int),
             label="cold sector",color="darkblue",lw=2)
        
        ax3.fill_betweenx(y=core.index.astype(float),
                          x1=core["sum_trans"],
                          x2=core_ideal["sum_trans"], 
                          color="green",alpha=0.3)
        ax3.fill_betweenx(y=warm_sector.index.astype(float),
                          x1=warm_sector["sum_trans"],
                          x2=warm_sector_ideal["sum_trans"], 
                          color="orange",alpha=0.3)
        ax3.fill_betweenx(y=core.index.astype(float),
                          x1=cold_sector["sum_trans"],
                          x2=cold_sector_ideal["sum_trans"], 
                          color="blue",alpha=0.3)
    
        ax3.axvline(0,ls="--",lw=2,color="k")
        ax3.spines['left'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(True)
        
        for axis in ['top','bottom','left','right']:
            ax3.spines[axis].set_linewidth(3)
        ax3.yaxis.set_tick_params(width=3,length=6)
        ax3.xaxis.set_tick_params(width=3,length=6)
        ax3.set_title("moist. transp.",fontsize=20)
        ax3.set_xlim([-3e-4,3e-4])
        ax3.set_xticks([-3e-4,0,3e-4])
        ax3.set_xticklabels(["-3e-4","0","3e-4"])

        ax1.text(-4.5e-4,225,"(a)",fontsize=18)#transform=ax1.transAxes)
        ax2.text(-4.5e-4,225,"(b)",fontsize=18)#,transform=ax1.transAxes)
        ax3.text(-4.5e-4,225,"(c)",fontsize=18)#,transform=ax1.transAxes)
        ax1.set_ylabel("Pressure in hPa")
        ax1.set_yticks([300,500,700,850,1000])
        ax1.set_yticklabels(["300","500","700","850","1000"])
        ax2.set_yticks([300,500,700,850,1000])
        ax2.set_yticklabels([])
    
        ax3.yaxis.tick_right()
        ax3.invert_yaxis()
        ax3.set_ylim([1000,200])
        for loc, spine in ax3.spines.items():
            if loc in ["right","bottom"]:
                spine.set_position(('outward', 10)) 
        ax3.set_yticks([])
        ax3.yaxis.set_minor_formatter(NullFormatter())
        ax3.set_yticks([300,500,700,850,1000])
        ax3.set_yticklabels(["300","500","700","850","1000"])
                
        plt.subplots_adjust(wspace=0.4)
        file_end=".png"
        fig_name=self.flight+"_"+self.grid_name+"_sonde_no_"+\
            str(self.sonde_no)+"_Moisture_transport_Divergence"
        if not self.scalar_based_div:
            fig_name+="_vectorised"
        fig_name+=file_end
        if not save_as_manuscript_figure:
            fig_plot_path=self.cmpgn_cls.plot_path+"/budget/"
            if not os.path.exists(fig_plot_path):
                os.makedirs(fig_plot_path)
        else:
            fig_plot_path=self.cmpgn_cls.plot_path+"/../../../"+\
                    "Synthetic_AR_Paper/"+"/Manuscript/Paper_Plots/"
            fig_name="fig12_"+fig_name
        profile.savefig(fig_plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",fig_plot_path+fig_name)
    # Figure 13
    def moisture_convergence_cases_overview(self,Campaign_Budgets={},
            Campaign_Ideal_Budgets={},Campaign_Inst_Budgets={},
            Campaign_Inst_Ideal_Budgets={},save_as_manuscript_figure=False,
            instantan_comparison=False,with_mean_error=False
            ):
        """
        This plotting routine is used to create Figure 13 of the manuscript
        being the boxplots of moisture budget contributions in mm/h occuring 
        from moisture advection (ADV) and mass convergence (CONV) across the 
        frontal sectors. The continuous and sonde-based statistics are depicted
        if instantan comparison is False (then it is equivalent to Figure 13),
        otherwise it just compares in the instantaneous time perspective.

        Parameters
        ----------
        Campaign_Budgets : dict, optional
            Budget compoents for sonde-based. The default is {}.
        Campaign_Ideal_Budgets : dict, optional
            Budget components for continuous representation. The default is {}.
        Campaign_Inst_Budgets : dict, optional
            Budget components for instantaneous sondes. The default is {}.
        Campaign_Inst_Ideal_Budgets : dict, optional
            Budget components for continuous and instantaneous representation.
            The default is {}.
        save_as_manuscript_figure : boolean, optional
            specifies if figure has to be saved as manuscript figure with 
            respective naming and file directory. The default is False.
        instantan_comparison : boolean, optional
            Switcher if instantaneous analysis/comparison should be conducted.
            The default is False for figure 13.
        with_mean_error : boolean, optional
            Switcher if mean errors should be added in Boxplots, is included in 
            final figure 13. The default is False.

        Returns
        -------
        None.

        """
        # Allocate variables and calc budget contributions in mm/h
        self.allocate_budgets(Campaign_Budgets,Campaign_Ideal_Budgets,
                              Campaign_Inst_Budgets,Campaign_Inst_Ideal_Budgets,
                              already_in_mm_h=False)
        self.calc_budgets_in_mm_h()
        #Start plotting
        budget_boxplot=plt.figure(figsize=(12,9), dpi= 300)
        matplotlib.rcParams.update({'font.size': 24})
            
        color_palette=["darkorange","orange","lightgreen",
                   "green","lightblue","blue"]                    

        ax1=budget_boxplot.add_subplot(111)
        ax1.axhline(0,color="grey",ls="--",lw=2,zorder=1)
    
        budget_regions=self.budget_regions
        budget_regions.columns=["Pre-fr. \n$IADV_{\mathrm{q}}$",
            "Pre-fr. \n$IDIV_{\mathrm{mass}}$",
            "Core \n$IADV_{\mathrm{q}}$",
            "Core \n$IDIV_{\mathrm{mass}}$",
            "Post-fr. \n$IADV_{\mathrm{q}}$",
            "Post-fr. \n$IDIV_{\mathrm{mass}}$"]
        
        if not instantan_comparison:
            budget_ideal_regions=self.budget_ideal_regions
        else:
            budget_ideal_regions=self.budget_inst_ideal_regions
        budget_ideal_regions.columns=["$IADV_{\mathrm{q}}$ \n(Pre-\nfrontal)",
            "$IDIV_{\mathrm{mass}}$ \n(Pre-\nfrontal)",
            "$IADV_{\mathrm{q}}$ \n(Core)",
            "$IDIV_{\mathrm{mass}}$ \n(Core)",
            "$IADV_{\mathrm{q}}$ \n(Post-\nfrontal)",
            "$IDIV_{\mathrm{mass}}$ \n(Post-\nfrontal)"]
            
        if self.grid_name=="CARRA":
            budget_ideal_regions=-1*budget_ideal_regions
            budget_regions=-1*budget_regions
            
        sns.boxplot(data=self.hours_to_use*budget_ideal_regions,
                    notch=False,zorder=0,linewidth=3.5,
                    palette=color_palette)
        
        for patch in ax1.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .5))
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(3.0)
        
        ax1.xaxis.set_tick_params(width=2,length=10)
        ax1.yaxis.set_tick_params(width=2,length=10)
                        #ax1.xaxis.spines(width=3)
        ax1.set_ylabel("Contribution to \nmoisture budget ("+self.unit+")")
        
        if not hasattr(self, "haloac3_div"):
            # Plot synthetic sondes    
            sns.boxplot(data=self.hours_to_use*budget_regions,
                        width=0.4,linewidth=3.0,notch=False,color="k",
                        palette=["lightgrey"],zorder=1)
        
            if with_mean_error:
                ax12=ax1.twinx()
            
                # Mean difference mean(ideal-sondes)
                sector_divergence_sonde_errors=\
                    self.hours_to_use*(budget_ideal_regions.iloc[:,0:6]-\
                    budget_regions.iloc[:,0:6])
                mean_sector_divergence_sonde_errors=sector_divergence_sonde_errors.mean()
            
                ax12.scatter(mean_sector_divergence_sonde_errors.index,
                         mean_sector_divergence_sonde_errors,marker="o",s=100,
                         color="red",edgecolor="k")
                if self.hours_to_use==24:
                    ax12.set_ylim([-5,5])
                else:
                    ax12.set_ylim([-0.75,0.75])
                    ax12.set_yticks([-.75,-.5,-.25,0,.25,.5,.75])
                ax12.spines["right"].set_linewidth(3.0)
                ax12.tick_params(axis='y', colors='darkred')
                ax12.xaxis.set_tick_params(width=2,length=10)
                ax12.yaxis.set_tick_params(width=2,length=10)
                ax12.set_ylabel("Mean Error in \n Contribution ("+self.unit+")",
                            color="darkred")
                ax12.spines["left"].set_visible(False)
                ax12.spines["top"].set_visible(False)
                ax12.spines["bottom"].set_visible(False)
                if self.hours_to_use==24:
                    ax1.set_ylim([-40,40])
                else:
                    ax1.set_ylim([-2.5,2.5])
                
        else:
            # Plot synthetic sondes    
            ax1.scatter([1,1,1,1],
                        -self.hours_to_use*self.haloac3_div.iloc[0:4,0],
                         marker="v",s=500,color="whitesmoke",lw=3,edgecolor="k",zorder=2)
            ax1.scatter([0,0,0,0],
                        -self.hours_to_use*self.haloac3_div.iloc[4:8,0],
                        marker="v",s=500,color="whitesmoke",lw=3,edgecolor="k",
                        zorder=2,label="HALO-$(\mathcal{AC})^{3}$ AR")
            if self.hours_to_use==24:
                ax1.set_ylim([-40,45])
            else:
                ax1.set_ylim([-2,2.5])
            ax1.legend(loc="upper right")
            
        sns.despine(ax=ax1,offset=10)
        fileend=".pdf"
        if not self.do_instantan:
            fig_name=self.grid_name+"_Water_Vapour_Budget"
        else:
            fig_name=self.grid_name+"_inst"+"_Water_Vapour_Budget"
        if with_mean_error:
            fig_name+="_with_mean_error"
        fig_name+=fileend
        if not save_as_manuscript_figure:
            plot_path=self.plot_path
            if hasattr(self,"haloac3_div"):
                plot_path=self.plot_path+"/../../../../../"+\
                    "my_GIT/Arctic_ARs_Thesis/plots/"
                fig_name="Fig3_5_IVTdiv_sector_comparison_synth_CARRA_HALO_AC3.pdf"
        else:
            plot_path=self.plot_path+\
                "/../../../../Synthetic_AR_paper/Manuscript/Paper_Plots/"
            if not instantan_comparison:
                fig_name="fig13_"+fig_name
            else:
                fig_name="Fig16_"+fig_name
        budget_boxplot.savefig(plot_path+fig_name,
                       dpi=300,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)
    
    # Figure 14
    def moisture_convergence_time_instantan_comparison(
        self,Campaign_Budgets={},Campaign_Ideal_Budgets={},
        Campaign_Inst_Budgets={},Campaign_Inst_Ideal_Budgets={},
        save_as_manuscript_figure=False,plot_mean_error=False,
        use_flight_tracks=False):
        """
        This plotting routine is used to create Figure 14 of the manuscript
        being the boxplots of moisture budget contributions in mm/d occuring 
        from moisture advection (ADV) and mass convergence (CONV) across the 
        frontal sectors. Both continuous representation (flight duration and 
        instantaneous) are contrasted. Mean errors are added if set true.
        (then it is equivalent to Figure 14).

        Parameters
        ----------
        Campaign_Budgets : dict, optional
            Budget compoents for sonde-based. The default is {}.
        Campaign_Ideal_Budgets : dict, optional
            Budget components for continuous representation. The default is {}.
        Campaign_Inst_Budgets : dict, optional
            Budget components for instantaneous sondes. The default is {}.
        Campaign_Inst_Ideal_Budgets : dict, optional
            Budget components for continuous and instantaneous representation.
            The default is {}.
        save_as_manuscript_figure : boolean, optional
            specifies if figure has to be saved as manuscript figure with 
            respective naming and file directory. The default is False.
        plot_mean_error : boolean, optional
            Switcher if mean errors should be added in Boxplots, is included in 
            final figure 14. The default is False.
        
        use_flight_tracks : TYPE, optional
            DESCRIPTION. The default is False.

        instantan_comparison : boolean, optional
            Switcher if instantaneous analysis/comparison should be conducted.
            The default is False for figure 13.
        """
        # Allocate variables and calc budget contributions in mm/h
        self.allocate_budgets(Campaign_Budgets,Campaign_Ideal_Budgets,
                              Campaign_Inst_Budgets,Campaign_Inst_Ideal_Budgets,
                              already_in_mm_h=False)
        self.calc_budgets_in_mm_h()
        #Start plotting
        budget_boxplot=plt.figure(figsize=(12,9), dpi= 300)
        matplotlib.rcParams.update({'font.size': 24})
            
        color_palette=["darkorange","orange","lightgreen",
                   "green","lightblue","blue"]                    

        ax1=budget_boxplot.add_subplot(111)
        ax1.axhline(0,color="grey",ls="--",lw=2,zorder=1)
    
        budget_continuous_regions=self.budget_ideal_regions
        
        budget_ideal_regions=self.budget_inst_ideal_regions
        if self.grid_name=="CARRA":
            budget_ideal_regions=-budget_ideal_regions
            budget_continuous_regions=-budget_continuous_regions
        budget_ideal_regions.columns=["$IADV_{\mathrm{q}}$\n(Pre-\nfrontal)",
            "$IDIV_{\mathrm{mass}}$\n(Pre-\nfrontal)",
            "$IADV_{\mathrm{q}}$\n(Core)",
            "$IDIV_{\mathrm{mass}}$\n(Core)",
            "$IADV_{\mathrm{q}}$\n(Post-\nfrontal)",
            "$IDIV_{\mathrm{mass}}$\n(Post-\nfrontal)"]
        budget_continuous_regions.columns=["$IADV_{\mathrm{q}}$\n(Pre-\nfrontal)",
            "$IDIV_{\mathrm{mass}}$\n(Pre-\nfrontal)",
            "$IADV_{\mathrm{q}}$\n(Core)",
            "$IDIV_{\mathrm{mass}}$\n(Core)",
            "$IADV_{\mathrm{q}}$\n(Post-\nfrontal)",
            "$IDIV_{\mathrm{mass}}$\n(Post-\nfrontal)"]
        
        budget_continuous_regions["Time"]="Non-instantaneous"
        budget_continuous_regions["Sector_Term"]=budget_continuous_regions.index.copy()
        budget_continuous_regions.index=np.arange(9)
        budget_ideal_regions["Time"]="Instantaneous"
        budget_ideal_regions["Sector_Term"]=budget_ideal_regions.index.copy()
        budget_ideal_regions.index=np.arange(9)
        
        
        budget_regions=pd.concat([budget_continuous_regions,budget_ideal_regions],
                                 ignore_index=True)
        budget_values=pd.DataFrame(data=np.nan,index=range(6*18),
                                   columns=["DIV","Sector_Term","Time"])
        for com,sector_comp in enumerate(budget_ideal_regions.iloc[:,:-2].columns):
            budget_values["Time"][18*com:18*com+18]=budget_regions["Time"].values
            budget_values["Sector_Term"][18*com:18*com+18]=sector_comp                                        
            budget_values["DIV"][18*com:18*com+18]=\
                self.hours_to_use*budget_regions[sector_comp].values
            
        ax1=sns.boxplot(data=budget_values,x="Sector_Term",y="DIV",
                    hue="Time",zorder=0,linewidth=2,color="k",
                    palette=["k","grey"],
                    medianprops=dict(color="yellow", alpha=0.7,linewidth=4,
                                     zorder=2),width=0.4)
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(3.0)
        
            ax1.xaxis.set_tick_params(width=2,length=10)
            ax1.yaxis.set_tick_params(width=2,length=10)

        ax1.set_ylabel("Contribution to \nMoisture Budget ("+self.unit+")")
        ax1.set_xlabel("")
        if self.hours_to_use==24:
            ax1.set_ylim([-50,50])
        else:
            ax1.set_ylim([-2.5,2.5])
        ax1.legend(loc="lower left", title="Time reference",fontsize=20)
        file_end=".pdf"
        if not self.do_instantan:
            fig_name=self.grid_name+"_Water_Vapour_Budget"
        else:
            fig_name=self.grid_name+"_inst"+"_Water_Vapour_Budget"
        if use_flight_tracks:
            fig_name=fig_name+"_on_flight"
        if not self.scalar_based_div:
            fig_name+="_vectorised"
        fig_name=fig_name+file_end
        if not save_as_manuscript_figure:
            plot_path=self.plot_path
        else:
            plot_path=self.plot_path+\
                "/../../../../Synthetic_AR_paper/Manuscript/Paper_Plots/"
        if plot_mean_error:
            sector_divergence_inst_errors=\
                self.hours_to_use*(budget_continuous_regions.iloc[:,0:6]-\
                    budget_ideal_regions.iloc[:,0:6])
            mean_sector_divergence_inst_errors=sector_divergence_inst_errors.mean()
            
            ax12=ax1.twinx()
            ax12.scatter(mean_sector_divergence_inst_errors.index,
                         mean_sector_divergence_inst_errors,marker="o",s=100,
                         color="red",edgecolor="k")
            if self.hours_to_use==24:
                ax12.set_ylim([-20,20])
            else:
                ax12.set_ylim([-.75,.75])
                ax12.set_yticks([-.75,-.5,-.25,0,.25,.5,.75])
            #for axis in ["right"]:
            ax12.spines["right"].set_linewidth(3.0)
            ax12.tick_params(axis='y', colors='darkred')
            ax12.xaxis.set_tick_params(width=2,length=10)
            ax12.yaxis.set_tick_params(width=2,length=10)
            ax12.set_ylabel("Mean Deviation in \n Contribution ("+self.unit+")",
                            color="darkred")
            ax12.spines["left"].set_visible(False)
            ax12.spines["top"].set_visible(False)
            ax12.spines["bottom"].set_visible(False)
        
        if hasattr(self, "haloac3_div"):
            ax1.scatter([-.1,-0.1,-0.1,-0.1],
                        -self.hours_to_use*self.era5_div.iloc[4:8,0],
                        marker="v",s=300,color="darkgrey",
                        edgecolor="k",zorder=2)
            ax1.scatter([0.1,0.1,0.1,0.1],
                        -self.hours_to_use*self.inst_div.iloc[4:8,0],
                        marker="v",s=300,color="whitesmoke",
                        edgecolor="k",zorder=2)
            ax1.scatter([0.9,0.9,0.9,0.9],
                        -self.hours_to_use*self.era5_div.iloc[0:4,0],
                        marker="v",s=300,color="darkgrey",
                        edgecolor="k",zorder=2)
            ax1.scatter([1.1,1.1,1.1,1.1],
                        -self.hours_to_use*self.inst_div.iloc[0:4,0],
                        marker="v",s=300,color="whitesmoke",
                        edgecolor="k",zorder=2)
            #ax1.scatter([1,1,1,1],-24*self.haloac3_div.iloc[0:4,0],
            #             marker="v",s=500,color="whitesmoke",
            #             lw=3,edgecolor="k",zorder=2)
            #ax1.scatter([0,0,0,0],-24*self.haloac3_div.iloc[4:8,0],
            #            marker="v",s=500,color="whitesmoke",lw=3,edgecolor="k",
            #            zorder=2,label="HALO-$(\mathcal{AC})^{3}$ AR")
            
        if not save_as_manuscript_figure:
                plot_path=self.plot_path
                if hasattr(self,"haloac3_div"):
                    plot_path=self.plot_path+"/../../../../../"+\
                        "my_GIT/Arctic_ARs_Thesis/plots/"
                    fig_name="Fig4_3_IVT_div_inst_comparison_synth_CARRA_HALO_AC3"
        else:
            fig_name="fig14_"+fig_name
        file_end=".pdf"
        fig_name+=file_end
        sns.despine(ax=ax1,offset=10)
        budget_boxplot.savefig(plot_path+fig_name,
                       dpi=200,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)
    #Figure 15
    def plot_rmse_instantan_sonde(self,save_as_manuscript_figure=False):
        """
        
        This plot routines illustrates the root-mean squared error (RMSE) to 
        the continuous and instantaneous representation of divergence components
        as a reference (Fig. 15). Three bars specify the instantaneous error 
        (for both continuous representations). Then error for sounding just in 
        the instantanenous representation and the combined error. 
        
        Parameters
        ----------
        save_as_manuscript_figure : boolean, optional
            if figure should be located and named as included in the manuscript.
            The default is False.

        Returns
        -------
        None.

        """
        budget_inst_ideal_regions=self.hours_to_use*self.budget_inst_ideal_regions
        budget_ideal_regions=self.hours_to_use*self.budget_ideal_regions
        budget_regions=self.hours_to_use*self.budget_regions
        
        sector_divergence_inst_errors=\
                            budget_ideal_regions-budget_inst_ideal_regions
        
        sector_divergence_inst_sonde_errors=\
                            budget_regions-budget_inst_ideal_regions
        sector_divergence_sonde_errors=\
                            budget_regions-budget_ideal_regions
                            
        # Continuous to instantan errors
        sector_squared_divergence_inst_errors = \
            (sector_divergence_inst_errors)**2
        # Entire errors
        sector_squared_divergence_inst_sonde_errors      = \
            (sector_divergence_inst_sonde_errors)**2
        # Pure sonde errors
        sector_squared_divergence_sonde_errors= \
            (sector_divergence_sonde_errors)**2
        
        rmse_inst       = np.sqrt(sector_squared_divergence_inst_errors.mean())
        rmse_inst_sonde = np.sqrt(sector_squared_divergence_inst_sonde_errors.mean())
        rmse_pure_sonde = np.sqrt(sector_squared_divergence_sonde_errors.mean())    
        
        #### Statistics of errors
        mean_sector_divergence_inst_errors=\
            sector_divergence_inst_errors.mean()
        mean_sector_divergence_inst_sonde_errors=\
            sector_divergence_inst_sonde_errors.mean()
        mean_sector_divergence_sonde_errors=\
            sector_divergence_sonde_errors.mean()
        std_sector_divergence_inst_errors=\
            sector_divergence_inst_errors.std()
        std_sector_divergence_inst_sonde_errors=\
            sector_divergence_inst_sonde_errors.std()
        std_sector_divergence_sonde_errors=\
            sector_divergence_sonde_errors.std()
        
        error_fig=plt.figure(figsize=(12,9))
        ax1=error_fig.add_subplot(111)
        ax1.bar(np.arange(rmse_inst.shape[0])+1,
                rmse_inst,facecolor="darkgrey",
                width=0.15,edgecolor="k",linewidth=3,
                alpha=0.9,label="Non-instantaneous")
        ax1.bar(np.arange(rmse_inst.shape[0])+0.85,
                rmse_inst_sonde,facecolor="peachpuff",
                width=0.15,edgecolor="saddlebrown",
                linewidth=3,linestyle=":",
                alpha=0.9,label="Non-instantaneous &\nDiscrete")
        ax1.bar(np.arange(rmse_inst.shape[0])+1.15,
                rmse_pure_sonde,facecolor="lightgreen",width=0.15,linestyle="--",
                edgecolor="darkgreen",linewidth=3,alpha=0.9,label="Discrete")
        
        ax1.set_xticks(np.arange(rmse_inst.shape[0])+1)
        ax1.set_xticklabels(rmse_inst.index,fontsize=10)
        ax1.set_xlabel("Frontal Sector and Component")
        
        ax1.set_ylabel("Deviation (inst-evolving) in \nIVT Divergence ("+self.unit+")")
        # Axis linewidth
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(3.0)
        ax1.xaxis.set_tick_params(width=2,length=10)
        ax1.yaxis.set_tick_params(width=2,length=10)
        sns.despine(offset=10)
        ##### Instationarity error
        eb1=ax1.errorbar(np.arange(
                        mean_sector_divergence_inst_errors.shape[0])+1,
                        mean_sector_divergence_inst_errors,markeredgewidth=2,
                        fmt="s",markersize=15,markeredgecolor="k",ecolor="k",
                        yerr=std_sector_divergence_inst_errors,
                        color="white",zorder=2)
        eb2=ax1.errorbar(np.arange(
                        mean_sector_divergence_inst_sonde_errors.shape[0])+0.85,
                        mean_sector_divergence_inst_sonde_errors,markeredgewidth=2,
                        fmt="d",markersize=15,markeredgecolor="k",ecolor="k",
                        yerr=std_sector_divergence_inst_sonde_errors,
                        color="seashell",zorder=2)
        eb3=ax1.errorbar(np.arange(
                        mean_sector_divergence_sonde_errors.shape[0])+1.15,
                        mean_sector_divergence_sonde_errors,markeredgewidth=2,
                        fmt="v",markersize=15,markeredgecolor="k",ecolor="k",
                        yerr=std_sector_divergence_sonde_errors,
                        color="lightgreen",zorder=2)
        eb1[-1][0].set_linestyle("-")
        eb1[-1][0].set_linewidth(3)
        eb2[-1][0].set_linestyle(":")
        eb2[-1][0].set_linewidth(3)
        eb3[-1][0].set_linestyle("--")
        eb3[-1][0].set_linewidth(3)
        ax1.axhline(y=0,color="k",ls="-.",lw=2)
        ax1.legend(loc="lower left",fontsize=18,title="RMSE")
        file_end=".pdf"
        fig_name=self.grid_name+"_Inst_Sonde_Error_Overview"
        if not self.scalar_based_div:
            fig_name+="vectorised"
        fig_name=fig_name+file_end
        if not save_as_manuscript_figure:
            plot_path=self.plot_path
        else:
            plot_path=self.plot_path+\
                "/../../../../Synthetic_AR_paper/Manuscript/Paper_Plots/"
        fig_name="fig15_"+fig_name
        error_fig.savefig(plot_path+fig_name,
                       dpi=200,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)

    # Supplementary Plots
    def plot_single_flight_and_sector_regression_divergence(self,sector,
                                                            number_of_sondes,
                                                            div_qv,
                                                            div_scalar_mass,
                                                            adv_q_calc,
                                                            adv_q_scalar):
        fig=plt.figure(figsize=(9,12))
        ax1=fig.add_subplot(111)
        ax1.plot(div_qv.values,div_qv.index,label="div: transp")
        ax1.axvline(x=0,ls="--",color="grey",lw=2)
        #plt.plot(div_mass.values,div_mass.index,label="div: mass")
        ax1.plot(div_scalar_mass.values,div_scalar_mass.index,
                 label="div: scalar mass")
        #plt.plot(adv_q.values,adv_q.index,label="adv: q")
        ax1.plot(adv_q_calc,adv_q_calc.index,
                 label="adv_calc:q",c="darkgreen")
        ax1.plot(adv_q_scalar,adv_q_scalar.index,
                 label="adv_scalar:q",c="green",ls="--")
        ax1.invert_yaxis()
        ax1.set_xlim([-2e-4,1e-4])
        ax1.set_xticks([-2e-4,0,2e-4])
        ax1.set_ylim([1000,300])
        ax1.legend()
                    
        budget_plot_file_name=self.flight+"_"+self.grid_name+"_AR_"+sector+\
                    "_regr_sonde_no_"+\
                        str(number_of_sondes)+".png"
            
        if self.plot_path.endswith("budget/"):
            plot_path=self.plot_path+"/supplementary/"
        else:
            pass
        fig.savefig(plot_path+budget_plot_file_name,
                        dpi=300,bbox_inches="tight")
        plt.close()
        
        print("Figure saved as:",plot_path+budget_plot_file_name)
    def plot_comparison_sector_leg_wind_q_transport(self,
                        mean_trpz_wind,mean_trpz_q,mean_trpz_moist_transport,
                        mean_core_trpz_q,mean_core_trpz_wind,
                        mean_core_trpz_moist_transport,
                        pressure,sector):
    
        profile=plt.figure(figsize=(12,9))
        ax1=profile.add_subplot(111)
        ax1.plot(mean_trpz_wind,pressure,label="wind, domain",color="purple")
        ax1.plot(mean_trpz_q*1000,pressure,label="q,domain",color="blue")
        ax1.plot(1/9.82*mean_trpz_moist_transport*1000,pressure,
             label="transport,domain",color="k")
        ax1.plot(mean_core_trpz_wind,pressure,label="wind,"+sector,
             ls="--",color="purple")
        ax1.plot(mean_core_trpz_q*1000,pressure,label="q,"+sector,
                 ls="--",color="blue")
        ax1.plot(1/9.82*mean_core_trpz_moist_transport*1000,
             pressure,label="transport,"+sector,ls="--",color="k")
        
        ax1.set_xlim([0,40])
        ax1.set_yscale("log")
        ax1.invert_yaxis()
        ax1.set_ylabel("Pressure in hPa")
        ax1.set_yticks([300,500,700,850,1000])
        ax1.set_yticklabels(["300","500","700","850","1000"])
        ax1.legend(loc="best",fontsize=12)
        fig_name=self.flight+"_"+self.grid_name+"_"+sector+"_Mean_Moisture_Transport.png"
        if not self.plot_path.endswith("budget/"):
            plot_path=self.plot_path+"/budget/supplementary/"
        else:
            plot_path=self.plot_path+"/supplementary/"
        
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        
        sns.despine(offset=10)
        profile.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)
    
    def plot_AR_TIVT_cumsum_quicklook(self,ar_inflow,ar_outflow):
        fig=plt.figure(figsize=(10,10))
        ax1=fig.add_subplot(111)
        ax1.plot(range(ar_inflow.shape[0]),
                     (ar_inflow["IVT_max_distance"].diff()*\
                      ar_inflow["Interp_IVT"].values).cumsum(),
                         label="inflow")
        ax1.plot(range(ar_outflow.shape[0]),
                     (ar_outflow["IVT_max_distance"].diff()*\
                      ar_outflow["Interp_IVT"].values).cumsum(),
                         label="outflow")
        ax1.set_ylabel("TIVT in kg/s")
        ax1.set_xlabel("Cumsum distance as timesteps")
        ax1.legend()
        fig_name=self.flight+"_"+self.grid_name+"_TIVT_inflow_outflow_cumsum.png"
        plot_path=self.plot_path+"/supplementary/"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig.savefig(plot_path+fig_name,
                dpi=200,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)
    
    def plot_moisture_budget_divergence_components(self,q_sector_inflow,
        q_sector_outflow,wind_sector_inflow,wind_sector_outflow,
        moist_transport_sector_inflow,moist_transport_sector_outflow,
        mean_sector_trpz_q,mean_sector_trpz_wind,
        mean_sector_trpz_moist_transport,
        pressure,mean_distance,sector):
    
        budget_profile_df=pd.DataFrame(data=np.nan,index=pressure.astype(int),
                                   columns=["ADV","CONV","moist_transp"])
        
        profile=plt.figure(figsize=(12,9))
        #######################################################################
        #Moisture advection
        q_diff=(q_sector_outflow.mean()-q_sector_inflow.mean())*1000
        
        budget_profile_df["ADV"]=np.array(mean_sector_trpz_wind*q_diff/\
                                      (mean_distance*1000))
        
        ax1=profile.add_subplot(131)
        ax1.plot(mean_sector_trpz_wind*q_diff/(mean_distance*1000),
         pressure,label="q: out-in",color="darkblue")
        ax1.axvline(0,ls="--",lw=2,color="k")
        ax1.fill_betweenx(y=q_diff.index.astype(float),
                      x1=mean_sector_trpz_wind*q_diff/(mean_distance*1000),
                      x2=0, where=(q_diff > 0),
                  color="saddlebrown",alpha=0.3)
        ax1.fill_betweenx(y=q_diff.index.astype(float),
                      x1=mean_sector_trpz_wind*q_diff/(mean_distance*1000),
                      where=(q_diff<0),x2=0,color="teal",alpha=0.3)
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(3)

        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.yaxis.set_tick_params(width=2,length=6)
        ax1.legend(loc="upper left",fontsize=14)
        ax1.set_title("ADV")
        ax1.set_xlim([-2e-4,1e-4])
        ax1.set_xticks([-2e-4,0,1e-4])
        ax1.set_xticklabels(["-2e-4","0","1e-4"])
        ax1.xaxis.set_tick_params(width=3,length=6)
    
        sns.despine(ax=ax1,offset=10)
        #######################################################################
        wind_diff=(wind_sector_outflow.mean()-wind_sector_inflow.mean())
        budget_profile_df["CONV"]=np.array(wind_diff*mean_sector_trpz_q*1000/\
                                       (mean_distance*1000))
    
    
        ax2=profile.add_subplot(132)
        ax2.axvline(0,ls="--",lw=2,color="k")
        ax2.plot(wind_diff*mean_sector_trpz_q*1000/(mean_distance*1000),
             pressure,label="wind: out-in",color="purple")

        ax2.fill_betweenx(y=wind_diff.index.astype(float),
                  x1=wind_diff*mean_sector_trpz_q*1000/(mean_distance*1000),
                  where=(wind_diff*mean_sector_trpz_q>0),
                  x2=0,color="saddlebrown",alpha=0.3)
        ax2.fill_betweenx(y=wind_diff.index.astype(float),
                  x1=wind_diff*mean_sector_trpz_q*1000/(mean_distance*1000),
                  where=(wind_diff*mean_sector_trpz_q<0),
                  x2=0,color="teal",alpha=0.3)
        for axis in ['top','bottom','left','right']:
            ax2.spines[axis].set_linewidth(3)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.yaxis.set_tick_params(width=2,length=6)
        ax2.set_title("CONV")
        ax2.set_xlim([-2e-4,1e-4])
        ax2.set_xticks([-2e-4,0,1e-4])
        ax2.set_xticklabels(["-2e-4","0","1e-4"])
        ax2.xaxis.set_tick_params(width=3,length=6)
    
        
        ax2.legend(loc="upper center",fontsize=14)
        sns.despine(ax=ax2,offset=10)
        ###########################################################################
        transport_diff=moist_transport_sector_outflow.mean()-\
                    moist_transport_sector_inflow.mean()
        budget_profile_df["moist_transp"]=np.array(transport_diff*1000/\
                                               (mean_distance*1000))
        ax3=profile.add_subplot(133)
        ax3.plot(transport_diff/mean_distance,pressure,
                 label="transport: out-in",color="k")
        ax3.axvline(0,ls="--",lw=2,color="k")
        ax3.fill_betweenx(y=transport_diff.index.astype(float),
                      x1=transport_diff*1000/(mean_distance*1000),
                      where=(transport_diff*1000<0),x2=0,color="teal",alpha=0.5)
        ax3.fill_betweenx(y=transport_diff.index.astype(float),
                      x1=transport_diff*1000/(mean_distance*1000),
                      where=(transport_diff*1000>0),x2=0,
                      color="saddlebrown",alpha=0.5)
        ax3.spines['left'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax3.spines[axis].set_linewidth(3)
        ax3.yaxis.set_tick_params(width=3,length=6)
        ax3.xaxis.set_tick_params(width=3,length=6)
    
        ax3.legend(loc="upper right",fontsize=14)
        ax3.set_title("moist.\ntransport")
        ax3.set_xlim([-2e-4,1e-4])
        ax3.set_xticks([-2e-4,0,1e-4])
        ax3.set_xticklabels(["-2e-4","0","1e-4"])

        """
        #ax1.plot(.mean()*1000,pressure,label="q-outflow")
        #ax1.plot(mean_trpz_moist_transport*1000,pressure,label="transport")
        """

        ax1.set_ylim([300,1000])
        ax2.set_ylim([300,1000])
        ax3.set_ylim([300,1000])
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax3.invert_yaxis()
        ax1.set_ylabel("Pressure in hPa")
        ax1.set_yticks([300,500,700,850,1000])
        ax1.set_yticklabels(["300","500","700","850","1000"])
        ax2.set_yticks([300,500,700,850,1000])
        ax3.set_yticks([300,500,700,850,1000])
        
        ax2.set_yticklabels([])
        
        ax3.yaxis.tick_right()

        for loc, spine in ax3.spines.items():
            if loc in ["right","bottom"]:
                spine.set_position(('outward', 10)) 
        plt.subplots_adjust(wspace=0.4)
        fig_name=self.flight+"_"+self.grid_name+"_"+\
            sector+"_Moisture_transport_Divergence.png"
        if not self.plot_path.endswith("budget/"):
            plot_path=self.plot_path+"/budget/"
        else:
            plot_path=self.plot_path
        plot_path=plot_path+"/supplementary/"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        profile.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)
        return budget_profile_df
    def compare_inst_sonde_pos(self,flight_dates,Campaign_Budgets={},
                               Campaign_Inst_Budgets={},
                               save_as_manuscript_figure=False,
                               use_flight_tracks=False):
        import gridonhalo
        major_data_path="C:/Users/u300737/Desktop/PhD_UHH_WIMI/Work/GIT_Repository/"
        self.flight_dates=flight_dates
        # Allocate variables and calc budget contributions in mm/h
        sonde_pos_fig,ax=plt.subplots(3,3,figsize=(18,12))
        axes=ax.flatten()
        p=0
        sector_colors={"warm_sector":"orange","core":"green","cold_sector":"blue"}
        inst_sector_colors={"warm_sector":"yellow","cold_sector":"lightblue"}
        sector_distances=pd.DataFrame(data=np.nan,
                                      index=range(9),
                                      columns=["warm_sector","cold_sector"])
                
        for campaign in [*self.flight_dates.keys()]:
            if campaign=="North_Atlantic_Run":    
                campaign_data_path=major_data_path+"NA_February_Run/data/"
            elif campaign=="Second_Synthetic_Study":
                campaign_data_path=major_data_path+"Second_Synthetic_Study/data/"
            sonde_pos_path=campaign_data_path+"budget/"
            for flight in [*self.flight_dates[campaign].keys()]:
                print(flight)
                inst_flight=flight+"_instantan"
                # load sonde positions
                for s,sector in enumerate(["warm_sector","cold_sector"]):
                    if flight=="SRF12":
                        if sector=="cold_sector":
                            continue
                    file_end=".csv"
                    pos_fname_arg=""
                    if use_flight_tracks:
                        pos_fname_arg="_on_flight"
                    sonde_sector_fname=flight+"_Sonde_Location_"+sector+"_"+\
                        self.grid_name+"_regr_sonde_no_3"+file_end                    
                    inst_sonde_sector_fname=inst_flight+"_Sonde_Location_"+\
                                            sector+"_"+self.grid_name+\
                                                "_regr_sonde_no_3"+pos_fname_arg+\
                                                    file_end
                    flight_sector_sondes=pd.read_csv(sonde_pos_path+\
                                                     sonde_sector_fname)
                    inst_sector_sondes=pd.read_csv(sonde_pos_path+\
                                                   inst_sonde_sector_fname)
                    axes[p].scatter(flight_sector_sondes["Halo_Lon"],
                                    flight_sector_sondes["Halo_Lat"],s=100,
                                    color=sector_colors[sector],
                                    edgecolor="k")#,edgewidth=2)
                    axes[p].scatter(inst_sector_sondes["Halo_Lon"],
                                    inst_sector_sondes["Halo_Lat"],
                                    color=sector_colors[sector],
                                    s=100,marker="v",edgecolor="grey")
                    axes[p].text(x=0.05,y=0.9,
                                 s=self.flight_dates[campaign][flight],
                                 transform=axes[p].transAxes)
                    distances=gridonhalo.vectorized_harvesine_distance(
                                            flight_sector_sondes["Halo_Lat"],
                                            flight_sector_sondes["Halo_Lon"],
                                            inst_sector_sondes["Halo_Lat"],
                                            inst_sector_sondes["Halo_Lon"])
                    sector_distances[sector].iloc[p]=distances.mean()
                sector_distances.rename(index={p:self.flight_dates[campaign]\
                                                    [flight]},inplace=True)
                p+=1
        sector_distances.loc["mean"]=sector_distances.mean()
        # save the data in the supplements folder of the manuscript
        supplement_path=self.cmpgn_cls.plot_path+\
                "/../../../Synthetic_AR_Paper/Manuscript/Supplements/"
        
        if not os.path.exists(supplement_path):
            os.makedirs(supplement_path)
        distances_fname="Sector_Inst_Sondes_distances"
        if use_flight_tracks:
            distances_fname+="_on_flight"
        distances_fname+=file_end
        sector_distances.to_csv(path_or_buf=supplement_path+\
                                    distances_fname,index=True)
        print("Sector differences saved as:",supplement_path+distances_fname)
    def mean_errors_per_flight(self,flight_dates,
                               flight_Ideal_Budgets,
                               Inst_Ideal_Budgets,
                               save_as_manuscript_figure=False):
        # Allocate variables and calc budget contributions in mm/h
        
        budget_ideal_regions      = flight_Ideal_Budgets#self.budget_ideal_regions
        budget_inst_ideal_regions = Inst_Ideal_Budgets#self.budget_inst_ideal_regions
            
        #Start plotting
        #budget_boxplot=plt.figure(figsize=(12,9), dpi= 300)
        matplotlib.rcParams.update({'font.size': 12})
        err_fig,ax=plt.subplots(figsize=(18,12),nrows=3,ncols=3)
        axes=ax.flatten()
        p=0
        
        div_errors=pd.DataFrame(data=np.nan,index=["Warm\nADV","Warm\nCONV",
                                "Core\nADV","Core\nCONV","Cold\nADV","Cold\nCONV"],
                                columns=["flight","instantaneous"])
        div_errors_dict={}
        relevant_long_index = budget_ideal_regions["core"]["ADV"].index
        relevant_index      = [idx.split("_")[0] for idx in relevant_long_index]
        mean_absolute_relative_error=pd.DataFrame(data=np.nan,
                            index=relevant_index,
                            columns=div_errors.index)

        for campaign in [*flight_dates.keys()]:
            for flight in [*flight_dates[campaign].keys()]:
                if campaign.startswith("N"):
                    index_start="NA"
                else:
                    index_start="Snd"
                index_end=flight
                index_inst_end=flight+"_instantan"
                index=index_start+index_end
                inst_index=index_start+index_inst_end
                for term in div_errors.index:
                    front_sector=term.split("\n")[0].lower()#
                    if not front_sector=="core":
                        front_sector+="_sector"
                    term_comp=term.split("\n")[-1]
                    div_errors["flight"].loc[term]=-24*budget_ideal_regions[\
                                                front_sector][term_comp][\
                                                index+"_sonde_100"+term_comp]
                    div_errors["instantaneous"].loc[term]=-24*budget_inst_ideal_regions[\
                                                        front_sector][term_comp][\
                                            inst_index+"_sonde_100"+term_comp]
                    mean_absolute_relative_error[term].loc[index_start+index_end]=\
                        abs(abs(div_errors["flight"].loc[term]-\
                             div_errors["instantaneous"].loc[term])/\
                                div_errors["instantaneous"].loc[term])
                
                div_errors_dict[flight_dates[campaign][flight]]=div_errors.copy()
                
        div_errors_dict=dict(sorted(div_errors_dict.items()))
        plot_index=np.arange(0,len(div_errors_dict["20150314"].index))
        for date in [*div_errors_dict.keys()]:
            axes[p].scatter(plot_index,
                            div_errors_dict[date]["flight"].values,
                            marker="s",s=100,color="k")
            axes[p].scatter(plot_index,
                            div_errors_dict[date]["instantaneous"].values,
                            marker="v",s=100,color="grey")
            axes[p].text(x=0.1,y=0.9,s=date,transform=axes[p].transAxes)
            axes[p].set_xticks(plot_index)
            
            axes[p].set_xticklabels("")
            axes[p].set_ylim([-36,36])
            axes2=axes[p].twinx()
            axes2.scatter(plot_index,div_errors_dict[date]["flight"].values-\
                          div_errors_dict[date]["instantaneous"].values,
                          color="darkred", ls="--",marker="o",s=75)
            axes2.set_ylim([-12,12])
            
            axes[p].set_yticks([-36,-24,-12,0,12,24,36])
            axes2.set_yticks([-12,-8,-4,0,4,8,12])
            axes[p].axhline(0,color="grey",ls="--",lw=2)
            axes[p].set_yticklabels("")
            axes2.set_yticklabels("")
            if p%3==0:
                axes[p].set_ylabel("Contribution to \nMoisture Budget ($\mathrm{mmd}^{-1}$)")
                axes[p].set_yticklabels(["-36","-24","-12","0","12","24","36"])
                
            elif p%3==2:
                axes2.set_yticklabels(["-12","-8","-4","0","4","8","12"])
                axes2.set_ylabel("Error in \n Moisture Budget ($\mathrm{mmd}^{-1}$)")
            if p>=6:
                axes[p].set_xticklabels(div_errors_dict[date].index)
            
            axes[p].set_xlim([-0.5,5.5])
            
            p+=1
        sns.despine(offset=10)             
        supplement_path=self.cmpgn_cls.plot_path+\
                "/../../../Synthetic_AR_Paper/Manuscript/Supplements/"
        fig_name="Mean_div_errors_per_flight.png"
        err_fig.savefig(supplement_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", supplement_path+fig_name)
    def sonde_divergence_error_bar(self,save_as_manuscript_figure=False):
            
        budget_inst_ideal_regions=24*self.budget_inst_ideal_regions
        budget_ideal_regions=24*self.budget_ideal_regions
        budget_regions=24*self.budget_regions
        
        sector_divergence_inst_errors=budget_regions-budget_inst_ideal_regions
        sector_divergence_errors=budget_regions-budget_ideal_regions
        mean_sector_divergence_inst_errors=sector_divergence_inst_errors.mean()
        mean_sector_divergence_errors=sector_divergence_errors.mean()
        std_sector_divergence_inst_errors=sector_divergence_inst_errors.std()
        std_sector_divergence_errors=sector_divergence_errors.std()
        
        abs_std_sector_divergence_inst_errors=abs(sector_divergence_inst_errors).std()
        abs_std_sector_divergence_errors=abs(sector_divergence_errors).std()
        
        abs_mean_sector_divergence_inst_error=\
            abs(sector_divergence_inst_errors).mean()
        abs_mean_sector_divergence_errors=\
            abs(sector_divergence_errors).mean()
        
        ##### Mean Error between sondes and 
        error_fig=plt.figure(figsize=(12,9))
        ax1=error_fig.add_subplot(111)
        ax1.bar(np.arange(abs_mean_sector_divergence_errors.shape[0]),
                abs_mean_sector_divergence_errors,color="lightgrey",alpha=0.6)
        ax1.bar(np.arange(abs_mean_sector_divergence_errors.shape[0]),
                abs_mean_sector_divergence_inst_error-\
                    abs_mean_sector_divergence_errors,
                bottom=abs_mean_sector_divergence_errors,
                color="darkgray",alpha=0.6)
        ax1.bar(np.arange(abs_mean_sector_divergence_errors.shape[0]),
                abs_mean_sector_divergence_inst_error,
                fill=False, edgecolor="k",lw=2,alpha=0.6)
        
        # Frequency error
        eb1=ax1.errorbar(np.arange(abs_mean_sector_divergence_errors.shape[0])-0.1,
                     mean_sector_divergence_errors,fmt="v",markersize=20,
                     markeredgecolor="k",ecolor="k",
                     yerr=std_sector_divergence_errors,
                     color="lightgrey",zorder=2,label="Pure Sonde Err")
        eb1[-1][0].set_linestyle("--")
        
        # Instationarity error
        eb2=ax1.errorbar(np.arange(abs_mean_sector_divergence_errors.shape[0])+0.1,
                     mean_sector_divergence_inst_errors,fmt="d",markersize=25,
                     markeredgecolor="k",ecolor="k",
                     yerr=std_sector_divergence_inst_errors,
                     color="darkgrey",zorder=2,label="Sonde Instationarity Err")
        eb2[-1][0].set_linestyle("-")
        eb2[-1][0].set_linewidth(2)
        ax1.axhline(y=0,color="brown",ls="--",lw=2)
        ax1.set_xticks(np.arange(mean_sector_divergence_errors.shape[0]))
        ax1.set_xticklabels(mean_sector_divergence_errors.index,fontsize=10)
        ax1.set_ylabel("IVT Convergence Error ($\mathrm{mmd}^{-1}$)")
        # Axis linewidth
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(3.0)
        ax1.xaxis.set_tick_params(width=2,length=10)
        ax1.yaxis.set_tick_params(width=2,length=10)
        ax1.legend(loc="lower left",fontsize=20)
        sns.despine(offset=10)
        fig_name="Error_Bars_Inst_Real_and_Sondes_DIV.png"
        if not save_as_manuscript_figure:
            plot_path=self.plot_path
        else:
            plot_path=self.plot_path+\
                "/../../../../Synthetic_AR_paper/Manuscript/Paper_Plots/"
        fig_name="old_Fig15_"+fig_name
        error_fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:", plot_path+fig_name)
        return None
    
    def plot_flight_specific_budget_components(self):
        budget_inst_ideal_regions=24*self.budget_inst_ideal_regions
        budget_ideal_regions=24*self.budget_ideal_regions
        budget_regions=24*self.budget_regions
        
        sector_divergence_inst_errors=\
                            budget_ideal_regions-budget_inst_ideal_regions
        
        sector_divergence_inst_sonde_errors=\
                            budget_regions-budget_inst_ideal_regions
        sector_divergence_sonde_errors=\
                            budget_regions-budget_ideal_regions
                            
        # Continuous to instantan errors
        sector_squared_divergence_inst_errors = \
            (sector_divergence_inst_errors)**2
        # Entire errors
        sector_squared_divergence_inst_sonde_errors      = \
            (sector_divergence_inst_sonde_errors)**2
        # Pure sonde errors
        sector_squared_divergence_sonde_errors= \
            (sector_divergence_sonde_errors)**2
        
        # Add flight specific values as table
        #relative Error
        rel_sec_div_inst_errors=sector_divergence_inst_errors/\
                                    budget_inst_ideal_regions
        rel_sec_div_inst_sonde_errors=sector_divergence_inst_sonde_errors/\
                                        budget_inst_ideal_regions
        rel_sec_div_sonde_errors=sector_divergence_sonde_errors/\
                                    budget_ideal_regions
        
        rel_sec_div_inst_errors.index+=1
        rel_sec_div_inst_sonde_errors.index+=1
        rel_sec_div_sonde_errors.index+=1
        error_case_fig=plt.figure(figsize=(16,22))
        ax1=error_case_fig.add_subplot(321)
        ax2=error_case_fig.add_subplot(322)
        ax3=error_case_fig.add_subplot(323)
        ax4=error_case_fig.add_subplot(324)
        ax5=error_case_fig.add_subplot(325)
        ax6=error_case_fig.add_subplot(326)

        color_palette=["darkorange","orange","lightgreen",
                   "green","lightblue","blue"]                    

        # Warm ADV
        ax1.plot(sector_divergence_inst_errors["Warm\nADV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax1.plot(sector_divergence_sonde_errors["Warm\nADV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax1.plot(sector_divergence_inst_sonde_errors["Warm\nADV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax1.set_yticks(np.arange(0,9)+1)
        ax1.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax1.text(-5,9,"Warm ADV", color=color_palette[0],
                 bbox={"facecolor":"lightgrey","edgecolor":"k",
                       "linewidth":2},fontsize=16)
        ax1.set_ylim([0,10])
        # Warm CONV
        ax2.plot(sector_divergence_inst_errors["Warm\nCONV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax2.plot(sector_divergence_sonde_errors["Warm\nCONV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax2.plot(sector_divergence_inst_sonde_errors["Warm\nCONV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax2.set_yticks(np.arange(0,9)+1)
        ax2.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax2.text(-5,9,"Warm CONV", color=color_palette[1],
                 bbox={"facecolor":"lightgrey","edgecolor":"k",
                       "linewidth":2},fontsize=14)
        ax2.set_ylim([0,10])
        # Core ADV
        ax3.plot(sector_divergence_inst_errors["Core\nADV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax3.plot(sector_divergence_sonde_errors["Core\nADV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax3.plot(sector_divergence_inst_sonde_errors["Core\nADV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax3.set_yticks(np.arange(0,9)+1)
        ax3.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax3.text(-3,9,"Core ADV", color=color_palette[2],
                 bbox={"facecolor":"darkgrey","edgecolor":"k",
                       "linewidth":2},fontsize=14)
        ax3.set_ylim([0,10])
        # Core CONV
        ax4.plot(sector_divergence_inst_errors["Core\nCONV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax4.plot(sector_divergence_sonde_errors["Core\nCONV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax4.plot(sector_divergence_inst_sonde_errors["Core\nCONV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax4.set_yticks(np.arange(0,9)+1)
        ax4.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax4.text(-3,9,"Core CONV", color=color_palette[3],
                 bbox={"facecolor":"lightgrey","edgecolor":"k",
                       "linewidth":2},fontsize=14)
        ax4.set_ylim([0,10])
        # Cold ADV
        ax5.plot(sector_divergence_inst_errors["Cold\nADV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax5.plot(sector_divergence_sonde_errors["Cold\nADV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax5.plot(sector_divergence_inst_sonde_errors["Cold\nADV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax5.set_yticks(np.arange(0,9)+1)
        ax5.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax5.text(-3,9,"Cold ADV", color=color_palette[4],
                 bbox={"facecolor":"lightgrey","edgecolor":"k",
                       "linewidth":2},fontsize=14)
        ax5.set_ylim([0,10])
        # Cold CONV
        ax6.plot(sector_divergence_inst_errors["Cold\nCONV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax6.plot(sector_divergence_sonde_errors["Cold\nCONV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax6.plot(sector_divergence_inst_sonde_errors["Cold\nCONV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax6.set_yticks(np.arange(0,9)+1)
        ax6.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax6.text(-3,9,"Core CONV", color=color_palette[5],
                 bbox={"facecolor":"lightgrey","edgecolor":"k",
                       "linewidth":2},fontsize=14)
        ax6.set_ylim([0,10])
        plt.suptitle("Absolute Errors in mm/d for\n(flight-inst)",y=0.92)
        ax1.set_xlim([-6,6])
        ax2.set_xlim([-6,6])
        ax3.set_xlim([-6,6])
        ax4.set_xlim([-6,6])
        ax5.set_xlim([-6,6])
        ax6.set_xlim([-6,6])
        ax1.axvline(x=0,ls="--",color="grey",lw=2)
        ax2.axvline(x=0,ls="--",color="grey",lw=2)
        ax3.axvline(x=0,ls="--",color="grey",lw=2)
        ax4.axvline(x=0,ls="--",color="grey",lw=2)
        ax5.axvline(x=0,ls="--",color="grey",lw=2)
        ax6.axvline(x=0,ls="--",color="grey",lw=2)
        plot_path=self.plot_path+\
                "/../../../../Synthetic_AR_paper/Manuscript/Supplements/"
        fig_name=plot_path+"Single case errors.png"
        error_case_fig.savefig(fig_name, dpi=300,bbox_inches="tight")
        print("Supplementary fig saved as:",fig_name)
        # Relative errors
        rel_error_case_fig=plt.figure(figsize=(16,22))
        ax1=rel_error_case_fig.add_subplot(321)
        ax2=rel_error_case_fig.add_subplot(322)
        ax3=rel_error_case_fig.add_subplot(323)
        ax4=rel_error_case_fig.add_subplot(324)
        ax5=rel_error_case_fig.add_subplot(325)
        ax6=rel_error_case_fig.add_subplot(326)

        color_palette=["darkorange","orange","lightgreen",
                   "green","lightblue","blue"]                    

        # Warm ADV
        ax1.plot(rel_sec_div_inst_errors["Warm\nADV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax1.plot(rel_sec_div_sonde_errors["Warm\nADV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax1.plot(rel_sec_div_inst_sonde_errors["Warm\nADV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax1.set_yticks(np.arange(0,9)+1)
        ax1.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax1.text(-5,9,"Warm ADV", color=color_palette[0],
                 bbox={"facecolor":"lightgrey","edgecolor":"k",
                       "linewidth":2},fontsize=16)
        ax1.set_ylim([0,10])
        # Warm CONV
        
        ax2.plot(rel_sec_div_inst_errors["Warm\nCONV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax2.plot(rel_sec_div_sonde_errors["Warm\nCONV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax2.plot(rel_sec_div_inst_sonde_errors["Warm\nCONV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax2.set_yticks(np.arange(0,9)+1)
        ax2.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax2.text(-5,9,"Warm CONV", color=color_palette[1],
                 bbox={"facecolor":"lightgrey","edgecolor":"k",
                       "linewidth":2},fontsize=14)
        ax2.set_ylim([0,10])
        # Core ADV
        ax3.plot(rel_sec_div_inst_errors["Core\nADV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax3.plot(rel_sec_div_sonde_errors["Core\nADV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax3.plot(rel_sec_div_inst_sonde_errors["Core\nADV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax3.set_yticks(np.arange(0,9)+1)
        ax3.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax3.text(-3,9,"Core ADV", color=color_palette[2],
                 bbox={"facecolor":"lightgrey","edgecolor":"k",
                       "linewidth":2},fontsize=14)
        ax3.set_ylim([0,10])
        # Core CONV
        ax4.plot(rel_sec_div_inst_errors["Core\nCONV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax4.plot(rel_sec_div_sonde_errors["Core\nCONV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax4.plot(rel_sec_div_inst_sonde_errors["Core\nCONV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax4.set_yticks(np.arange(0,9)+1)
        ax4.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax4.text(-3,9,"Core ADV", color=color_palette[3],
                 bbox={"facecolor":"lightgrey","edgecolor":"k",
                       "linewidth":2},fontsize=14)
        ax4.set_ylim([0,10])
        # Cold ADV
        ax5.plot(rel_sec_div_inst_errors["Core\nADV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax5.plot(rel_sec_div_sonde_errors["Core\nADV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax5.plot(rel_sec_div_inst_sonde_errors["Core\nADV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax5.set_yticks(np.arange(0,9)+1)
        ax5.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax5.text(-3,9,"Core ADV", color=color_palette[4],
                 bbox={"facecolor":"lightgrey","edgecolor":"k",
                       "linewidth":2},fontsize=14)
        ax5.set_ylim([0,10])
        # Cold CONV
        ax6.plot(rel_sec_div_inst_errors["Core\nCONV"],
                 rel_sec_div_inst_errors.index,
                 color="black",lw=2,ls="-")
        ax6.plot(rel_sec_div_sonde_errors["Core\nCONV"],
                 rel_sec_div_sonde_errors.index,
                 color="green",lw=2,ls="--")
        ax6.plot(rel_sec_div_inst_sonde_errors["Core\nCONV"],
                 rel_sec_div_inst_sonde_errors.index,
                 color="darkorange",lw=2,ls=":")
        ax6.set_yticks(np.arange(0,9)+1)
        ax6.set_yticklabels(["AR"+str(i+1) for i in range(9)])
        ax6.text(-3,9,"Core CONV", color=color_palette[5],
                 bbox={"facecolor":"lightgrey","edgecolor":"k",
                       "linewidth":2},fontsize=14)
        ax6.set_ylim([0,10])
        plt.suptitle("Absolute Errors in mm/d",y=0.92)
        ax1.set_xlim([-6,6])
        ax2.set_xlim([-6,6])
        ax3.set_xlim([-6,6])
        ax4.set_xlim([-6,6])
        ax5.set_xlim([-6,6])
        ax6.set_xlim([-6,6])
        ax1.axvline(x=0,ls="--",color="grey",lw=2)
        ax2.axvline(x=0,ls="--",color="grey",lw=2)
        ax3.axvline(x=0,ls="--",color="grey",lw=2)
        ax4.axvline(x=0,ls="--",color="grey",lw=2)
        ax5.axvline(x=0,ls="--",color="grey",lw=2)
        ax6.axvline(x=0,ls="--",color="grey",lw=2)
    
    ###########################################################################