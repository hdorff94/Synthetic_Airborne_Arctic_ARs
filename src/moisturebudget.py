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
    
import matplotlib
import matplotlib.pyplot as plt
#from matplotlib import ticker as tick_right

import seaborn as sns
from ivtvariability import IVT_Variability_Plotter

#import Grid_on_HALO
if not "flightcampaign" in sys.modules:
    import flightcampaign as Flight_Campaign
class Moisture_Budgets():
    def __init__(self):
        pass
class Moisture_Convergence(Moisture_Budgets):
    
    def __init__(self,cmpgn_cls,flight,config_file,
                 flight_dates={},sonde_no=3,
                 grid_name="ERA5",do_instantan=False):
        
        self.cmpgn_cls=cmpgn_cls
        self.grid_name=grid_name
        self.do_instantan=do_instantan
        self.flight=flight
        self.config_file=config_file
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
        self.sector_colors={"warm_sector":"orange",
                            "core":"darkgreen",
                            "cold_sector":"darkblue"}            
    
    #%% Budget functions
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
        for sector in sectors:
            #if not self.do_instantan:
            budget_file=self.flight+"_AR_"+sector+"_"+self.grid_name+\
                            "_regr_sonde_no_"+str(self.sonde_no)+".csv"
            if sector=="core":
                print("Read budget file",budget_file)
            budget_ideal_file=self.flight+"_AR_"+sector+"_"+self.grid_name+\
                                "_regr_sonde_no_100"+".csv"
            #else:
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
    def get_overall_budgets(self):
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
                if self.do_instantan:
                    flight=flight+"_instantan"
                
                #Core
                core_file=flight+"_AR_core_"+self.grid_name+\
                "_regr_sonde_no_"+str(self.sonde_no)+".csv"
                core_ideal_file=flight+"_AR_core_"+self.grid_name+\
                "_regr_sonde_no_100.csv"
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
                    "_regr_sonde_no_"+str(self.sonde_no)+".csv"
                warm_ideal_file=flight+"_AR_warm_sector_"+self.grid_name+\
                    "_regr_sonde_no_100"+".csv"
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
                        "_regr_sonde_no_"+str(self.sonde_no)+".csv"
                    cold_ideal_file=flight+"_AR_cold_sector_"+self.grid_name+\
                    "_regr_sonde_no_100.csv"
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
                g=9.82
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
            use_icon=False,do_supplements=False):
        
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
            Hydrometeors,HALO_Dict,cmpgn_cls=campaignAR_plotter.main(
                                        campaign=campaign,flights=flights,
                                        era_is_desired=use_era, 
                                        icon_is_desired=use_icon,
                                        carra_is_desired=use_carra,
                                        do_daily_plots=do_plotting,
                                        calc_hmp=True,calc_hmc=False,
                                        do_instantaneous=self.do_instantan)

            HMCs,HALO_Dict,cmpgn_cls=campaignAR_plotter.main(
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
            
            # create mean values plots
            mean_profile_fig=plt.figure(figsize=(16,9))
            ax1_mean=mean_profile_fig.add_subplot(131)
            ax2_mean=mean_profile_fig.add_subplot(132)
            ax3_mean=mean_profile_fig.add_subplot(133)
        
            
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
                #working_path=os.getcwd()
                grid_name=HMCs[analysed_flight]\
                                        ["AR_internal"]["name"]
                
                AR_inflow,AR_outflow=atmospheric_rivers.Atmospheric_Rivers.\
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
                        #flight_dates=["2016"]
                        
                
                        #%%
                        # Sonde number
                        sondes_selection={}
                        sondes_selection["inflow_"+sector]=np.linspace(
                                0,AR_inflow["AR_inflow_"+sector].shape[0]-1,
                                num=number_of_sondes).astype(int)
                        sondes_selection["outflow_"+sector]=np.linspace(
                                0,AR_outflow["AR_outflow_"+sector].shape[0]-1,
                                num=number_of_sondes).astype(int)
                        #%% Loc and locate sondes for regression method
                        inflow_sondes_times=\
                                AR_inflow["AR_inflow_"+sector].index[\
                                    sondes_selection["inflow_"+sector]]
                        outflow_sondes_times=\
                                AR_outflow["AR_outflow_"+sector].index[\
                                        sondes_selection["outflow_"+sector]]
                
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
                #%%
                        if not "q" in HMCs[analysed_flight]["AR_internal"].keys():
                            HMCs[analysed_flight]["AR_internal"]["q"]=\
                                    HMCs[analysed_flight]["AR_internal"]\
                                        ["specific_humidity"].copy()
                        q_inflow_sondes=\
                                HMCs[analysed_flight]["AR_internal"]["q"].loc[\
                                                        inflow_sondes_times]
                        q_outflow_sondes=\
                            HMCs[analysed_flight]["AR_internal"]["q"].loc[\
                                                        outflow_sondes_times]
                
                        u_inflow_sondes=\
                            HMCs[analysed_flight]["AR_internal"]["u"].loc[\
                                                        inflow_sondes_times]
                        u_outflow_sondes=\
                            HMCs[analysed_flight]["AR_internal"]["u"].loc[\
                                                        outflow_sondes_times]
                
                        v_inflow_sondes=\
                            HMCs[analysed_flight]["AR_internal"]["v"].loc[\
                                                        inflow_sondes_times]
                        v_outflow_sondes=\
                            HMCs[analysed_flight]["AR_internal"]["v"].loc[\
                                                        outflow_sondes_times]
                
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
                        #%%
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
                
                        mean_u,dx_u,dy_u=self.run_regression(sondes_pos_all,
                                                     domain_values,"u")
                        mean_v,dx_v,dy_v=self.run_regression(sondes_pos_all,
                                                     domain_values,"v")
                
                        mean_qv,dx_qv,dy_qv=self.run_regression(sondes_pos_all,
                                                        domain_values,"transport")
                
                        mean_q,dx_q_calc,dy_q_calc=self.run_regression(sondes_pos_all,
                                                               domain_values,"q")
                
                        mean_scalar_wind,dx_scalar_wind,dy_scalar_wind=self.run_regression(
                                        sondes_pos_all,domain_values,"wind")
                
                        div_qv=(dx_qv+dy_qv)*1000
                        div_scalar_wind=(dx_scalar_wind+dy_scalar_wind)
                        #div_mass=div_wind*domain_values["q"].mean(axis=0).values*1000
                        div_scalar_mass=div_scalar_wind*\
                            domain_values["q"].mean(axis=0).values*1000
                        # Adv term based on divergence of q from run_regression
                        adv_q_calc=(dx_q_calc+dy_q_calc)*\
                            domain_values["wind"].mean(axis=0).values*1000
                        # Simply the difference of Moisture transport divergence and
                        # and the scalar based mass divergence
                        adv_q_scalar=div_qv-div_scalar_mass
                
                        if do_supplements:
                            Budget_plots.\
                            plot_single_flight_and_sector_regression_divergence(
                            sector,self.sonde_no,div_qv,div_scalar_mass,
                            adv_q_calc,adv_q_scalar)
                    
                            # Sector-based comparison of values
                            fig=plt.figure(figsize=(9,12))
                            ax1=fig.add_subplot(111)
                            ax1.plot(div_qv.values,div_qv.index,label="div: transp")
                            ax1.axvline(x=0,ls="--",color="grey",lw=2)
                            
                            ax1.plot(div_scalar_mass.values,
                                     div_scalar_mass.index,
                                     label="div: scalar mass")
                            
                            ax1.plot(adv_q_calc,adv_q_calc.index,
                                     label="adv_calc:q",c="darkgreen")
                            
                            ax1.plot(adv_q_scalar,adv_q_scalar.index,
                             label="adv_scalar:q",c="green",ls="--")
                    
                            ax1.invert_yaxis()
                            ax1.set_xlim([-2e-4,1e-4])
                            ax1.set_xticks([-2e-4,0,2e-4])
                            ax1.set_ylim([1000,300])
                            ax1.legend()
                            budget_plot_file_name=flight+"_"+grid_name+\
                                "_AR_"+sector+"_regr_sonde_no_"+\
                                    str(number_of_sondes)+".png"
                            fig.savefig(budget_plot_path+"/supplementary/"+
                                        budget_plot_file_name,
                                        dpi=300,bbox_inches="tight")
                            print("Figure saved as:",budget_plot_path+\
                                  "/supplements/"+budget_plot_file_name)
                        
                        if number_of_sondes<10:
                            ax1_mean.plot(domain_values["wind"].mean()*\
                              (q_outflow_sondes.mean()-q_inflow_sondes.mean()),
                              q_inflow_sondes.columns.astype(float),
                              color=Budget_plots.sector_colors[sector])
                            ax1_mean.text(-0.01,150,"ADV")
                            ax2_mean.plot(domain_values["q"].mean()*\
                              (wind_outflow_sondes.mean()-\
                               wind_inflow_sondes.mean()),
                              wind_outflow_sondes.columns.astype(float),
                              color=Budget_plots.sector_colors[sector])
                            ax2_mean.text(-0.01,150,"Mass Div")
                            ax3_mean.plot(moist_transport_outflow.mean()-\
                              moist_transport_inflow.mean(),
                              moist_transport_outflow.columns.astype(float))
                            ax3_mean.text(-0.01,150,s="Transp Div")
                        # Save sonde budget components as dataframe
                        budget_regression_profile_df=pd.DataFrame(data=np.nan,
                                        index=div_qv.index,
                                        columns=["CONV","ADV_calc","ADV_diff",
                                                 "TRANSP"])
                        budget_regression_profile_df["CONV"]=\
                            div_scalar_mass.values
                        budget_regression_profile_df["ADV_calc"]=\
                            adv_q_calc.values
                        budget_regression_profile_df["ADV_diff"]=\
                            adv_q_scalar.values
                        budget_regression_profile_df["TRANSP"]=\
                            div_qv.values
                
                    
                        budget_file_name=flight+"_AR_"+sector+"_"+\
                                    grid_name+"_regr_sonde_no_"+\
                                        str(number_of_sondes)+".csv"
                        budget_regression_profile_df.to_csv(
                            path_or_buf=budget_data_path+budget_file_name)    
                        print("Convergence components saved as: ",
                              budget_data_path+budget_file_name)
                if do_supplements:
                    ax1_mean.invert_yaxis()
                    ax2_mean.invert_yaxis()
                    ax3_mean.invert_yaxis() 
                    ax1_mean.set_xlim([-0.02,0.02])
                    ax2_mean.set_xlim([-0.02,0.02])
                    ax3_mean.set_xlim([-0.02,0.02])  
                    file_name=flight+"_simplified_divergence_sonde_no_"+\
                                str(number_of_sondes)+".png"
                    mean_profile_fig.savefig(budget_data_path+file_name)
                    print("Figure saved as:",budget_data_path+file_name)
            

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
    def run_haloac3_sondes_regression(geo_domain,domain_values,parameter):
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


class Moisture_Budget_Plots(Moisture_Convergence):
    #def __init__(self,Moisture_Convergence):
    def __init__(self,cmpgn_cls,flight,config_file,
                 grid_name="ERA5",do_instantan=False,sonde_no=3):
        
        super().__init__(cmpgn_cls,flight,config_file,grid_name,do_instantan)
        self.plot_path=self.cmpgn_cls.plot_path+"/budget/" # ----> to be filled
        self.grid_name=grid_name
        self.sonde_no=sonde_no
    # Quick functions of variables       
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
    
    def plot_comparison_sector_leg_wind_q_transport(self,
                        mean_trpz_wind,mean_trpz_q,mean_trpz_moist_transport,
                        mean_core_trpz_q,mean_core_trpz_wind,
                        mean_core_trpz_moist_transport,
                        pressure,sector):
    #if not ar_sector=="core":
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
        #ax1.legend()
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
        #print(mean_core_trpz_wind*q_diff/(mean_distance*1000))
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
    
        #sys.exit()
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
            #ax3.yaxis.set_label_position("right")
            #ax1.yaxis.set_ticks([])
        plt.subplots_adjust(wspace=0.4)
        #plot_path=cmpgn_cls.plot_path+"/budget/"
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
    ###############################################################################
    #%% Major functions
    def plot_single_case(self,Sectors,Ideal_Sectors,
                         save_as_manuscript_figure=False):
        """
    
        This function plots vertical profiles of moisture ADV and mass CONV and 
        moisture transport convergence.
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
        ax1.plot(core["ADV_calc"],
             core["ADV_calc"].index.astype(int),lw=2,label="core",color="darkgreen")
    
        ax1.plot(warm_sector["ADV_calc"],
             warm_sector["ADV_calc"].index.astype(int),lw=2,
             label="warm sector",color="darkorange")
    
        ax1.plot(cold_sector["ADV_calc"],
             cold_sector["ADV_calc"].index.astype(int),lw=2,
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
    
        #ax1.fill_betweenx(y=q_diff.index.astype(float),
        #                      x1=mean_core_trpz_wind*q_diff/(mean_distance*1000),
        #                      where=(q_diff<0),x2=0,color="teal",alpha=0.3)
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(3)
    
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.yaxis.set_tick_params(width=3,length=6)
        #ax1.xaxis.set_tick_params(width=2,length=6)
    
        ax1.legend(loc="upper left",fontsize=14)
        ax1.set_title("moisture \nADV",fontsize=20)
        ax1.set_xlim([-3e-4,3e-4])
        ax1.set_xticks([-3e-4,0,3e-4])
        ax1.set_xticklabels(["-3e-4","0","3e-4"])
        ax1.xaxis.set_tick_params(width=3,length=6)
        #ax1.yscale("log")    
        sns.despine(ax=ax1,offset=10)
    
        ax1.set_ylim([200,1000])
        #ax2.set_ylim([300,1000])
        #ax3.set_ylim([300,1000])
        ax1.invert_yaxis()
    
        #ax3.invert_yaxis()
    
        ax2=profile.add_subplot(132)
        ax2.axvline(0,ls="--",lw=2,color="k")
        ax2.plot(core["CONV"],core["CONV"].index.astype(int),
                 label="core",color="darkgreen")
        ax2.plot(warm_sector["CONV"],warm_sector["CONV"].index.astype(int),
             label="warm sector",color="orange")
        ax2.plot(cold_sector["CONV"],cold_sector["CONV"].index.astype(int),
             label="core",color="darkblue")
        ax2.set_ylim([200,1000])
        ####
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
        ax2.yaxis.set_tick_params(width=2,length=6)
        ax2.set_title("mass \nCONV",fontsize=20)
        ax2.set_xlim([-3e-4,3e-4])
        ax2.set_xticks([-3e-4,0,3e-4])
        ax2.set_xticklabels(["-3e-4","0","3e-4"])
        ax2.xaxis.set_tick_params(width=3,length=6)
        ax2.yaxis.set_tick_params(width=3,length=6)
    
        ax2.set_xlabel("Flux divergence in $\mathrm{gkg}^{-1}\mathrm{s}^{-1}$")
        #ax2.legend(loc="upper center",fontsize=20)
        ax2.invert_yaxis()
        sns.despine(ax=ax2,offset=10)
    
        #######################################################################
        ax3=profile.add_subplot(133)
        ax3.plot(core["TRANSP"],
             core["TRANSP"].index.astype(int),label="core",color="darkgreen")
        ax3.plot(warm_sector["TRANSP"],
             warm_sector["TRANSP"].index.astype(int),
             label="warm sector",color="orange")
        ax3.plot(cold_sector["TRANSP"],
             cold_sector["TRANSP"].index.astype(int),
             label="cold sector",color="darkblue")
    
        ax3.fill_betweenx(y=core.index.astype(float),
                          x1=core["TRANSP"],
                          x2=core_ideal["TRANSP"], 
                          color="green",alpha=0.3)
        ax3.fill_betweenx(y=warm_sector.index.astype(float),
                          x1=warm_sector["TRANSP"],
                          x2=warm_sector_ideal["TRANSP"], 
                          color="orange",alpha=0.3)
        ax3.fill_betweenx(y=core.index.astype(float),
                          x1=cold_sector["TRANSP"],
                          x2=cold_sector_ideal["TRANSP"], 
                          color="blue",alpha=0.3)
    
        ax3.axvline(0,ls="--",lw=2,color="k")
        ax3.spines['left'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        for axis in ['top','bottom','left','right']:
            ax3.spines[axis].set_linewidth(3)
        ax3.yaxis.set_tick_params(width=3,length=6)
        ax3.xaxis.set_tick_params(width=3,length=6)
        
        ax3.set_title("moisture \ntransport",fontsize=20)
        ax3.set_xlim([-3e-4,3e-4])
        ax3.set_xticks([-3e-4,0,3e-4])
        ax3.set_xticklabels(["-3e-4","0","3e-4"])

        """
        #ax1.plot(.mean()*1000,pressure,label="q-outflow")
        #ax1.plot(mean_trpz_moist_transport*1000,pressure,label="transport")
        """
        
        ax1.set_ylabel("Pressure in hPa")
        ax1.set_yticks([300,500,700,850,1000])
        ax1.set_yticklabels(["300","500","700","850","1000"])
        ax2.set_yticks([300,500,700,850,1000])
        ax3.set_yticks([300,500,700,850,1000])
        
        ax2.set_yticklabels([])
    
        ax3.yaxis.tick_right()
        ax3.invert_yaxis()
        ax3.set_ylim([1000,200])
        for loc, spine in ax3.spines.items():
            if loc in ["right","bottom"]:
                spine.set_position(('outward', 10)) 
        plt.subplots_adjust(wspace=0.4)
        fig_name=self.flight+"_"+self.grid_name+"_sonde_no_"+\
            str(self.sonde_no)+"_Moisture_transport_Divergence.png"
        if not save_as_manuscript_figure:
            fig_plot_path=self.cmpgn_cls.plot_path+"/budget/"
            if not os.path.exists(fig_plot_path):
                os.makedirs(fig_plot_path)
        else:
            fig_plot_path=self.cmpgn_cls.plot_path+"/../../../"+\
                    "Synthetic_AR_Paper/"+"/Manuscript/Paper_Plots/"
            fig_name="Fig13_"+fig_name
        profile.savefig(fig_plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",fig_plot_path+fig_name)
    
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
            
            
    ###########################################################################
    #%%
    def allocate_budgets(self,Campaign_Budgets={},
                         Campaign_Ideal_Budgets={},
                         Campaign_Inst_Budgets={},
                         Campaign_Inst_Ideal_Budgets={}):
        
        if Campaign_Budgets!={}:
            self.Campaign_Budgets=Campaign_Budgets
        if Campaign_Ideal_Budgets!={}:
            self.Campaign_Ideal_Budgets=Campaign_Ideal_Budgets
        if Campaign_Inst_Budgets!={}:
            self.Campaign_Inst_Budgets=Campaign_Inst_Budgets
        if Campaign_Inst_Ideal_Budgets!={}:
            self.Campaign_Inst_Ideal_Budgets=Campaign_Inst_Ideal_Budgets
    
    # Summarizing plots of campaign
    def calc_budgets_in_mm_h(self):
        if hasattr(self,"Campaign_Budgets"):
            warm_budgets=self.Campaign_Budgets["warm_sector"]
            core_budgets=self.Campaign_Budgets["core"]
            cold_budgets=self.Campaign_Budgets["cold_sector"]
            gravit_norm=1/9.82
            self.budget_regions=pd.DataFrame()
            #    index=self.Campaign_Budgets["core"]["ADV"].index)
            self.budget_regions["Warm\nADV"]=gravit_norm*warm_budgets["ADV"].values/\
                                                1000*3600
            self.budget_regions["Warm\nCONV"]=gravit_norm*warm_budgets["CONV"].values/\
                                                1000*3600
            self.budget_regions["Core\nADV"]=gravit_norm*core_budgets["ADV"].values/\
                                                1000*3600
            self.budget_regions["Core\nCONV"]=gravit_norm*core_budgets["CONV"].values/\
                                                1000*3600
            self.budget_regions["Cold\nADV"]=gravit_norm*cold_budgets["ADV"].values/\
                                                1000*3600
            self.budget_regions["Cold\nCONV"]=gravit_norm*cold_budgets["CONV"].values/\
                                                1000*3600
        
        if hasattr(self,"Campaign_Ideal_Budgets"):
            warm_ideal_budgets=self.Campaign_Ideal_Budgets["warm_sector"]
            core_ideal_budgets=self.Campaign_Ideal_Budgets["core"]
            cold_ideal_budgets=self.Campaign_Ideal_Budgets["cold_sector"]
            
            self.budget_ideal_regions=pd.DataFrame()
            #    index=self.Campaign_Ideal_Budgets["core"]["ADV"].index)
            self.budget_ideal_regions["Warm\nADV"]=gravit_norm*warm_ideal_budgets["ADV"].values/\
                                                        1000*3600
            self.budget_ideal_regions["Warm\nCONV"]=gravit_norm*warm_ideal_budgets["CONV"].values/\
                                                        1000*3600
            self.budget_ideal_regions["Core\nADV"]=gravit_norm*core_ideal_budgets["ADV"].values/\
                                                        1000*3600
            self.budget_ideal_regions["Core\nCONV"]=gravit_norm*core_ideal_budgets["CONV"].values/\
                                                        1000*3600
            self.budget_ideal_regions["Cold\nADV"]=gravit_norm*cold_ideal_budgets["ADV"].values/\
                                                        1000*3600
            self.budget_ideal_regions["Cold\nCONV"]=gravit_norm*cold_ideal_budgets["CONV"].values/\
                                                        1000*3600
        if hasattr(self,"Campaign_Inst_Budgets"):
            warm_inst_budgets=self.Campaign_Inst_Budgets["warm_sector"]
            core_inst_budgets=self.Campaign_Inst_Budgets["core"]
            cold_inst_budgets=self.Campaign_Inst_Budgets["cold_sector"]
            
            self.budget_inst_regions=pd.DataFrame()
                #index=self.Campaign_Inst_Budgets["core"]["ADV"].index)
            self.budget_inst_regions["Warm\nADV"]=gravit_norm*warm_inst_budgets["ADV"].values/\
                                                        1000*3600
            self.budget_inst_regions["Warm\nCONV"]=gravit_norm*warm_inst_budgets["CONV"].values/\
                                                        1000*3600
            self.budget_inst_regions["Core\nADV"]=gravit_norm*core_inst_budgets["ADV"].values/\
                                                        1000*3600
            self.budget_inst_regions["Core\nCONV"]=gravit_norm*core_inst_budgets["CONV"].values/\
                                                        1000*3600
            self.budget_inst_regions["Cold\nADV"]=gravit_norm*cold_inst_budgets["ADV"].values/\
                                                        1000*3600
            self.budget_inst_regions["Cold\nCONV"]=gravit_norm*cold_inst_budgets["CONV"].values/\
                                                        1000*3600
        
        if hasattr(self,"Campaign_Inst_Ideal_Budgets"):
            warm_inst_ideal_budgets=self.Campaign_Inst_Ideal_Budgets["warm_sector"]
            core_inst_ideal_budgets=self.Campaign_Inst_Ideal_Budgets["core"]
            cold_inst_ideal_budgets=self.Campaign_Inst_Ideal_Budgets["cold_sector"]
            
            self.budget_inst_ideal_regions=pd.DataFrame()
                    #index=self.Campaign_Inst_Ideal_Budgets["core"]["ADV"].index)
            self.budget_inst_ideal_regions["Warm\nADV"]=gravit_norm*\
                                        warm_inst_ideal_budgets["ADV"].values/\
                                                        1000*3600
            self.budget_inst_ideal_regions["Warm\nCONV"]=gravit_norm*\
                                        warm_inst_ideal_budgets["CONV"].values/\
                                                        1000*3600
            self.budget_inst_ideal_regions["Core\nADV"]=gravit_norm*\
                core_inst_ideal_budgets["ADV"].values/\
                                                        1000*3600
            self.budget_inst_ideal_regions["Core\nCONV"]=gravit_norm*\
                core_inst_ideal_budgets["CONV"].values/\
                                                        1000*3600
            self.budget_inst_ideal_regions["Cold\nADV"]=gravit_norm*\
                cold_inst_ideal_budgets["ADV"].values/\
                                                        1000*3600
            self.budget_inst_ideal_regions["Cold\nCONV"]=gravit_norm*\
                        cold_inst_ideal_budgets["CONV"].values/\
                                                        1000*3600
        
            
    def moisture_convergence_cases_overview(self,Campaign_Budgets={},
                                            Campaign_Ideal_Budgets={},
                                            Campaign_Inst_Budgets={},
                                            Campaign_Inst_Ideal_Budgets={},
                                            save_as_manuscript_figure=False,
                                            instantan_comparison=False):
        
        # Allocate variables and calc budget contributions in mm/h
        self.allocate_budgets(Campaign_Budgets,Campaign_Ideal_Budgets,
                              Campaign_Inst_Budgets,Campaign_Inst_Ideal_Budgets)
        self.calc_budgets_in_mm_h()
        #Start plotting
        budget_boxplot=plt.figure(figsize=(12,9), dpi= 300)
        matplotlib.rcParams.update({'font.size': 24})
            
        color_palette=["darkorange","orange","lightgreen",
                   "green","lightblue","blue"]                    

        ax1=budget_boxplot.add_subplot(111)
        ax1.axhline(0,color="grey",ls="--",lw=2,zorder=1)
    
        budget_regions=self.budget_regions
        if not instantan_comparison:
            budget_ideal_regions=self.budget_ideal_regions
        else:
            budget_ideal_regions=self.budget_inst_ideal_regions
            
        if self.grid_name=="CARRA":
            budget_ideal_regions=-1*budget_ideal_regions
            budget_regions=-1*budget_regions
            
        sns.boxplot(data=24*budget_ideal_regions,notch=False,
                       zorder=0,linewidth=3.5,palette=color_palette)
        
        for patch in ax1.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .5))
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(3.0)
        
        ax1.xaxis.set_tick_params(width=2,length=10)
        ax1.yaxis.set_tick_params(width=2,length=10)
                        #ax1.xaxis.spines(width=3)
        ax1.set_ylabel("Contribution to \nWater Vapour Budget ($\mathrm{mmd}^{-1}$)")
        sns.boxplot(data=24*budget_regions,width=0.4,linewidth=3.0,
            notch=False,color="k",palette=["lightgrey"],zorder=1)
    
        sns.despine(offset=10)
        ax1.set_ylim([-8,8])
        if not self.do_instantan:
            fig_name=self.grid_name+"_Water_Vapour_Budget.png"
        else:
            fig_name=self.grid_name+"_inst"+"_Water_Vapour_Budget.png"
        if not save_as_manuscript_figure:
            plot_path=self.plot_path
        else:
            plot_path=self.plot_path+\
                "/../../../../Synthetic_AR_paper/Manuscript/Paper_Plots/"
            if not instantan_comparison:
                fig_name="Fig14_"+fig_name
            else:
                fig_name="Fig18_"+fig_name
        budget_boxplot.savefig(plot_path+fig_name,
                       dpi=300,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)
        
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
        #ax1.bar(np.arange(sector_divergence_errors.shape[0])+1,
        #            sector_divergence_errors)
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
                     color="lightgrey",zorder=2)
        eb1[-1][0].set_linestyle("--")
        
        # Instationarity error
        eb2=ax1.errorbar(np.arange(abs_mean_sector_divergence_errors.shape[0])+0.1,
                     mean_sector_divergence_inst_errors,fmt="s",markersize=30,
                     markeredgecolor="k",ecolor="k",
                     yerr=std_sector_divergence_inst_errors,
                     color="darkgrey",zorder=2)
        eb2[-1][0].set_linestyle("-")
        eb2[-1][0].set_linewidth(2)
        ax1.axhline(y=0,color="brown",ls="--",lw=2)
        #yerr=abs_std_sector_divergence_errors,color="lightgrey")
        ax1.set_xticks(np.arange(mean_sector_divergence_errors.shape[0]))
        ax1.set_xticklabels(mean_sector_divergence_errors.index,fontsize=10)
        ax1.set_ylabel("Error in $\mathrm{mmh}^{-1} $")
        # Axis linewidth
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(3.0)
        ax1.xaxis.set_tick_params(width=2,length=10)
        ax1.yaxis.set_tick_params(width=2,length=10)
        
        sns.despine(offset=10)
        fig_name="Error_Bars_Inst_Real_and_Sondes_DIV.png"
        if not save_as_manuscript_figure:
            plot_path=self.plot_path
        else:
            plot_path=self.plot_path+\
                "/../../../../Synthetic_AR_paper/Manuscript/Paper_Plots/"
        fig_name="Fig18_"+fig_name
        error_fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:", plot_path+fig_name)
        return None
        