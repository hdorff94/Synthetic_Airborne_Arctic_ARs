# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:24:54 2022

@author: u300737
"""
import numpy as np
import pandas as pd 

import gridonhalo as GridHalo
import flightcampaign
# Plot scripts
import matplotlib 
import matplotlib.pyplot as plt
        
try:
    from typhon.plots import styles
except:
    print("Typhon module cannot be imported")
import seaborn as sns
matplotlib.rcParams.update({"font.size":16})

class Instationarity(GridHalo.ERA_on_HALO,GridHalo.CARRA_on_HALO):
    
    def __init__(self,temporary_cmpgn_cls,config_file):
        self.cmpgn_cls=temporary_cmpgn_cls # just used to define a default 
        #                                    plot path   
        self.cfg_file=config_file
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
        self.na_flights  = [*self.flight_dates[self.na_campaign_name].keys()]
        self.snd_flights = [*self.flight_dates[self.snd_campaign_name].keys()]
        self.use_era     = True
        self.use_carra   = True
        if self.use_carra:
            self.ivt_arg="highres_Interp_IVT"
            self.grid_name="CARRA"
        else:
            self.ivt_arg="Interp_IVT"
            self.grid_name="ERA5"
        self.path_declarer()
        #self.import_plot_modules()
        
    
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
        path_dict["plot_figures_path"] = path_dict["aircraft_base_path"]+\
                            "/../Synthetic_AR_Paper/Manuscript/Paper_Plots/"

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
        
    def load_hmp_flights(self):
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
                    era_is_desired=self.use_era,icon_is_desired=False,
                    carra_is_desired=self.use_carra,do_daily_plots=False,
                    calc_hmp=True,calc_hmc=False,
                    do_instantaneous=True)                                                        
        
        key_list=[*NA_Hydrometeors.keys()]
        # rename dictionary keys from SRFs to flight dates"
        # first campaign
        for key in key_list:
            new_dict_entry=int(self.flight_dates["North_Atlantic_Run"][key])
            # standard flight"
            NA_Hydrometeors[new_dict_entry]=NA_Hydrometeors[key]
            NA_HALO_Dict[new_dict_entry]=NA_HALO_Dict[key]
            del NA_Hydrometeors[key], NA_HALO_Dict[key]
            # instantan flight"
            NA_Hydrometeors_inst[new_dict_entry]=NA_Hydrometeors_inst[key]
            NA_HALO_Dict_inst[new_dict_entry]=NA_HALO_Dict_inst[key]
            del NA_Hydrometeors_inst[key], NA_HALO_Dict_inst[key]
        
        #second campaign"
        key_list=[*SND_Hydrometeors.keys()]
        for key in key_list:
            #standard flight
            new_dict_entry=int(self.flight_dates["Second_Synthetic_Study"][key])
            SND_Hydrometeors[new_dict_entry]=SND_Hydrometeors[key]
            SND_HALO_Dict[new_dict_entry]=SND_HALO_Dict[key]
            del SND_Hydrometeors[key], SND_HALO_Dict[key]
            #instantan flight
            SND_Hydrometeors_inst[new_dict_entry]=SND_Hydrometeors_inst[key]
            SND_HALO_Dict_inst[new_dict_entry]=SND_HALO_Dict_inst[key]
            del SND_Hydrometeors_inst[key], SND_HALO_Dict_inst[key]
    
        # Merge both campaigns"
        # standard flight
        self.campaign_Hydrometeors= dict(list(NA_Hydrometeors.items()) +\
                                    list(SND_Hydrometeors.items()))
        self.campaign_Hydrometeors=dict(
                                    sorted(self.campaign_Hydrometeors.items()))

        self.campaign_HALO = dict(list(NA_HALO_Dict.items()) +\
                             list(SND_HALO_Dict.items()))

        self.campaign_HALO=dict(sorted(self.campaign_HALO.items()))
        # instantan flight
        self.campaign_Hydrometeors_inst=\
                dict(list(NA_Hydrometeors_inst.items()) +\
                     list(SND_Hydrometeors_inst.items()))
        self.campaign_Hydrometeors_inst= dict(sorted(\
                                    self.campaign_Hydrometeors_inst.items()))
        self.campaign_HALO_inst = dict(list(NA_HALO_Dict_inst.items()) +\
                                  list(SND_HALO_Dict_inst.items()))
        self.campaign_HALO_inst=dict(sorted(self.campaign_HALO_inst.items()))
        self.grid_name=SND_Hydrometeors[new_dict_entry]["AR_internal"].name
    
    def preprocess_loaded_single_flight_data(self,flight):
            flight_hmp_df=self.campaign_Hydrometeors[flight]["AR_internal"]
            flight_hmp_df_inst=self.campaign_Hydrometeors_inst[\
                                                    flight]["AR_internal"]
            # Define sectors
            inflow_index=self.campaign_HALO[flight]["inflow"].index
            inflow_inst_index=self.campaign_HALO_inst[flight]["inflow"].index
            outflow_index=self.campaign_HALO[flight]["outflow"].index
            outflow_inst_index=self.campaign_HALO_inst[flight]["outflow"].index
    
            # Inflow sectors
            self.ivt_inflow=pd.DataFrame()
            self.ivt_inflow["flight"]=flight_hmp_df[self.ivt_arg]\
                                    .loc[inflow_index].values
            self.ivt_inflow["inst"]=flight_hmp_df_inst[self.ivt_arg]\
                                    .loc[inflow_inst_index].values
            self.ivt_inflow["IVT_max_distance"]=flight_hmp_df["IVT_max_distance"]\
                                    .loc[inflow_index].values
            # Outflow sectors
            self.ivt_outflow=pd.DataFrame()
            self.ivt_outflow["IVT_max_distance"]=flight_hmp_df["IVT_max_distance"].\
                                            loc[outflow_index].values
            self.ivt_outflow["flight"]=flight_hmp_df[self.ivt_arg]\
                                    .loc[outflow_index].values
            self.ivt_outflow["inst"]=flight_hmp_df_inst[self.ivt_arg].\
                                    loc[outflow_inst_index].values
    
            max_inflow_center=self.ivt_inflow["IVT_max_distance"]\
                                .iloc[self.ivt_inflow["flight"].argmax()]
            max_inflow_inst_center=self.ivt_inflow["IVT_max_distance"].iloc[\
                                        self.ivt_inflow["inst"].argmax()]
            
            self.ivt_inflow_center=self.ivt_inflow["IVT_max_distance"]-\
                                    max_inflow_center
            self.ivt_inflow_inst_center=self.ivt_inflow["IVT_max_distance"]-\
                                        max_inflow_inst_center
        
            max_outflow_center=self.ivt_outflow["IVT_max_distance"].iloc[\
                                    self.ivt_outflow["flight"].argmax()]
            max_outflow_inst_center=self.ivt_outflow["IVT_max_distance"].iloc[\
                                    self.ivt_outflow["inst"].argmax()]
    
            #ivt_outflow["IVT_max_distance"]=ivt_outflow["IVT_max_distance"]-max_outflow_center
            self.ivt_outflow_center=self.ivt_outflow["IVT_max_distance"]-\
                                    max_outflow_center
            self.ivt_outflow_inst_center=self.ivt_outflow["IVT_max_distance"]-\
                                    max_outflow_inst_center
    
    def plot_in_outflow_instantan_comparison(self,
                                    save_as_manuscript_figure=False):
        HMP_dict=self.campaign_Hydrometeors.copy()
        row_number=3
        col_number=int(len(HMP_dict.keys())/row_number)+\
                        len(HMP_dict.keys()) % row_number

        f,ax=plt.subplots(nrows=row_number,ncols=col_number,
                          figsize=(18,12),sharex=True,sharey=True)
        i=0
        for flight in HMP_dict.keys():        
            self.preprocess_loaded_single_flight_data(flight)
            
            if len(ax.shape)>=2:
                if i<col_number:
                    horizontal_field=i
                    plot_ax=ax[0,horizontal_field]
                elif i<2*col_number:
                    horizontal_field=i-col_number
                    plot_ax=ax[1,horizontal_field]
                    #plot_ax.set_xlabel("IVT max distance (km)")        
                else:
                    horizontal_field=i-2*col_number
                    plot_ax=ax[2,horizontal_field]
                    plot_ax.set_xlabel("IVT max distance (km)")
            if horizontal_field==0:
                plot_ax.set_ylabel("IVT ($\mathrm{kgm}^{-1}\mathrm{s}^{-1})$")
                
            else:
                horizontal_field=i
            #plot_ax=ax[i]
            #plot_ax.set_ylabel("IVT ($\mathrm{kgm}^{-1}\mathrm{s}^{-1})$")
                
            plot_ax.plot(self.ivt_inflow_center/1000,
                         self.ivt_inflow["flight"],color="k",lw=2)            
            plot_ax.plot(self.ivt_inflow_inst_center/1000,
                         self.ivt_inflow["inst"],color="k",lw=2,ls="-.")
            plot_ax.plot(self.ivt_outflow_center/1000,
                         self.ivt_outflow["flight"],color="darkred",lw=2)            
            plot_ax.plot(self.ivt_outflow_inst_center/1000,
                         self.ivt_outflow["inst"],color="darkred",lw=2,ls="-.")
            plot_ax.text(0.015,0.8,"AR"+str(i+1),color="k",
                     transform=plot_ax.transAxes,
                     bbox=dict(facecolor='lightgrey', edgecolor='black', 
                               boxstyle='round,pad=0.2'))
            plot_ax.set_xlim([-500,500])
            plot_ax.set_ylim([100,700])
            plot_ax.set_yticks([200,400,600])
            for axis in ["left","bottom"]:
                plot_ax.spines[axis].set_linewidth(2)
                plot_ax.tick_params(length=6,width=2)#

            i+=1
                
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_patches = [Patch(facecolor='k', edgecolor='k',label='Inflow'),
                Patch(facecolor='darkred', edgecolor='k',label='Outflow'),
                Line2D([0], [0],color="dimgray",ls="-",lw=3,label="flight"),
                Line2D([0], [0],color="dimgray",ls="-.",lw=3,label="instantan")]
        
        sns.despine(offset=10)

        plot_ax.legend(handles=legend_patches,loc='lower left',
               bbox_to_anchor=(-1.6, -0.6),ncol=4,
               fontsize=16, fancybox=True, shadow=True)

        fig_name=self.grid_name+"_AR_IVT_Stationarity.pdf"
        if not save_as_manuscript_figure:
            plot_path=self.cmpgn_cls.plot_path
        else:
            fig_name="Fig13_"+fig_name
            plot_path=self.path_dict["plot_figures_path"]
        f.savefig(plot_path+fig_name,dpi=60,bbox_inches="tight")
        print("Figure saved as:", plot_path+fig_name)
    
    def create_data_and_plot_of_instantan_in_outflow(self,
                                            save_as_manuscript_figure=False):
        self.load_hmp_flights()    
        self.plot_in_outflow_instantan_comparison(
            save_as_manuscript_figure=save_as_manuscript_figure)
    
    def plot_div_term_instantan_comparison(self,div_var="CONV",
                                           limit_min_max=pd.DataFrame(),
                                           save_as_manuscript_figure=False):
        """
        Parameters
        ----------
        div_var : str, optional
            Variable to show differences between flight and instantan values.
            The default is "CONV". The other accepted is "ADV".
        limit_min_max : pd.DataFrame, optional
            DataFrame specifying minimum and maximum values from other quantity
            to compare their impact
            
        Returns
        -------
        None.

        """
        if not div_var in ["ADV","CONV"]:
            raise Exception("Wrong divergence variable given.",
                            "You have to choose either ADV or CONV")
        import moisturebudget
        Moist_Convergence=moisturebudget.Moisture_Convergence
        ###
        # temporary values to initialize things
        flight="SRF02"

        ###
        
        # Vertical profiles
        from matplotlib.ticker import NullFormatter
        from matplotlib.lines import Line2D
        
        legend_elements = [Line2D([0],[0],color='orange',lw=3,ls="--",marker="o",
                              markerfacecolor="orange",markeredgecolor="k",
                              label='warm sector'),
                       Line2D([0],[0],color='g',lw=3,ls="--",marker="o",
                              markerfacecolor="g",markeredgecolor="k",
                              label='core'),
                       Line2D([0],[0],color='b',lw=3,ls="--",marker="o",
                              markerfacecolor="b",markeredgecolor="k",
                              label='cold sector')]        
        row_number=3
        col_number=3
        f,ax=plt.subplots(nrows=row_number,ncols=col_number,
                          figsize=(20,18),sharex=True,sharey=True)
    
        i=0
        min_max_convs=pd.DataFrame(columns=["min_CONV","max_CONV"])
        rf_date_values=[*self.flight_dates["North_Atlantic_Run"].values()]+\
                    [*self.flight_dates["Second_Synthetic_Study"].values()]
        rf_date_values=sorted(rf_date_values)
        ##self.campaign_Hydrometeors= dict(list(NA_Hydrometeors.items()) +\
        #                            list(SND_Hydrometeors.items()))
        #self.campaign_Hydrometeors=dict(
        #                            sorted(self.campaign_Hydrometeors.items()))

        for rf_date in rf_date_values:
            if rf_date in [*self.flight_dates["North_Atlantic_Run"].values()]:
                campaign="North_Atlantic_Run"
                cmpgn_cls=flightcampaign.North_Atlantic_February_Run(
                    is_flight_campaign=True,
                    major_path=self.cfg_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=flight,
                    instruments=["radar","radiometer","sonde"])
            
            else:
                campaign="Second_Synthetic_Study"
                cmpgn_cls=flightcampaign.Second_Synthetic_Study(
                    is_flight_campaign=True,
                    major_path=self.cfg_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=flight,
                    instruments=["radar","radiometer","sonde"])

            flight=[f for f,d in self.flight_dates[campaign].items() \
                        if d==rf_date][0]
            #rf_date=self.flight_dates[campaign][flight]
            Moisture_CONV=Moist_Convergence(cmpgn_cls,
                                    flight+"_instantan",self.cfg_file,
                                    grid_name=self.grid_name,do_instantan=True)
            Sectors,Ideal_Sectors,cmpgn_cls=\
                        Moisture_CONV.load_moisture_convergence_single_case()
                    
            Flight_Moisture_CONV=Moist_Convergence(
                        cmpgn_cls,flight,self.cfg_file,
                        grid_name=self.grid_name,do_instantan=False)    
            Flight_Sectors,Flight_Ideal_Sectors,cmpgn_cls=\
                    Flight_Moisture_CONV.load_moisture_convergence_single_case()
            
            if div_var=="CONV":        
                    xlabel="Error in mass \ndivergence ($\mathrm{gkg}^{-1}\mathrm{s}^{-1}$)"
            else:
                    xlabel="Error in advection ($\mathrm{gkg}^{-1}\mathrm{s}^{-1}$)"
                    
            if len(ax.shape)>=2:
                if i<col_number:
                        horizontal_field=i
                        plot_ax=ax[0,horizontal_field]
                elif i<2*col_number:
                        horizontal_field=i-col_number
                        plot_ax=ax[1,horizontal_field]
                else:
                        horizontal_field=i-2*col_number
                        plot_ax=ax[2,horizontal_field]
                        plot_ax.set_xlabel(xlabel)
                if horizontal_field==0:        
                        plot_ax.set_ylabel("Pressure (hPa)")
                
            # not instantan - instantan
            if div_var=="CONV":
                div_var_arg=div_var
            else:
                div_var_arg=div_var+"_calc"
                
            core_relative_error=\
                    (Flight_Ideal_Sectors["core"][div_var_arg]-\
                             Ideal_Sectors["core"][div_var_arg])    
            core_sign_change=np.sign(Flight_Ideal_Sectors["core"][div_var_arg])\
                                !=np.sign(Ideal_Sectors["core"][div_var_arg])
            core_sign_change=core_sign_change.astype(int)
            warm_relative_error=\
                    (Flight_Ideal_Sectors["warm_sector"][div_var_arg]-\
                             Ideal_Sectors["warm_sector"][div_var_arg])
            if not flight.startswith("SRF12"):
                cold_relative_error=\
                        (Flight_Ideal_Sectors["cold_sector"][div_var_arg]-\
                             Ideal_Sectors["cold_sector"][div_var_arg])
            else:
                cold_relative_error=pd.Series(data=np.nan,
                                            index=warm_relative_error.index.values)
            
            plot_ax.plot(core_relative_error.values,
                             Ideal_Sectors["core"].index.values,marker="o",
                             color="green",markeredgecolor="k",ls="--",zorder=2)
            plot_ax.plot(warm_relative_error.values,
                             Ideal_Sectors["warm_sector"].index.values,
                             marker="o",color="orange",markeredgecolor="k",ls="--",
                             zorder=3)
                
            if not flight.startswith("SRF12"):    
                plot_ax.plot(cold_relative_error.values,
                             Ideal_Sectors["cold_sector"].index.values,
                             marker="o",color="blue",markeredgecolor="k",ls="--",
                             zorder=4)
                
            day_stats=pd.DataFrame(data=np.nan,columns=["min","max"],
                                        index=["warm","core","cold"])
                
            day_stats["min"]=pd.Series(data=np.array([warm_relative_error.min(),
                                                 core_relative_error.min(),
                                                 cold_relative_error.min()]),
                                           index=["warm","core","cold"])
            day_stats["max"]=pd.Series(data=np.array([warm_relative_error.max(),
                                                 core_relative_error.max(),
                                                 cold_relative_error.max()]),
                                           index=["warm","core","cold"])
                
            temporary_stat_df=pd.DataFrame(
                    data=np.nan,
                    columns=["min","max"],
                    index=[rf_date])
                
            temporary_stat_df["min"]=day_stats["min"].min()
            temporary_stat_df["max"]=day_stats["max"].max()
                
            min_max_convs=min_max_convs.append(temporary_stat_df)
                
            plot_ax.axvline(x=0,ls="--",color="k",lw=2)
            plot_ax.set_ylim([200,1000])
            plot_ax.set_yscale("log")
            plot_ax.yaxis.set_major_formatter(NullFormatter())
            plot_ax.yaxis.set_minor_formatter(NullFormatter())
            plot_ax.get_yaxis().set_major_formatter(
                                        matplotlib.ticker.ScalarFormatter())
            plot_ax.set_yticks([300,500,700,850,1000])
            plot_ax.set_yticklabels(["300","500","700","850","1000"])
            if limit_min_max.shape[0]>0:
                plot_ax.axvspan(limit_min_max["min"].loc[rf_date],
                                    limit_min_max["max"].loc[rf_date],
                                    alpha=0.5,color="darkgrey")
            #plot_ax.set_xscale("log")
            plot_ax.invert_yaxis()
            if div_var=="CONV":
                plot_ax.set_xlim([-1e-4,1e-4])
                plot_ax.set_xticks([-1e-4,-0.5e-4,0,0.5e-4,1e-4])
                plot_ax.set_xticklabels(["-1e-4","-5e-5","0","5e-5","1e-4"])
            
                plot_ax.text(x=0.5e-4,y=250,s=rf_date,
                         color="gray",fontsize=16)
            else:
                plot_ax.set_xlim([-1.5e-4,1.5e-4])
                plot_ax.set_xticks([-1.5e-4,-0.75e-4,0,0.75e-4,1.5e-4])
                plot_ax.set_xticklabels(["-1.5e-4","-0.75e-4",
                                         "0","0.75e-4","1.5e-4"])
                
                plot_ax.text(x=0.5e-4,y=250,s=rf_date,
                         color="gray",fontsize=16)
                    
            if i==1:
                plot_ax.legend(handles=legend_elements,
                               loc="upper center", ncol=3,
                               bbox_to_anchor=(0.4,1.25))
            plot_ax.tick_params(length=4,width=2)
            for axis in ['bottom','left']:
                plot_ax.spines[axis].set_linewidth(2)
            i+=1
        sns.despine(offset=10)
        fig_name=self.grid_name+"_instantan_comparison_"+div_var+"_flights.png"
        if not save_as_manuscript_figure:
            plot_path=self.cmpgn_cls.plot_path
        else:
            if div_var=="CONV":
                fig_name="Fig16_"+fig_name
            else:
                fig_name="Fig17_"+fig_name
            plot_path=self.path_dict["plot_figures_path"]
        f.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)
        min_max_convs.name=div_var
        #if div_var=="CONV":
        return min_max_convs 
    
    def plot_div_term_instantan_deviation(self,div_var="CONV",
                                           limit_min_max=pd.DataFrame(),
                                           save_as_manuscript_figure=False):
        """
        Parameters
        ----------
        div_var : str, optional
            Variable to show differences between flight and instantan values.
            The default is "CONV". The other accepted is "ADV".
        limit_min_max : pd.DataFrame, optional
            DataFrame specifying minimum and maximum values from other quantity
            to compare their impact
            
        Returns
        -------
        None.

        """
        if not div_var in ["ADV","CONV"]:
            raise Exception("Wrong divergence variable given.",
                            "You have to choose either ADV or CONV")
        import moisturebudget
        Moist_Convergence=moisturebudget.Moisture_Convergence
        ###
        # temporary values to initialize things
        flight="SRF02"

        ###
        
        # Vertical profiles
        from matplotlib.ticker import NullFormatter
        from matplotlib.lines import Line2D
        
        legend_elements = [Line2D([0],[0],color='orange',lw=3,ls="--",marker="o",
                              markerfacecolor="orange",markeredgecolor="k",
                              label='warm sector'),
                       Line2D([0],[0],color='g',lw=3,ls="--",marker="o",
                              markerfacecolor="g",markeredgecolor="k",
                              label='core'),
                       Line2D([0],[0],color='b',lw=3,ls="--",marker="o",
                              markerfacecolor="b",markeredgecolor="k",
                              label='cold sector')]        
        row_number=3
        col_number=3
        f,ax=plt.subplots(nrows=row_number,ncols=col_number,
                          figsize=(20,18),sharex=True,sharey=True)
    
        i=0
        min_max_convs=pd.DataFrame(columns=["min_CONV","max_CONV"])
        rf_date_values=[*self.flight_dates["North_Atlantic_Run"].values()]+\
                    [*self.flight_dates["Second_Synthetic_Study"].values()]
        rf_date_values=sorted(rf_date_values)
        ##self.campaign_Hydrometeors= dict(list(NA_Hydrometeors.items()) +\
        #                            list(SND_Hydrometeors.items()))
        #self.campaign_Hydrometeors=dict(
        #                            sorted(self.campaign_Hydrometeors.items()))

        for rf_date in rf_date_values:
            if rf_date in [*self.flight_dates["North_Atlantic_Run"].values()]:
                campaign="North_Atlantic_Run"
                cmpgn_cls=flightcampaign.North_Atlantic_February_Run(
                    is_flight_campaign=True,
                    major_path=self.cfg_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=flight,
                    instruments=["radar","radiometer","sonde"])
            
            else:
                campaign="Second_Synthetic_Study"
                cmpgn_cls=flightcampaign.Second_Synthetic_Study(
                    is_flight_campaign=True,
                    major_path=self.cfg_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=flight,
                    instruments=["radar","radiometer","sonde"])

            flight=[f for f,d in self.flight_dates[campaign].items() \
                        if d==rf_date][0]
            #rf_date=self.flight_dates[campaign][flight]
            Moisture_CONV=Moist_Convergence(cmpgn_cls,
                                    flight+"_instantan",self.cfg_file,
                                    grid_name=self.grid_name,do_instantan=True)
            Sectors,Ideal_Sectors,cmpgn_cls=\
                        Moisture_CONV.load_moisture_convergence_single_case()
                    
            Flight_Moisture_CONV=Moist_Convergence(
                        cmpgn_cls,flight,self.cfg_file,
                        grid_name=self.grid_name,do_instantan=False)    
            Flight_Sectors,Flight_Ideal_Sectors,cmpgn_cls=\
                    Flight_Moisture_CONV.load_moisture_convergence_single_case()
            
            if div_var=="CONV":        
                    xlabel="Error in mass \ndivergence ($\mathrm{gkg}^{-1}\mathrm{s}^{-1}$)"
            else:
                    xlabel="Error in advection ($\mathrm{gkg}^{-1}\mathrm{s}^{-1}$)"
                    
            if len(ax.shape)>=2:
                if i<col_number:
                        horizontal_field=i
                        plot_ax=ax[0,horizontal_field]
                elif i<2*col_number:
                        horizontal_field=i-col_number
                        plot_ax=ax[1,horizontal_field]
                else:
                        horizontal_field=i-2*col_number
                        plot_ax=ax[2,horizontal_field]
                        plot_ax.set_xlabel(xlabel)
                if horizontal_field==0:        
                        plot_ax.set_ylabel("Pressure (hPa)")
                
            # not instantan - instantan
            if div_var=="CONV":
                div_var_arg=div_var
            else:
                div_var_arg=div_var+"_calc"
                
            core_relative_error=\
                    (Flight_Ideal_Sectors["core"][div_var_arg]-\
                             Ideal_Sectors["core"][div_var_arg])    
            core_sign_change=np.sign(Flight_Ideal_Sectors["core"][div_var_arg])\
                                !=np.sign(Ideal_Sectors["core"][div_var_arg])
            core_sign_change=core_sign_change.astype(int)
            warm_relative_error=\
                    (Flight_Ideal_Sectors["warm_sector"][div_var_arg]-\
                             Ideal_Sectors["warm_sector"][div_var_arg])
            if not flight.startswith("SRF12"):
                cold_relative_error=\
                        (Flight_Ideal_Sectors["cold_sector"][div_var_arg]-\
                             Ideal_Sectors["cold_sector"][div_var_arg])
            else:
                cold_relative_error=pd.Series(data=np.nan,
                                            index=warm_relative_error.index.values)
            
            plot_ax.plot(core_relative_error.values,
                             Ideal_Sectors["core"].index.values,marker="o",
                             color="green",markeredgecolor="k",ls="--",zorder=2)
            plot_ax.plot(warm_relative_error.values,
                             Ideal_Sectors["warm_sector"].index.values,
                             marker="o",color="orange",markeredgecolor="k",ls="--",
                             zorder=3)
                
            if not flight.startswith("SRF12"):    
                plot_ax.plot(cold_relative_error.values,
                             Ideal_Sectors["cold_sector"].index.values,
                             marker="o",color="blue",markeredgecolor="k",ls="--",
                             zorder=4)
                
            day_stats=pd.DataFrame(data=np.nan,columns=["min","max"],
                                        index=["warm","core","cold"])
                
            day_stats["min"]=pd.Series(data=np.array([warm_relative_error.min(),
                                                 core_relative_error.min(),
                                                 cold_relative_error.min()]),
                                           index=["warm","core","cold"])
            day_stats["max"]=pd.Series(data=np.array([warm_relative_error.max(),
                                                 core_relative_error.max(),
                                                 cold_relative_error.max()]),
                                           index=["warm","core","cold"])
                
            temporary_stat_df=pd.DataFrame(
                    data=np.nan,
                    columns=["min","max"],
                    index=[rf_date])
                
            temporary_stat_df["min"]=day_stats["min"].min()
            temporary_stat_df["max"]=day_stats["max"].max()
                
            min_max_convs=min_max_convs.append(temporary_stat_df)
                
            plot_ax.axvline(x=0,ls="--",color="k",lw=2)
            plot_ax.set_ylim([200,1000])
            plot_ax.set_yscale("log")
            plot_ax.yaxis.set_major_formatter(NullFormatter())
            plot_ax.yaxis.set_minor_formatter(NullFormatter())
            plot_ax.get_yaxis().set_major_formatter(
                                        matplotlib.ticker.ScalarFormatter())
            plot_ax.set_yticks([300,500,700,850,1000])
            plot_ax.set_yticklabels(["300","500","700","850","1000"])
            if limit_min_max.shape[0]>0:
                plot_ax.axvspan(limit_min_max["min"].loc[rf_date],
                                    limit_min_max["max"].loc[rf_date],
                                    alpha=0.5,color="darkgrey")
            #plot_ax.set_xscale("log")
            plot_ax.invert_yaxis()
            if div_var=="CONV":
                plot_ax.set_xlim([-1e-4,1e-4])
                plot_ax.set_xticks([-1e-4,-0.5e-4,0,0.5e-4,1e-4])
                plot_ax.set_xticklabels(["-1e-4","-5e-5","0","5e-5","1e-4"])
            
                plot_ax.text(x=0.5e-4,y=250,s=rf_date,
                         color="gray",fontsize=16)
            else:
                plot_ax.set_xlim([-1.5e-4,1.5e-4])
                plot_ax.set_xticks([-1.5e-4,-0.75e-4,0,0.75e-4,1.5e-4])
                plot_ax.set_xticklabels(["-1.5e-4","-0.75e-4",
                                         "0","0.75e-4","1.5e-4"])
                
                plot_ax.text(x=0.5e-4,y=250,s=rf_date,
                         color="gray",fontsize=16)
                    
            if i==1:
                plot_ax.legend(handles=legend_elements,
                               loc="upper center", ncol=3,
                               bbox_to_anchor=(0.4,1.25))
            plot_ax.tick_params(length=4,width=2)
            for axis in ['bottom','left']:
                plot_ax.spines[axis].set_linewidth(2)
            i+=1
        sns.despine(offset=10)
        fig_name=self.grid_name+"_instantan_comparison_"+div_var+"_flights.png"
        if not save_as_manuscript_figure:
            plot_path=self.cmpgn_cls.plot_path
        else:
            if div_var=="CONV":
                fig_name="Fig16_"+fig_name
            else:
                fig_name="Fig17_"+fig_name
            plot_path=self.path_dict["plot_figures_path"]
        f.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)
        min_max_convs.name=div_var
        #if div_var=="CONV":
        return min_max_convs  

    
def main(figure_to_create="fig15_in_outflow_instantan"):
    import os
    import sys
    sys.path.insert(1,os.getcwd()+"/../config/")
    import data_config
    # Load config file
    config_file_path=os.getcwd()+"/../../../Work/GIT_Repository/"
    config_file=data_config.load_config_file(config_file_path,
                                             "data_config_file")
    
    cpgn_cls_name="Second_Synthetic_Study"
    cpgn_cls=flightcampaign.Second_Synthetic_Study(
        is_flight_campaign=True,major_path=config_file["Data_Paths"]\
            ["campaign_path"],aircraft="HALO",
            interested_flights=["SRF02","SRF04","SRF07","SRF08"],
            instruments=["radar","radiometer","sonde"])       
    instantan_cls=Instationarity(cpgn_cls,config_file)
    if figure_to_create.lower().startswith("fig14new"):
        instantan_cls.plot_div_term_instantan_deviations()
        
    if figure_to_create.lower().startswith("fig15"):
        instantan_cls.create_data_and_plot_of_instantan_in_outflow(
                            save_as_manuscript_figure=True)
    #if figure_to_create.lower().startswith("fig16"):
    #    conv_limits=instantan_cls.plot_div_term_instantan_comparison("CONV",
    #                        save_as_manuscript_figure=True)
    #if figure_to_create.lower().startswith("fig17"):
    #    conv_limits=instantan_cls.plot_div_term_instantan_comparison(
    #    "CONV",save_as_manuscript_figure=False)
    
     #   instantan_cls.plot_div_term_instantan_comparison("ADV",
     #                       save_as_manuscript_figure=True,
     #                       limit_min_max=conv_limits)
    return None

if __name__=="__main__":
    main(figure_to_create="fig14new")
