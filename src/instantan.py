# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:24:54 2022

@author: u300737
"""
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
    
    def __init__(self,temporary_cmpgn_cls):
        self.cmpgn_cls=temporary_cmpgn_cls # just used to define a default 
        #                                    plot path   
        
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
        else:
            self.ivt_arg="Interp_IVT"
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
            
            plot_ax.set_title(flight,fontsize=16,loc="left",y=0.9)
            plot_ax.set_xlim([-500,500])
            plot_ax.set_ylim([100,650])
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
            fig_name="Fig15_"+fig_name
            plot_path=self.path_dict["plot_figures_path"]
        f.savefig(plot_path+fig_name,dpi=60,bbox_inches="tight")
        print("Figure saved as:", plot_path+fig_name)
    
    def create_data_and_plot_of_instantan_in_outflow(self,
                                            save_as_manuscript_figure=False):
        self.load_hmp_flights()    
        self.plot_in_outflow_instantan_comparison(
            save_as_manuscript_figure=save_as_manuscript_figure)

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
    instantan_cls=Instationarity(cpgn_cls)
    if figure_to_create.lower().startswith("fig15"):
        instantan_cls.create_data_and_plot_of_instantan_in_outflow(
                            save_as_manuscript_figure=True)
    return None

if __name__=="__main__":
    main()
    #legend_loc="upper right"
     #       if hmp_inflow[ivt_var_arg].max()>450:
     #           legend_loc="lower right"
            #line_core_in[0],line_core_out[0],
    #        lgd = plot_ax.legend(handles=[\
    #                                  legend_patches[0],legend_patches[1]],
    #                         loc=legend_loc,fontsize=10,ncol=1)
    
    #plot_ax.fill_betweeny(ivt_inflow["IVT_max_distance"]/1000,ivt_inflow["inst"],
    #                        y2=ivt_inflow["flight"],color="grey",alpha=0.7)
    #line_core_in=plot_ax.plot(inflow_core["IVT_max_distance"]/1000,
    #                  inflow_core[ivt_var_arg],lw=2,color="darkblue",
    #                  label="AR core (in): TIVT="+\
    #                              str((TIVT_inflow_core/1e6).round(1)))
    #
    #        plot_ax.plot(hmp_outflow["IVT_max_distance"]/1000,
    #             hmp_outflow[ivt_var_arg],color="orange",lw=8)
    
    #        line_core_out=plot_ax.plot(outflow_core["IVT_max_distance"]/1000,
    #                           outflow_core[ivt_var_arg],
    #                           lw=2,color="darkred",
    #                           label="AR core (out): TIVT="+\
    #                               str((TIVT_outflow_core/1e6).round(1)))

        
    #        plot_ax.plot(ar_inflow_warm_sector["IVT_max_distance"]/1000,
    #             ar_inflow_warm_sector[ivt_var_arg],
    #             lw=3,ls=":",color="darkblue")
        
    #        plot_ax.plot(ar_inflow_cold_sector["IVT_max_distance"]/1000,
    #             ar_inflow_cold_sector[ivt_var_arg],
    #             lw=3,ls="-.",color="darkblue")
   # 
    #        plot_ax.plot(ar_outflow_warm_sector["IVT_max_distance"]/1000,
    #             ar_outflow_warm_sector[ivt_var_arg],
    #             lw=3,ls=":",color="darkred")
    #        plot_ax.plot(ar_outflow_cold_sector["IVT_max_distance"]/1000,
    #             ar_outflow_cold_sector[ivt_var_arg],
    #             lw=3,ls="-.",color="darkred")
                

#inflow_maxima=flight_hmp_df["IVT_max_distance"].loc[]
#ax1.plot(ivt_inflow["IVT_max_distance"],ivt_inflow["flight"],
#          color="k",lw=2,ls="-.")
#ax1.plot(ivt_outflow["IVT_max_distance"],flight_hmp_df[ivt_arg].loc[outflow_index],
#         color="grey",lw=2,ls="--")

#ax1.plot(flight_hmp_df["IVT_max_distance"].loc[inflow_index])


#ax1.plot(flight_hmp_df["IVT_max_distance"].loc[outflow_index],color="darkred",
#         lw=2,ls="-.",flight_hmp_df_inst[ivt_arg].loc[outflow_index])


#ax_in=ivt_inflow.plot()
#ax_in.set_ylabel("IVT in $\\mathrm{kg}{\\mathrm{m}}^{-1}{\\mathrm{s}}^{-1}$")
#print(NA_Hydrometeors.keys())#
#ax_in.set_ylim([100,650])
#grid_name="ERA5"
#if use_carra:
#    grid_name="CARRA"
#ax.figure.savefig('demo-file.pdf')"
            
