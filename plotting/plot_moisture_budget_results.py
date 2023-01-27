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

def main(figure_to_create="fig13"):
    
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

    # Moisture Classes
    Moisture_CONV=\
        Budgets.Moisture_Convergence(cmpgn_cls,flight,config_file,
                 flight_dates=flight_dates,grid_name=grid_name,
                 do_instantan=do_instantan,sonde_no=sonde_no)
    Budget_plots=Budgets.Moisture_Budget_Plots(cmpgn_cls,flight,config_file,
                 grid_name=grid_name,do_instantan=do_instantan,sonde_no=sonde_no)
    Inst_Budget_plots=Budgets.Moisture_Budget_Plots(cmpgn_cls,flight,
                                        config_file,grid_name=grid_name,
                                        do_instantan=True,sonde_no=sonde_no)
    on_flight_tracks=True
    
    if figure_to_create.startswith("fig11"):
        Sectors,Ideal_Sectors,cmpgn_cls=\
            Moisture_CONV.load_moisture_convergence_single_case()
        if do_plotting:
            Budget_plots.plot_single_case(Sectors,Ideal_Sectors,
                                save_as_manuscript_figure=save_for_manuscript)
    elif figure_to_create.startswith("fig12_"):
        Campaign_Budgets,Campaign_Ideal_Budgets=\
            Moisture_CONV.get_overall_budgets()
        if do_plotting:
            Budget_plots.moisture_convergence_cases_overview(
                            Campaign_Budgets,Campaign_Ideal_Budgets,
                            save_as_manuscript_figure=save_for_manuscript,
                            with_mean_error=True)
    
    elif figure_to_create.startswith("fig14_"):
        # Here we take the continuous representation
        # so reset the sonde no
        #Moisture_CONV.sonde_no="100"
        # Instantan Budgets
        Inst_Moisture_CONV=Budgets.Moisture_Convergence(cmpgn_cls,
                                    flight+"_instantan",config_file,
                                    flight_dates=flight_dates,
                                    grid_name=grid_name,do_instantan=True)
        #Inst_Moisture_CONV.sonde_no="100"
        Inst_Budgets,Inst_Ideal_Budgets=Inst_Moisture_CONV.get_overall_budgets(
                        use_flight_tracks=on_flight_tracks)
        # Airborne Budgets
        Campaign_Budgets,Campaign_Ideal_Budgets=\
            Moisture_CONV.get_overall_budgets(use_flight_tracks=on_flight_tracks)
        
        #Inst_Budget_plots.moisture_convergence_cases_overview(
        #                    Campaign_Budgets=Campaign_Budgets,
        #                    Campaign_Ideal_Budgets=Campaign_Ideal_Budgets,
        #                    Campaign_Inst_Budgets={},
        #                    Campaign_Inst_Ideal_Budgets=Inst_Ideal_Budgets,
        #                    instantan_comparison=True,
        #                    save_as_manuscript_figure=False)
        Inst_Budget_plots.moisture_convergence_time_instantan_comparison(
                Campaign_Budgets=Campaign_Budgets,
                Campaign_Ideal_Budgets=Campaign_Ideal_Budgets,
                Campaign_Inst_Budgets={},
                Campaign_Inst_Ideal_Budgets=Inst_Ideal_Budgets,
                save_as_manuscript_figure=True,
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
                                    grid_name=grid_name,do_instantan=True)
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
            #Inst_Budget_plots.sonde_divergence_error_bar(
            #    save_as_manuscript_figure=True)            
#        Flight_Moisture_CONV=Moist_Convergence(
#                        cmpgn_cls,flight,self.cfg_file,
#                        grid_name=self.grid_name,do_instantan=False)    
#            Flight_Sectors,Flight_Ideal_Sectors,cmpgn_cls=\
#                    Flight_Moisture_CONV.load_moisture_convergence_single_case()            

    elif figure_to_create.startswith("fig_supplements"):
        import interpdata_plotting
        #---------------------------------------------------------------------#
        #%% Vertical distribution of moisture & wind speed cross-section contour
        calc_hmp=False
        calc_hmc=True
        do_plotting=False
        synthetic_campaign=True
        ar_of_day=["AR_internal"]
        #campaign_name="Second_Synthetic_Study"##"HALO_AC3"
        #campaign_name="North_Atlantic_Run"#"Second_Synthetic_Study"
    
        NA_flights_to_analyse={"SRF02":"20180224",#,#,
                               "SRF04":"20190319",#}#,#,
                               "SRF07":"20200416",#}#,#,#}#,#}#,
                               "SRF08":"20200419"}
        #Second Synthetic Study
        SND_flights_to_analyse={"SRF02":"20110317",
                                "SRF03":"20110423",#,
                                "SRF08":"20150314",#,
                                "SRF09":"20160311",#,
                                "SRF12":"20180225"}
        use_era=True
        use_carra=True
        use_icon=False
        na_flights=[*NA_flights_to_analyse.keys()]
        snd_flights=[*SND_flights_to_analyse.keys()]
        
        do_instantaneous=True
        #interpdata_plotting.ar_cross_sections_overview_flights_vertical_profile(
        #    flight_dates,use_era,use_carra,use_icon,na_flights,snd_flights,
        #    do_meshing=False)
        
        
        #%%
        """
        #plt.plot(mean_absolute_relative_error.mean(axis=0))        
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
            
            #else:
            #    axes2.set_yticks()
            p+=1
        
        sns.despine(offset=10)             
        supplement_path=self.cmpgn_cls.plot_path+\
                "/../../../Synthetic_AR_Paper/Manuscript/Supplements/"
        fig_name="Mean_div_errors_per_flight.png"
        err_fig.savefig(supplement_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", supplement_path+fig_name)
        """
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
                                                 save_as_manuscript_figure=False)
        
if __name__=="__main__":
    # Figures to create choices:
    #figure_to_create="fig11_single_case_sector_profiles"
    figure_to_create="fig12_campaign_divergence_overviews"
    #figure_to_create="fig14_divergence_instantan_errorbars"
    #figure_to_create="fig15_campaign_divergence_overview_instantan_comparison"
    
    #figure_to_create="fig_supplements_sonde_pos_comparison"
    #figure_to_create="fig12_campaign_divergence_overviews"
    main(figure_to_create=figure_to_create)

"""
if do_instantan:
    Flight_Moisture_CONV=Budgets.Moisture_Convergence(cmpgn_cls,
            original_flight,config_file,grid_name=grid_name,
            do_instantan=False)    
    Flight_Sectors,Flight_Ideal_Sectors,cmpgn_cls=\
        Flight_Moisture_CONV.load_moisture_convergence_single_case()
    import matplotlib.pyplot as plt
    import seaborn
    # Scatter plot
    stationary_comparison_fig=plt.figure(figsize=(18,6))
    ax1=stationary_comparison_fig.add_subplot(131)
    ax1.scatter(Ideal_Sectors["warm_sector"]["CONV"].values,
            Flight_Ideal_Sectors["warm_sector"]["CONV"].values,
            c="blue")
    ax1.scatter(Ideal_Sectors["warm_sector"]["ADV"].values,
            Flight_Ideal_Sectors["warm_sector"]["ADV"].values,
            c="orange")
    
    ax1.plot([-0.1,0.1],[-0.1,0.1],lw=3,ls="--",color="k")
    ax1.set_xlim([-2.5e-4,2.5e-4])
    ax1.set_ylim([-2.5e-4,2.5e-4])
    ax1.set_xticks([-2.5e-4,0,2.5e-4])
    ax1.set_yticks([-2.5e-4,0,2.5e-4])
    ax1.set_yticklabels(["-2.5e-4","0","2.5e-4"])
    ax1.set_xticklabels(["-2.5e-4","0","2.5e-4"])
    
    ax1.set_title("Warm Sector")
    ax2=stationary_comparison_fig.add_subplot(132)
    ax2.scatter(Ideal_Sectors["core"]["CONV"].values,
            Flight_Ideal_Sectors["core"]["CONV"].values,
            c="blue")
    ax2.scatter(Ideal_Sectors["core"]["ADV"].values,
            Flight_Ideal_Sectors["core"]["ADV"].values,
            c="orange")
    
    ax2.plot([-5,5],[-5,5],lw=3,ls="--",color="k")
    ax2.set_xlim([-5,5])
    ax2.set_ylim([-5,5])
    ax2.set_xlim([-2.5e-4,2.5e-4])
    ax2.set_ylim([-2.5e-4,2.5e-4])
    ax2.set_xticks([-2.5e-4,0,2.5e-4])
    ax2.set_yticks([-2.5e-4,0,2.5e-4])
    ax2.set_yticklabels(["-2.5e-4","0","2.5e-4"])
    ax2.set_xticklabels(["-2.5e-4","0","2.5e-4"])
    
    ax2.set_title("Core")
    
    ax3=stationary_comparison_fig.add_subplot(133)
    ax3.scatter(Ideal_Sectors["cold_sector"]["CONV"].values,
            Flight_Ideal_Sectors["cold_sector"]["CONV"].values,
            c="blue")
    ax3.scatter(Ideal_Sectors["cold_sector"]["ADV"].values,
            Flight_Ideal_Sectors["cold_sector"]["ADV"].values,
            c="orange")
    
    ax3.plot([-5,5],[-5,5],lw=3,ls="--",color="k")
    ax3.set_xlim([-5,5])
    ax3.set_ylim([-5,5])
    ax3.set_xlim([-2.5e-4,2.5e-4])
    ax3.set_ylim([-2.5e-4,2.5e-4])
    ax3.set_xticks([-2.5e-4,0,2.5e-4])
    ax3.set_yticks([-2.5e-4,0,2.5e-4])
    ax3.set_yticklabels(["-2.5e-4","0","2.5e-4"])
    ax3.set_xticklabels(["-2.5e-4","0","2.5e-4"])
    
    ax3.set_title("Cold Sector")
    seaborn.despine(offset=20)
    plt.subplots_adjust(wspace=0.8)
    #sys.exit()
    ### Overviews
    conv_limits=Budget_plots.plot_div_term_instantan_comparison(
        flight_dates,div_var="CONV")
    Budget_plots.plot_div_term_instantan_comparison(flight_dates,div_var="ADV",
                                                    limit_min_max=conv_limits)    
    
                #hmp_inflow["IVT_max_distance"]/1000,hmp_inflow[ivt_var_arg],
            #             color="lightblue",lw=8,label="Total AR (in): TIVT=")
            #line_core_in=plot_ax.plot(inflow_core["IVT_max_distance"]/1000,
            #                  inflow_core[ivt_var_arg],lw=2,color="darkblue",
            #                  label="AR core (in): TIVT="+\
            #                              str((TIVT_inflow_core/1e6).round(1)))
    
            #plot_ax.plot(hmp_outflow["IVT_max_distance"]/1000,
            #     hmp_outflow[ivt_var_arg],color="orange",lw=8)
    
            #line_core_out=plot_ax.plot(outflow_core["IVT_max_distance"]/1000,
            #                   outflow_core[ivt_var_arg],
            #                   lw=2,color="darkred",
            #                   label="AR core (out): TIVT="+\
            #                       str((TIVT_outflow_core/1e6).round(1)))#

        
            #plot_ax.plot(ar_inflow_warm_sector["IVT_max_distance"]/1000,
            #     ar_inflow_warm_sector[ivt_var_arg],
            #     lw=3,ls=":",color="darkblue")
        
            #plot_ax.plot(ar_inflow_cold_sector["IVT_max_distance"]/1000,
            #     ar_inflow_cold_sector[ivt_var_arg],
            #     lw=3,ls="-.",color="darkblue")
    
            #plot_ax.plot(ar_outflow_warm_sector["IVT_max_distance"]/1000,
             #    ar_outflow_warm_sector[ivt_var_arg],
            #     lw=3,ls=":",color="darkred")
            #plot_ax.plot(ar_outflow_cold_sector["IVT_max_distance"]/1000,
            #     ar_outflow_cold_sector[ivt_var_arg],
            #     lw=3,ls="-.",color="darkred")
            #plot_ax.set_title(flight,fontsize=16,loc="left",y=0.9)
            #plot_ax.set_xlim([-500,500])
            #plot_ax.set_ylim([100,650])
            #for axis in ["left","bottom"]:
            #    plot_ax.spines[axis].set_linewidth(2)
            #    plot_ax.tick_params(length=6,width=2)#

            #from matplotlib.patches import Patch
            #from matplotlib.lines import Line2D

            #legend_patches = [Patch(facecolor='darkblue', edgecolor='k',
            #                        label='TIVT (in)='+\
            #                        str((TIVT_inflow_total/1e6).round(1))+\
            #                        "$\cdot 1\mathrm{e}6\,\mathrm{kgs}^{-1}$"),
            #                  Patch(facecolor='darkred', edgecolor='k',
            #                        label='TIVT (out)='+\
            #                        str((TIVT_outflow_total/1e6).round(1))+\
            #                        "$\cdot 1\mathrm{e}6\,\mathrm{kgs}^{-1}$")]
            #legend_loc="upper right"
            #if hmp_inflow[ivt_var_arg].max()>450:
            #    legend_loc="lower right"
            #line_core_in[0],line_core_out[0],
            #lgd = plot_ax.legend(handles=[\
            #                          legend_patches[0],legend_patches[1]],
            #                 loc=legend_loc,fontsize=10,ncol=1)
                
            #i+=1
        #fig_name=grid_name+"_AR_TIVT_cases_overview.pdf"
        #plot_path=cmpn_cls.plot_path
        #f.savefig(plot_path+fig_name,
        #            dpi=60,bbox_inches="tight")
        #print("Figure saved as:", cmpn_cls.plot_path+fig_name)
        
#Start plotting
#Budget_Plots.moi
"""
