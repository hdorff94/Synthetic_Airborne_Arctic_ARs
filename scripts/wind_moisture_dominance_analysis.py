# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 07:40:25 2022

@author: u300737
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xarray as xr

#import matplotlib
#import matplotlib.pyplot as plt
#import seaborn as sns
# Change path to working script directory
def importer(current_path=os.getcwd()):
    paths_dict={}
    print(current_path)
    major_path = os.path.abspath("../../../")
    base_working_path=major_path+"/my_GIT/Synthetic_Airborne_Arctic_ARs/"
    aircraft_base_path=major_path+"/Work/GIT_Repository/"
    working_path  = base_working_path+"/src/"
    config_path   = base_working_path+"/config/"
    plotting_path = base_working_path+"/plotting/"
    plot_figures_path = aircraft_base_path+"/../Synthetic_AR_Paper/Manuscript/Paper_Plots/"
    sys.path.insert(1, os.path.join(sys.path[0], working_path))
    sys.path.insert(2, os.path.join(sys.path[0], config_path))
    sys.path.insert(3, os.path.join(sys.path[0], plotting_path))
    
    paths_dict["major_path"]            = major_path
    paths_dict["base_working_path"]     = base_working_path
    paths_dict["aircraft_base_path"]    = aircraft_base_path
    paths_dict["working_path"]          = working_path
    paths_dict["config_path"]           = config_path
    paths_dict["plotting_path"]         = plotting_path
    paths_dict["plot_figures_path"]     = plot_figures_path
    
    import flight_track_creator
    import data_config
    # Config File
    config_file=data_config.load_config_file(
                	paths_dict["aircraft_base_path"],"data_config_file")
    
    
    return paths_dict,config_file


def prepare_data(paths_dict,config_file,flight_dates,reanalysis_to_use,
                 shifted_lat=0,shifted_lon=0,ar_of_day="SAR_internal"):
    import flightcampaign as Campaign
    import flight_track_creator
    #import atmospheric_rivers
    import gridonhalo
    from reanalysis import ERA5,CARRA
    
    moisture_transport_flights_dict={}
    
    merged_profiles={}
    merged_profiles["name"]=reanalysis_to_use
    
    i=0
    for campaign in [*flight_dates.keys()]:
        for flight in [*flight_dates[campaign]]:
            date=flight_dates[campaign][flight]
            print(date)
            moisture_transport_flights_dict[date]={}
            if campaign=="Second_Synthetic_Study":
                cmpgn_cls=Campaign.Second_Synthetic_Study(
                             is_flight_campaign=True,
                             major_path=config_file["Data_Paths"]["campaign_path"],
                             aircraft="HALO",interested_flights=[flight],
                             instruments=["radar","radiometer","sonde"])               
            else:
                cmpgn_cls=Campaign.North_Atlantic_February_Run(
                    is_flight_campaign=True,
                    major_path=config_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=[flight],
                    instruments=["radar","radiometer","sonde"])
            # Allocate classes for campaign
            cmpgn_cls.specify_flights_of_interest(flight[0])
            #na_run.create_directory(directory_types=["data"])
            #cmpgn_cls=na_run
        
            ###################################################################
            # for flight
            Flight_Tracker=flight_track_creator.Flighttracker(cmpgn_cls,flight,
                                ar_of_day,track_type="internal",
                                shifted_lat=0,shifted_lon=0,
                                load_save_instantan=False)
            track_df,campaign_path=Flight_Tracker.get_synthetic_flight_track(
                as_dict=True)
            inflow_df=track_df["inflow"]
            ################################################################################
            # for reanalysis (CARRA)
            hydrometeor_lvls_path=cmpgn_cls.campaign_path+"/data/"+\
                                    reanalysis_to_use+"/"
            
            if reanalysis_to_use=="ERA-5":
                hydrometeor_lvls_file="hydrometeors_pressure_levels_"+date+".nc"
                if ar_of_day is not None:
                    interpolated_iwc_file="Synthetic_"+flight+"_"+ar_of_day+\
                                            "_IWC_"+date+".csv"        
                ##### Load ERA5-data
                Reanalysis=ERA5(for_flight_campaign=True,
                                campaign=cmpgn_cls.name,
                                research_flights=flight,
                                era_path=cmpgn_cls.campaign_path+"/data/ERA-5/")
                Reanalysis_on_HALO=gridonhalo.ERA_on_HALO(
                                inflow_df,hydrometeor_lvls_path,
                                hydrometeor_lvls_file,interpolated_iwc_file,
                                True,campaign,
                                config_file["Data_Paths"]["campaign_path"],
                                [flight],date,config_file,ar_of_day=ar_of_day,
                                synthetic_flight=True,
                                do_instantaneous=False)
                open_hwc_fct=Reanalysis_on_HALO.load_hwc
       
            elif reanalysis_to_use=="CARRA":
                hydrometeor_lvls_path=cmpgn_cls.campaign_path+"/data/CARRA/"
                #print(carra_lvls_path)    
                Reanalysis=CARRA(for_flight_campaign=True,
                            campaign=campaign,research_flights=None,
                            carra_path=hydrometeor_lvls_path) 
                Reanalysis_on_HALO=gridonhalo.CARRA_on_HALO(inflow_df,
                                    hydrometeor_lvls_path,True,campaign,
                                    config_file["Data_Paths"]["campaign_path"],
                                    [flight],flight_dates[campaign][flight],
                                    config_file,ar_of_day=ar_of_day,
                                    synthetic_flight=True,do_instantaneous=False)
                open_hwc_fct=\
                    Reanalysis_on_HALO.load_or_calc_interpolated_hmc_data
            else:
                raise Exception("Your reanalysis name is wrong.")
            inflow_df=inflow_df.groupby(level=0).first()
            # Open aircraft interpolated hwc from reanalysis
            halo_reanalysis_hmc=open_hwc_fct()
            store_halo_reanalysis_hmc=halo_reanalysis_hmc.copy()
            for key in halo_reanalysis_hmc.keys():
                halo_reanalysis_hmc[key]=halo_reanalysis_hmc[key].groupby(level=0).first()
                halo_reanalysis_hmc[key]=halo_reanalysis_hmc[key].reindex(
                   inflow_df.index)
            #sys.exit()
            if not "wind" in halo_reanalysis_hmc.keys():
                halo_reanalysis_hmc["wind"]=np.sqrt(halo_reanalysis_hmc["u"]**2+\
                                                    halo_reanalysis_hmc["v"]**2)
            halo_reanalysis_hmc["q"]*=1000
            halo_reanalysis_hmc["transport"]=halo_reanalysis_hmc["q"]*\
                                                halo_reanalysis_hmc["wind"]
            profile_stats=pd.DataFrame(index=halo_reanalysis_hmc["q"].columns,
                           columns=["transport_mean","transport_25",
                                    "transport_75","wind_mean","wind_std",
                                    "wind_25","wind_75",
                                    "q_mean","q_std","q_25","q_75","qv_corr"])

            
            # Get the profile stats
            profile_stats["q_mean"]=halo_reanalysis_hmc["q"].mean(axis=0)
            profile_stats["q_std"]=halo_reanalysis_hmc["q"].std(axis=0)
            profile_stats["q_25"]=halo_reanalysis_hmc["q"].quantile(0.25,axis=0)                        
            profile_stats["q_75"]=halo_reanalysis_hmc["q"].quantile(0.75,axis=0)
            
            profile_stats["wind_mean"]=halo_reanalysis_hmc["wind"].mean(axis=0)
            profile_stats["wind_std"]=halo_reanalysis_hmc["wind"].std(axis=0)
            profile_stats["wind_25"]=halo_reanalysis_hmc["wind"].quantile(0.25,
                                                                          axis=0)
            profile_stats["wind_75"]=halo_reanalysis_hmc["wind"].quantile(0.75,
                                                                          axis=0)
            
            profile_stats["transport_mean"]=halo_reanalysis_hmc["transport"].mean(axis=0)
            profile_stats["transport_25"]=halo_reanalysis_hmc["transport"].quantile(
                                                0.25,axis=0)
            profile_stats["transport_75"]=halo_reanalysis_hmc["transport"].quantile(0.75,
                                                                          axis=0)
            profile_stats["qv_corr"]=halo_reanalysis_hmc["q"].corrwith(
                                        halo_reanalysis_hmc["wind"],axis=0)
            pres_index=pd.Series(halo_reanalysis_hmc["q"].columns.astype(float)*100)
            g=9.81
            #iwv_temporary=-1/g*np.trapz(q_loc,axis=0,x=pres_index)
            #ivt_u_temporary=-1/g*np.trapz(qu,axis=0,x=pres_index)
            #ivt_v_temporary=-1/g*np.trapz(qv,axis=0,x=pres_index)
            ivt_mean=1/g*np.trapz(profile_stats["transport_mean"],
                                   x=pres_index)/1000
            profile_stats["q_v_dash"]=profile_stats["q_mean"]*(profile_stats["wind_mean"]+\
                                              profile_stats["wind_std"])
            profile_stats["q_dash_v"]=profile_stats["wind_mean"]*(profile_stats["q_mean"]+\
                                                 profile_stats["q_std"])
            
            #ivt_q_v_dash=1/g*np.trapz(profile_stats["q_v_dash"],
            #                          x=pres_index)/1000
            
            #ivt_q_dash_v=1/g*np.trapz(profile_stats["q_dash_v"],
            #                          x=pres_index)/1000
            if i==0:
                merged_profiles["q"]=halo_reanalysis_hmc["q"].copy()
                merged_profiles["wind"]=halo_reanalysis_hmc["wind"].copy()
                merged_profiles["transport"]=halo_reanalysis_hmc["transport"].copy()
                merged_profiles["Geopot_Z"]=halo_reanalysis_hmc["Geopot_Z"].copy()
            else:
                merged_profiles["q"]=merged_profiles["q"].append(
                                        halo_reanalysis_hmc["q"].copy())
                merged_profiles["wind"]=merged_profiles["wind"].append(
                                        halo_reanalysis_hmc["wind"].copy())
                merged_profiles["transport"]=merged_profiles["transport"].append(
                                            halo_reanalysis_hmc["transport"].copy())
                merged_profiles["Geopot_Z"]=merged_profiles["Geopot_Z"].append(
                                            halo_reanalysis_hmc["Geopot_Z"].copy())
            moisture_transport_flights_dict[date]["stats"]=profile_stats 
            moisture_transport_flights_dict[date]["pres_index"]=pres_index
            
            moisture_transport_flights_dict[date]["ivt_q_v_dash"]=\
                1/g*np.trapz(profile_stats["q_v_dash"],x=pres_index)/1000
            moisture_transport_flights_dict[date]["ivt_q_dash_v"]=\
                1/g*np.trapz(profile_stats["q_dash_v"],x=pres_index)/1000
            moisture_transport_flights_dict[date]["ivt_mean"]=ivt_mean
            i+=1
    moisture_transport_flights_dict = dict(sorted(
                                    moisture_transport_flights_dict.items()))
    extra_output={}
    extra_output["inflow"]=inflow_df
    extra_output["cmpgn_cls"]=cmpgn_cls
    return merged_profiles,profile_stats,moisture_transport_flights_dict,extra_output
    
def create_fig10_q_v_vertical_variability(paths_dict,config_file,flight_dates,
                                          reanalysis_to_use):
    #campaigns=[*flight_dates.keys()]
    merged_profiles,profile_stats,moisture_transport_flights_dict,extra_output=\
        prepare_data(paths_dict,config_file,flight_dates,reanalysis_to_use)
    
    import ivtvariability as IVT_handler
    from ivtvariability import IVT_variability
    
    #%%
    log_file_name="logging_ivt_variability_icon.log"
    ivt_logger=IVT_handler.ICON_IVT_Logger(log_file_path=os.getcwd(),
                                            file_name=log_file_name)
    ivt_logger.create_plot_logging_file()

    merged_profiles["moist_transport"]=merged_profiles["transport"].copy()
    IVT_handler_cls=IVT_handler.IVT_variability(None,
                                    merged_profiles,None,
                                    False,None,extra_output["inflow"],
                                    extra_output["cmpgn_cls"].plot_path,
                                    "SAR_internal",
                                    "all",ivt_logger)
        
    IVT_handler_cls.calc_vertical_quantiles(use_grid=True,quantiles=["50","75",
                                                              "90","97","100"],
                                    do_all_preps=True)
    # Plot routine
    IVT_var_Plotter=IVT_handler.IVT_Variability_Plotter(None,merged_profiles,None,
                                    False,None,extra_output["inflow"],
                                    extra_output["cmpgn_cls"].plot_path,
                                    "SAR_internal","all",ivt_logger)

    IVT_var_Plotter.plot_IVT_vertical_variability(subsample_day="2011-03-17",
            save_figure=True,undefault_path=paths_dict["plot_figures_path"])

def create_fig11_q_v_flavor(paths_dict,config_file,flight_dates,
                                          reanalysis_to_use):
    
    import matplotlib
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.rcParams.update({"font.size":15})
    g=9.82#campaigns=[*flight_dates.keys()]
    merged_profiles,profile_stats,moisture_transport_flights_dict,extra_output=\
        prepare_data(paths_dict,config_file,flight_dates,reanalysis_to_use)
    
    #Figure 11 of Manuscript
    fig,axs=plt.subplots(3,3,figsize=(12,16),sharey=True,sharex=True)
    axes=axs.flatten()
    d=0
    pres_index=moisture_transport_flights_dict[\
                [*moisture_transport_flights_dict.keys()][-1]]["pres_index"]
    for date in moisture_transport_flights_dict.keys():
        relative_v_dash=moisture_transport_flights_dict[date]["ivt_q_v_dash"]/\
                        moisture_transport_flights_dict[date]["ivt_mean"]
        relative_q_dash=moisture_transport_flights_dict[date]["ivt_q_dash_v"]/\
                        moisture_transport_flights_dict[date]["ivt_mean"]
        axes[d].plot(1/g*moisture_transport_flights_dict[date]["stats"][\
                        "transport_mean"],pres_index/100,color="black",lw=2,ls="--")
        axes[d].fill_betweenx(x1=1/g*moisture_transport_flights_dict[date]["stats"][\
                                                                "transport_25"],
                              x2=1/g*moisture_transport_flights_dict[date]["stats"][\
                                                                "transport_75"],
                              y=pres_index/100,color="lightgrey")
        
        axes[d].plot(1/g*moisture_transport_flights_dict[date]["stats"]["q_dash_v"],
                     pres_index/100,color="blue",lw=2,label="NIVTq'="+\
                         str(round(relative_q_dash,2)))
        axes[d].plot(1/g*moisture_transport_flights_dict[date]["stats"]["q_v_dash"],
                     pres_index/100,color="magenta",lw=2,label="NIVTv'="+\
                         str(round(relative_v_dash,2))) 
        axes[d].text(0.7,0.9,transform=axes[d].transAxes,s=date,color="black",
                  fontsize=14)
        #axes[d].set_yticklabels([])
        axes[d].invert_yaxis()
        axes[d].set_ylim([1000,200])
        axes[d].set_xlim([0,15])
        axes[d].semilogy()
        axes[d].set_yticks(np.log10([1000,850,700,500,300]))
        axes[d].set_xticks([0,5,10,15])
        if d>=6:
            axes[d].set_xlabel("Moisture Transport (kg/s)",fontsize=16)
        if d%3==0:
            axes[d].set_ylabel("Pressure (hPa)",fontsize=16)
        axes[d].legend(loc="center right")
        d+=1#
            
    sns.despine(offset=10)
    fig_name="Fig11_IVT_Q_V_Variability.pdf"
    fig.savefig(paths_dict["plot_figures_path"]+fig_name,dpi=200,
                bbox_inches="tight")
    print("Figure saved as:", paths_dict["plot_figures_path"]+fig_name)                           

def create_pre_fig11_q_v_flavor(paths_dict,config_file,
                                flight_dates,reanalysis_to_use):
    
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.rcParams.update({"font.size":15})
    g=9.82#campaigns=[*flight_dates.keys()]
    merged_profiles,profile_stats,moisture_transport_flights_dict,extra_output=\
        prepare_data(paths_dict,config_file,flight_dates,reanalysis_to_use)
    
    #Figure 11 of Manuscript
    fig,axs=plt.subplots(3,3,figsize=(12,16),sharey=True,sharex=True)
    axes=axs.flatten()
    d=0
    pres_index=moisture_transport_flights_dict[\
                [*moisture_transport_flights_dict.keys()][-1]]["pres_index"]
    for date in moisture_transport_flights_dict.keys():
        relative_v_std=moisture_transport_flights_dict[date]["stats"]["wind_std"]/\
                        moisture_transport_flights_dict[date]["stats"]["wind_mean"]
        relative_q_std=moisture_transport_flights_dict[date]["stats"]["q_std"]/\
                        moisture_transport_flights_dict[date]["stats"]["q_mean"]
        
        var_factor=moisture_transport_flights_dict[date]["stats"]["qv_corr"]*\
                        relative_v_std*relative_q_std
        axes[d].plot(var_factor,pres_index/100,color="black",lw=1,ls="--")
        axes[d].text(-.2,220,s="$\overline{IVT}=$"+\
                str(int(moisture_transport_flights_dict[date]["ivt_mean"]))+\
                        "$\mathrm{kgm}^{-1}\mathrm{s}^{-1}$",fontsize=9,
                        bbox=dict(facecolor='lightgrey', edgecolor='k',
                                  boxstyle='round,pad=0.3'))
        axes[d].scatter(var_factor,pres_index/100,
                            s=moisture_transport_flights_dict[date]\
                                ["stats"]["transport_mean"]/\
                                    moisture_transport_flights_dict[date]\
                                ["stats"]["transport_mean"].max()*100,color="k")
        #axes[d].plot(relative_q_std,pres_index/100,color="blue",ls="--",lw=1)
        #axes[d].scatter(relative_q_std,pres_index/100,
        #                s=moisture_transport_flights_dict[date]\
        #                    ["stats"]["q_mean"]/\
        #                     moisture_transport_flights_dict[date]\
        #                    ["stats"]["q_mean"].max()*100,color="blue")
        #
        # 
        #axes[d].plot(relative_v_std,pres_index/100,color="magenta",ls="--",lw=1)
        #axes[d].scatter(relative_v_std,pres_index/100,
        #                s=moisture_transport_flights_dict[date]\
        #                    ["stats"]["wind_mean"]/\
        #                     moisture_transport_flights_dict[date]\
        #                    ["stats"]["wind_mean"].max()*100,color="magenta")
            
        axes[d].text(-0.2,700,s=date,color="grey",fontsize=10)
        axes[d].axvline(0,ls="--",color="grey",lw=3)
        #axes[d].legend(loc="upper left")
        #axes[d].set_yticklabels([])
        axes[d].invert_yaxis()
        axes[d].set_ylim([1000,200])
        axes[d].set_xlim([-0.2,0.2])
        axes[d].semilogy()
        if d==7:
            axes[d].set_xlabel("$corr(q,v) \cdot s_{v} \cdot s_{q}$")
        #axes[d].set_yticks(np.log10([1000,850,700,500,300]))
        axes[d].set_xticks([-0.2,-0.1,0,0.1,0.2])
        for axis in ["left","bottom"]:
                axes[d].spines[axis].set_linewidth(2)
                axes[d].tick_params(length=6,width=2)#

        #if d>=6:
        #    axes[d].set_xlabel("Moisture Transport (kg/s)",fontsize=16)
        #if d%3==0:
        #    axes[d].set_ylabel("Pressure (hPa)",fontsize=16)
        #axes[d].legend(loc="upper left",fontsize=10)                                                        
        d+=1
    
    sns.despine(offset=10)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    custom_lines = [#Line2D([0], [0], color="k", lw=1,ls="--",label="corr"),
                #Line2D([0], [0], color="b", lw=1,ls="--",label="$s_{q}$"),
                #Line2D([0], [0], color="magenta", lw=1,ls="--",label="$s_{v}$"),
                Line2D([0],[0],marker="o",color="w",markeredgecolor="k",
                       markerfacecolor="lightgrey",
                       markersize=11,label="Max value")]

    fig.legend(handles=custom_lines,loc="top",ncol=4,bbox_to_anchor=[0.5,0.93],
               bbox_transform=fig.transFigure)
    fig_name="Fig11pre_IVT_Q_V_Variability.pdf"
    fig.savefig(paths_dict["plot_figures_path"]+fig_name,dpi=200,
                bbox_inches="tight")
    print("Figure saved as:", paths_dict["plot_figures_path"]+fig_name)

def create_summarized_fig11_q_v_flavor(paths_dict,config_file,
                                       flight_dates,reanalysis_to_use):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.rcParams.update({"font.size":18})
    g=9.82#campaigns=[*flight_dates.keys()]
    merged_profiles,profile_stats,moisture_transport_flights_dict,extra_output=\
        prepare_data(paths_dict,config_file,
                     flight_dates,reanalysis_to_use)
    
    #Figure 11 of Manuscript
    cov_fig=plt.figure(figsize=(12,9))#
    ax1=cov_fig.add_subplot(111)
    pres_index=moisture_transport_flights_dict[\
                [*moisture_transport_flights_dict.keys()][-1]]["pres_index"]/100

    cov_df=pd.DataFrame(data=np.nan,index=moisture_transport_flights_dict.keys(),
                        columns=pres_index)
    corr_df=pd.DataFrame(data=np.nan,index=moisture_transport_flights_dict.keys(),
                         columns=pres_index)
    moisture_df=pd.DataFrame(data=np.nan,index=moisture_transport_flights_dict.keys(),
                        columns=pres_index)
    
    wind_df=pd.DataFrame(data=np.nan,index=moisture_transport_flights_dict.keys(),
                        columns=pres_index)
    
    for date in cov_df.index:
        relative_v_std=moisture_transport_flights_dict[date]["stats"]["wind_std"]/\
                        moisture_transport_flights_dict[date]["stats"]["wind_mean"]
        relative_q_std=moisture_transport_flights_dict[date]["stats"]["q_std"]/\
                        moisture_transport_flights_dict[date]["stats"]["q_mean"]
        
        cov_df.loc[date,:]=(moisture_transport_flights_dict[date]["stats"]["qv_corr"]*\
                            relative_v_std*relative_q_std).values
        corr_df.loc[date,:]=moisture_transport_flights_dict[date]["stats"]["qv_corr"].values
        moisture_df.loc[date,:]=relative_q_std.values
        wind_df.loc[date,:]=relative_v_std.values
        
        ax1.plot(cov_df.loc[date,:].values, cov_df.columns,
                         marker="o",markersize=2, ls="--", color="lightgrey")
        ax1.plot(wind_df.loc[date,:].values,wind_df.columns,
                 marker="v",markersize=2,ls="--",color="thistle")
        ax1.plot(moisture_df.loc[date,:].values,moisture_df.columns,
                 marker="s",markersize=2,ls="--",color="powderblue")
        #ax1.plot(corr_df.loc[date,:].values,corr_df.columns,
        #         marker="^",markersize=2,ls="--",color="peachpuff")
    mean_moisture_df=moisture_df.mean(axis=0)
    std_moisture_df=moisture_df.std(axis=0)
    mean_corr_df=corr_df.mean(axis=0)
    std_corr_df=corr_df.std(axis=0)
    mean_wind_df=wind_df.mean(axis=0)
    std_wind_df=wind_df.std(axis=0)
    mean_cov_df=cov_df.mean(axis=0)
    std_cov_df=cov_df.std(axis=0)
    print(mean_cov_df)
   
    # Variability term
    ax1.plot(mean_cov_df.values,
                mean_cov_df.index,
                marker="o",markersize=2,ls="-",lw=2,color="k",
                label="$cov \cdot s_{q} \cdot s_{v}$")
    ax1.scatter(mean_cov_df.values,mean_cov_df.index,
                 s=moisture_transport_flights_dict[date]["stats"]["transport_mean"]/\
                     moisture_transport_flights_dict[date]\
                        ["stats"]["transport_mean"].max()*100,
                        color="k",zorder=5)
    
    # Relative Wind variability
    ax1.plot(mean_wind_df.values,
            mean_wind_df.index,
            marker="o",markersize=2,ls="-",lw=2,color="purple",label="$s_{v}$")

    ax1.scatter(mean_wind_df.values,mean_wind_df.index,
                s=moisture_transport_flights_dict[date]["stats"]["wind_mean"]/\
                moisture_transport_flights_dict[date]\
                       ["stats"]["wind_mean"].max()*100,
                marker="v",
                color="purple",zorder=5)
#    # Relative moisture variability
    ax1.plot(mean_moisture_df.values,
            mean_moisture_df.index,
            marker="o",markersize=2,ls="-",lw=2,color="darkblue",
            label="$s_{q}$")
    ax1.scatter(mean_moisture_df.values,mean_moisture_df.index,
                s=moisture_transport_flights_dict[date]["stats"]["q_mean"]/\
                moisture_transport_flights_dict[date]\
                       ["stats"]["q_mean"].max()*100,
                        color="darkblue",zorder=5)
    # Correlation term
    #ax1.plot(mean_corr_df.values,
    #        mean_corr_df.index,
    #        marker="o",markersize=2,ls="-",lw=2,color="saddlebrown",
    #        label="$s_{q}$")
    
    #ax1.scatter(mean_corr_df.values,mean_corr_df.index,
    #            s=moisture_transport_flights_dict[date]["stats"]["qv_corr"]/\
    #            moisture_transport_flights_dict[date]\
    #                   ["stats"]["qv_corr"].max()*100,
    #                    color="saddlebrown",zorder=5)
    
    ax1.axvline(x=0,ls="-.",color="k",lw=2)
    ax1.invert_yaxis()
    ax1.set_ylabel("Pressure (hPa)")
    ax1.set_xlabel("Variability relative value")
    
    for axis in ["left","bottom"]:
        ax1.spines[axis].set_linewidth(2)
        ax1.tick_params(length=6,width=2)#

    sns.despine(offset=10)
    fig_name="Fig11summarized_IVT_Q_V_Variability.pdf"
    cov_fig.savefig(paths_dict["plot_figures_path"]+fig_name,dpi=200,
                bbox_inches="tight")
    print("Figure saved as:", paths_dict["plot_figures_path"]+fig_name)                           
    
    sys.exit()    
    
def create_updated_fig11_q_v_flavor(paths_dict,config_file,
                                    flight_dates,reanalysis_to_use):
    import matplotlib
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    matplotlib.rcParams.update({"font.size":15})
    g=9.82#campaigns=[*flight_dates.keys()]
    merged_profiles,profile_stats,moisture_transport_flights_dict,extra_output=\
        prepare_data(paths_dict,config_file,flight_dates,reanalysis_to_use)
    
    #Figure 11 of Manuscript
    fig,axs=plt.subplots(3,3,figsize=(12,16),sharey=True,sharex=True)
    axes=axs.flatten()
    d=0
    pres_index=moisture_transport_flights_dict[\
                [*moisture_transport_flights_dict.keys()][-1]]["pres_index"]
    for date in moisture_transport_flights_dict.keys():
        relative_v_std=moisture_transport_flights_dict[date]["stats"]["wind_std"]/\
                        moisture_transport_flights_dict[date]["stats"]["wind_mean"]
        relative_q_std=moisture_transport_flights_dict[date]["stats"]["q_std"]/\
                        moisture_transport_flights_dict[date]["stats"]["q_mean"]
        axes[d].plot(moisture_transport_flights_dict[date]["stats"]["qv_corr"],
            pres_index/100,color="black",lw=1,ls="--")
        axes[d].text(-.99,220,s="$\overline{IVT}=$"+\
                str(int(moisture_transport_flights_dict[date]["ivt_mean"]))+\
                        "$\mathrm{kgm}^{-1}\mathrm{s}^{-1}$",fontsize=9,
                        bbox=dict(facecolor='lightgrey', edgecolor='k',
                                  boxstyle='round,pad=0.3'))
        axes[d].scatter(moisture_transport_flights_dict[date]["stats"]["qv_corr"],
            pres_index/100,s=moisture_transport_flights_dict[date]\
                                ["stats"]["transport_mean"]/\
                                    moisture_transport_flights_dict[date]\
                                ["stats"]["transport_mean"].max()*100,color="k")
        axes[d].plot(relative_q_std,pres_index/100,color="blue",ls="--",lw=1)
        axes[d].scatter(relative_q_std,pres_index/100,
                        s=moisture_transport_flights_dict[date]\
                            ["stats"]["q_mean"]/\
                             moisture_transport_flights_dict[date]\
                            ["stats"]["q_mean"].max()*100,color="blue")
        
        
        axes[d].plot(relative_v_std,pres_index/100,color="magenta",ls="--",lw=1)
        axes[d].scatter(relative_v_std,pres_index/100,
                        s=moisture_transport_flights_dict[date]\
                            ["stats"]["wind_mean"]/\
                             moisture_transport_flights_dict[date]\
                            ["stats"]["wind_mean"].max()*100,color="magenta")
            
        axes[d].text(-0.9,700,s=date,color="grey",fontsize=10)
        axes[d].axvline(0,ls="--",color="grey",lw=3)
        #axes[d].legend(loc="upper left")
        #axes[d].set_yticklabels([])
        axes[d].invert_yaxis()
        axes[d].set_ylim([1000,200])
        axes[d].set_xlim([-1,1])
        axes[d].semilogy()
        #axes[d].set_yticks(np.log10([1000,850,700,500,300]))
        axes[d].set_xticks([-1.0,-0.5,0,0.5,1.0])
        for axis in ["left","bottom"]:
                axes[d].spines[axis].set_linewidth(2)
                axes[d].tick_params(length=6,width=2)#

        #if d>=6:
        #    axes[d].set_xlabel("Moisture Transport (kg/s)",fontsize=16)
        #if d%3==0:
        #    axes[d].set_ylabel("Pressure (hPa)",fontsize=16)
        #axes[d].legend(loc="upper left",fontsize=10)                                                        
        d+=1
    
    sns.despine(offset=10)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    custom_lines = [Line2D([0], [0], color="k", lw=1,ls="--",label="corr"),
                Line2D([0], [0], color="b", lw=1,ls="--",label="$s_{q}$"),
                Line2D([0], [0], color="magenta", lw=1,ls="--",label="$s_{v}$"),
                Line2D([0],[0],marker="o",color="w",markeredgecolor="k",
                       markerfacecolor="lightgrey",
                       markersize=12,label="Max value")]

    fig.legend(handles=custom_lines,loc="top",ncol=4,bbox_to_anchor=[0.75,0.93],
               bbox_transform=fig.transFigure)
    fig_name="Fig11updated_IVT_Q_V_Variability.pdf"
    fig.savefig(paths_dict["plot_figures_path"]+fig_name,dpi=200,
                bbox_inches="tight")
    print("Figure saved as:", paths_dict["plot_figures_path"]+fig_name)                           
    
def plotter(figures_to_create,flight_dates,reanalysis_to_use):
    paths_dict,config_file=importer()
    
    #import matplotlib.pyplot as plt
    #import seaborn as sns
    plot_fct_dict={"fig10":[create_fig10_q_v_vertical_variability],
                   "fig11":[create_fig11_q_v_flavor],
                   "updated_fig11":[create_summarized_fig11_q_v_flavor,
                                    create_pre_fig11_q_v_flavor,
                                    create_updated_fig11_q_v_flavor],
                   "both":[create_fig10_q_v_vertical_variability,
                           create_fig11_q_v_flavor]}
    plot_fct_list=plot_fct_dict[figures_to_create]
    for fct in plot_fct_list:
        fct(paths_dict,config_file,flight_dates,reanalysis_to_use)
         
#%%
def main(reanalysis_to_use="ERA-5",figures_to_create="updated_fig11"):
    #plot_fct_kwargs={"fig10":[paths_dict,config_file,
    #                          flight_dates,reanalysis_to_use],
    #                 "fig11":[],
    #                 "both":[]}
    
    analyse_all_flights=False
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
    
    flight_tracks_dict={}
    # Dummy definitions needed for initiating the campaign classes
    plotter(figures_to_create,flight_dates,reanalysis_to_use)    

if __name__=="__main__":
    main()