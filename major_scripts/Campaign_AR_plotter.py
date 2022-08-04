# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 14:09:33 2021

@author: u300737
"""
import numpy as np
import os
import pandas as pd

import data_config
import Flight_Campaign
import run_grid_data_on_halo
from AR import Atmospheric_Rivers

def plot_IVT_shapes(init_campaign_cls,hmp_dict,config_file,
                    show_AR_detection=False,
                    meteo_var="IVT", with_cross_sections=False):
    # Define the plot specifications for the given variables
    met_var_dict={}
    met_var_dict["ERA_name"]    = {"IWV":"tcwv","IVT":"IVT",
                                       "IVT_u":"IVT_u","IVT_v":"IVT_v"}
    met_var_dict["colormap"]    = {"IWV":"density","IVT":"speed",
                                       "IVT_v":"speed",
                                       "IVT_u":"speed"}
    met_var_dict["levels"]      = {"IWV":np.linspace(0,50,101),
                                   "IVT":np.linspace(50,600,101),
                                   "IVT_v":np.linspace(0,500,101),
                                   "IVT_u":np.linspace(0,500,101)}
    met_var_dict["units"]       = {"IWV":"(kg$\mathrm{m}^{-2}$)",
                                   "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                   "IVT_v":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                   "IVT_u":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
    import matplotlib    
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    from matplotlib import gridspec
        
    ivt_fig=plt.figure(figsize=(16,9))
    if not with_cross_sections:
        gs=gridspec.GridSpec(1,1)
    else:
        gs=gridspec.GridSpec(1,2,width_ratios=[2,1])
    
    ax1 = plt.subplot(gs[0],projection=ccrs.AzimuthalEquidistant(
                                central_longitude=20.0,central_latitude=70))
    ax1.set_extent([-25,90,55,90])
    #ax1.set_extent([-70,80,40,90])
        
    ax1.coastlines(resolution="50m")
    #gl1=ax1.gridlines(draw_labels=True,dms=True,
    #                      x_inline=False,y_inline=False)
    if with_cross_sections:    
        ax2=plt.subplot(gs[1])
    import seaborn as sns
            
    if show_AR_detection:    
        import AR
    
        AR=AR.Atmospheric_Rivers("ERA")
        AR_era_ds=AR.open_AR_catalogue()
    i=len(hmp_dict.keys())   
    handles = []
    levels=[0,5000]
    for flight in hmp_dict.keys():
        if not flight.startswith("S"):
            campaign_cls=Flight_Campaign.NAWDEX(is_flight_campaign=True,
                major_path=config_file["Data_Paths"]["campaign_path"],
                aircraft="HALO",instruments=["radar","radiometer","sonde"])
            campaign_cls.dropsonde_path=campaign_cls.major_path+\
                                    campaign_cls.name+"/data/HALO-Dropsonden/"
        else:
            campaign_cls=init_campaign_cls
    
        flight_colors={"RF10":"salmon","SRF02":"purple","SRF03":"darkgreen",
                   "SRF04":"lightpink","SRF05":"orange","SRF06":"skyblue",
                   "SRF07":"navy","SRF08":"teal"}   
        cmap_color={"RF10":"Reds","SRF02":"Purples","SRF03":"Greens",
                   "SRF04":"PuRd","SRF05":"Oranges","SRF06":"Blues",
                   "SRF07":"PuBu","SRF08":"BuGn"}
        flight_date=campaign_cls.years[flight]+"-"+\
                    campaign_cls.flight_month[flight]+"-"+\
                    campaign_cls.flight_day[flight]
        if show_AR_detection: 
            if flight=="SRF03" or flight=="SRF06" \
                or flight=="SRF07" or flight=="RF10":
                AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)
                time_step=AR_era_data["model_runs"].start+2
                if flight=="SRF03":
                   time_step=time_step-1 
                #if flight=="SRF02":
                    #   time_step=time_step+1 
                AR_pattern=AR_era_ds.kidmap[0,time_step,0,:,:]
                AR_pattern_cline=AR_pattern.fillna(0)
                AR_pattern_cline=AR_pattern_cline.where(AR_pattern_cline<1.0,1)
                AR1=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                     AR_pattern,alpha=0.1,
                #     linecolors=flight_colors[flight],linewidths=3,
                     labels=flight,cmap=plt.get_cmap("Greys"),levels=2,
                     transform=ccrs.PlateCarree(),zorder=i)
                ax1.contour(AR_era_ds.lon,AR_era_ds.lat,
                     AR_pattern_cline,colors=flight_colors[flight],
                     linewidths=1,levels=[0,0.25],linestyles="-.",
                     labels=flight,zorder=i,transform=ccrs.PlateCarree())
            
            i-=1
        
            handles.append(mpatches.Patch(color=flight_colors[flight],
                                          label="{:s}".format(flight)))

        for ar in hmp_dict[flight].keys():
            halo_ar_era5=hmp_dict[flight][ar]
            if flight=="RF10":
                campaign_cls.specify_flights_of_interest(flight)
                real_halo_df,campaign_path=campaign_cls.load_aircraft_position()
                ax1.plot(real_halo_df["longitude"],real_halo_df["latitude"],
                         ls="--",lw=1,color="darkgrey",
                         transform=ccrs.PlateCarree())
                
            C1=ax1.scatter(halo_ar_era5["Halo_Lon"],halo_ar_era5["Halo_Lat"],
                       c=halo_ar_era5["Interp_IVT"],cmap=cmap_color[flight],s=5,
                       vmin=50, vmax=600,transform=ccrs.PlateCarree())
            #if i==0:
            #    cb=ivt_fig.colorbar(C1,ax=ax1,shrink=0.95,extend="both")
            #    cb.set_label(meteo_var+" "+met_var_dict["units"][meteo_var])
            #    if meteo_var=="IWV":
             #       cb.set_ticks([0,10,20,30,40,50])
             #   elif meteo_var=="IVT":
             #           cb.set_ticks([50,200,400,600])
             #   else:
             #       pass

            #plot Dropsonde releases
            date=campaign_cls.year+campaign_cls.flight_month[flight]
            date=date+campaign_cls.flight_day[flight]
            if not campaign_cls.is_synthetic_campaign:
                if not flight=="RF06":                           
                     Dropsondes=campaign_cls.load_dropsonde_data(
                                                    date,print_arg="yes",
                                                    dt="all",plotting="no")
                     print("Dropsondes loaded")
                     # in some cases the Dropsondes variable can be a dataframe or
                     # just a series, if only one sonde has been released
                     if isinstance(Dropsondes["Lat"],pd.DataFrame):
                         dropsonde_releases=pd.DataFrame(index=\
                                pd.DatetimeIndex(Dropsondes["LTS"].index))
                         dropsonde_releases["Lat"]=Dropsondes["Lat"].loc[\
                                                        :,"6000.0"].values
                         dropsonde_releases["Lon"]=Dropsondes["Lon"].loc[\
                                                        :,"6000.0"].values
    
                else:
                         index_var=Dropsondes["Time"].loc["6000.0"]
                         dropsonde_releases=pd.Series()
                         dropsonde_releases["Lat"]=np.array(\
                                        Dropsondes["Lat"].loc["6000.0"])
                         dropsonde_releases["Lon"]=np.array(\
                                        Dropsondes["Lon"].loc["6000.0"])
                         dropsonde_releases["Time"]=index_var
    
    
            try:
                dropsonde_releases["Lon"]=dropsonde_releases["Lon"].loc[\
                                                halo_ar_era5.index[0]:\
                                                halo_ar_era5.index[-1]]
                dropsonde_releases["Lat"]=dropsonde_releases["Lat"].loc[\
                                                halo_ar_era5.index[0]:\
                                                halo_ar_era5.index[-1]]
                
                ax1.scatter(dropsonde_releases["Lon"],
                            dropsonde_releases["Lat"]+0.1,
                            s=50,marker="^",color=flight_colors[flight],
                            edgecolors="black",
                            transform=ccrs.PlateCarree())
            except:
                         pass
            if with_cross_sections:     
                ax2.plot(hmp_dict[flight][ar]["IVT_max_distance"]/1000,
                         hmp_dict[flight][ar]["Interp_IVT"],
                         ls="-",lw=2,color=flight_colors[flight])
                ax2.set_xlabel("Centered AR distance (km)")
                ax2.set_xlim([-1500,1500])
                ax2.set_ylim([50,650])
                sns.despine(ax=ax2,offset=10)
    
        ###
    mappable = plt.cm.ScalarMappable(cmap='gray_r')
    # the mappable usually contains an array of data, here we can
    # use that to set the limits
    mappable.set_array([50,600])   
    
    cb=ivt_fig.colorbar(mappable,ax=ax1,orientation="horizontal",
                        shrink=0.7,extend="both")
    
    cb.set_label("IVT in $\mathrm{kgm}^{-1}\mathrm{s}^{-1}$")
    cb.set_ticks([50,200,400,600])
    
    ax1.legend(handles=handles,loc="upper right",fontsize=12)    
    #Save figure
    fig_name=campaign_cls.name+"_AR_IVT_cross_sections_ERA5.png"
    #plt.suptitle(campaign_cls.name)
    fig_path=os.getcwd()+"/"+campaign_cls.name+"/plots/"
    if not os.path.exists(fig_path):
                os.makedirs(fig_path)
    print("Figure saved as:",fig_path+fig_name)
    plt.subplots_adjust(hspace=0.6)
    #plt.close()
    ivt_fig.savefig(fig_path+fig_name,bbox_inches="tight",dpi=300)
    return None

def plot_ivt_in_outflow(cmpgn_cls,hmp_dict,halo_dict,
                        config_file,high_res=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    from matplotlib import gridspec
    import matplotlib
    flight_colors={"RF10":"salmon","SRF02":"purple","SRF03":"darkgreen",
                   "SRF04":"lightpink","SRF05":"orange","SRF06":"skyblue",
                   "SRF07":"navy","SRF08":"teal"}   
            
    fig=plt.figure(figsize=(12,7))
    matplotlib.rcParams.update({"font.size":20})
    ax1=fig.add_subplot(111)
    variable="Interp_IVT"
    if high_res:
        variable="highres_Interp_IVT"
    for flight in hmp_dict.keys():
        if high_res:
            if flight=="SRF06":
                continue
        halo_inflow=halo_dict[flight]["inflow"]
        halo_inflow.index=pd.DatetimeIndex(halo_inflow.index)
        halo_outflow=halo_dict[flight]["outflow"]
        halo_outflow.index=pd.DatetimeIndex(halo_outflow.index)
        
        ivt_inflow=hmp_dict[flight]["AR_internal"].loc[halo_inflow.index]
        ivt_inflow.index=pd.DatetimeIndex(ivt_inflow.index)
        
        ivt_outflow=hmp_dict[flight]["AR_internal"].loc[halo_outflow.index]
        ivt_outflow.index=pd.DatetimeIndex(ivt_outflow.index)
        
        ivt_inflow=cmpgn_cls.calc_distance_to_IVT_max(halo_inflow,ivt_inflow)
        ivt_outflow=cmpgn_cls.calc_distance_to_IVT_max(halo_outflow,ivt_outflow)
        
        ax1.plot(ivt_inflow["IVT_max_distance"]/1000,
                 ivt_inflow[variable],ls="-",lw=3,
                 color=flight_colors[flight],label=flight,zorder=0)
        ax1.plot(ivt_outflow["IVT_max_distance"]/1000,
                 ivt_outflow[variable],ls="--",lw=2,
                 color=flight_colors[flight],zorder=0)
        if flight=="RF10":
           import metpy.calc as mpcalc 
           date="20161013"
           ar_of_day="AR3"
           nawdex=Flight_Campaign.NAWDEX(is_flight_campaign=True,
                major_path=config_file["Data_Paths"]["campaign_path"],
                aircraft="HALO",instruments=["radar","radiometer","sonde"])
        
           nawdex.specify_flights_of_interest(flight)
           nawdex.dropsonde_path=nawdex.major_path+nawdex.name+\
                                   "/data/HALO-Dropsonden/"
        
           Dropsondes=nawdex.load_dropsonde_data(date,print_arg="yes",
                                          dt="all",plotting="no")
           Dropsondes["q"]=mpcalc.specific_humidity_from_mixing_ratio(
                                            Dropsondes["MR"])
           Dropsondes["q"].columns=Dropsondes["Wspeed"].columns
                
           Dropsondes=nawdex.calculate_dropsonde_ivt(Dropsondes,
                                                  date,ar_of_day,flight)
                 
           sonde_ivt=Dropsondes["IVT"]
           sonde_ivt=sonde_ivt.to_frame(name="IVT")
           sonde_ivt["IVT_max_distance"]=ivt_inflow["IVT_max_distance"].reindex(\
                                            sonde_ivt.index).dropna()/1000
        
           ax1.scatter(sonde_ivt["IVT_max_distance"],
                       sonde_ivt["IVT"],s=200,marker="^",
                       color=flight_colors[flight],
                       edgecolors="black",zorder=1)

    ax1.set_xlabel("Centered AR distance (km)")
    for axis in ["left","bottom"]:
        ax1.spines[axis].set_linewidth(3)
        ax1.tick_params(length=10,width=3)
    
    ax1.set_xlim([-750,750])
    ax1.set_ylim([50,650])
    ax1.legend(loc="upper right")
    ax1.set_ylabel("IVT (kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)")
    plot_path=cmpgn_cls.plot_path
    fig_name="AR_IVT_cases_cross_section.pdf"
    if high_res:
        fig_name="High_res_"+fig_name
    print("Figure saved as:",plot_path+fig_name)
    sns.despine(ax=ax1,offset=10)
    fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")

    

    
###############################################################################
#### Main plotter    
def main(campaign="North_Atlantic_Run",flights=["RF10","SRF02","SRF03","SRF04","SRF05","SRF06"],
         do_daily_plots=True,calc_hmp=True,calc_hmc=True,
         era_is_desired=True,carra_is_desired=False,icon_is_desired=False,
         do_instantaneous=False):

    #campaign="North_Atlantic_Run"#"North_Atlantic_Run"

    do_plots=do_daily_plots
    if (campaign=="North_Atlantic_Run") or (campaign=="Second_Synthetic_Study"):
        synthetic_campaign=True
        synthetic_flight=True
    else:
        synthetic_campaign=False
        synthetic_flight=False

    # Load config file
    config_file=data_config.load_config_file(os.getcwd(),"data_config_file")
    
    if campaign=="NAWDEX":
        cpgn_cls_name="NAWDEX"
        NAWDEX=Flight_Campaign.NAWDEX(is_flight_campaign=True,
                    major_path=config_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",instruments=["radar","radiometer","sonde"])
        NAWDEX.dropsonde_path=NAWDEX.major_path+NAWDEX.name+"/data/HALO-Dropsonden/"
        cmpgn_cls=NAWDEX
    elif campaign=="North_Atlantic_Run":
        cpgn_cls_name="NA_February_Run"
        na_run=Flight_Campaign.North_Atlantic_February_Run(
                                        is_flight_campaign=True,
                                        major_path=config_file["Data_Paths"]\
                                                    ["campaign_path"],
                                        aircraft="HALO",
                                        interested_flights=flights,
                                        instruments=["radar","radiometer","sonde"])
        cmpgn_cls=na_run
    elif campaign=="Second_Synthetic_Study":
        cpgn_cls_name="Second_Synthetic_Study"
        na_run=Flight_Campaign.Second_Synthetic_Study(
            is_flight_campaign=True,major_path=config_file["Data_Paths"]["campaign_path"],
            aircraft="HALO",interested_flights=flights,
            instruments=["radar","radiometer","sonde"])
        cmpgn_cls=na_run               
    HMCs={}
    HMPs={}
    HALO_dict_dict={}
    AR_radar=pd.DataFrame()
    i=0
    for flight in flights:
        
        HMCs[flight]={}
        HMPs[flight]={}
    
        for ar_of_day in ["AR_internal"]:
            if not flight.startswith("S"):
                cpgn_cls_name="NAWDEX"
                #ar_of_day="AR3"
                synthetic_campaign=False
            else:
                if cpgn_cls_name=="NAWDEX":
                    print("Campaign name has to be changed")
                    cpgn_cls_name="NA_February_Run"
                if synthetic_campaign==False:
                    synthetic_campaign=True
            if calc_hmp:
                #try:
                    HMPs[flight][ar_of_day],ar_rf_radar,HALO_dict_dict[flight]=\
                                run_grid_data_on_halo.main(
                                    campaign=cpgn_cls_name,
                                    hmp_plotting_desired=calc_hmp,
                                    hmc_plotting_desired=calc_hmc,
                                    plot_data=do_plots,
                                    ar_of_day=ar_of_day,flight=[flight],
                                    era_is_desired=era_is_desired,
                                    carra_is_desired=carra_is_desired,
                                    icon_is_desired=icon_is_desired,
                                    synthetic_campaign=synthetic_campaign,
                                    synthetic_flight=synthetic_flight,
                                    do_instantaneous=do_instantaneous)
                #except:
                #    pass
            if calc_hmc:
                #try:
                    HMCs[flight][ar_of_day],ar_rf_radar,HALO_dict_dict[flight]=\
                        run_grid_data_on_halo.main(
                                    campaign=cpgn_cls_name,
                                    hmp_plotting_desired=False,
                                    hmc_plotting_desired=calc_hmc,
                                    plot_data=do_plots,
                                    ar_of_day=ar_of_day,flight=[flight],
                                    era_is_desired=era_is_desired,
                                    carra_is_desired=carra_is_desired,
                                    icon_is_desired=icon_is_desired,
                                    synthetic_campaign=synthetic_campaign,
                                    synthetic_flight=synthetic_flight,
                                    do_instantaneous=do_instantaneous)
                #except:
                #    pass
        
            #if 'Reflectivity' in ar_rf_radar.keys():
            #    if i==0:
            #        AR_radar=ar_rf_radar["Reflectivity"]
            #    else:
            #        AR_radar=pd.concat([AR_radar,ar_rf_radar["Reflectivity"]],
            #                       ignore_index=True)
            i+=1
    if calc_hmp:
        print("DATASET NAME:",HMPs[flight][ar_of_day].name)
        return HMPs,HALO_dict_dict,cmpgn_cls
    if calc_hmc:
        print("DATASET NAME:",HMCs[flight][ar_of_day]["name"])
        return HMCs,HALO_dict_dict,cmpgn_cls

    #    plot_IVT_shapes(cmpgn_cls,HMPs,config_file,show_AR_detection=True)
    #    plot_ivt_in_outflow(cmpgn_cls,HMPs,HALO_dict_dict)
    #    run_plot_IVT_long_term_stats(cmpgn_cls, HMPs)    

if __name__=="__main__":
    # This part runs the main funciton meaning all the stuff from run_grid_on_halo
    # as well as plots and then (!) additionally is used for testing of 
    # IVT variability handling that will be runned in the ipnyb later on in order 
    # to not confuse people too much. 
    
    # Load config file
    config_file=data_config.load_config_file(os.getcwd(),"data_config_file")
    
    # Relevant specifications for running , those are default values
    calc_hmp=False
    calc_hmc=True
    do_plotting=True
    flights_to_analyse={#"SRF02":"20180224",#,#,
                        #"SRF04":"20190319",#}#,#,
                        #"SRF07":"20200416",#}#,#,#}#,#}#,
                        #"SRF08":"20200419"#,}
        #Second Synthetic Study
        
        #"SRF02":"20110317",
        #"SRF03":"20110423",#,
            #"SRF06":"20140325",#,                    
                        #"SRF07":"20150307"}#,
        
        "SRF08":"20150314",#,
        #"SRF09":"20160311",#,
        #"SRF12":"20180225"
        }        
    campaign_name="Second_Synthetic_Study"#"North_Atlantic_Run"##
    use_era=True
    use_carra=True
    use_icon=False
    flights=[*flights_to_analyse.keys()]
    do_instantaneous=False

    Hydrometeors,HALO_Dict,cmpgn_cls=main(campaign=campaign_name,flights=flights,
                                          era_is_desired=use_era, 
                                          icon_is_desired=use_icon,
                                          carra_is_desired=use_carra,
                                          do_daily_plots=do_plotting,
                                          calc_hmp=calc_hmp,calc_hmc=calc_hmc,
                                          do_instantaneous=do_instantaneous)
    if do_instantaneous:
        import sys
        sys.exit()
    #%%
    ### IVT climatology
    
    #run_plot_IVT_long_term_stats(cmpgn_cls, Hydrometeors,flights_to_analyse)    
    
    ###
    #%% IVT Variability
    flight_leg_to_take="inflow" # this can be either "inflow" or "outflow" or "both"
    single_flight_to_use=[*flights_to_analyse.keys()][0]
    analysed_flight=single_flight_to_use
    analysed_flight_df=HALO_Dict[analysed_flight][flight_leg_to_take]
    major_data_pah=config_file["Data_Paths"]\
                 ["campaign_path"]
    # sounding analysis
    # Define number of sondes per cross-section
    sonde_no=6
    # Logging
    import IVT_Variability_handler as IVT_handler
    
    
    log_file_name="logging_ivt_variability_icon.log"
    ivt_logger=IVT_handler.ICON_IVT_Logger(log_file_path=os.getcwd(),
                                            file_name=log_file_name)
    ivt_logger.create_plot_logging_file()
    sounding_frequency="standard"    
    # Dropsondes
    if sounding_frequency=="standard":
        sounding_name="Sounding"
        ivt_logger.icon_ivt_logger.info(
                        "Consider Standard Sounding")
    elif sounding_frequency=="Upsampled":
        sounding_name="Upsampled_Sounding"
        ivt_logger.icon_ivt_logger.info(
                    "Consider Upsampled Sounding")
    else: 
        sounding_name="Sounding"
        ivt_logger.icon_ivt_logger.info(
                    "Consider Standard Sounding")
    print(Hydrometeors)
    if not calc_hmc:
        grid_dict_hmc=None
        grid_dict_hmp=Hydrometeors[single_flight_to_use]["AR_internal"]
    if not calc_hmp:
        grid_dict_hmp=None
        grid_dict_hmc=Hydrometeors[single_flight_to_use]["AR_internal"]
    
    import IVT_Variability_handler as IVT_handler
    
    # Get sondes
    # Flight tracks where no sondes have been launched use synthetic soundings
    # So far, they are spaced equidistantly along cross-section
    sonde_dict={}
    sonde_dict["Pres"]=pd.DataFrame()
    sonde_dict["q"]=pd.DataFrame()
    sonde_dict["Wspeed"]=pd.DataFrame()
    sonde_dict["IVT"]=pd.DataFrame()
    
    ### here a new function should be added that creates 
    ### synthetic soundings at certain intervals along 
    ### HALO AR cross-sections
    #sonde_dict["IVT"]
    if analysed_flight.startswith("S"):
        only_model_sounding_profiles=True
        ar_of_day="AR_internal"
    else:
        only_model_sounding_profiles=False

    
    ## apparently need to be defined 
    # plot_path, ar_of_day,flight,
    if calc_hmc:
        if grid_dict_hmc["name"]=="ERA5":
            grid_dict_hmc["p"]=pd.DataFrame(data=np.tile(
                np.array(grid_dict_hmc["IWC"].columns[:].astype(float)),
                (grid_dict_hmc["IWC"].shape[0],1)),
                columns=[grid_dict_hmc["IWC"].columns[:]],
                index=grid_dict_hmc["IWC"].index)
        elif grid_dict_hmc["name"]=="CARRA":
            grid_dict_hmc["p"]=pd.DataFrame(data=np.tile(
                np.array(grid_dict_hmc["u"].columns[:].astype(float)),
                (grid_dict_hmc["u"].shape[0],1)),
                columns=[grid_dict_hmc["u"].columns[:]],
                index=grid_dict_hmc["u"].index)
    
    sonde_aircraft_df,sonde_dict=IVT_handler.create_synthetic_sondes(
                                        analysed_flight_df,
                            Hydrometeors[single_flight_to_use]["AR_internal"],
                            sonde_dict,hmps_used=calc_hmp,no_of_sondes=sonde_no)
    
    # Open classes
    # IVT Processing
    IVT_handler_cls=IVT_handler.IVT_variability(grid_dict_hmp,
                                    grid_dict_hmc,sonde_dict,
                                    only_model_sounding_profiles,
                                    sounding_frequency,
                                    HALO_Dict[single_flight_to_use]["inflow"],
                                    cmpgn_cls.plot_path,ar_of_day,
                                    analysed_flight,ivt_logger)
    # Plot routine
    IVT_var_Plotter=IVT_handler.IVT_Variability_Plotter(grid_dict_hmp,
                                    grid_dict_hmc,sonde_dict,
                                    only_model_sounding_profiles,sounding_frequency,
                                    HALO_Dict[single_flight_to_use]["inflow"],
                                    cmpgn_cls.plot_path,ar_of_day,
                                    analysed_flight,ivt_logger)

        #%% IVT Analysis
    if calc_hmc:
#        IVT_handler_cls.calc_vertical_quantiles(do_all_preps=True)
#        IVT_handler_cls.calc_vertical_quantiles(use_grid=False,do_all_preps=True)
        # Vertical moisture transport variability
        IVT_var_Plotter.plot_IVT_vertical_variability()    
    
    if calc_hmp:
        if analysed_flight.startswith("S"):
            synthetic_flight=True
        else:
            synthetic_flight=False
        
        IVT_var_Plotter.plot_model_sounding_frequency_comparison(
                            name_of_grid_data=grid_dict_hmp.name)
        IVT_var_Plotter.plot_distance_based_IVT(False,
                            synthetic_flight=synthetic_flight,delete_sondes=None,
                            name_of_grid_data=grid_dict_hmp.name)
        IVT_var_Plotter.plot_distance_based_IVT(False,
                            synthetic_flight=synthetic_flight,delete_sondes=None,
                            name_of_grid_data=grid_dict_hmp.name,show_sondes=False)
        IVT_handler_cls.study_TIVT_sondes_grid_frequency_dependency()
        
        if use_era:
            IVT_handler_cls.TIVT["grid_name"]="ERA5"
        if use_carra:
            IVT_handler_cls.TIVT["grid_name"]="CARRA"
        if use_icon:
            IVT_handler_cls.TIVT["grid_name"]="ICON_2km"
        IVT_var_Plotter.TIVT=IVT_handler_cls.TIVT
        IVT_var_Plotter.plot_TIVT_error_study()
        
        #if calc_hmp:
        #    plot_ivt_in_outflow(cmpgn_cls,Hydrometeors,Halo_dict,config_file,high_res=True)
    

        #    cfad_ar_df=NAWDEX.calculate_cfad_radar_reflectivity(AR_radar)
        #    NAWDEX.plot_cfad_2d_hist(cfad_ar_df,os.getcwd()+"/NAWDEX/plots/",True)