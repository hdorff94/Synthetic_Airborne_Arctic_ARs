# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:17:14 2022

@author: u300737
"""
import numpy as np
import os
import sys
    
#%% Predefining all paths to take scripts and data from and where to store
def importer():
    paths_dict={}
    
    paths_dict["actual_working_path"]=os.getcwd()+"/../"
    os.chdir(paths_dict["actual_working_path"]+"/config/")
    import init_paths
    import data_config

    paths_dict["working_path"]=init_paths.main()
        
    paths_dict["airborne_data_importer_path"]       =\
            paths_dict["working_path"]+"/Work/GIT_Repository/"
    paths_dict["airborne_script_module_path"]       =\
            paths_dict["actual_working_path"]+"/scripts/"
    paths_dict["airborne_processing_module_path"]   =\
        paths_dict["actual_working_path"]+"/src/"
    paths_dict["airborne_plotting_module_path"]     =\
        paths_dict["actual_working_path"]+"/plotting/"
    paths_dict["manuscript_path"]                   =\
        paths_dict["working_path"]+\
            "Work/Synthetic_AR_Paper/Manuscript/Paper_Plots/"
    paths_dict["scripts_path"]                      =\
        paths_dict["actual_working_path"]+"/major_scripts/"
                            
    os.chdir(paths_dict["airborne_processing_module_path"])
    sys.path.insert(1,paths_dict["airborne_script_module_path"])
    sys.path.insert(2,paths_dict["airborne_processing_module_path"])
    sys.path.insert(3,paths_dict["airborne_plotting_module_path"])
    sys.path.insert(4,paths_dict["airborne_data_importer_path"])
    sys.path.insert(5,paths_dict["scripts_path"])
    return paths_dict

def main(save_in_manuscript_path=False,figure_to_create="fig01"):
    """
    This routine creates overview plots for the ARs depicted (nine cases).
    
    Two different manuscript multiplots can be provided (Fig01, AR IVT maps,
                                    and Fig10, moisture transport contours)
    
    Parameters
    ----------
    save_in_manuscript_path : boolean, optional
        Specifies if the figure should be saved as ready for the manuscript.
        The default is False.
    
    figure_to_create : str, optional
        String that specifies with figure to be created. Two opportunities as
        specified above. The default is "fig01". Other possible is "fig10".

    Raises
    ------
    Exception
        if another Figure name than "fig01" or "fig10" are given.

    Returns
    -------
    None.

    """
    
    paths_dict=importer()
    #%% Define the flight campaign classes
    import flightcampaign
    if "data_config" in sys.modules:
        import data_config
    config_file=data_config.load_config_file(
                    paths_dict["airborne_data_importer_path"],
                        "data_config_file")
    
    

    na_run=flightcampaign.North_Atlantic_February_Run(
                    is_flight_campaign=True,
                    major_path=config_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=["SRF02","SRF04",
                                                        "SRF07","SRF08"],
                    instruments=["radar","radiometer","sonde"])
    snd_run=flightcampaign.Second_Synthetic_Study(
                    is_flight_campaign=True,
                    major_path=config_file["Data_Paths"]["campaign_path"],
                    aircraft="HALO",interested_flights=["SRF02","SRF03",
                                        "SRF08","SRF09","SRF12"],
                    instruments=["radar","radiometer","sonde"])
    
    #%% Get the flight data    
    import flight_track_creator
    Flight_Tracker=flight_track_creator.Flighttracker(
                                                na_run,"SRF02","AR_internal",
                                                track_type="internal",
                                                shifted_lat=0,
                                                shifted_lon=0)
                
    flight_dict={"20110317":[snd_run,"SRF02"],
                 "20160311":[snd_run,"SRF09"],
                 "20190319":[na_run,"SRF04"],
                 "20110423":[snd_run,"SRF03"],
                 "20180224":[na_run,"SRF02"],
                 "20200416":[na_run,"SRF07"],
                 "20150314":[snd_run,"SRF08"],
                 "20180225":[snd_run,"SRF12"],
                 "20200419":[na_run,"SRF08"]
                 }
    
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
    
    flight_tracks_dict=Flight_Tracker.get_all_synthetic_flights(flight_dict)
    if figure_to_create.startswith("fig01"):    
        #%% Plot the map
        import matplotlib
        import matplotlib.patches as patches
        matplotlib.rcParams.update({"font.size":16})
        import matplotlib.pyplot as plt
        import cartopy
        import cartopy.crs as ccrs
        try:
            from typhon.plots import styles
        except:
            print("Typhon module cannot be imported")
        
        from reanalysis import ERA5        
        # Define the plot specifications for the given variables
        met_var_dict={}
        met_var_dict["ERA_name"]    = {"IWV":"tcwv","IVT":"IVT",
                                       "IVT_u":"IVT_u","IVT_v":"IVT_v"}
        met_var_dict["colormap"]    = {"IWV":"density","IVT":"ocean_r",
                                       "IVT_v":"speed","IVT_u":"speed"}
        met_var_dict["levels"]      = {"IWV":np.linspace(10,25,101),
                                       "IVT":np.linspace(50,500,101),
                                       "IVT_v":np.linspace(0,500,101),
                                       "IVT_u":np.linspace(0,500,101)}
        met_var_dict["units"]       = {"IWV":"(kg$\mathrm{m}^{-2}$)",
                                "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                "IVT_v":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                "IVT_u":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
                
        
        col_no=3
        row_no=3
        projection=ccrs.AzimuthalEquidistant(central_longitude=-5.0,
                                             central_latitude=70)
        fig,axs=plt.subplots(row_no,col_no,sharex=True,sharey=True,
                             figsize=(12,16),
                             subplot_kw={'projection': projection})
        key=0
        era_index_dict={"20110317":15,
                     "20160311":17,
                     "20190319":19,
                     "20110423":19,
                     "20180224":17,
                     "20200416":9,
                     "20150314":19,
                     "20180225":10,
                     "20200419":8
                     }
        ar_label={"20110317":"AR1","20160311":"AR4",
                  "20190319":"AR7","20110423":"AR2",
                  "20180224":"AR5","20200416":"AR8",
                  "20150314":"AR3","20180225":"AR6",
                  "20200419":"AR9"}
        meteo_var="IVT"
        pressure_color="purple"                 #"royalblue"
        sea_ice_colors=["gold","saddlebrown"]   #["mediumslateblue", "indigo"]
        plot_labels=["(a)","(d)","(g)",
                     "(b)","(e)","(h)",
                     "(c)","(f)","(i)"]
        for col in range(col_no):
            for row in range(row_no):
               
               flight_date= [*flight_tracks_dict.keys()][key]
               print("Flight date",flight_date)
               
               cmpgn_cls=[*flight_dict.values()][key][0]
               flight=[*flight_dict.values()][key][1]
               
               ##### Load ERA5-data
               era5=ERA5(for_flight_campaign=True,campaign=cmpgn_cls.name,
                          research_flights=flight,
                          era_path=cmpgn_cls.campaign_path+"/data/ERA-5/")
               
               hydrometeor_lvls_path=cmpgn_cls.campaign_path+"/data/ERA-5/"
               file_name="total_columns_"+flight_date[0:4]+"_"+\
                           flight_date[4:6]+"_"+\
                           flight_date[6:8]+".nc"    
                
               era_ds,era_path=era5.load_era5_data(file_name)
               era_index=era_index_dict[flight_date] 
               era_ds["IVT_v"]=era_ds["p72.162"]
               era_ds["IVT_u"]=era_ds["p71.162"]
               era_ds["IVT"]=np.sqrt(era_ds["IVT_u"]**2+era_ds["IVT_v"]**2)
               
               # Plot IVT
               C1=axs[row,col].contourf(era_ds["longitude"],era_ds["latitude"],
                    era_ds[met_var_dict["ERA_name"][meteo_var]][era_index,:,:],
                    levels=met_var_dict["levels"][meteo_var],extend="max",
                    transform=ccrs.PlateCarree(),
                    cmap=met_var_dict["colormap"][meteo_var],alpha=0.95)
               
               # Plot surface presure
               C_p=axs[row,col].contour(era_ds["longitude"],era_ds["latitude"],
                    era_ds["msl"][era_index,:,:]/100,
                    levels=np.linspace(950,1050,11),linestyles="-.",
                    linewidths=2,colors="grey",transform=ccrs.PlateCarree())
               axs[row,col].clabel(C_p, inline=1, fmt='%03d hPa',fontsize=12)
               
               # mean sea ice cover
               C_i=axs[row,col].contour(era_ds["longitude"],era_ds["latitude"],
                        era_ds["siconc"][era_index,:,:]*100,levels=[15,85],
                        linestyles="-",linewidths=[1.5,2],
                        colors=sea_ice_colors,
                        transform=ccrs.PlateCarree())
               axs[row,col].clabel(C_i, inline=1, fmt='%02d %%',fontsize=14)
                    
                           
               halo_df=flight_tracks_dict[flight_date]
               axs[row,col].coastlines(resolution="50m",zorder=0)
               axs[row,col].set_extent([-20,25,60,90])
               
               # Date and Timestep
               axs[row,col].text(-12, 62, str(flight_date)+" "+\
                                 str(era_index)+" UTC",
                    fontsize=15,transform=ccrs.PlateCarree(),color="darkblue",
                    bbox=dict(facecolor='lightgrey',edgecolor="black"))
               
               axs[row,col].plot(halo_df["longitude"],halo_df["latitude"],
                        color="white",lw=4,transform=ccrs.PlateCarree())
               
               axs[row,col].plot(halo_df["longitude"],halo_df["latitude"],
                        color="indianred",lw=2,transform=ccrs.PlateCarree())
               
               # AR label (AR1)
               axs[row,col].text(-60,82,ar_label[flight_date],
                                 transform=ccrs.PlateCarree(),
                                 color="k",bbox=dict(facecolor="lightgrey",
                                                edgecolor="black"),zorder=10)
               
               axs[row,col].text(-76,83,plot_labels[key],
                                 transform=ccrs.PlateCarree(),
                                 color="k",fontsize=14,zorder=10)
               
               step=20
               quiver_lon=np.array(era_ds["longitude"][::step])
               quiver_lat=np.array(era_ds["latitude"][::step])
               u=era_ds["IVT_u"][era_index,::step,::step]
               v=era_ds["IVT_v"][era_index,::step,::step]
               v=v.where(v>200)
               v=np.array(v)
               u=np.array(u)
               quiver=axs[row,col].quiver(quiver_lon,quiver_lat,
                                          u,v,color="white",
                                          edgecolor="k",lw=1,
                                          scale=800,scale_units="inches",
                                              pivot="mid",width=0.015,
                                              transform=ccrs.PlateCarree())
               if key==8:
                   x=[-16,21,23,-16.75]
                   y=[63.5, 61.4, 63.2, 65.6]
                   q_typ=600.0
                   qk=axs[row, col].quiverkey(quiver,0.4,0.13,q_typ,
                    label=str(q_typ)+' $\mathrm{kgm}^{-1}\mathrm{s}^{-1}$',
                    coordinates="axes",labelpos="E",fontproperties={"size":12},
                    zorder=10)
                   qk.text.set_zorder(50)
                   qk.Q.set_zorder(50)#
                   
                   rect = axs[row,col].add_patch(
                           patches.Polygon(xy=list(zip(x,y)),fill=True,
                           linewidth=1, edgecolor='k', facecolor='lightgrey',
                           transform=ccrs.Geodetic(),zorder=1,alpha=0.8))
                   # Create a Rectangle patch
                   
               key+=1
        
        # Adjust the location of the subplots on the page
        # to make room for the colorbar
        fig.subplots_adjust(bottom=0.15,top=0.9, left=0.15, right=0.9,
                            wspace=0.05, hspace=0.05)
        # Add a colorbar axis at the bottom of the graph
        cbar_ax = fig.add_axes([0.905, 0.225, 0.02, 0.6])
        cbar=fig.colorbar(C1, cax=cbar_ax,extend="max")
        cbar_ax.text(3.2,0.375,meteo_var+" "+met_var_dict["units"][meteo_var],
                     rotation=90,fontsize=18,transform=cbar_ax.transAxes)
        cbar.set_ticks([50,250,500])
        if not save_in_manuscript_path:
            fig_path=paths_dict["airborne_plotting_module_path"]
        else:
            fig_path=paths_dict["manuscript_path"]
        fig_name="fig01_AR_cases_overview.png"
        fig.savefig(fig_path+fig_name,dpi=600,bbox_inches="tight")
        print("Figure saved as:",fig_path+fig_name)
        
        return None
    
    elif figure_to_create.startswith("fig10"):
        ar_of_day=["AR_internal"]
        
        NA_flights_to_analyse={"SRF02":"20180224",
                               "SRF04":"20190319",
                               "SRF07":"20200416",
                               "SRF08":"20200419"}
        #Second Synthetic Study
        SND_flights_to_analyse={"SRF02":"20110317",
                                "SRF03":"20110423",
                                "SRF08":"20150314",
                                "SRF09":"20160311",
                                "SRF12":"20180225"}
        use_era=True
        use_carra=True
        use_icon=False
        na_flights=[*NA_flights_to_analyse.keys()]
        snd_flights=[*SND_flights_to_analyse.keys()]
        
        do_instantaneous=True
        
        import interpdata_plotting
        interpdata_plotting.\
            ar_cross_sections_overview_flights_vertical_profile(
                flight_dates,use_era,use_carra,use_icon,
                na_flights,snd_flights,do_meshing=False)
    else:
        raise Exception("You have given the wrong figure name.",
                        " No figure created")
if __name__=="__main__":
    main(save_in_manuscript_path=True,figure_to_create="fig10_")