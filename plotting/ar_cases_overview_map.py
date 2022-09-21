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
        
    paths_dict["airborne_data_importer_path"]=\
            paths_dict["working_path"]+"/Work/GIT_Repository/"
    paths_dict["airborne_script_module_path"]=\
            paths_dict["actual_working_path"]+"/scripts/"
    paths_dict["airborne_processing_module_path"]=\
        paths_dict["actual_working_path"]+"/src/"
    paths_dict["airborne_plotting_module_path"]=\
        paths_dict["actual_working_path"]+"/plotting/"
    paths_dict["manuscript_path"]=paths_dict["working_path"]+\
        "Synthetic_AR_Paper/Manuscript/Paper_Plots/"
    os.chdir(paths_dict["airborne_processing_module_path"])
    sys.path.insert(1,paths_dict["airborne_script_module_path"])
    sys.path.insert(2,paths_dict["airborne_processing_module_path"])
    sys.path.insert(3,paths_dict["airborne_plotting_module_path"])
    sys.path.insert(4,paths_dict["airborne_data_importer_path"])
    return paths_dict

def main(save_in_manuscript_path=False):
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
                                        major_path=config_file["Data_Paths"]\
                                                    ["campaign_path"],
                                        aircraft="HALO",
                                        interested_flights=["SRF02","SRF04",
                                                            "SRF07","SRF08"],
                                        instruments=["radar","radiometer","sonde"])
    snd_run=flightcampaign.Second_Synthetic_Study(
                                        is_flight_campaign=True,
                                        major_path=config_file["Data_Paths"]\
                                            ["campaign_path"],aircraft="HALO",
                                        interested_flights=["SRF02","SRF03",
                                                            "SRF08","SRF09",
                                                            "SRF12"],
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
    flight_tracks_dict=Flight_Tracker.get_all_synthetic_flights(flight_dict)
    #%% Plot the map
    import matplotlib
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
    met_var_dict["colormap"]    = {"IWV":"density","IVT":"speed",
                                   "IVT_v":"speed","IVT_u":"speed"}
    met_var_dict["levels"]      = {"IWV":np.linspace(10,25,101),
                                   "IVT":np.linspace(50,600,101),
                                   "IVT_v":np.linspace(0,500,101),
                                   "IVT_u":np.linspace(0,500,101)}
    met_var_dict["units"]       = {"IWV":"(kg$\mathrm{m}^{-2}$)",
                                   "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                   "IVT_v":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                   "IVT_u":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
            
    
    col_no=3
    row_no=3
    projection=ccrs.AzimuthalEquidistant(central_longitude=-5.0,central_latitude=70)
    fig,axs=plt.subplots(row_no,col_no,sharex=True,sharey=True,figsize=(12,16),
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
    meteo_var="IVT"
    pressure_color="royalblue"
    sea_ice_colors=["mediumslateblue", "indigo"]
    
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
                                levels=np.linspace(950,1050,11),
                                linestyles="-.",linewidths=2,
                                colors="grey",transform=ccrs.PlateCarree())
           axs[row,col].clabel(C_p, inline=1, fmt='%03d hPa',fontsize=12)
           # mean sea ice cover
           C_i=axs[row,col].contour(era_ds["longitude"],era_ds["latitude"],
                                era_ds["siconc"][era_index,:,:]*100,levels=[15,85],
                                linestyles="-",linewidths=[1.5,2],
                                colors=sea_ice_colors,
                                transform=ccrs.PlateCarree())
           axs[row,col].clabel(C_i, inline=1, fmt='%02d %%',fontsize=14)
                
                
           #cb=map_fig.colorbar(C1,ax=ax)
           #     cb.set_label(meteo_var+" "+met_var_dict["units"][meteo_var])
           #     if meteo_var=="IWV":
           #         cb.set_ticks([10,15,20,25,30])
           #cb.set_ticks([50,100,200,300,400,500,600])
                       
           halo_df=flight_tracks_dict[flight_date]
           axs[row,col].coastlines(resolution="50m")
           axs[row,col].set_extent([-20,25,60,90])
           #axs[row,col].gridlines()
           axs[row,col].text(-12, 62, str(flight_date)+" "+str(era_index)+" UTC",
                             fontsize=15,transform=ccrs.PlateCarree(),
                             color="darkblue",bbox=dict(
                                 facecolor='lightgrey',edgecolor="black"))
           axs[row,col].plot(halo_df["longitude"],
                             halo_df["latitude"],
                             color="white",lw=4,
                             transform=ccrs.PlateCarree())
           axs[row,col].plot(halo_df["longitude"],
                             halo_df["latitude"],
                             color="blue",lw=2,
                             transform=ccrs.PlateCarree())
           
           step=20
           quiver_lon=np.array(era_ds["longitude"][::step])
           quiver_lat=np.array(era_ds["latitude"][::step])
           u=era_ds["IVT_u"][era_index,::step,::step]
           v=era_ds["IVT_v"][era_index,::step,::step]
           v=v.where(v>200)
           v=np.array(v)
           u=np.array(u)
           quiver=axs[row,col].quiver(quiver_lon,quiver_lat,u,v,color="white",
                             edgecolor="k",lw=1,
                             scale=800,scale_units="inches",
                                      pivot="mid",width=0.015,
                                      transform=ccrs.PlateCarree())
           key+=1
    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.9,
                        wspace=0.05, hspace=0.05)
    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.905, 0.225, 0.02, 0.6])
    cbar=fig.colorbar(C1, cax=cbar_ax,extend="max")
    cbar_ax.text(1.3,0.32,meteo_var+" "+met_var_dict["units"][meteo_var],
                 rotation=90,fontsize=22,transform=cbar_ax.transAxes)
    cbar.set_ticks([50,200,400,600])
    if not save_in_manuscript_path:
        fig_path=paths_dict["airborne_plotting_module_path"]
    else:
        fig_path=paths_dict["manuscript_path"]
    fig_name="Fig01_AR_cases_overview.png"
    fig.savefig(fig_path+fig_name,dpi=300,bbox_inches="tight")
    print("Figure saved as:",fig_path+fig_name)
    #plt.adjust_subplots(hspace=0.1,vspace=0.1)
    return None

if __name__=="__main__":
    main(save_in_manuscript_path=True)