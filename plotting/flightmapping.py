import os
import sys

sys.path.insert(1,os.getcwd()+"/../scripts/")
sys.path.insert(2,os.getcwd()+"/../src/")
sys.path.insert(3,os.getcwd()+"/../config/")

#import netCDF4
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

try:
    from typhon.plots import styles
except:
    print("Typhon module cannot be imported")

import data_config
# Real Campaigns
from flightcampaign import Campaign as flight_campaign
from flightcampaign import NAWDEX
from flightcampaign import HALO_AC3
# Synthetic Campaigns
from flightcampaign import Synthetic_Campaign as synthetic_campaign
from flightcampaign import North_Atlantic_February_Run
from flightcampaign import Second_Synthetic_Study

"""
FlightMaps inherits from Campaign of Flight_Campaign
    map functions:
    - flight_map_iop(self,campaign_cls,sea_ice_desired=False)
        --> plots major flights of NAWDEX IOPs (Interesting Observation Periods)
        
    - plot_flight_map_era(self,campaign_cls,coords_station,
                          flight,meteo_var,show_AR_detection=True,
                          show_supersites=True)
        --> plots the Flighttrack together with given meteorological variable
            (IWV or IVT) from ERA-5 and if desired with AR Detection and Supersites
            
    - plot_flight_map_complete_NAWDEX_with_ARs(self,campaign_cls,flights,
                                               with_dropsondes=False, 
                                               include_ARs=False)
        --> plots all flights given from the campaign together with the ARs in 
            the overreaching domain and highlight AR cross_sections
            
    - plot_flight_map_AR_crossing(self,cut_radar,Dropsondes,campaign_cls,
                                  flight,AR_number)
        --> plots the AR cross-section of given AR-Number together with IVT and
            IWV (ERA-5), defining dropsonde releases and high radar reflectivity
    
    - plot_flight_combined_IWV_map_AR_crossing(self,cut_radar,Dropsondes,campaign_cls,
                                               flight,AR_number,last_hour)
        --> plots the IWV (not IVT even if listed) map from AR crossing for 
            ERA-5 and ICON, additional data is similar to function before. 
            
    - plot_flight_map_Hydrometeorpaths_AR_crossing(self,cut_radar,Dropsondes,
                                                 campaign_cls,flight,
                                                 AR_number,last_hour):
        --> plots HMP map from AR crossing for ERA-5 and ICON, additional data
            is similar to function before
            
    - to be continued
    
    -
    
    -

"""

class FlightMaps(flight_campaign):
    def __init__(self,major_path,campaign_path,
                     aircraft,instruments,interested_flights,
                     plot_path=os.getcwd(),
                     analysing_campaign=True,
                     synthetic_campaign=False,
                     flight="",
                     ar_of_day="",
                     synthetic_icon_lat=0,
                     synthetic_icon_lon=0,
                     track_type=None,
                     track_dict=None,
                     pick_legs=[]):
            
            super().__init__(self,major_path,
                             aircraft,instruments)
            
            self.ar_of_day=ar_of_day
            self.plot_path=plot_path
            self.interested_flights=interested_flights
            self.campaign_path=campaign_path
            self.analysing_campaign=analysing_campaign
            self.synthetic_campaign=synthetic_campaign
            self.flight=flight
            self.synthetic_icon_lat=synthetic_icon_lat
            self.synthetic_icon_lon=synthetic_icon_lon
            self.track_type=track_type
            if self.track_type=="internal":
                self.track_dict=track_dict
            self.pick_legs=pick_legs
            import cartopy.crs as ccrs
            
    def flight_map_iop(self,campaign_cls,sea_ice_desired=False):
        from Cloudnet import Cloudnet_Data
        
        print(self.interested_flights)
        if self.interested_flights==["RF01","RF03","RF04","RF05","RF10"]:
            pass
        else:
            raise AssertionError("The specified flights are wrong.", 
                                 "they are not RF01,RF03,RF04,RF05,RF10")
        # get station_locations
        campaign_cloudnet=Cloudnet_Data(self.campaign_path)
        station_coords=campaign_cloudnet.get_cloudnet_station_coordinates(\
                                                    campaign_cls.campaign_path)
        
        # HALO position
        aircraft_position=campaign_cls.get_aircraft_position(campaign_cls.flight_day.keys(),
                                                       campaign_cls.name)
        
        # Map plotting
        import cartopy.crs as ccrs
        map_fig=plt.figure(figsize=(12,12))
        ax = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-25.0,
                                                           central_latitude=55))
        ax.coastlines(resolution="50m")
        ax.set_extent([-50,10,40,90])
        ax.gridlines()
        print(station_coords)
        
        #Summit
        ax.text(station_coords["Summit"]["Lon"]-8,
                station_coords["Summit"]["Lat"]-1.5, 'Summit\n(Greenland)',
                horizontalalignment='left', transform=ccrs.Geodetic(),
                fontsize=10,color="blue")
        plt.scatter(station_coords["Summit"]["Lon"],
                    station_coords["Summit"]["Lat"],
                    marker='x', transform=ccrs.PlateCarree(),color="blue")
        
        #Ny-Alesund
        ax.text(station_coords["Ny-Alesund"]["Lon"]-15,
                station_coords["Ny-Alesund"]["Lat"]-1.75, 
                'Ny Alesund\n(Spitzbergen)', horizontalalignment='left',
                transform=ccrs.Geodetic(),fontsize=10,color="blue")
        plt.scatter(station_coords["Ny-Alesund"]["Lon"],
                    station_coords["Ny-Alesund"]["Lat"],
                    marker='x', transform=ccrs.PlateCarree(),color="blue")
        
        #Mace-Head
        ax.text(station_coords["Mace-Head"]["Lon"]-8,
                station_coords["Mace-Head"]["Lat"]+3.6, 
                'Mace Head\n(Ireland)', horizontalalignment='left', 
                transform=ccrs.Geodetic(),fontsize=10,color="blue")
        plt.scatter(station_coords["Mace-Head"]["Lon"],
                    station_coords["Mace-Head"]["Lat"]+3.5,
                    marker='x', transform=ccrs.PlateCarree(),color="blue")
        
        # Draw Triangle around stations
        plt.plot([station_coords["Summit"]["Lon"],
                  station_coords["Ny-Alesund"]["Lon"]],
                 [station_coords["Summit"]["Lat"],
                  station_coords["Ny-Alesund"]["Lat"]],
                 transform=ccrs.Geodetic(),linestyle='--',color="black")
        plt.plot([station_coords["Ny-Alesund"]["Lon"],
                  station_coords["Mace-Head"]["Lon"]],
                 [station_coords["Ny-Alesund"]["Lat"],
                  station_coords["Mace-Head"]["Lat"]+3.6],
                 transform=ccrs.Geodetic(),linestyle='--',color="black")
        plt.plot([station_coords["Mace-Head"]["Lon"],
                  station_coords["Summit"]["Lon"]],
                 [station_coords["Mace-Head"]["Lat"]+3.6,
                  station_coords["Summit"]["Lat"]],
                 transform=ccrs.Geodetic(),linestyle='--',color="black")
        
        iop=[": WCB Outflow",": WCB Ascent /Outflow",
             ": WCB Ascent",": AR Moisture Transport",": Arctic "]
        i=0
        
        
        for f in campaign_cls.flight_day.keys():
            if f in self.interested_flights:
                plt.plot(aircraft_position[f]["longitude"],
                         aircraft_position[f]["latitude"],
                         transform=ccrs.Geodetic(),label=f+iop[i])
                i+=1    
            else:
                plt.plot(aircraft_position[f]["longitude"],
                         aircraft_position[f]["latitude"],
                         transform=ccrs.Geodetic(),
                         color="lightgrey",alpha=0.7)
        if sea_ice_desired:
            print("Add mean sea ice extension.")
            print("For now this is work in progress and pass")
            pass
        ax.legend(fontsize=10)        
        file_name="Campaign_Flights_IOP_Map.png"
        #plt.show()
        map_fig.savefig(campaign_cls.plot_path+file_name,
                        dpi=600,bbox_inches="tight")
        print("Map Figure saved as: ",campaign_cls.plot_path+file_name)

    #Create map plots while looping over data
    ###########################################################################
    #ERA5
    def plot_flight_map_era(self,campaign_cls,coords_station,
                            flight,meteo_var,show_AR_detection=True,
                            show_supersites=True,use_era5_ARs=False):
        """
        

        Parameters
        ----------
        coords_station : dict
            Dictionary containing the coordinates of each station. Can be 
            assessed by Cloudnet_Data.get_cloudnet_station_coordinates
        flight : str
            research flight to analyse.
        meteo_var : str
            meteorological variable from ERA-analysis to map.
        show_AR_detection : bool
            Insert the AR-Detection of algorithm from Guan & Waliser,2015
        show_supersites :  bool
            boolean for showing the three cloud net supersite data if accessible.
        Returns
        -------
        None.

        """
        import matplotlib
        import cartopy.crs as ccrs
        
        #from era5_on_halo_backup import ERA5
        from reanalysis import ERA5
        
        set_font=16
        matplotlib.rcParams.update({'font.size':set_font})
        plt.rcParams.update({'hatch.color': 'k'})  
        plt.rcParams.update({'hatch.linewidth':1.5})
        
        # Define the plot specifications for the given variables
        met_var_dict={}
        met_var_dict["ERA_name"]    = {"IWV":"tcwv","IVT":"IVT",
                                       "IVT_u":"IVT_u","IVT_v":"IVT_v"}
        met_var_dict["colormap"]    = {"IWV":"density","IVT":"speed",
                                       "IVT_v":"speed",
                                       "IVT_u":"speed"}
        met_var_dict["levels"]      = {"IWV":np.linspace(10,25,101),
                                       "IVT":np.linspace(50,600,101),
                                       "IVT_v":np.linspace(0,500,101),
                                       "IVT_u":np.linspace(0,500,101)}
        met_var_dict["units"]       = {"IWV":"(kg$\mathrm{m}^{-2}$)",
                                       "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                       "IVT_v":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                       "IVT_u":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
        
        flight_date=campaign_cls.year+"-"+campaign_cls.flight_month[flight]
        flight_date=flight_date+"-"+campaign_cls.flight_day[flight]
        
        
        era5=ERA5(for_flight_campaign=True,campaign=campaign_cls.name,
                  research_flights=flight,
                  era_path=campaign_cls.campaign_path+"/data/ERA-5/")
        plot_path=campaign_cls.campaign_path+"/plots/"+flight+"/"
        hydrometeor_lvls_path=campaign_cls.campaign_path+"/data/ERA-5/"
    
        file_name="total_columns_"+campaign_cls.year+"_"+\
                                    campaign_cls.flight_month[flight]+"_"+\
                                    campaign_cls.flight_day[flight]+".nc"    
        
        ds,era_path=era5.load_era5_data(file_name)
        
        #if meteo_var.startswith("IVT"):
        ds["IVT_v"]=ds["p72.162"]
        ds["IVT_u"]=ds["p71.162"]
        ds["IVT"]=np.sqrt(ds["IVT_u"]**2+ds["IVT_v"]**2)
        # Load Flight Track
        halo_dict={}
        if not self.analysing_campaign:
            if not (campaign_cls.campaign_name=="NAWDEX") and \
                not (campaign_cls.campaign_name=="HALO_AC3"):
                    halo_dict=campaign_cls.get_aircraft_position(self.ar_of_day)
            if campaign_cls.campaign_name=="NAWDEX":
                from flight_track_creator import Flighttracker
                Tracker=Flighttracker(campaign_cls,flight,self.ar_of_day,
                          shifted_lat=0,
                          shifted_lon=0,
                          track_type="internal")   
                halo_dict,cmpgn_path=Tracker.run_flight_track_creator()
    
            if campaign_cls.campaign_name=="HALO_AC3":
                campaign_cls.load_AC3_bahamas_ds(flight)
                halo_dict=campaign_cls.bahamas_ds
        print(halo_dict)
        if isinstance(halo_dict,pd.DataFrame):
            halo_df=halo_dict.copy() 
        elif isinstance(halo_dict,xr.Dataset):
            halo_df=pd.DataFrame(data=np.nan,columns=["alt","Lon","Lat"],
                                index=pd.DatetimeIndex(np.array(halo_dict["TIME"][:])))
            halo_df["Lon"]=halo_dict["IRS_LON"].data
            halo_df["Lat"]=halo_dict["IRS_LAT"].data
        else:
            if len(halo_dict.keys())==1:
                halo_df=halo_dict.values()[0]
            else:   
                    halo_df=pd.concat([halo_dict["inflow"],halo_dict["internal"],
                                       halo_dict["outflow"]])
                    halo_df.index=pd.DatetimeIndex(halo_df.index)
            if campaign_cls.name=="NAWDEX" and flight=="RF10":
                real_halo_df,_=campaign_cls.load_aircraft_position()
                real_halo_df.index=pd.DatetimeIndex(real_halo_df.index)
                real_halo_df["Hour"]=real_halo_df.index.hour
                real_halo_df=real_halo_df.rename(columns={"Lon":"longitude",
                                    "Lat":"latitude"})
                
        halo_df["Hour"]=halo_df.index.hour
        halo_df=halo_df.rename(columns={"Lon":"longitude",
                                    "Lat":"latitude"})
        
        for i in range(24):
            print("Hour of the day:",i)
            calc_time=era5.hours[i]
            map_fig=plt.figure(figsize=(12,9))
            ax = plt.axes(projection=ccrs.AzimuthalEquidistant(
                central_longitude=-5.0,central_latitude=70))
            if flight=="SRF06":
                ax = plt.axes(projection=ccrs.AzimuthalEquidistant(
                central_longitude=30.0,central_latitude=70))
            if flight=="SRF07":
                ax = plt.axes(projection=ccrs.AzimuthalEquidistant(
                central_longitude=40.0,central_latitude=70))
            
            ax.coastlines(resolution="50m")
            ax.gridlines()
            if campaign_cls.name=="NAWDEX":
                if flight=="RF01" or flight=="RF13":
                    ax.set_extent([-45,5,30,70])
                elif flight=="RF02":
                    ax.set_extent([-45,5,30,70])
                elif flight=="RF03":
                    ax.set_extent([-45,5,30,70])
                elif flight=="RF04":
                    ax.set_extent([-45,5,35,70])
                elif flight=="RF05" or flight=="RF06":
                    ax.set_extent([-45,5,35,70])
                elif (flight=="RF08") or (flight=="RF09") or (flight=="RF11"):
                    ax.set_extent([-40, 0, 40, 70])
                elif flight=="RF10":
                    ax.set_extent([-30,5,40,85])
                elif (flight=="RF07") or flight=="RF12":
                    ax.set_extent([-75,-20,45,70])
                else:
                    pass
            elif campaign_cls.name=="HALO_AC3_Dry_Run":
                if flight=="SRF04":
                    ax.set_extent([-10,40,60,90])
                elif flight=="SRF01":
                    ax.set_extent([-20,70,60,90])
               
                else:
                    raise Exception("Other flights are not yet provided")
            elif campaign_cls.name=="HALO_AC3":
                if (flight=="RF01") or (flight=="RF02") or (flight=="RF03") or \
                   (flight=="RF04") or (flight=="RF05") or \
                   (flight=="RF06") or (flight == "RF07"):
                   ax.set_extent([-40,30,55,90]) 
            elif campaign_cls.name=="NA_February_Run":
                ax.set_extent([-30,5,40,90])
                if flight=="SRF04":
                    ax.set_extent([-25,10,55,90])
                if flight=="SRF06" :
                    ax.set_extent([20,90,50,90])
                if flight=="SRF07":
                    ax.set_extent([10,70,50,90])
            elif campaign_cls.name=="Second_Synthetic_Study":
                ax.set_extent([-25,30,55,90])
            #-----------------------------------------------------------------#
            # Meteorological Data plotting
            # Plot Water Vapour Quantity    
            C1=plt.contourf(ds["longitude"],ds["latitude"],
                            ds[met_var_dict["ERA_name"][meteo_var]][i,:,:],
                            levels=met_var_dict["levels"][meteo_var],
                            extend="max",transform=ccrs.PlateCarree(),
                            cmap=met_var_dict["colormap"][meteo_var],alpha=0.95)
            
            cb=map_fig.colorbar(C1,ax=ax)
            cb.set_label(meteo_var+" "+met_var_dict["units"][meteo_var])
            if meteo_var=="IWV":
                cb.set_ticks([10,15,20,25,30])
            elif meteo_var=="IVT":
                cb.set_ticks([50,100,200,300,400,500,600])
            else:
                pass
            # Mean surface level pressure
            if meteo_var.startswith("IVT"):
                pressure_color="royalblue"
                sea_ice_colors=["mediumslateblue", "indigo"]
            else:
                pressure_color="green"
                sea_ice_colors=["peru","sienna"]
            C_p=plt.contour(ds["longitude"],ds["latitude"],
                            ds["msl"][i,:,:]/100,levels=np.linspace(950,1050,11),
                            linestyles="-.",linewidths=1.5,colors=pressure_color,
                            transform=ccrs.PlateCarree())
            plt.clabel(C_p, inline=1, fmt='%03d hPa',fontsize=12)
            # mean sea ice cover
            C_i=plt.contour(ds["longitude"],ds["latitude"],
                            ds["siconc"][i,:,:]*100,levels=[15,85],
                            linestyles="-",linewidths=[1,1.5],colors=sea_ice_colors,
                            transform=ccrs.PlateCarree())
            plt.clabel(C_i, inline=1, fmt='%02d %%',fontsize=10)
            
            #-----------------------------------------------------------------#
            # Quiver-Plot
            step=15
            quiver_lon=np.array(ds["longitude"][::step])
            quiver_lat=np.array(ds["latitude"][::step])
            u=ds["IVT_u"][i,::step,::step]
            v=ds["IVT_v"][i,::step,::step]
            v=v.where(v>200)
            v=np.array(v)
            u=np.array(u)
            quiver=plt.quiver(quiver_lon,quiver_lat,
                                  u,v,color="lightgrey",edgecolor="k",lw=1,
                                  scale=800,scale_units="inches",
                                  pivot="mid",width=0.008,
                                  transform=ccrs.PlateCarree())
            plt.rcParams.update({'hatch.color': 'lightgrey'})
            #-----------------------------------------------------------------#
            # Show Guan & Waliser 2020 Quiver-Plot if available (up to 2019)
                #if int(flight_date[0:4])>=2020:
                #   show_AR_detection=False
            if show_AR_detection:    
                import atmospheric_rivers as AR
                AR=AR.Atmospheric_Rivers("ERA",use_era5=use_era5_ARs)
                AR_era_ds=AR.open_AR_catalogue(after_2019=int(flight_date[0:4])>2019,
                                               year=campaign_cls.year,
                                               month=campaign_cls.flight_month[flight])
                AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)

                if not use_era5_ARs:
            
                    if i<6:
                        hatches=plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start,
                                                 0,:,:],
                                 hatches=['//'],colors='none',cmap="bone_r",
                                 alpha=0.8,transform=ccrs.PlateCarree())
                        for i,collection in enumerate(hatches.collections):
                            collection.set_edgecolor("green")
                    elif 6<=i<12:
                        plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,
                                                 0,:,:],
                                 hatches=[ '//'],cmap='bone',alpha=0.2,
                                 transform=ccrs.PlateCarree())
                    elif 12<=i<18:
                        plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,
                                                 0,:,:],
                                 hatches=['//'],cmap='bone_r',alpha=0.2,
                                 transform=ccrs.PlateCarree())
                    else:
                        hatches=plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,
                                                 0,:,:],
                                 hatches=['//'],cmap='bone_r',
                                 alpha=0.1,
                                 transform=ccrs.PlateCarree())
                        for i,collection in enumerate(hatches.collections):
                            collection.set_edgecolor("k")
                   
                else:
                    hatches=plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                 AR_era_ds.shape[0,AR_era_data["model_runs"].start+i,
                                                 0,:,:],
                                 hatches=["//"],cmap="bone_r",
                                 alpha=0.2,transform=ccrs.PlateCarree())
                    for c,collection in enumerate(hatches.collections):
                            collection.set_edgecolor("k")
            #-----------------------------------------------------------------#
            # Flight Track (either real or synthetic)
            if self.analysing_campaign or self.synthetic_campaign:
                 # Load aircraft data
                 plot_halo_df=halo_df[halo_df.index.hour<i]
                 if flight=="RF10":
                     plot_real_halo_df=real_halo_df[real_halo_df.index.hour<i]
                 
                 if i<= pd.DatetimeIndex(halo_df.index).hour[0]:
                      ax.scatter(halo_df["longitude"].iloc[0],
                                 halo_df["latitude"].iloc[0],
                                s=30,marker='x',color="red",
                                transform=ccrs.PlateCarree())
                 
                 elif i>pd.DatetimeIndex(halo_df.index).hour[-1]:
                     ax.scatter(halo_df["longitude"].iloc[-1],
                                 halo_df["latitude"].iloc[-1],
                                s=30,marker='x',color="red",
                                transform=ccrs.PlateCarree())
                 else:
                      if flight=="RF10":
                          ax.plot(plot_real_halo_df["longitude"],
                                  plot_real_halo_df["latitude"],
                                  lw=2,ls="-.",color="grey",
                                  transform=ccrs.PlateCarree())
                      ax.plot(plot_halo_df["longitude"],plot_halo_df["latitude"],
                          linewidth=3.0,color="red",transform=ccrs.PlateCarree(),
                          alpha=0.8)
                 #------------------------------------------------------------#
                 # plot Cloudnet Locations
                 if show_supersites:
                    
                    if meteo_var=="IWV":
                        station_marker_color="green"
                    else:
                        station_marker_color="red"
                    for station in coords_station.keys():
                        try:
                            if station=="Mace-Head":
                                ax.scatter(coords_station[station]["Lon"],
                                            coords_station[station]["Lat"]+3.6,
                                            s=100,marker="s",color=station_marker_color,
                                            edgecolors="black",
                                            transform=ccrs.PlateCarree())
                            else:
                                ax.scatter(coords_station[station]["Lon"],
                                            coords_station[station]["Lat"],
                                            s=100,marker="s",color=station_marker_color,
                                            edgecolors="black",
                                            transform=ccrs.PlateCarree())        
                        except:
                            pass
            #-----------------------------------------------------------------#
                 #plot Dropsonde releases
                 date=campaign_cls.year+campaign_cls.flight_month[flight]
                 date=date+campaign_cls.flight_day[flight]
                 if self.analysing_campaign:
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
                             plotting_dropsondes=dropsonde_releases.loc[\
                                            dropsonde_releases.index.hour<i]
                             ax.scatter(plotting_dropsondes["Lon"],
                                        plotting_dropsondes["Lat"],
                                        s=100,marker="v",color="orange",
                                        edgecolors="black",
                                        transform=ccrs.PlateCarree())
                         except:
                            pass
                 
                     if flight=="RF08":
                        if i>=12:
                            ax.scatter(dropsonde_releases["Lon"],
                                   dropsonde_releases["Lat"],
                                   s=100,marker="v",color="orange",
                                   edgecolors="black",
                                   transform=ccrs.PlateCarree()) 
            #-----------------------------------------------------------------#
            ax.set_title(campaign_cls.name+" "+flight+": "+campaign_cls.year+\
                         "-"+campaign_cls.flight_month[flight]+\
                         "-"+campaign_cls.flight_day[flight]+" "+calc_time)
            
            #Save figure
            fig_name=campaign_cls.name+"_"+flight+'_'+era5.hours_time[i][0:2]+\
                "H"+era5.hours_time[i][3:6]+"_"+str(meteo_var)+".png"
            if not show_AR_detection:
                fig_name="no_AR_"+fig_name
            if show_supersites:
                fig_name="supersites_"+fig_name
            fig_path=plot_path+meteo_var+"/"
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            map_fig.savefig(fig_path+fig_name,bbox_inches="tight",dpi=150)
            print("Figure saved as:",fig_path+fig_name)
            plt.close()
            
        return None
    
    def plot_map_era_all_AR_cross_sections(self,campaign_cls,coords_station,
                                           flight,meteo_var,ar_list,
                                           halo_df,halo_ar,
                                           show_AR_detection=True,
                                           show_supersites=True,
                                           invert_flight=False):
        """
        

        Parameters
        ----------
        coords_station : dict
            Dictionary containing the coordinates of each station. Can be 
            assessed by Cloudnet_Data.get_cloudnet_station_coordinates
        flight : str
            research flight to analyse.
        meteo_var : str
            meteorological variable from ERA-analysis to map.
        ar_list : list
            list of ARs 
        halo_df  : pd.DataFrame
            containing entire flight track
        halo_ar : dict
            dictionary with flight track for all AR cross-section
        show_AR_detection : bool
            Insert the AR-Detection of algorithm from Guan & Waliser,2015
        show_supersites :  bool
            boolean for showing the three cloud net supersite data if accessible.
        Returns
        -------
        None.

        """
        import matplotlib
        import cartopy.crs as ccrs
        
        from era5_on_halo_backup import ERA5
        from matplotlib import gridspec
        
        set_font=16
        matplotlib.rcParams.update({'font.size':set_font})
                
        
        # Define the plot specifications for the given variables
        met_var_dict={}
        met_var_dict["ERA_name"]    = {"IWV":"tcwv","IVT":"IVT",
                                       "IVT_u":"IVT_u","IVT_v":"IVT_v"}
        met_var_dict["colormap"]    = {"IWV":"density","IVT":"speed",
                                       "IVT_v":"speed",
                                       "IVT_u":"speed"}
        met_var_dict["levels"]      = {"IWV":np.linspace(0,50,101),
                                       "IVT":np.linspace(50,350,101),
                                       "IVT_v":np.linspace(0,500,101),
                                       "IVT_u":np.linspace(0,500,101)}
        met_var_dict["units"]       = {"IWV":"(kg$\mathrm{m}^{-2}$)",
                                       "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                       "IVT_v":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)",
                                       "IVT_u":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
        
        flight_date=campaign_cls.year+"-"+campaign_cls.flight_month[flight]
        flight_date=flight_date+"-"+campaign_cls.flight_day[flight]
        
        
        era5=ERA5(for_flight_campaign=True,campaign=campaign_cls.name,
                  research_flights=flight,
                  era_path=campaign_cls.campaign_path+"/data/ERA-5/")
        plot_path=campaign_cls.campaign_path+"/plots/"+flight+"/"
        hydrometeor_lvls_path=campaign_cls.campaign_path+"/data/ERA-5/"
    
        file_name="total_columns_"+campaign_cls.year+"_"+\
                                    campaign_cls.flight_month[flight]+"_"+\
                                    campaign_cls.flight_day[flight]+".nc"    
        
        ds,era_path=era5.load_era5_data(file_name)
        
        if meteo_var.startswith("IVT"):
            ds["IVT_v"]=ds["p72.162"]
            ds["IVT_u"]=ds["p71.162"]
            ds["IVT"]=np.sqrt(ds["IVT_u"]**2+ds["IVT_v"]**2)
        
        hour_to_plot=halo_ar[ar_list[-1]].index.hour[-1]
        calc_time=str(hour_to_plot)+":00"
        print("Hour of the day:",hour_to_plot)
        
        map_fig=plt.figure(figsize=(16,9))
        gs=gridspec.GridSpec(1,2,width_ratios=[2,1])
    
        ax=plt.subplot(gs[0],projection=ccrs.AzimuthalEquidistant(
                                central_longitude=20.0,central_latitude=70))
        #ax = plt.axes()
        ax.coastlines(resolution="50m")
        ax.gridlines()
        
        if campaign_cls.name=="HALO_AC3_Dry_Run":
            if flight=="RF04":
                ax.set_extent([-10,40,60,90])
            else:
                raise Exception("Other flights are not yet provided")
        C1=plt.contourf(ds["longitude"],ds["latitude"],
                        ds[met_var_dict["ERA_name"][meteo_var]][hour_to_plot,:,:],
                        levels=met_var_dict["levels"][meteo_var],
                        extend="max",transform=ccrs.PlateCarree(),
                        cmap=met_var_dict["colormap"][meteo_var],alpha=0.95)
        
        if meteo_var.startswith("IVT"):
            step=10
            quiver_lon=np.array(ds["longitude"][::step])
            quiver_lat=np.array(ds["latitude"][::step])
            u=np.array(ds["IVT_u"][hour_to_plot,::step,::step])
            v=np.array(ds["IVT_v"][hour_to_plot,::step,::step])
            quiver=plt.quiver(quiver_lon,quiver_lat,
                              u,v,color="dimgrey",
                              scale=600,scale_units="inches",
                              pivot="mid",width=0.005,
                              transform=ccrs.PlateCarree())
        plt.rcParams.update({'hatch.color': 'lightgrey'})
        
        if show_AR_detection:    
            import atmospheric_rivers as AR
    
            AR=AR.Atmospheric_Rivers("ERA")
            AR_era_ds=AR.open_AR_catalogue(after_2019=int(flight_date[0:4])>2019)
            AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)

    
            plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
                             AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                             hatches=['//'],cmap='bone_r',alpha=0.1,
                             transform=ccrs.PlateCarree())
            
        cb=map_fig.colorbar(C1,ax=ax,shrink=0.95,extend="both")
        cb.set_label(meteo_var+" "+met_var_dict["units"][meteo_var])
        if meteo_var=="IWV":
            cb.set_ticks([0,10,20,30,40,50])
        elif meteo_var=="IVT":
            cb.set_ticks([50,100,150,200,250,300,350])
        else:
            pass
            
        if self.analysing_campaign or self.synthetic_campaign:
             # Load aircraft data
             #plot_halo_df=halo_df#
             ax.plot(halo_df["longitude"],halo_df["latitude"],
                            ls="--",color="plum",
                            transform=ccrs.PlateCarree())
             colors_ar_list={"AR1":"mediumorchid","AR2":"blueviolet",
                             "AR3":"mediumblue","AR4":"navy"}
             for ar in ar_list:
                 ax.scatter(halo_ar[ar]["longitude"],halo_ar[ar]["latitude"],
                            s=5,marker="x",color=colors_ar_list[ar],
                            transform=ccrs.PlateCarree(),
                            label=ar)
             if show_supersites:
                #plot Cloudnet Locations
                if meteo_var=="IWV":
                    station_marker_color="green"
                else:
                    station_marker_color="red"
                for station in coords_station.keys():
                    try:
                        if station=="Mace-Head":
                            ax.scatter(coords_station[station]["Lon"],
                                        coords_station[station]["Lat"]+3.6,
                                        s=100,marker="s",color=station_marker_color,
                                        edgecolors="black",
                                        transform=ccrs.PlateCarree())
                        else:
                            ax.scatter(coords_station[station]["Lon"],
                                        coords_station[station]["Lat"],
                                        s=100,marker="s",color=station_marker_color,
                                        edgecolors="black",
                                        transform=ccrs.PlateCarree())        
                    except:
                        pass
        
             #plot Dropsonde releases
             date=campaign_cls.year+campaign_cls.flight_month[flight]
             date=date+campaign_cls.flight_day[flight]
             if not self.synthetic_campaign:
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
                         ax.scatter(dropsonde_releases["Lon"],
                                    dropsonde_releases["Lat"],
                                    s=100,marker="v",color="orange",
                                    edgecolors="black",
                                    transform=ccrs.PlateCarree())
                     except:
                         pass
                 
        ax.set_title(flight+": "+campaign_cls.year+\
                     "-"+campaign_cls.flight_month[flight]+\
                     "-"+campaign_cls.flight_day[flight]+" "+calc_time)
        ax2=plt.subplot(gs[1])
        import seaborn as sns
        for ar in ar_list:
            hmp_path=campaign_cls.campaign_path+"/data/ERA-5/"
            if invert_flight:
                hmp_path=hmp_path+"inverted/"
            hmp_file=ar+"_HMP_ERA_HALO_"+flight+"_"+date+".csv"
            ar_hmp_df=pd.read_csv(hmp_path+hmp_file)
            ar_hmp_df.index=pd.DatetimeIndex(ar_hmp_df["index"])
            ar_hmp_df["Distance"]=halo_ar[ar]['Cum. dist. (km)']
            #find ivt max:
            interp_var_name="Interp_"+met_var_dict["ERA_name"][meteo_var]
            print("MAX "+met_var_dict["ERA_name"][meteo_var]+": ",
                  ar_hmp_df[interp_var_name].max())
            ivt_max_loc=ar_hmp_df[interp_var_name].idxmax()
            ar_hmp_df["Center_Max_Distance"]=ar_hmp_df["Distance"]-\
                                    ar_hmp_df["Distance"].loc[ivt_max_loc]
            ax2.plot(ar_hmp_df["Center_Max_Distance"],ar_hmp_df[interp_var_name],
                     ls="--",lw=3,color=colors_ar_list[ar])
            del ar_hmp_df["index"]
            ax2.set_xlabel("Centered AR distance (km)")
            ax2.set_xlim([-400,400])
            ax2.set_ylim([50,350])
            sns.despine(ax=ax2,offset=10)
            #ax2.set_ylabel(met_var_dict["ERA_name"][meteo_var]+" "+
            #               met_var_dict["units"][meteo_var])
        #Save figure
        fig_name=campaign_cls.name+"_"+flight+'_'+\
            era5.hours_time[hour_to_plot][0:2]+"H"+\
            era5.hours_time[hour_to_plot][3:6]+"_"+str(meteo_var)+".png"
        plt.suptitle("HALO-(AC)³ Dry Run ")
        if not show_AR_detection:
            fig_name="no_AR_"+fig_name
        if not show_supersites:
            fig_name="_no_supersites_"+fig_name
        fig_name="Cross_Sections_"+fig_name
        fig_path=plot_path
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        ax.legend(loc="upper left",title="Cross-Sections")
        map_fig.savefig(fig_path+fig_name,bbox_inches="tight",dpi=200)
        print("Figure saved as:",fig_path+fig_name)
        plt.subplots_adjust(hspace=0.3)
        plt.close()
        
        return None
    
    def plot_flight_map_complete_NAWDEX_with_ARs(self,campaign_cls,flights,
                                                 rf_colors,
                                                 with_dropsondes=False, 
                                                 include_ARs=False):
       
       from matplotlib.lines import Line2D
       custom_lines=[] 
       if flights=="all":
           self.interested_flights=campaign_cls.flight_day.keys()
       elif isinstance(flights,list): #print(self.interested_flights)
            self.interested_flights=flights
       else:
            raise AssertionError("The specified flights ",flights, "are of wrong type.", 
                                 "Redefine your argument")
       # HALO position
       aircraft_position=campaign_cls.get_aircraft_position(campaign_cls.flight_day.keys(),
                                                       campaign_cls.name)
        
       # Map plotting
       import cartopy.crs as ccrs
       all_rfs_map_fig=plt.figure(figsize=(16,12))
       matplotlib.rcParams.update({"font.size":16})
       ax = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-25.0,
                                                           central_latitude=55))
       ax.coastlines(resolution="50m",color="gray")
       ax.set_extent([-60,5,40,80])
       ax.gridlines()
       AR_unique=np.empty(0)
       i=0
       flight_colors={"RF01":"grey","RF02":"paleturquoise","RF03":"salmon",
                      "RF04":"peru","RF05":"skyblue","RF06":"moccasin",
                      "RF07":"slateblue","RF08":"bisque","RF09":"thistle",
                      "RF10":"lightgreen","RF11":"lightpink","RF12":"gold","RF13":"rosybrown"}
       for f in campaign_cls.flight_day.keys():
           
           if f in self.interested_flights:
               print(i)
               if include_ARs:
                   line_color=[*rf_colors][i]
                   line_obj=Line2D([0], [0], color=line_color, lw=2)
               else:
                   line_color=flight_colors[f]
                   line_obj=Line2D([0], [0], color=line_color, lw=2)
               
               custom_lines.append(line_obj)
       
               date=campaign_cls.year+campaign_cls.flight_month[f]
               date=date+campaign_cls.flight_day[f]
               print(date)
               if include_ARs:
                   flight_color="grey"
                   plt.plot(aircraft_position[f]["longitude"],
                         aircraft_position[f]["latitude"],
                         transform=ccrs.Geodetic(),
                         label=f,lw=2,ls='--',color=flight_color)
               
               else: 
                   flight_color=flight_colors[f]
                   plt.plot(aircraft_position[f]["longitude"],
                         aircraft_position[f]["latitude"],
                         transform=ccrs.Geodetic(),
                         label=f,lw=3,ls='-',color=flight_color)
               if with_dropsondes:
                    if not f=="RF06":
                                   
                        Dropsondes=campaign_cls.load_dropsonde_data(date,
                                                                    print_arg="yes",
                                                                    dt="all",
                                                                    plotting="no")
        
                        # in some cases the Dropsondes variable can be a dataframe or
                        # just a series, if only one sonde has been released
                        if isinstance(Dropsondes["Lat"],pd.DataFrame):
                            dropsonde_releases=pd.DataFrame(index=pd.DatetimeIndex(Dropsondes["LTS"].index))
                            dropsonde_releases["Lat"]=Dropsondes["Lat"].loc[:,"6000.0"].values
                            dropsonde_releases["Lon"]=Dropsondes["Lon"].loc[:,"6000.0"].values
        
                        else:
                            index_var=Dropsondes["Time"].loc["6000.0"]
                            dropsonde_releases=pd.Series()
                            dropsonde_releases["Lat"]=np.array(Dropsondes["Lat"].loc["6000.0"])
                            dropsonde_releases["Lon"]=np.array(Dropsondes["Lon"].loc["6000.0"])
                            dropsonde_releases["Time"]=index_var
        
                          #  if not f=="RF08":
                        relevant_dropsondes=dropsonde_releases#.loc[cut_radar["Reflectivity"].index[0]:cut_radar["Reflectivity"].index[-1]]
                        if relevant_dropsondes.shape[0]>0:
                                ax.scatter(relevant_dropsondes["Lon"],
                                            relevant_dropsondes["Lat"],
                                            s=50,marker="v",color="lightgrey",
                                            edgecolors="black",
                                            transform=ccrs.PlateCarree())
               if include_ARs:
                   import atmospheric_rivers as AR
                   from era5_on_halo_backup import ERA_on_HALO
                   era_on_halo=ERA_on_HALO()
                   print("Use AR Catalogue from Guan & Waliser")
                   ARs_of_day=["AR1","AR2","AR3","AR4"]
                   
                   AR=AR.Atmospheric_Rivers("ERA")
                   AR_era_ds=AR.open_AR_catalogue(after_2019=int(flight_date[0:4])>2019)
                   flight_date=campaign_cls.year+"-"+campaign_cls.flight_month[f]
                   flight_date=flight_date+"-"+campaign_cls.flight_day[f]
                   
                   AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)

                   
                   color=[*rf_colors][i]
                   colormaps=plt.get_cmap(rf_colors[color])
                   dark_colormap=colormaps(np.linspace(0.5,1.0,10))    
                   
                   for ar in ARs_of_day:
                       try:
                           ar_aircraft_position,ar_sondes,ar_of_day=era_on_halo.cut_halo_to_AR_crossing(
                                                                   ar,f,aircraft_position,
                                                                   dropsonde_releases,
                                                                   device="sondes")
                           print("Loaded data within cross-section of ",ar)
                           plt.plot(ar_aircraft_position[f]["longitude"],
                                    ar_aircraft_position[f]["latitude"],
                                    transform=ccrs.Geodetic(),
                                    lw=2,ls='-',color=color)
                           ax.scatter(ar_sondes["Lon"],
                                            ar_sondes["Lat"],
                                            s=150,marker="v",color="orange",
                                            edgecolors="black",transform=ccrs.PlateCarree())
                       except:
                           print("Cross-section ", ar,
                             " does not exist for flight ",f)
                       last_hour=12
                       if ar == "AR3":
                          #try:
                          #    last_hour=ar_aircraft_position[f].index[-1].hour+1
                          #except:
                              last_hour=12
                   AR_field=pd.DataFrame(np.array(AR_era_ds.kidmap[0,AR_era_data["model_runs"].start+2,0,:,:])) 
                   AR_field.replace(to_replace=AR_unique,value=np.nan)
                   
                   x,y=np.meshgrid(AR_era_ds.lon,AR_era_ds.lat)
                   if last_hour<6:
                       AR_c=ax.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                    AR_era_ds.kidmap[0,AR_era_data["model_runs"].start,0,:,:],
                                    levels=30,
                                    cmap=rf_colors[color],linewidths=2,alpha=0.3,
                                    transform=ccrs.PlateCarree())
                   
                   elif 6<=last_hour<12:
                       AR_c=ax.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                    AR_era_ds.kidmap[0,AR_era_data["model_runs"].start+1,0,:,:],
                                    cmap=rf_colors[color],linewidths=2,alpha=0.3,
                                    transform=ccrs.PlateCarree())
                   elif 12<=last_hour<18:
                       AR_c=ax.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                    AR_era_ds.kidmap[0,AR_era_data["model_runs"].start+2,0,:,:],
                                    cmap=rf_colors[color],linewidths=2,alpha=0.3,
                                    transform=ccrs.PlateCarree())
                   #else:
                   #    AR_c=ax.contour(AR_era_ds.lon,AR_era_ds.lat,
                   #                 AR_era_ds.kidmap[0,AR_era_data["model_runs"].start+3,0,:,:],
                   #                 cmap=rf_colors[color],alpha=0.6,
                   #                 transform=ccrs.PlateCarree())
                   
                   """
                       AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                       AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                       hatches=['//'],cmap='bone_r',alpha=0.1,
                       transform=ccrs.PlateCarree())
                       AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                         AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                                         hatches=[ '//'],cmap='bone_r',alpha=0.1,
                                         transform=ccrs.PlateCarree())
                       AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                         AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                                         hatches=['//'],cmap='bone_r',alpha=0.1,
                                         transform=ccrs.PlateCarree())
                        AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                                          AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                                          hatches=['//'],cmap='bone_r',alpha=0.1,
                                          transform=ccrs.PlateCarree())
                    """
                   
                   AR_field=AR_field.replace(to_replace=np.nan,value=-999)
                   
                   AR_unique=np.append(AR_unique,np.unique(np.array(AR_field)))
                   
                   i+=1
       #from mpl_toolkits.axes_grid1.inset_locator import inset_axes
       #axins1=inset_axes(ax,width="3%",
       #                   height="80%",
       #                   loc="center",
       #                   bbox_to_anchor=(0.55,0,1,1),
       #                   bbox_transform=ax.transAxes,
       #                   borderpad=0)        
       #cb=all_rfs_map_fig.colorbar(AR_c,cax=axins1)
       #cb.set_label("Atmospheric River: ID")                     
                
       #if sea_ice_desired:
       #     print("Add mean sea ice extension.")
       #     print("For now this is work in progress and pass")
       #     pass
       
       
       if include_ARs:
           legend_title="AR cross-sections:"
       else: 
           legend_title="Research Flights"
       ax.legend(custom_lines, self.interested_flights,
                 fontsize=16,loc="best",title=legend_title)        
       file_name=campaign_cls.name+"_Research_Flights_Map.png"
       if include_ARs:
           file_name="ARs_"+file_name
       #plt.show()
       all_rfs_map_fig.savefig(campaign_cls.plot_path+file_name,
                       dpi=600,bbox_inches="tight")
       print("Map Figure will be saved as: ",campaign_cls.plot_path+file_name)
       print("For now this function is in current work and won't be do more")
       return None
    
    # Define AR_Crossing Flight Mapping
    def plot_flight_map_AR_crossing(self,era_on_halo_cls,cut_radar,
                                    Dropsondes,campaign_cls,
                                    opt_plot_path=os.getcwd(),
                                    invert_flight=False):
        """
        

        Parameters
        ----------
        cut_radar : dict
            Dictionary containing the radar data cutted to the AR. 
            assessed by Cloudnet_Data.get_cloudnet_station_coordinates
        dropsondes : dict 
            Dictionary containing the dropsonde releases and their data.
        campaign_cls : class
            class of the flight campaign, for now applicable for NAWDEX
        flight : str
            flight to analyse
        AR_number : str
            id of AR Crossing this figure considers
        Returns
        -------
        None.

        """
        import matplotlib
        import cartopy.crs as ccrs
        
        import atmospheric_rivers as AR
        
        #if not "ERA" in sys.modules:
        import reanalysis
            #from era5_on_halo_backup import ERA5
        #if not "ICON" in sys.modules:
        #    import ICON
        #if not "ERA_on_HALO" in sys.modules:
        #    from Grid_on_HALO import ERA_on_HALO
        if self.flight.endswith("instantan"):
            flight_str=str.split(self.flight,"_")[0]
        else:
            flight_str=self.flight
        ERA5_on_HALO=era_on_halo_cls
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        set_font=16
        matplotlib.rcParams.update({'font.size':set_font})
                
        
        # Define the plot specifications for the given variables
        met_var_dict={}
        met_var_dict["ERA_name"]    = {"IWV":"tcwv","IVT":"IVT"}
        met_var_dict["colormap"]    = {"IWV":"density","IVT":"speed"}
        met_var_dict["levels"]      = {"IWV":np.linspace(10,50,51),
                                       "IVT":np.linspace(50,600,61)}
        met_var_dict["units"]       = {"IWV":"(kg/$\mathrm{m}^2$)",
                                       "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
        
        if campaign_cls.is_flight_campaign:
            AR=AR.Atmospheric_Rivers("ERA")
            flight_date=campaign_cls.years[flight_str]+"-"+\
                            campaign_cls.flight_month[flight_str]
            flight_date=flight_date+"-"+\
                        campaign_cls.flight_day[flight_str]
            AR_era_ds=AR.open_AR_catalogue(after_2019=int(flight_date[0:4])>2019)
            
            AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)

        
        plot_path=opt_plot_path
        hydrometeor_lvls_path=campaign_cls.campaign_path+"/data/ERA-5/"
    
        file_name="total_columns_"+campaign_cls.years[flight_str]+"_"+\
                    campaign_cls.flight_month[flight_str]+"_"+\
                    campaign_cls.flight_day[flight_str]+".nc"    
        
        era5=reanalysis.ERA5(for_flight_campaign=True,campaign="NAWDEX",
                  research_flights=self.flight,
                  era_path=campaign_cls.campaign_path+"/data/ERA-5/")
        
        ds,era_path=era5.load_era5_data(file_name)
        
        #IVT Processing
        ds["IVT_v"]=ds["p72.162"]
        ds["IVT_u"]=ds["p71.162"]
        ds["IVT"]=np.sqrt(ds["IVT_u"]**2+ds["IVT_v"]**2)
        
        
        #Aircraft Position
        if not self.synthetic_campaign:
            if campaign_cls.is_flight_campaign:
                halo_dict=campaign_cls.get_aircraft_position([flight_str],
                                                         campaign_cls.name)
            #print(halo_dict)
            else:
                # Load Halo Dataset
                halo_waypoints=campaign_cls.get_aircraft_waypoints(filetype=".csv")
                if invert_flight:
                    halo_waypoints=campaign_cls.invert_flight_from_waypoints(
                                halo_waypoints,[flight_str])
                halo_dict={}
                for flight in campaign_cls.interested_flights:
                    halo_dict[flight_str]=campaign_cls.\
                                        interpolate_flight_from_waypoints(\
                                        halo_waypoints[flight_str])
        
            halo_df=halo_dict[flight_str] 
        else:
            import flight_track_creator
            
            Tracker=flight_track_creator.Flighttracker(campaign_cls,
                                                flight_str,
                                                self.ar_of_day,
                                                track_type=self.track_type,
                                                shifted_lat=self.synthetic_icon_lat,
                                                shifted_lon=self.synthetic_icon_lon)
            halo_df,cmpgn_path=Tracker.run_flight_track_creator()
            if isinstance(halo_df,dict):
                halo_dict=halo_df.copy()
                halo_df,time_legs_df=Tracker.concat_track_dict_to_df(
                                                        merge_all=False,
                                                        pick_legs=self.pick_legs)
            
            print("Synthetic flight track loaded")
        
        halo_df=halo_df.rename(columns={"Lon":"longitude",
                                 "Lat":"latitude"})
        
        map_fig=plt.figure(figsize=(18,12))
        
        ax1 = plt.subplot(1,2,1,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=-5.0,central_latitude=60))
        ax1.coastlines(resolution="50m")
        gl1=ax1.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
        
        ax2 = plt.subplot(1,2,2,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=-5.0,central_latitude=60))
        ax2.coastlines(resolution="50m")
        gl2=ax2.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
        
        gl1.top_labels=True
        gl2.top_labels=True
        gl1.bottom_labels=False
        gl2.bottom_labels=False
        
        gl2.left_labels=False
        
        gl1.right_labels=False
        gl2.right_labels=False
        
        ticklabel_color="dimgrey"
        tick_size=14
        gl1.xlabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        gl2.xlabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        gl1.ylabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        gl2.ylabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        
        #     if flight=="RF01" or flight=="RF13":
        lat_extension=2.0
        lon_extension=2.0
            # 
        axins1=inset_axes(ax1,width="3%",
                              height="80%",
                              loc="center",
                              bbox_to_anchor=(0.55,0,1,1),
                              bbox_transform=ax1.transAxes,
                              borderpad=0)        
            
        #Plot HALO flight course
        ax1.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),
                  color="salmon",linestyle='--',linewidth=1.0,alpha=0.9)    
        ax2.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),color="salmon",
                  linestyle='--',linewidth=1.0,alpha=0.9)    
        
        
        
        if not cut_radar=={}:
            last_minute=cut_radar["Reflectivity"].index.minute[-1]
            if last_minute>30:
                last_hour=cut_radar["Reflectivity"].index.hour[-1]+1
            else:
                last_hour=cut_radar["Reflectivity"].index.hour[-1]
            # ERA data
            C1=ax1.contourf(ds["longitude"],ds["latitude"],
                            ds[met_var_dict["ERA_name"]["IWV"]][last_hour,:,:],
                            levels=met_var_dict["levels"]["IWV"],
                            extend="max",transform=ccrs.PlateCarree(),
                            cmap=met_var_dict["colormap"]["IWV"],alpha=0.95)
        
            # Plot AR Detection 
            plt.rcParams.update({'hatch.color': 'lightgrey'})
                
            cb=map_fig.colorbar(C1,cax=axins1)
            cb.set_label("IWV"+" "+met_var_dict["units"]["IWV"])
            cb.set_ticks([0,10,20,30,40,50])
            
            C2=ax2.contourf(ds["longitude"],ds["latitude"],
                            ds[met_var_dict["ERA_name"]["IVT"]][last_hour,:,:],
                            levels=met_var_dict["levels"]["IVT"],
                            extend="max",transform=ccrs.PlateCarree(),
                            cmap=met_var_dict["colormap"]["IVT"],alpha=0.95)
            axins2=inset_axes(ax2,width="3%",
                          height="80%",
                          loc="center",
                          bbox_to_anchor=(0.55,0,1,1),
                          bbox_transform=ax2.transAxes,
                          borderpad=0)        
        
            cb2=map_fig.colorbar(C2,cax=axins2)
            cb2.set_label("IVT"+" "+met_var_dict["units"]["IVT"])
            cb2.set_ticks([50,100,150,200,250,300,350,400,450,500,550,600])
        
            #Identify periods of strong radar reflectivity
            if not self.synthetic_campaign:
                high_dbZ_index=cut_radar["Reflectivity"][\
                                    cut_radar["Reflectivity"]>15].any(axis=1)
                high_dbZ=cut_radar["Reflectivity"].loc[high_dbZ_index]
        
            start_pos=halo_df.loc[cut_radar["Reflectivity"].index[0]]
            end_pos=halo_df.loc[cut_radar["Reflectivity"].index[-1]]
            ax1.scatter(halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                             cut_radar["Reflectivity"].index[-1]],
                        halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                         cut_radar["Reflectivity"].index[-1]],
                        transform=ccrs.PlateCarree(),marker=".",s=3,color="red",
                        alpha=0.95,zorder=1)    
            ax2.scatter(halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                             cut_radar["Reflectivity"].index[-1]],
                        halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                         cut_radar["Reflectivity"].index[-1]],
                        transform=ccrs.PlateCarree(),marker='.',s=3,
                        color="red",alpha=0.95,zorder=1)    
            #------------------------------------------------------------------#
            ## Add quiver
            step=15
            quiver_lon=np.array(ds["longitude"][::step])
            quiver_lat=np.array(ds["latitude"][::step])
            u=np.array(ds["IVT_u"][last_hour,::step,::step])
            v=np.array(ds["IVT_v"][last_hour,::step,::step])
            #u[u<50]=np.nan
            v[v<50]=np.nan
            quiver=ax1.quiver(quiver_lon,quiver_lat,u,v,color="white",
                              edgecolor="k",linewidth=1,scale=900,
                              scale_units="inches",pivot="mid",
                              width=0.008,transform=ccrs.PlateCarree())
            quiver2=ax2.quiver(quiver_lon,quiver_lat,u,v,color="lightskyblue",
                               edgecolor="k",linewidth=1,
                               scale=900,scale_units="inches",
                               pivot="mid",width=0.008,
                               transform=ccrs.PlateCarree())
        
            #-----------------------------------------------------------------#    
            if not self.synthetic_campaign:
                #Plot high reflectivity values
                ax1.scatter(halo_df["longitude"].loc[high_dbZ.index],
                    halo_df["latitude"].loc[high_dbZ.index],
                    s=30,color="white",marker="D",linewidths=0.5,
                    label="Radar dbZ > 15",edgecolor="k",
                    transform=ccrs.PlateCarree())
        
                ax2.scatter(halo_df["longitude"].loc[high_dbZ.index],
                    halo_df["latitude"].loc[high_dbZ.index],
                    s=30,color="white",marker="D",
                    linewidths=0.5,edgecolor="k",
                    transform=ccrs.PlateCarree())
        
        else:
            if not self.synthetic_campaign:
                AR_cutted_halo,_,ar_of_day=ERA5_on_HALO.cut_halo_to_AR_crossing(\
                                                    self.ar_of_day,flight_str, 
                                                    halo_df,None,
                                                    campaign=campaign_cls.name,
                                                    device="halo",
                                                    invert_flight=invert_flight)
            else:
                AR_cutted_halo=halo_df.copy()
                AR_cutted_halo.index=pd.DatetimeIndex(AR_cutted_halo.index)
            last_hour=AR_cutted_halo.index.hour[-1]
            start_pos=AR_cutted_halo.iloc[0]
            end_pos  =AR_cutted_halo.iloc[-1]
            # ERA data
            C1=ax1.contourf(ds["longitude"],ds["latitude"],
                            ds[met_var_dict["ERA_name"]["IWV"]][last_hour,:,:],
                            levels=met_var_dict["levels"]["IWV"],
                            extend="max",transform=ccrs.PlateCarree(),
                            cmap=met_var_dict["colormap"]["IWV"],alpha=0.95)
        
            # Plot AR Detection 
            plt.rcParams.update({'hatch.color': 'lightgrey'})
                
            cb=map_fig.colorbar(C1,cax=axins1)
            cb.set_label("IWV"+" "+met_var_dict["units"]["IWV"])
            cb.set_ticks([10,20,30,40,50])
            
            C2=ax2.contourf(ds["longitude"],ds["latitude"],
                            ds[met_var_dict["ERA_name"]["IVT"]][last_hour,:,:],
                            levels=met_var_dict["levels"]["IVT"],
                            extend="max",transform=ccrs.PlateCarree(),
                            cmap=met_var_dict["colormap"]["IVT"],alpha=0.95)
            axins2=inset_axes(ax2,width="3%",
                          height="80%",
                          loc="center",
                          bbox_to_anchor=(0.55,0,1,1),
                          bbox_transform=ax2.transAxes,
                          borderpad=0)        
        
            cb2=map_fig.colorbar(C2,cax=axins2)
            cb2.set_label("IVT"+" "+met_var_dict["units"]["IVT"])
            cb2.set_ticks([50,100,200,300,400,500,600,700,800,900,1000])
            
            #------------------------------------------------------------------#
            ## Add quiver
            step=10
            quiver_lon=np.array(ds["longitude"][::step])
            quiver_lat=np.array(ds["latitude"][::step])
            u=pd.DataFrame(np.array(ds["IVT_u"][last_hour,::step,::step]))
            v=pd.DataFrame(np.array(ds["IVT_v"][last_hour,::step,::step]))
            #u=u.where(abs(u)>100,np.nan)
            v=v.where(abs(v)>200,np.nan)
            quiver=ax2.quiver(quiver_lon,quiver_lat,u.values,v.values,
                              color="orange",edgecolor="k",lw=1,
                              scale=800,scale_units="inches",
                              pivot="mid",width=0.005,
                              transform=ccrs.PlateCarree())
            quiver2=ax1.quiver(quiver_lon,quiver_lat,u.values,v.values,
                               color="white",edgecolor="k",lw=1,
                               scale=800,scale_units="inches",
                               pivot="mid",width=0.005,
                               transform=ccrs.PlateCarree())
        
            #-----------------------------------------------------------------#    
        
            ax1.scatter(AR_cutted_halo["longitude"],
                        AR_cutted_halo["latitude"],
                        transform=ccrs.PlateCarree(),marker=".",
                        s=3,color="red",
                        alpha=0.95,zorder=1)    
            ax2.scatter(AR_cutted_halo["longitude"],
                        AR_cutted_halo["latitude"],
                        transform=ccrs.PlateCarree(),marker='.',s=3,
                        color="red",alpha=0.95,zorder=1)    
        
            
        print("Hour of the day:",last_hour)
        calc_time=era5.hours[last_hour]
        
        deg_ratio=(start_pos["longitude"]-end_pos["longitude"])/\
            (start_pos["latitude"]-end_pos["latitude"])
        print("Meridional-Zonal ratio: ",deg_ratio)
        resizing_done=False
        if abs((start_pos["longitude"]-end_pos["longitude"])/\
               (start_pos["latitude"]-end_pos["latitude"]))>1.5:
            lat_extension=4
            resizing_done=True
        elif abs((start_pos["latitude"]-end_pos["latitude"])/\
                 (start_pos["longitude"]-end_pos["longitude"]))>1.5:
            lon_extension=4
            resizing_done=True
        else:
            pass
        lower_lon=np.min([start_pos["longitude"],end_pos["longitude"]])-\
                                                lon_extension*5
        upper_lon=np.max([start_pos["longitude"],end_pos["longitude"]])+\
                                                lon_extension*5
        lower_lat=np.min([start_pos["latitude"],end_pos["latitude"]])-\
                                                lat_extension*5
        upper_lat=np.max([start_pos["latitude"],end_pos["latitude"]])+\
                                                lat_extension*5
        if upper_lat>90:
            upper_lat=90
        ax1.set_extent([lower_lon,upper_lon,lower_lat,upper_lat])
        ax2.set_extent([lower_lon,upper_lon,lower_lat,upper_lat])
        
            
        
        if not campaign_cls.is_synthetic_campaign:
            if last_hour<6:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                    hatches=['//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
        
            elif 6<=last_hour<12:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                        hatches=[ '//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                    hatches=[ '//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
        
            
            elif 12<=last_hour<18:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                    hatches=['//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
            else:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())    
                ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                    hatches=['//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
            AR_C.set_label="AR (Guan & Waliser, 2019)"
            
        
                
        #plot Dropsonde releases
        date=campaign_cls.year+campaign_cls.flight_month[flight_str]
        date=date+campaign_cls.flight_day[flight_str]
        #          if not flight=="RF06":                           
        #              Dropsondes=campaign_cls.load_dropsonde_data(\
        #              date,print_arg="yes",
        #              dt="all",plotting="no")
        
        # in some cases the Dropsondes variable can be a dataframe or
        # just a series, if only one sonde has been released
        
        if not Dropsondes=={}:
            if isinstance(Dropsondes["Lat"],pd.DataFrame):
                dropsonde_releases=pd.DataFrame(\
                            index=pd.DatetimeIndex(Dropsondes["LTS"].index))
                dropsonde_releases["Lat"]=Dropsondes["Lat"].loc[:,"6000.0"].values
                dropsonde_releases["Lon"]=Dropsondes["Lon"].loc[:,"6000.0"].values
        
            else:
                index_var=Dropsondes["Time"].loc["6000.0"]
                dropsonde_releases=pd.Series()
                dropsonde_releases["Lat"]=np.array(Dropsondes["Lat"].loc["6000.0"])
                dropsonde_releases["Lon"]=np.array(Dropsondes["Lon"].loc["6000.0"])
                dropsonde_releases["Time"]=index_var
        
            if not self.flight=="RF08":
                relevant_dropsondes=dropsonde_releases.loc[\
                                        cut_radar["Reflectivity"].index[0]:\
                                        cut_radar["Reflectivity"].index[-1]]
                if relevant_dropsondes.shape[0]>0:
                    ax1.scatter(relevant_dropsondes["Lon"],
                            relevant_dropsondes["Lat"],
                            s=100,marker="v",color="orange",
                            edgecolors="black",label="Dropsondes",
                            transform=ccrs.PlateCarree())
                
                    ax2.scatter(relevant_dropsondes["Lon"],
                            relevant_dropsondes["Lat"],
                            s=100,marker="v",color="orange",
                            edgecolors="black",label="Dropsondes",
                            transform=ccrs.PlateCarree())
        
        if not resizing_done:    
            map_fig.suptitle(campaign_cls.name+" ERA-5 data for "+\
                             flight_str+": "+campaign_cls.year+"-"+\
                             campaign_cls.flight_month[flight_str]+"-"+\
                             campaign_cls.flight_day[flight_str]+" "+\
                             calc_time,y=0.94)
        else:
            map_fig.suptitle(campaign_cls.name+" ERA-5 data for "+\
                             flight_str+": "+campaign_cls.year+"-"+\
                             campaign_cls.flight_month[flight_str]+\
                             "-"+campaign_cls.flight_day[flight_str]+\
                             " "+calc_time,y=0.94)
        legend=ax1.legend(bbox_to_anchor=(0.65,-0.15,1.5,0),
                          facecolor='lightgrey',
                          loc="lower center",
                          ncol=2,mode="expand")
        
        legend.get_frame().set_linewidth(2.0)
        legend.get_frame().set_edgecolor("k")
        
        #Save figure
        fig_name=self.ar_of_day+"_"+campaign_cls.name+"_"+\
                    self.flight+'_MAP_ERA5'+".png"
        if not self.plot_path==None:
            fig_path=self.plot_path
        else:
            fig_path=opt_plot_path
        print("PLot path ",fig_path)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        map_fig.savefig(fig_path+fig_name,bbox_inches="tight",dpi=300)
        print("Figure saved as:",fig_path+fig_name)
            
        return None
    
    
    
    
    # Moisture budget analysis
    def plot_AR_moisture_components_map(self,era_on_halo_cls,cut_radar,
                                    Dropsondes,campaign_cls,
                                    opt_plot_path=os.getcwd(),
                                    invert_flight=False):
        """
        

        Parameters
        ----------
        cut_radar : dict
            Dictionary containing the radar data cutted to the AR. 
            assessed by Cloudnet_Data.get_cloudnet_station_coordinates
        dropsondes : dict 
            Dictionary containing the dropsonde releases and their data.
        campaign_cls : class
            class of the flight campaign, for now applicable for NAWDEX
        flight : str
            flight to analyse
        AR_number : str
            id of AR Crossing this figure considers
        Returns
        -------
        None.

        """
        import matplotlib
        import cartopy.crs as ccrs
        
        import atmospheric_rivers as AR
        import reanalysis as Reanalysis
        ERA5_on_HALO=era_on_halo_cls
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        set_font=16
        matplotlib.rcParams.update({'font.size':set_font})
                
        
        # Define the plot specifications for the given variables
        met_var_dict={}
        met_var_dict["ERA_name"]    = {"EV":"e","TP":"tp",
                                       "IWV":"tcwv","IVT":"IVT",
                                       "IVT_conv":"IVT_conv"}
        met_var_dict["colormap"]    = {"EV":"Blues","IVT_conv":"BrBG_r",
                                       "TP":"Blues","IVT":"speed"}
        met_var_dict["levels"]      = {"IWV":np.linspace(10,50,51),
                                       "EV":np.linspace(0,1.5,51),
                                       "TP":np.linspace(0,1.5,51),
                                       "IVT_conv":np.linspace(-2,2,101),
                                       "IVT":np.linspace(50,600,61)}
        met_var_dict["units"]       = {"EV":"(kg$\mathrm{m}^{-2}$)",
                                       "TP":"(kg$\mathrm{m}^{-2}$)",
                                       "IVT_conv":"(kg$\mathrm{m}^{-2}$)",
                                       "IWV":"(kg$\mathrm{m}^{-2}\mathrm{h}^{-1}$)",
                                       "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
        
        if self.flight.endswith("instantan"):
            flight_str=str.split(self.flight,"_")[0]
        else:
            flight_str=self.flight
        if campaign_cls.is_flight_campaign:
            flight_date=campaign_cls.years[flight_str]+"-"+\
                            campaign_cls.flight_month[flight_str]
            flight_date=flight_date+"-"+\
                        campaign_cls.flight_day[flight_str]
            
            AR=AR.Atmospheric_Rivers("ERA")
            AR_era_ds=AR.open_AR_catalogue(after_2019=int(flight_date[0:4])>2019)
            
            AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)
        
        
        file_name="total_columns_"+campaign_cls.years[flight_str]+"_"+\
                    campaign_cls.flight_month[flight_str]+"_"+\
                    campaign_cls.flight_day[flight_str]+".nc"    
        
        era5=Reanalysis.ERA5(for_flight_campaign=True,campaign="NAWDEX",
                  research_flights=self.flight,
                  era_path=campaign_cls.campaign_path+"/data/ERA-5/")
        
        ds,era_path=era5.load_era5_data(file_name)
        
        #IVT Processing
        ds["IVT_v"]=ds["p72.162"]
        ds["IVT_u"]=ds["p71.162"]
        ds["IVT"]=np.sqrt(ds["IVT_u"]**2+ds["IVT_v"]**2)
        ds["IVT_conv"]=ds["p84.162"]*3600 # units in seconds
        ds["e"]=ds["e"]*-1000
        ds["tp"]=ds["tp"]*1000
        #ds["TP"]=ds["tp"]
        
        
        #Aircraft Position
        if not self.synthetic_campaign:
            if campaign_cls.is_flight_campaign:
                halo_dict=campaign_cls.get_aircraft_position([flight_str],
                                                         campaign_cls.name)
        else:
            if campaign_cls.name=="HALO_AC3_Dry_Run":
            # Load Halo Dataset
                halo_waypoints=campaign_cls.get_aircraft_waypoints(filetype=".csv")
                if invert_flight:
                    halo_waypoints=campaign_cls.invert_flight_from_waypoints(
                                halo_waypoints,[flight_str])
                halo_dict={}
                for flight in campaign_cls.interested_flights:
                    halo_dict[flight_str]=campaign_cls.\
                                        interpolate_flight_from_waypoints(\
                                        halo_waypoints[flight_str])
        
                halo_df=halo_dict[flight_str] 
            elif campaign_cls.name=="NA_February_Run" or \
                campaign_cls.name=="NAWDEX" or \
                    campaign_cls.name=="Second_Synthetic_Study":
                import flight_track_creator
                Tracker=flight_track_creator.Flighttracker(
                                                campaign_cls,
                                                flight_str,
                                                self.ar_of_day,
                                                track_type=self.track_type,
                                                shifted_lat=self.synthetic_icon_lat,
                                                shifted_lon=self.synthetic_icon_lon)
            
            halo_df,cmpgn_path=Tracker.run_flight_track_creator()
            if isinstance(halo_df,dict):
                halo_dict=halo_df.copy()
                halo_df,time_legs_df=Tracker.concat_track_dict_to_df(
                                                    merge_all=False,
                                                    pick_legs=self.pick_legs)
            
            print("Synthetic flight track loaded")
        if not "halo_df" in locals():
            halo_df=halo_dict[self.flight]
        halo_df=halo_df.rename(columns={"Lon":"longitude",
                                 "Lat":"latitude"})
        halo_df.index=pd.DatetimeIndex(halo_df.index)
        #map_fig=plt.figure(figsize=(13,10))
        
        map_fig=plt.figure(figsize=(22,20))
        
        ax1 = plt.subplot(2,2,1,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=-5.0,central_latitude=60))
        ax1.coastlines(resolution="50m")
        gl1=ax1.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
        
        ax2 = plt.subplot(2,2,2,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=-5.0,central_latitude=60))
        ax2.coastlines(resolution="50m")
        gl2=ax2.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
        
        ax3 = plt.subplot(2,2,3,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=-5.0,central_latitude=60))
        ax3.coastlines(resolution="50m")
        gl3=ax3.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
        
        ax4 = plt.subplot(2,2,4,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=-5.0,central_latitude=60))
        ax4.coastlines(resolution="50m")
        gl4=ax4.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
        
        gl1.top_labels=True
        gl2.top_labels=True
        gl1.bottom_labels=False
        gl2.bottom_labels=False
        
        gl2.left_labels=False
        
        gl1.right_labels=False
        gl2.right_labels=False
        
        gl3.top_labels=True
        gl4.top_labels=True
        gl3.bottom_labels=False
        gl4.bottom_labels=False
        
        gl3.left_labels=False
        
        gl3.right_labels=False
        gl4.right_labels=False
        
        ticklabel_color="dimgrey"
        tick_size=14
        gl1.xlabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        gl2.xlabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        gl1.ylabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        gl2.ylabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        
        #     if flight=="RF01" or flight=="RF13":
        lat_extension=2.0
        lon_extension=2.0
            # 
        axins1=inset_axes(ax1,width="3%",
                              height="80%",
                              loc="center",
                              bbox_to_anchor=(0.55,0,1,1),
                              bbox_transform=ax1.transAxes,
                              borderpad=0)        
            
        #Plot HALO flight course
        ax1.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),
                  color="salmon",linestyle='--',linewidth=1.0,alpha=0.9)    
        ax2.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),color="salmon",
                  linestyle='--',linewidth=1.0,alpha=0.9)    
        ax3.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),
                  color="salmon",linestyle='--',linewidth=1.0,alpha=0.9)    
        ax4.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),color="salmon",
                  linestyle='--',linewidth=1.0,alpha=0.9)
        
        
        if not cut_radar=={}:
            last_minute=cut_radar["Reflectivity"].index.minute[-1]
            if last_minute>30:
                last_hour=cut_radar["Reflectivity"].index.hour[-1]+1
            else:
                last_hour=cut_radar["Reflectivity"].index.hour[-1]
        else:
            last_minute=halo_df.index.minute[-1]
            if last_minute>30:
                last_hour=halo_df.index.hour[-1]+1
            else:
                last_hour=halo_df.index.hour[-1]
            
        # ERA data
        C1=ax1.contourf(ds["longitude"],ds["latitude"],
                            ds[met_var_dict["ERA_name"]["IVT_conv"]][last_hour,:,:],
                            levels=met_var_dict["levels"]["IVT_conv"],
                            extend="both",transform=ccrs.PlateCarree(),
                            cmap=met_var_dict["colormap"]["IVT_conv"],alpha=0.95)
        print("IVT conv. mapped")
        
        # Plot AR Detection 
        plt.rcParams.update({'hatch.color': 'lightgrey'})
                
        cb=map_fig.colorbar(C1,cax=axins1)
        cb.set_label("$ div\,IVT$"+" "+met_var_dict["units"]["IVT_conv"])
        cb.set_ticks([-1.0,0,1.0])
            
        C2=ax2.contourf(ds["longitude"],ds["latitude"],
                            ds[met_var_dict["ERA_name"]["EV"]][last_hour,:,:],
                            levels=met_var_dict["levels"]["EV"],
                            extend="max",transform=ccrs.PlateCarree(),
                            cmap=met_var_dict["colormap"]["EV"],alpha=0.95)
        print("Evaporation mapped")
        
        axins2=inset_axes(ax2,width="3%",
                          height="80%",
                          loc="center",
                          bbox_to_anchor=(0.55,0,1,1),
                          bbox_transform=ax2.transAxes,
                          borderpad=0)        
            
        cb2=map_fig.colorbar(C2,cax=axins2)
        cb2.set_label("EV"+" "+met_var_dict["units"]["EV"])
        cb2.set_ticks([0,0.5,1.0,1.5])
            
        C3=ax3.contourf(ds["longitude"],ds["latitude"],
                        ds[met_var_dict["ERA_name"]["TP"]][last_hour,:,:],
                        levels=met_var_dict["levels"]["TP"],
                        extend="max",transform=ccrs.PlateCarree(),
                        cmap=met_var_dict["colormap"]["TP"],alpha=0.95)
        axins3=inset_axes(ax3,width="3%",
                          height="80%",
                          loc="center",
                          bbox_to_anchor=(0.55,0,1,1),
                          bbox_transform=ax3.transAxes,
                          borderpad=0)        
        
        cb3=map_fig.colorbar(C3,cax=axins3)
        cb3.set_label("TP"+" "+met_var_dict["units"]["TP"])
        cb3.set_ticks([0,0.5,1.0,1.5])
        print("Total precipitation mapped")
            
        dIWV_dt=(ds["tcwv"][last_hour+1,:,:]-ds["tcwv"][last_hour-1,:,:])/2
        C4=ax4.contourf(ds["longitude"],ds["latitude"],dIWV_dt,
                        levels=met_var_dict["levels"]["IVT_conv"],
                        extend="both",transform=ccrs.PlateCarree(),
                        cmap="gist_earth_r",alpha=0.95)
        axins4=inset_axes(ax4,width="3%",
                          height="80%",
                          loc="center",
                          bbox_to_anchor=(0.55,0,1,1),
                          bbox_transform=ax4.transAxes,
                          borderpad=0)        
        print("IWV Tendency mapped")
        
        cb4=map_fig.colorbar(C4,cax=axins4)
        cb4.set_label("$\delta \mathrm{IWV}/ \delta \mathrm{t} $"+" "+
                          met_var_dict["units"]["IWV"])
        cb4.set_ticks([-1.5,-1.0,-0.5,0,0.5,1.0,1.5])
      
        budget_epsilon=ds[met_var_dict["ERA_name"]["TP"]]-\
                ds[met_var_dict["ERA_name"]["EV"]]+\
                    ds[met_var_dict["ERA_name"]["IVT_conv"]]
            
            
        #Identify periods of strong radar reflectivity
        if not self.synthetic_campaign:
                high_dbZ_index=cut_radar["Reflectivity"][\
                                    cut_radar["Reflectivity"]>15].any(axis=1)
                high_dbZ=cut_radar["Reflectivity"].loc[high_dbZ_index]
        else:
            # Just for not creating coding mess-up, no radar data is given
            cut_radar=dict()
            cut_radar["Reflectivity"]=halo_df.copy()
        start_pos=halo_df.loc[cut_radar["Reflectivity"].index[0]]
        end_pos=halo_df.loc[cut_radar["Reflectivity"].index[-1]]
        ax1.scatter(halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                             cut_radar["Reflectivity"].index[-1]],
                    halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                     cut_radar["Reflectivity"].index[-1]],
                    transform=ccrs.PlateCarree(),marker=".",s=3,color="red",
                    alpha=0.95,zorder=1)    
        ax2.scatter(halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                             cut_radar["Reflectivity"].index[-1]],
                    halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                         cut_radar["Reflectivity"].index[-1]],
                        transform=ccrs.PlateCarree(),marker='.',s=3,
                        color="red",alpha=0.95,zorder=1)    
        ax3.scatter(halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                             cut_radar["Reflectivity"].index[-1]],
                    halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                         cut_radar["Reflectivity"].index[-1]],
                    transform=ccrs.PlateCarree(),marker='.',s=3,
                    color="red",alpha=0.95,zorder=1)    
        
        ax4.scatter(halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                             cut_radar["Reflectivity"].index[-1]],
                    halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                         cut_radar["Reflectivity"].index[-1]],
                    transform=ccrs.PlateCarree(),marker='.',s=3,
                    color="red",alpha=0.95,zorder=1)    
        #------------------------------------------------------------------#
        ## Add quiver
        step=15
        quiver_lon=np.array(ds["longitude"][::step])
        quiver_lat=np.array(ds["latitude"][::step])
        u=np.array(ds["IVT_u"][last_hour,::step,::step])
        v=np.array(ds["IVT_v"][last_hour,::step,::step])
        #u[u<50]=np.nan
        v[v<100]=np.nan
        quiver=ax1.quiver(quiver_lon,quiver_lat,u,v,color="white",
                          edgecolor="k",linewidth=1,scale=900,
                          scale_units="inches",pivot="mid",
                          width=0.008,transform=ccrs.PlateCarree())
        quiver2=ax2.quiver(quiver_lon,quiver_lat,u,v,color="white",
                           edgecolor="k",linewidth=1,
                           scale=900,scale_units="inches",
                           pivot="mid",width=0.008,
                           transform=ccrs.PlateCarree())
        quiver3=ax3.quiver(quiver_lon,quiver_lat,u,v,color="lightskyblue",
                           edgecolor="k",linewidth=1,
                           scale=900,scale_units="inches",
                           pivot="mid",width=0.008,
                           transform=ccrs.PlateCarree())
        quiver4=ax4.quiver(quiver_lon,quiver_lat,u,v,color="lightskyblue",
                           edgecolor="k",linewidth=1,
                           scale=900,scale_units="inches",
                           pivot="mid",width=0.008,
                           transform=ccrs.PlateCarree())
        #-----------------------------------------------------------------#    
        if not self.synthetic_campaign:
            #Plot high reflectivity values
            ax1.scatter(halo_df["longitude"].loc[high_dbZ.index],
                    halo_df["latitude"].loc[high_dbZ.index],
                    s=30,color="white",marker="D",linewidths=0.5,
                    label="Radar dbZ > 15",edgecolor="k",
                    transform=ccrs.PlateCarree())
        
            ax2.scatter(halo_df["longitude"].loc[high_dbZ.index],
                    halo_df["latitude"].loc[high_dbZ.index],
                    s=30,color="white",marker="D",
                    linewidths=0.5,edgecolor="k",
                    transform=ccrs.PlateCarree())
                
            ax3.scatter(halo_df["longitude"].loc[high_dbZ.index],
                    halo_df["latitude"].loc[high_dbZ.index],
                    s=30,color="white",marker="D",linewidths=0.5,
                    edgecolor="k",transform=ccrs.PlateCarree())
        
            ax4.scatter(halo_df["longitude"].loc[high_dbZ.index],
                    halo_df["latitude"].loc[high_dbZ.index],
                    s=30,color="white",marker="D",linewidths=0.5,
                    edgecolor="k",transform=ccrs.PlateCarree())
        #######################################################################
        # if no cut_radar data, e.g for Synthetic campaign or flight tracks
            AR_cutted_halo,_,ar_of_day=ERA5_on_HALO.cut_halo_to_AR_crossing(\
                                                    self.ar_of_day,flight_str, 
                                                    halo_df,None,
                                                    campaign=campaign_cls.name,
                                                    device="halo",
                                                    invert_flight=invert_flight)
        else:
                AR_cutted_halo=halo_df.copy()
                AR_cutted_halo.index=pd.DatetimeIndex(AR_cutted_halo.index)
        last_hour=AR_cutted_halo.index.hour[-1]
        start_pos=AR_cutted_halo.iloc[0]
        end_pos  =AR_cutted_halo.iloc[-1]
            # ERA data
            # C1=ax1.contourf(ds["longitude"],ds["latitude"],
            #                 ds[met_var_dict["ERA_name"]["IVT_conv"]][last_hour,:,:],
            #                 levels=met_var_dict["levels"]["IVT_conv"],
            #                 extend="both",transform=ccrs.PlateCarree(),
            #                 cmap=met_var_dict["colormap"]["IVT_conv"],alpha=0.95)
        
            # # Plot AR Detection 
            # plt.rcParams.update({'hatch.color': 'lightgrey'})
                
            # cb=map_fig.colorbar(C1,cax=axins1)
            # cb.set_label("$ div\,IVT$"+" "+met_var_dict["units"]["IVT_conv"])
            # cb.set_ticks([-1.0,0,1.0])
            
            # C2=ax2.contourf(ds["longitude"],ds["latitude"],
            #                 ds[met_var_dict["ERA_name"]["EV"]][last_hour,:,:],
            #                 levels=met_var_dict["levels"]["EV"],
            #                 extend="max",transform=ccrs.PlateCarree(),
            #                 cmap=met_var_dict["colormap"]["EV"],alpha=0.95)
            # axins2=inset_axes(ax2,width="3%",
            #               height="80%",
            #               loc="center",
            #               bbox_to_anchor=(0.55,0,1,1),
            #               bbox_transform=ax2.transAxes,
            #               borderpad=0)        
            
            # cb2=map_fig.colorbar(C2,cax=axins2)
            # cb2.set_label("EV"+" "+met_var_dict["units"]["EV"])
            # cb2.set_ticks([0,0.5,1.0,1.5])
            
            # C3=ax3.contourf(ds["longitude"],ds["latitude"],
            #                 ds[met_var_dict["ERA_name"]["TP"]][last_hour,:,:],
            #                 levels=met_var_dict["levels"]["TP"],
            #                 extend="max",transform=ccrs.PlateCarree(),
            #                 cmap=met_var_dict["colormap"]["TP"],alpha=0.95)
            # axins3=inset_axes(ax3,width="3%",
            #               height="80%",
            #               loc="center",
            #               bbox_to_anchor=(0.55,0,1,1),
            #               bbox_transform=ax3.transAxes,
            #               borderpad=0)        
        
            # cb3=map_fig.colorbar(C3,cax=axins3)
            # cb3.set_label("TP"+" "+met_var_dict["units"]["TP"])
            # cb3.set_ticks([0,0.5,1.0,1.5])
            
            # dIWV_dt=(ds["tcwv"][last_hour+1,:,:]-ds["tcwv"][last_hour-1,:,:])/2
            # C4=ax4.contourf(ds["longitude"],ds["latitude"],
            #                 dIWV_dt,
            #                 levels=met_var_dict["levels"]["IVT_conv"],
            #                 extend="both",transform=ccrs.PlateCarree(),
            #                 cmap="gist_earth_r",alpha=0.95)
            # axins4=inset_axes(ax4,width="3%",
            #               height="80%",
            #               loc="center",
            #               bbox_to_anchor=(0.55,0,1,1),
            #               bbox_transform=ax4.transAxes,
            #               borderpad=0)        
        
            # cb4=map_fig.colorbar(C4,cax=axins4)
            # cb4.set_label("$\delta \mathrm{IWV}/ \delta \mathrm{t} $"+" "+
            #               met_var_dict["units"]["IWV"])
            # cb4.set_ticks([-1.5,-1.0,-0.5,0,0.5,1.0,1.5])
      
            # budget_epsilon=ds[met_var_dict["ERA_name"]["TP"]]-\
            #     ds[met_var_dict["ERA_name"]["EV"]]+\
            #         ds[met_var_dict["ERA_name"]["IVT_conv"]]
            
            # #------------------------------------------------------------------#
            # ## Add quiver
            # step=10
            # quiver_lon=np.array(ds["longitude"][::step])
            # quiver_lat=np.array(ds["latitude"][::step])
            # u=pd.DataFrame(np.array(ds["IVT_u"][last_hour,::step,::step]))
            # v=pd.DataFrame(np.array(ds["IVT_v"][last_hour,::step,::step]))
            # u=u.where(abs(v)>100,np.nan)
            # v=v.where(abs(v)>100,np.nan)
            # quiver=ax2.quiver(quiver_lon,quiver_lat,u.values,v.values,
            #                   color="orange",edgecolor="k",lw=1,
            #                   scale=800,scale_units="inches",
            #                   pivot="mid",width=0.005,
            #                   transform=ccrs.PlateCarree())
            # quiver2=ax1.quiver(quiver_lon,quiver_lat,u.values,v.values,
            #                    color="white",edgecolor="k",lw=1,
            #                    scale=800,scale_units="inches",
            #                    pivot="mid",width=0.005,
            #                    transform=ccrs.PlateCarree())
        
            # #-----------------------------------------------------------------#    
        
        ax1.scatter(AR_cutted_halo["longitude"],
                        AR_cutted_halo["latitude"],
                        transform=ccrs.PlateCarree(),marker=".",
                        s=3,color="red",
                        alpha=0.95,zorder=1)    
        ax2.scatter(AR_cutted_halo["longitude"],
                        AR_cutted_halo["latitude"],
                        transform=ccrs.PlateCarree(),marker='.',s=3,
                        color="red",alpha=0.95,zorder=1)    
        #######################################################################
            
        print("Hour of the day:",last_hour)
        calc_time=era5.hours[last_hour]
        
        deg_ratio=(start_pos["longitude"]-end_pos["longitude"])/\
            (start_pos["latitude"]-end_pos["latitude"])
        print("Meridional-Zonal ratio: ",deg_ratio)
        resizing_done=False
        if abs((start_pos["longitude"]-end_pos["longitude"])/\
               (start_pos["latitude"]-end_pos["latitude"]))>1.5:
            lat_extension=4
            resizing_done=True
        elif abs((start_pos["latitude"]-end_pos["latitude"])/\
                 (start_pos["longitude"]-end_pos["longitude"]))>1.5:
            lon_extension=4
            resizing_done=True
        else:
            pass
        lower_lon=np.min([start_pos["longitude"],end_pos["longitude"]])-\
                                                lon_extension*5
        upper_lon=np.max([start_pos["longitude"],end_pos["longitude"]])+\
                                                lon_extension*5
        lower_lat=np.min([start_pos["latitude"],end_pos["latitude"]])-\
                                                lat_extension*5
        upper_lat=np.max([start_pos["latitude"],end_pos["latitude"]])+\
                                                lat_extension*5
        if upper_lat>90:
            upper_lat=90
        ax1.set_extent([lower_lon,upper_lon,lower_lat,upper_lat])
        ax2.set_extent([lower_lon,upper_lon,lower_lat,upper_lat])
        ax3.set_extent([lower_lon,upper_lon,lower_lat,upper_lat])
        ax4.set_extent([lower_lon,upper_lon,lower_lat,upper_lat])
        
            
        
        if not campaign_cls.is_synthetic_campaign:
            if last_hour<6:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                    hatches=['//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
                ax3.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                ax4.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                    hatches=['//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
        
            elif 6<=last_hour<12:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                        hatches=[ '//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                    hatches=[ '//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
                ax3.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                ax4.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                    hatches=['//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
            
            elif 12<=last_hour<18:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                    hatches=['//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
                ax3.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                ax4.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                    hatches=['//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
            else:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())    
                ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                    hatches=['//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
                ax3.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                    hatches=['//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())
                ax4.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                    hatches=['//'],cmap='bone_r',alpha=0.1,
                    transform=ccrs.PlateCarree())


                
            AR_C.set_label="AR (Guan & Waliser, 2019)"
            
        
                
        #plot Dropsonde releases
        date=campaign_cls.year+campaign_cls.flight_month[flight_str]
        date=date+campaign_cls.flight_day[flight_str]
        
        # in some cases the Dropsondes variable can be a dataframe or
        # just a series, if only one sonde has been released
        
        if not Dropsondes=={}:
            if isinstance(Dropsondes["Lat"],pd.DataFrame):
                dropsonde_releases=pd.DataFrame(\
                            index=pd.DatetimeIndex(Dropsondes["LTS"].index))
                dropsonde_releases["Lat"]=Dropsondes["Lat"].loc\
                                                [:,"6000.0"].values
                dropsonde_releases["Lon"]=Dropsondes["Lon"].loc\
                                                [:,"6000.0"].values
        
            else:
                index_var=Dropsondes["Time"].loc["6000.0"]
                dropsonde_releases=pd.Series()
                dropsonde_releases["Lat"]=np.array(Dropsondes["Lat"]\
                                                   .loc["6000.0"])
                dropsonde_releases["Lon"]=np.array(Dropsondes["Lon"]\
                                                   .loc["6000.0"])
                dropsonde_releases["Time"]=index_var
        
            if not self.flight=="RF08":
                relevant_dropsondes=dropsonde_releases.loc[\
                                        cut_radar["Reflectivity"].index[0]:\
                                        cut_radar["Reflectivity"].index[-1]]
                if relevant_dropsondes.shape[0]>0:
                    ax1.scatter(relevant_dropsondes["Lon"],
                            relevant_dropsondes["Lat"],
                            s=100,marker="v",color="orange",
                            edgecolors="black",label="Dropsondes",
                            transform=ccrs.PlateCarree())
                
                    ax2.scatter(relevant_dropsondes["Lon"],
                            relevant_dropsondes["Lat"],
                            s=100,marker="v",color="orange",
                            edgecolors="black",label="Dropsondes",
                            transform=ccrs.PlateCarree())
                    ax3.scatter(relevant_dropsondes["Lon"],
                            relevant_dropsondes["Lat"],
                            s=100,marker="v",color="orange",
                            edgecolors="black",label="Dropsondes",
                            transform=ccrs.PlateCarree())
                
                    ax4.scatter(relevant_dropsondes["Lon"],
                            relevant_dropsondes["Lat"],
                            s=100,marker="v",color="orange",
                            edgecolors="black",label="Dropsondes",
                            transform=ccrs.PlateCarree())

        
        if not resizing_done:    
            map_fig.suptitle(campaign_cls.name+" ERA-5 Moisture Budget for "+\
                             flight_str+": "+campaign_cls.year+"-"+\
                             campaign_cls.flight_month[flight_str]+"-"+\
                             campaign_cls.flight_day[flight_str]+" "+\
                             calc_time,y=0.94)
        else:
            map_fig.suptitle(campaign_cls.name+" ERA-5 Moisture Budget for "+\
                             flight_str+": "+campaign_cls.year+"-"+\
                             campaign_cls.flight_month[flight_str]+\
                             "-"+campaign_cls.flight_day[flight_str]+\
                             " "+calc_time,y=0.94)
        legend=ax1.legend(bbox_to_anchor=(0.65,-0.15,1.5,0),
                          facecolor='lightgrey',
                          loc="lower center",
                          ncol=2,mode="expand")
        
        legend.get_frame().set_linewidth(2.0)
        legend.get_frame().set_edgecolor("k")
        
        #Save figure
        fig_name=self.ar_of_day+"_"+campaign_cls.name+"_"+\
                    self.flight+'_Moisture_Components_Map_ERA5'+".png"
        if not self.plot_path==None:
            fig_path=self.plot_path
        else:
            fig_path=opt_plot_path
        print("PLot path ",fig_path)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        map_fig.savefig(fig_path+fig_name,bbox_inches="tight",dpi=300)
        print("Figure saved as:",fig_path+fig_name)
            
        return None
    def plot_moisture_budget(self,era_on_halo_cls,cut_radar,
                             Dropsondes,campaign_cls,
                             opt_plot_path=os.getcwd(),
                             invert_flight=False):
        import matplotlib
        import cartopy.crs as ccrs
        
        import atmospheric_rivers as AR
        import reanalysis as Reanalysis
        ERA5_on_HALO=era_on_halo_cls
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        set_font=16
        matplotlib.rcParams.update({'font.size':set_font})
                
        
        # Define the plot specifications for the given variables
        met_var_dict={}
        met_var_dict["ERA_name"]    = {"EV":"e","TP":"tp",
                                       "IWV":"tcwv","IVT":"IVT",
                                       "IVT_conv":"IVT_conv"}
        met_var_dict["colormap"]    = {"EV":"Blues","IVT_conv":"BrBG_r",
                                       "TP":"Blues","IVT":"speed"}
        
        met_var_dict["levels"]      = {"IWV":np.linspace(10,50,51),
                                "EV":np.linspace(0,1.5,51),
                                "TP":np.linspace(0,1.5,51),
                                "IVT_conv":np.linspace(-2,2,101),
                                       "IVT":np.linspace(50,600,61)}
        
        met_var_dict["units"]       = {"EV":"(kg$\mathrm{m}^{-2}$)",
                                "TP":"(kg$\mathrm{m}^{-2}$)",
                                "IVT_conv":"(kg\mathrm{m}^{-2}$)",
                                "IWV":"(kg$\mathrm{m}^{-2}\mathrm{h}^{-1}$)",
                                "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
        
        if campaign_cls.is_flight_campaign:
            AR=AR.Atmospheric_Rivers("ERA")
            flight_date=campaign_cls.years[self.flight]+"-"+\
                            campaign_cls.flight_month[self.flight]
            flight_date=flight_date+"-"+\
                        campaign_cls.flight_day[self.flight]
            AR_era_ds=AR.open_AR_catalogue(after_2019=int(flight_date[0:4])>2019)
            AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)

        
        
        file_name="total_columns_"+campaign_cls.years[self.flight]+"_"+\
                    campaign_cls.flight_month[self.flight]+"_"+\
                    campaign_cls.flight_day[self.flight]+".nc"    
        
        era5=Reanalysis.ERA5(for_flight_campaign=True,campaign="NAWDEX",
                  research_flights=self.flight,
                  era_path=campaign_cls.campaign_path+"/data/ERA-5/")
        
        ds,era_path=era5.load_era5_data(file_name)
        
        #IVT Processing
        ds["IVT_v"]=ds["p72.162"]
        ds["IVT_u"]=ds["p71.162"]
        ds["IVT"]=np.sqrt(ds["IVT_u"]**2+ds["IVT_v"]**2)
        ds["IVT_conv"]=ds["p84.162"]*3600 # units in seconds
        ds["e"]=ds["e"]*-1000
        ds["tp"]=ds["tp"]*1000
        
        #Aircraft Position
        if not self.synthetic_campaign:
            if campaign_cls.is_flight_campaign:
                halo_dict=campaign_cls.get_aircraft_position([self.flight],
                                                         campaign_cls.name)
            else:
                # Load Halo Dataset
                halo_waypoints=campaign_cls.get_aircraft_waypoints(filetype=".csv")
                if invert_flight:
                    halo_waypoints=campaign_cls.invert_flight_from_waypoints(
                                halo_waypoints,[self.flight])
                halo_dict={}
                for flight in campaign_cls.interested_flights:
                    halo_dict[self.flight]=campaign_cls.\
                                        interpolate_flight_from_waypoints(\
                                        halo_waypoints[self.flight])
        
            halo_df=halo_dict[self.flight] 
        else:
            import flight_track_creator
            Tracker=flight_track_creator.Flighttracker(
                                                campaign_cls,
                                                self.flight,
                                                self.ar_of_day,
                                                track_type=self.track_type,
                                                shifted_lat=self.synthetic_icon_lat,
                                                shifted_lon=self.synthetic_icon_lon)
            
            halo_df,cmpgn_path=Tracker.run_flight_track_creator()
            if isinstance(halo_df,dict):
                halo_dict=halo_df.copy()
                halo_df,time_legs_df=Tracker.concat_track_dict_to_df(
                                                    merge_all=False,
                                                    pick_legs=self.pick_legs)
                halo_df.index=pd.DatetimeIndex(halo_df.index)
            
            print("Synthetic flight track loaded")
        
        halo_df=halo_df.rename(columns={"Lon":"longitude",
                                 "Lat":"latitude"})
        
        map_fig=plt.figure(figsize=(14,14))
        
        ax1 = plt.subplot(1,1,1,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=-5.0,central_latitude=60))
        ax1.coastlines(resolution="50m")
        gl1=ax1.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
        
        gl1.top_labels=False
        gl1.bottom_labels=False
        gl1.right_labels=False
        gl1.left_labels=False
        ticklabel_color="dimgrey"
        tick_size=14
        gl1.xlabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        
        lat_extension=2.0
        lon_extension=2.0
        
        axins1=inset_axes(ax1,width="3%",
                              height="80%",
                              loc="center",
                              bbox_to_anchor=(0.55,0,1,1),
                              bbox_transform=ax1.transAxes,
                              borderpad=0)        
            
        #Plot HALO flight course
        ax1.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),
                  color="salmon",linestyle='--',linewidth=1.0,alpha=0.9)    
        
        if not cut_radar=={}:
            last_minute=cut_radar["Reflectivity"].index.minute[-1]
            if last_minute>30:
                last_hour=cut_radar["Reflectivity"].index.hour[-1]+1
            else:
                last_hour=cut_radar["Reflectivity"].index.hour[-1]
        else:
            last_hour=halo_df.index.hour[-1]
            cut_radar["Reflectivity"]=halo_df
            
        # ERA Moisture Budget
        dIWV_dt=(ds["tcwv"][last_hour,:,:]-ds["tcwv"][last_hour-1,:,:])
            
        budget_epsilon=ds[met_var_dict["ERA_name"]["TP"]]-\
            ds[met_var_dict["ERA_name"]["EV"]]+\
                    ds[met_var_dict["ERA_name"]["IVT_conv"]]
        C1=ax1.contourf(ds["longitude"],ds["latitude"],
                            budget_epsilon[last_hour,:,:]+dIWV_dt,
                            levels=met_var_dict["levels"]["IVT_conv"],
                            extend="both",transform=ccrs.PlateCarree(),
                            cmap="PuOr",alpha=0.95)
            
        # Plot AR Detection 
        plt.rcParams.update({'hatch.color': 'lightgrey'})
                
        cb=map_fig.colorbar(C1,cax=axins1)
        cb.set_label("Moisture Budget \n Error"+" "+\
                     met_var_dict["units"]["TP"])
        cb.set_ticks([-2.0 -1.0,0,1.0,2.0])

        #Identify periods of strong radar reflectivity
        if not self.synthetic_campaign:
            high_dbZ_index=cut_radar["Reflectivity"][\
                                    cut_radar["Reflectivity"]>15].any(axis=1)
            high_dbZ=cut_radar["Reflectivity"].loc[high_dbZ_index]
        
        start_pos=halo_df.loc[cut_radar["Reflectivity"].index[0]]
        end_pos=halo_df.loc[cut_radar["Reflectivity"].index[-1]]
        ax1.scatter(halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                        cut_radar["Reflectivity"].index[-1]],
                    halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                        cut_radar["Reflectivity"].index[-1]],
                    transform=ccrs.PlateCarree(),marker=".",
                        s=3,color="red",alpha=0.95,zorder=1)
        #------------------------------------------------------------------#
        ## Add quiver
        step=15
        quiver_lon=np.array(ds["longitude"][::step])
        quiver_lat=np.array(ds["latitude"][::step])
        u=np.array(ds["IVT_u"][last_hour,::step,::step])
        v=np.array(ds["IVT_v"][last_hour,::step,::step])
        u[u<50]=np.nan
        v[v<50]=np.nan
        quiver=ax1.quiver(quiver_lon,quiver_lat,u,v,color="white",
                              edgecolor="k",linewidth=1,scale=900,
                              scale_units="inches",pivot="mid",
                              width=0.008,transform=ccrs.PlateCarree())
        #---------------------------------------------------------------------#    
        if not self.synthetic_campaign:
                #Plot high reflectivity values
                ax1.scatter(halo_df["longitude"].loc[high_dbZ.index],
                    halo_df["latitude"].loc[high_dbZ.index],
                    s=30,color="white",marker="D",linewidths=0.5,
                    label="Radar dbZ > 15",edgecolor="k",
                    transform=ccrs.PlateCarree())
        #######################################################################
        # if no cut_radar data, e.g mostly for Synthetic campaign
        else:
            if not self.synthetic_campaign:
                AR_cutted_halo,_,ar_of_day=ERA5_on_HALO.cut_halo_to_AR_crossing(\
                                                    self.ar_of_day,self.flight, 
                                                    halo_df,None,
                                                    campaign=campaign_cls.name,
                                                    device="halo",
                                                    invert_flight=invert_flight)
            else:
                AR_cutted_halo=halo_df.copy()
                AR_cutted_halo.index=pd.DatetimeIndex(AR_cutted_halo.index)    
            last_hour=AR_cutted_halo.index.hour[-1]
            start_pos=AR_cutted_halo.iloc[0]
            end_pos  =AR_cutted_halo.iloc[-1]
            # ERA Moisture Budget
            dIWV_dt=(ds["tcwv"][last_hour,:,:]-ds["tcwv"][last_hour-1,:,:])
            
            budget_epsilon=ds[met_var_dict["ERA_name"]["TP"]]-\
                ds[met_var_dict["ERA_name"]["EV"]]+\
                    ds[met_var_dict["ERA_name"]["IVT_conv"]]
            C1=ax1.contourf(ds["longitude"],ds["latitude"],
                            budget_epsilon[last_hour,:,:]-dIWV_dt,
                            levels=met_var_dict["levels"]["IVT_conv"],
                            extend="both",transform=ccrs.PlateCarree(),
                            cmap="PuOr",alpha=0.95)
        
            #------------------------------------------------------------------#
            ## Add quiver
            step=10
            quiver_lon=np.array(ds["longitude"][::step])
            quiver_lat=np.array(ds["latitude"][::step])
            u=pd.DataFrame(np.array(ds["IVT_u"][last_hour,::step,::step]))
            v=pd.DataFrame(np.array(ds["IVT_v"][last_hour,::step,::step]))
            u=u.where(abs(u)>100,np.nan)
            v=v.where(abs(v)>100,np.nan)
            quiver2=ax1.quiver(quiver_lon,quiver_lat,u.values,v.values,
                               color="white",edgecolor="k",lw=1,
                               scale=800,scale_units="inches",
                               pivot="mid",width=0.005,
                               transform=ccrs.PlateCarree())
        
            #-----------------------------------------------------------------#    
        
            ax1.scatter(AR_cutted_halo["longitude"],
                        AR_cutted_halo["latitude"],
                        transform=ccrs.PlateCarree(),marker=".",
                        s=3,color="red",
                        alpha=0.95,zorder=1)    
            ###################################################################
            
        print("Hour of the day:",last_hour)
        calc_time=era5.hours[last_hour]
        
        deg_ratio=(start_pos["longitude"]-end_pos["longitude"])/\
            (start_pos["latitude"]-end_pos["latitude"])
        print("Meridional-Zonal ratio: ",deg_ratio)
        resizing_done=False
        if abs((start_pos["longitude"]-end_pos["longitude"])/\
               (start_pos["latitude"]-end_pos["latitude"]))>1.5:
            lat_extension=4
            resizing_done=True
        elif abs((start_pos["latitude"]-end_pos["latitude"])/\
                 (start_pos["longitude"]-end_pos["longitude"]))>1.5:
            lon_extension=4
            resizing_done=True
        else:
            pass
        lower_lon=np.min([start_pos["longitude"],end_pos["longitude"]])-\
                                                lon_extension*5
        upper_lon=np.max([start_pos["longitude"],end_pos["longitude"]])+\
                                                lon_extension*5
        lower_lat=np.min([start_pos["latitude"],end_pos["latitude"]])-\
                                                lat_extension*5
        upper_lat=np.max([start_pos["latitude"],end_pos["latitude"]])+\
                                                lat_extension*5
        if upper_lat>90:
            upper_lat=90
        ax1.set_extent([lower_lon,upper_lon,lower_lat,upper_lat])
        
        #LAT LON Extent
        calc_time=era5.hours[last_hour]
        deg_ratio=(start_pos["longitude"]-end_pos["longitude"])/\
            (start_pos["latitude"]-end_pos["latitude"])
        resizing_done=False
        if abs((deg_ratio))>1.5:
            lat_extension=4
            resizing_done=True
        elif abs(1/deg_ratio)>1.5:
            lon_extension=4
            resizing_done=True
        else:
            pass
        lower_lon=np.min([start_pos["longitude"],end_pos["longitude"]])-\
                                                lon_extension*5
        upper_lon=np.max([start_pos["longitude"],end_pos["longitude"]])+\
                                                lon_extension*5
        lower_lat=np.min([start_pos["latitude"],end_pos["latitude"]])-\
                                                lat_extension*5
        upper_lat=np.max([start_pos["latitude"],end_pos["latitude"]])+\
                                                lat_extension*5
        if upper_lat>90:
            upper_lat=90
        ax1.set_extent([lower_lon,upper_lon,lower_lat,upper_lat])
        
        
        
            
        
        if not campaign_cls.is_synthetic_campaign:
            if last_hour<6:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                # ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #     AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                #     hatches=['//'],cmap='bone_r',alpha=0.1,
                #     transform=ccrs.PlateCarree())
                # ax3.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #         AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                #         hatches=['//'],cmap='bone_r',alpha=0.1,
                #         transform=ccrs.PlateCarree())
                # ax4.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #     AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                #     hatches=['//'],cmap='bone_r',alpha=0.1,
                #     transform=ccrs.PlateCarree())
        
            elif 6<=last_hour<12:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                        hatches=[ '//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                # ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #     AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                #     hatches=[ '//'],cmap='bone_r',alpha=0.1,
                #     transform=ccrs.PlateCarree())
                # ax3.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #         AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                #         hatches=['//'],cmap='bone_r',alpha=0.1,
                #         transform=ccrs.PlateCarree())
                # ax4.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #     AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                #     hatches=['//'],cmap='bone_r',alpha=0.1,
                #     transform=ccrs.PlateCarree())
            
            elif 12<=last_hour<18:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())
                # ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #     AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                #     hatches=['//'],cmap='bone_r',alpha=0.1,
                #     transform=ccrs.PlateCarree())
                # ax3.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #         AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                #         hatches=['//'],cmap='bone_r',alpha=0.1,
                #         transform=ccrs.PlateCarree())
                # ax4.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #     AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                #     hatches=['//'],cmap='bone_r',alpha=0.1,
                #     transform=ccrs.PlateCarree())
            else:
                AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                        AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                        hatches=['//'],cmap='bone_r',alpha=0.1,
                        transform=ccrs.PlateCarree())    
                # ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #     AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                #     hatches=['//'],cmap='bone_r',alpha=0.1,
                #     transform=ccrs.PlateCarree())
                # ax3.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #     AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                #     hatches=['//'],cmap='bone_r',alpha=0.1,
                #     transform=ccrs.PlateCarree())
                # ax4.contourf(AR_era_ds.lon,AR_era_ds.lat,
                #     AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                #     hatches=['//'],cmap='bone_r',alpha=0.1,
                #     transform=ccrs.PlateCarree())


                
            AR_C.set_label="AR (Guan & Waliser, 2019)"
          
        #plot Dropsonde releases
        date=campaign_cls.year+campaign_cls.flight_month[self.flight]
        date=date+campaign_cls.flight_day[self.flight]
        
        # in some cases the Dropsondes variable can be a dataframe or
        # just a series, if only one sonde has been released
        
        if not Dropsondes=={}:
            if isinstance(Dropsondes["Lat"],pd.DataFrame):
                dropsonde_releases=pd.DataFrame(\
                            index=pd.DatetimeIndex(Dropsondes["LTS"].index))
                dropsonde_releases["Lat"]=Dropsondes["Lat"].loc[:,"6000.0"].values
                dropsonde_releases["Lon"]=Dropsondes["Lon"].loc[:,"6000.0"].values        
            else:
                index_var=Dropsondes["Time"].loc["6000.0"]
                dropsonde_releases=pd.Series()
                dropsonde_releases["Lat"]=np.array(Dropsondes["Lat"].loc["6000.0"])
                dropsonde_releases["Lon"]=np.array(Dropsondes["Lon"].loc["6000.0"])
                dropsonde_releases["Time"]=index_var        
            if not self.flight=="RF08":
                relevant_dropsondes=dropsonde_releases.loc[\
                                        cut_radar["Reflectivity"].index[0]:\
                                        cut_radar["Reflectivity"].index[-1]]
                if relevant_dropsondes.shape[0]>0:
                    ax1.scatter(relevant_dropsondes["Lon"],
                            relevant_dropsondes["Lat"],
                            s=100,marker="v",color="orange",
                            edgecolors="black",label="Dropsondes",
                            transform=ccrs.PlateCarree())
        # Figure specifications
        map_fig.suptitle(campaign_cls.name+" ERA-5 Moisture Budget Error \n for "+\
                             self.flight+": "+campaign_cls.year+"-"+\
                             campaign_cls.flight_month[self.flight]+"-"+\
                             campaign_cls.flight_day[self.flight]+" "+\
                             calc_time,y=0.94)
        legend=ax1.legend(bbox_to_anchor=(0.5,-0.15,1.0,0),
                          facecolor='lightgrey',
                          loc="lower center",
                          ncol=2,mode="expand")
        
        legend.get_frame().set_linewidth(2.0)
        legend.get_frame().set_edgecolor("k")
        
        #Save figure
        fig_name=self.ar_of_day+"_"+campaign_cls.name+"_"+\
                    self.flight+'_Moisture_Budget_Error_Map_ERA5'+".png"
        if not self.plot_path==None:
            fig_path=self.plot_path
        else:
            fig_path=opt_plot_path
        print("PLot path ",fig_path)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        map_fig.savefig(fig_path+fig_name,bbox_inches="tight",dpi=200)
        print("Figure saved as:",fig_path+fig_name)
        
        return None
    ###########################################################################
    #ERA and ICON
    def plot_flight_combined_IWV_map_AR_crossing(self,cut_radar,Dropsondes,
                                                 campaign_cls,last_hour,
                                                 opt_plot_path=None):
        """
        

        Parameters
        ----------
        cut_radar : dict
            Dictionary containing the radar data cutted to the AR. 
            assessed by Cloudnet_Data.get_cloudnet_station_coordinates
        dropsondes : dict 
            Dictionary containing the dropsonde releases and their data.
        campaign_cls : class
            class of the flight campaign, for now applicable for NAWDEX
        flight : str
            flight to analyse
        AR_number : str
            id of AR Crossing this figure considers
        last_hour : str
            number of flight hour
        opt_plot_path : str
            plot path to store the figure in. Default is "None".
        Returns
        -------
        None.

        """
        import matplotlib
        from matplotlib.colors import BoundaryNorm
        
        import cartopy.crs as ccrs
        
        import atmospheric_rivers as AR
        import reanalysis as Reanalysis
        import ICON
        
        
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        
        set_font=18
        matplotlib.rcParams.update({'font.size':set_font})
                
        upper_iwv=25
        # Define the plot specifications for the given variables
        met_var_dict={}
        met_var_dict["ERA_name"]    = {"IWV":"tcwv","IVT":"IVT"}
        met_var_dict["colormap"]    = {"IWV":"density","IVT":"speed"}
        met_var_dict["levels"]      = {"IWV":np.linspace(0,upper_iwv,
                                                         2*upper_iwv+1),
                                       "IVT":np.linspace(0,1000,101)}
        met_var_dict["units"]       = {"IWV":"(kg/$\mathrm{m}^2$)",
                                "IVT":"(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)"}
        
        # Load Reanalysis
    
        file_name="total_columns_"+campaign_cls.year+"_"+\
                    campaign_cls.flight_month[self.flight]+"_"+\
                    campaign_cls.flight_day[self.flight]+".nc"    
        
        era5=Reanalysis.ERA5(for_flight_campaign=True,campaign="NAWDEX",
                  research_flights=self.flight,
                  era_path=campaign_cls.campaign_path+"/data/ERA-5/")
            
        ds,era_path=era5.load_era5_data(file_name)
        
        AR=AR.Atmospheric_Rivers("ERA")
        
        flight_date=campaign_cls.year+"-"+campaign_cls.flight_month[self.flight]
        flight_date=flight_date+"-"+campaign_cls.flight_day[self.flight]
        if campaign_cls.is_flight_campaign:
            AR_era_ds=AR.open_AR_catalogue(after_2019=int(flight_date[0:4])>2019)
       
            AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)

        
        
        if not self.synthetic_campaign:
            #Aircraft Position
            if not "latitude" in cut_radar["Position"].columns:
                print("Load aircraft position externally ")
                halo_dict=campaign_cls.get_aircraft_position([self.flight],
                                                         campaign_cls.name)
                halo_df=halo_dict[self.flight] 
            else:
                print("Position data already in radar dataframe")
                halo_df=cut_radar["Position"]
        else:
            import flight_track_creator
            Tracker=flight_track_creator.Flighttracker(campaign_cls,
                                                self.flight,
                                                self.ar_of_day,
                                                track_type=self.track_type,
                                                shifted_lat=self.synthetic_icon_lat,
                                                shifted_lon=self.synthetic_icon_lon)
            
            halo_df,campaign_path=Tracker.run_flight_track_creator(
                                                        track_type=self.track_type)
            if isinstance(halo_df,dict):
                halo_dict=halo_df.copy()
                halo_df,time_legs_df=flight_track_creator.concat_track_dict_to_df(
                                                halo_df,merge_all=False,
                                                pick_legs=self.pick_legs)
            
            print("Synthetic flight track loaded")
        
        ###This is now set but have to be ambigious for next time
        ##
        #last_hour=13
        hydrometeor_icon_path=campaign_cls.campaign_path+"/data/ICON_LEM_2KM/"
        resolution=2000 # units m
        icon=ICON.ICON_NWP(str(last_hour),resolution,for_flight_campaign=True,
                    campaign="NAWDEX",research_flights=None,
                    icon_path=hydrometeor_icon_path)
        var="Hydrometeor"#
        #Open first hour
        print("Open ICON-Simulations Start Hour")
        icon_file_name=var+"_ICON_"+self.flight+"_"+str(last_hour)+"UTC.nc"    
    
        icon_ds=icon.load_icon_data(icon_file_name)
        iwv_icon=icon_ds["tqv_dia"]

        
        
        print("Hour of the day:",last_hour)
        calc_time=era5.hours[last_hour]
        map_fig=plt.figure(figsize=(25,12))
        
        ax1 = plt.subplot(1,2,1,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=-25.0,central_latitude=55))
        ax1.coastlines(resolution="50m")
        gl1=ax1.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
        
        ax2 = plt.subplot(1,2,2,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=-25.0,central_latitude=55))
        ax2.coastlines(resolution="50m")
        gl2=ax2.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
        
        gl1.top_labels=True
        gl2.top_labels=True
        gl1.bottom_labels=False
        gl2.bottom_labels=False
        
        gl2.left_labels=False
        
        gl1.right_labels=False
        gl2.right_labels=False
        
        ticklabel_color="black"
        tick_size=set_font
        
        gl1.xlabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        gl2.xlabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        gl1.ylabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        gl2.ylabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        
        #Identify periods of strong radar reflectivity
        high_dbZ_index=cut_radar["Reflectivity"][cut_radar["Reflectivity"]>15].any(axis=1)
        high_dbZ=cut_radar["Reflectivity"].loc[high_dbZ_index]
        
        temp_halo_df=halo_df.truncate(before=cut_radar["Reflectivity"].index[0])
        temp_halo_df=halo_df.truncate(after=cut_radar["Reflectivity"].index[-1])
        
        start_pos=temp_halo_df.iloc[0,:]
        end_pos=temp_halo_df.iloc[-1,:]
        
        lat_extension=1.0#0.2
        lon_extension=1.0#0.2
    
        deg_ratio=(start_pos["longitude"]-end_pos["longitude"])/\
            (start_pos["latitude"]-end_pos["latitude"])
        print("Meridional-Zonal ratio: ",deg_ratio)
        resizing_done=False
        if abs((start_pos["longitude"]-end_pos["longitude"])/\
               (start_pos["latitude"]-end_pos["latitude"]))>1.5:
            lat_extension=1.0#0.1
            resizing_done=True
        elif abs((start_pos["latitude"]-end_pos["latitude"])/\
                 (start_pos["longitude"]-end_pos["longitude"]))>1.5:
            lon_extension=1.0#0.1
            resizing_done=True
        else:
            pass
        
        ax1.set_extent([np.min([start_pos["longitude"],
                                end_pos["longitude"]])-lon_extension*5,
                        np.max([start_pos["longitude"],
                                end_pos["longitude"]])+lon_extension*5,
                        np.min([start_pos["latitude"],
                                end_pos["latitude"]])-lat_extension*5,
                        np.max([start_pos["latitude"],
                                end_pos["latitude"]])+lat_extension*5,
                        ])
        
        ax2.set_extent([np.min([start_pos["longitude"],
                                end_pos["longitude"]])-lon_extension*5,
                        np.max([start_pos["longitude"],
                                end_pos["longitude"]])+lon_extension*5,
                        np.min([start_pos["latitude"],
                                end_pos["latitude"]])-lat_extension*5,
                        np.max([start_pos["latitude"],
                                end_pos["latitude"]])+lat_extension*5,
                        ])
        
        x,y=np.meshgrid(ds["longitude"],ds["latitude"])    
        norm=BoundaryNorm(met_var_dict["levels"]["IWV"],
                    ncolors=plt.get_cmap(met_var_dict["colormap"]["IWV"]).N,
                    clip=True)
        
        C1=ax1.pcolormesh(x,y,
                            ds[met_var_dict["ERA_name"]["IWV"]][last_hour,:,:],
                            norm=norm,
                            transform=ccrs.PlateCarree(),
                            cmap=met_var_dict["colormap"]["IWV"])
        
        #---------------------------------------------------------------------#
        ## Add quiver
        step=10
        quiver_lon=np.array(ds["longitude"][::step])
        quiver_lat=np.array(ds["latitude"][::step])
        ds["IVT_u"]=ds["p71.162"]
        ds["IVT_v"]=ds["p72.162"]
        u=np.array(ds["IVT_u"][last_hour,::step,::step])
        v=np.array(ds["IVT_v"][last_hour,::step,::step])
        quiver=ax1.quiver(quiver_lon,quiver_lat,u,v,color="lightgrey",
                                  scale=500,scale_units="inches",lw=1,
                                  edgecolor="black",
                                  pivot="mid",width=0.005,
                                  transform=ccrs.PlateCarree())
        quiver2=ax1.quiver(quiver_lon,quiver_lat,u,v,color="lightgrey",lw=1,
                                  edgecolor="black",
                                  scale=500,scale_units="inches",
                                  pivot="mid",width=0.005,
                                  transform=ccrs.PlateCarree())
        
        #---------------------------------------------------------------------#    
        C2=ax2.scatter(np.rad2deg(iwv_icon.clon),np.rad2deg(iwv_icon.clat),
                       c=iwv_icon[0,:],cmap=met_var_dict["colormap"]["IWV"],
                       s=1.25,vmin=0.0,vmax=upper_iwv,
                       transform=ccrs.PlateCarree())
        
        
        axins2=inset_axes(ax2,width="3%",
                          height="80%",
                          loc="center",
                          bbox_to_anchor=(0.55,0,1,1),
                          bbox_transform=ax2.transAxes,
                          borderpad=0)   
        
        cb2=map_fig.colorbar(C2,cax=axins2,extend="max")
        cb2.set_label("IWV"+" "+met_var_dict["units"]["IWV"])
        cb2.set_ticks([0,5,10,15,20,25,30,40,50])
        
        
        # Plot AR Detection 
        plt.rcParams.update({'hatch.color': 'lightgrey'})
        if last_hour<6:
            AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                    hatches=['/'],cmap='bone',alpha=0.1,
                    transform=ccrs.PlateCarree())
        elif 6<=last_hour<12:
            AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                    hatches=[ '/'],cmap='bone',alpha=0.1,
                    transform=ccrs.PlateCarree())
        elif 12<=last_hour<18:
            AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                    alpha=0.1,hatches=["/"],cmap=None,
                    transform=ccrs.PlateCarree())
        else:
            AR_C=ax1.contour(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                    hatches=['/'],cmap='bone',alpha=0.1,color="white",
                    transform=ccrs.PlateCarree())
        if last_hour<6:
            ax2.contour(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
                    hatches=['/'],cmap='bone',alpha=0.1,
                    transform=ccrs.PlateCarree())
        elif 6<=last_hour<12:
            ax2.contour(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
                    hatches=['/'],cmap='bone',alpha=0.1,
                    transform=ccrs.PlateCarree())
        elif 12<=last_hour<18:
            ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
                    alpha=0.1,hatches=['/'],cmap=None,
                    transform=ccrs.PlateCarree())
        else:
            ax2.contour(AR_era_ds.lon,AR_era_ds.lat,
                    AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
                    hatches=['/'],alpha=0.1,
                    transform=ccrs.PlateCarree())
        AR_C.set_label="AR (Guan & Waliser, 2019)"
        
        #C2=ax2.contourf(ds["longitude"],ds["latitude"],
        #                    ds[met_var_dict["ERA_name"]["IVT"]][last_hour,:,:],
        #                    levels=met_var_dict["levels"]["IVT"],
        #                    extend="max",transform=ccrs.PlateCarree(),
        #                    cmap=met_var_dict["colormap"]["IVT"],alpha=0.95)
        #cb2=map_fig.colorbar(C2,cax=axins2)
        
        
        #Plot HALO flight course
        ax1.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),
                  color="salmon",linestyle='--',linewidth=1.0,alpha=0.9)    
        ax1.scatter(halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                             cut_radar["Reflectivity"].index[-1]],
                    halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                            cut_radar["Reflectivity"].index[-1]],
                    transform=ccrs.PlateCarree(),marker=".",s=3,color="red",
                    alpha=0.95,zorder=1)    
        ax2.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),color="salmon",
                  linestyle='--',linewidth=1.0,alpha=0.9)    
        ax2.scatter(halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                             cut_radar["Reflectivity"].index[-1]],
                 halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                         cut_radar["Reflectivity"].index[-1]],
                 transform=ccrs.PlateCarree(),marker='.',s=3,
                 color="red",alpha=0.95,zorder=1)    
        
        if not self.synthetic_campaign:
            #Plot high reflectivity values
            ax1.scatter(halo_df["longitude"].reindex(high_dbZ.index),
                    halo_df["latitude"].reindex(high_dbZ.index),
                    s=30,color="white",marker="D",linewidths=0.5,
                    label="Radar dBZ > 15",edgecolor="k",
                    transform=ccrs.PlateCarree())
        
            ax2.scatter(halo_df["longitude"].reindex(high_dbZ.index),
                    halo_df["latitude"].reindex(high_dbZ.index),
                    s=30,color="white",marker="D",
                    linewidths=0.5,edgecolor="k",
                    transform=ccrs.PlateCarree())
        
                
            #plot Dropsonde releases
            date=campaign_cls.year+campaign_cls.flight_month[self.flight]
            date=date+campaign_cls.flight_day[self.flight]
            #          if not flight=="RF06":                           
            #              Dropsondes=campaign_cls.load_dropsonde_data(date,print_arg="yes",
            #                                                          dt="all",plotting="no")
        
            # in some cases the Dropsondes variable can be a dataframe or
            # just a series, if only one sonde has been released
            if isinstance(Dropsondes["Lat"],pd.DataFrame):
                dropsonde_releases=pd.DataFrame(index=pd.DatetimeIndex(Dropsondes["LTS"].index))
                dropsonde_releases["Lat"]=Dropsondes["Lat"].loc[:,"6000.0"].values
                dropsonde_releases["Lon"]=Dropsondes["Lon"].loc[:,"6000.0"].values
        
            else:
                index_var=Dropsondes["Time"].loc["6000.0"]
                dropsonde_releases=pd.Series()
                dropsonde_releases["Lat"]=np.array(Dropsondes["Lat"].loc["6000.0"])
                dropsonde_releases["Lon"]=np.array(Dropsondes["Lon"].loc["6000.0"])
                dropsonde_releases["Time"]=index_var
        
            if not self.flight=="RF08":
                relevant_dropsondes=dropsonde_releases.loc[\
                                        cut_radar["Reflectivity"].index[0]:\
                                        cut_radar["Reflectivity"].index[-1]]
            
            if relevant_dropsondes.shape[0]>0:
                ax1.scatter(relevant_dropsondes["Lon"],
                            relevant_dropsondes["Lat"],
                            s=100,marker="v",color="orange",
                            edgecolors="black",label="Dropsondes",
                            transform=ccrs.PlateCarree())
                
                ax2.scatter(relevant_dropsondes["Lon"],
                            relevant_dropsondes["Lat"],
                            s=100,marker="v",color="orange",
                            edgecolors="black",label="Dropsondes",
                            transform=ccrs.PlateCarree())
        ax1.set_title("ERA-5")
        ax2.set_title("ICON-NWP (2km)")
        if not resizing_done:    
            map_fig.suptitle(campaign_cls.name+" IWV data for "+\
                             self.flight+": "+campaign_cls.year+"-"+\
                             campaign_cls.flight_month[self.flight]+\
                             "-"+campaign_cls.flight_day[self.flight]+\
                             " "+calc_time,y=0.96)
        else:
            map_fig.suptitle(campaign_cls.name+" IWV data for "+\
                             self.flight+": "+campaign_cls.year+"-"+\
                             campaign_cls.flight_month[self.flight]+"-"+\
                             campaign_cls.flight_day[self.flight]+\
                             " "+calc_time,y=0.96)
        legend=ax1.legend(bbox_to_anchor=(0.65,-0.15,1.0,0),
                          facecolor='lightgrey',loc="lower center",
                          ncol=3,mode="expand")
        
        legend.get_frame().set_linewidth(2.0)
        legend.get_frame().set_edgecolor("k")
        
        #Save figure
        fig_name=self.ar_of_day+"_"+campaign_cls.name+"_"+\
                    self.flight+'_IWV_MAP_ERA5_ICON_'+str(last_hour)+"UTC.png"
        if opt_plot_path==None:
            fig_path=self.plot_path
        else:
            fig_path=opt_plot_path
        print("PLot path ",fig_path)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        map_fig.savefig(fig_path+fig_name,bbox_inches="tight",dpi=200)
        print("Figure saved as:",fig_path+fig_name)
        return None

    def plot_flight_map_Hydrometeorpaths_AR_crossing(self,cut_radar,Dropsondes,
                                                     campaign_cls,last_hour,
                                                     halo_df=None,
                                                     opt_plot_path=None, 
                                                     with_ICON=True,
                                                     AR_in_Catalogue=True):
        """
        

        Parameters
        ----------
        cut_radar : dict
            Dictionary containing the radar data cutted to the AR. 
            assessed by Cloudnet_Data.get_cloudnet_station_coordinates
        dropsondes : dict 
            Dictionary containing the dropsonde releases and their data.
        campaign_cls : class
            class of the flight campaign, for now applicable for NAWDEX
        flight : str
            flight to analyse
        AR_number : str
            id of AR Crossing this figure considers
        last_hour : str
            hour to consider for map
        Returns
        -------
        None.

        """
        import matplotlib
        from matplotlib.colors import BoundaryNorm
        import cartopy.crs as ccrs
        
        import atmospheric_rivers as AR
        
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        #---------------------------------------------------------------------#
        ## Specifications        
        set_font=16
        matplotlib.rcParams.update({'font.size':set_font})
                
        upper_iwv=500
        
        # plot specifications for the given variables
        met_var_dict={}
        met_var_dict["ERA_name"]    = {"LWP":"tclw","IWP":"tciw"}
        met_var_dict["colormap"]    = {"LWP":"YlGnBu","IWP":"YlGnBu"}
        met_var_dict["levels"]      = {"LWP":np.linspace(50,upper_iwv,
                                                         int(0.2*upper_iwv+1)),
                                       "IWP":np.linspace(50,upper_iwv,
                                                         int(0.2*upper_iwv+1))}
        met_var_dict["units"]       = {"LWP":"(g/$\mathrm{m}^2$)",
                                       "IWP":"(g/$\mathrm{m}^2$)"}
        if self.flight.endswith("instantan"):
            flight_str=str.split(self.flight,"_")[0]
        else:
            flight_str=self.flight
        flight_date=campaign_cls.years[flight_str]+"-"+campaign_cls.flight_month[flight_str]
        flight_date=flight_date+"-"+campaign_cls.flight_day[flight_str]

        #---------------------------------------------------------------------#
        # ARs and Aircraft Data
        if AR_in_Catalogue:
            AR=AR.Atmospheric_Rivers("ERA")
            AR_era_ds=AR.open_AR_catalogue(after_2019=int(flight_date[0:4])>2019)
            AR_era_data=AR.specify_AR_data(AR_era_ds,flight_date)

        
        
        #Aircraft Position
        if not self.synthetic_campaign:
            if not isinstance(halo_df,pd.DataFrame):
                if not "latitude" in cut_radar["Position"].columns:
                    print("Load aircraft position externally ")
                    halo_dict=campaign_cls.get_aircraft_position([flight_str],
                                                             campaign_cls.name)
                    halo_df=halo_dict[flight_str] 
                else:
                    print("Position data already in radar dataframe")
                    halo_df=cut_radar["Position"]
        else:
            import flight_track_creator
            Tracker=flight_track_creator.Flighttracker(campaign_cls,
                                                flight_str,
                                                self.ar_of_day,
                                                track_type=self.track_type,
                                                shifted_lat=self.synthetic_icon_lat,
                                                shifted_lon=self.synthetic_icon_lon)
            
            halo_dict,campaign_path=Tracker.run_flight_track_creator()
            print("Synthetic flight track loaded")
            halo_df=pd.concat([halo_dict["inflow"],halo_dict["internal"],
                               halo_dict["outflow"]])
        
        #---------------------------------------------------------------------#
        ## Grid Data
        # Load Reanalysis
        file_name="total_columns_"+campaign_cls.years[flight_str]+"_"+\
                    campaign_cls.flight_month[flight_str]+"_"+\
                    campaign_cls.flight_day[flight_str]+".nc"    
        #try:
        #    era5=ERA5(for_flight_campaign=True,campaign=campaign_cls.name,
        #          research_flights=flight,
        #          era_path=campaign_cls.campaign_path+"/data/ERA-5/")
        #except:
        import reanalysis as Reanalysis
        era5=Reanalysis.ERA5(for_flight_campaign=True,campaign=campaign_cls.name,
                  research_flights=self.flight,
                  era_path=campaign_cls.campaign_path+"/data/ERA-5/")
        ds,era_path=era5.load_era5_data(file_name)
        
        hydrometeor_icon_path=campaign_cls.campaign_path+"/data/ICON_LEM_2KM/"
        resolution=2000 # units m
        var="Hydrometeor"#
        
        #Open first hour
    
        
        
        print("Hour of the day:",last_hour)
        calc_time=era5.hours[last_hour]
        if with_ICON:
            import ICON
            icon=ICON.ICON_NWP(str(last_hour),resolution,
                               for_flight_campaign=True,
                               campaign="NAWDEX",research_flights=None,
                               icon_path=hydrometeor_icon_path)
        
            print("Open ICON-Simulations Start Hour")
            icon_file_name=var+"_ICON_"+self.flight+"_"+str(last_hour)+"UTC.nc"    
    
            icon_ds=icon.load_icon_data(icon_file_name)
            lwp_icon=icon_ds["tqc_dia"]*1000    # units g/m2
            iwp_icon=icon_ds["tqi_dia"]*1000    # units g/m2
            
            map_fig=plt.figure(figsize=(28,12))
        
            ax1 = plt.subplot(2,2,1,projection=ccrs.AzimuthalEquidistant(
                                    central_longitude=-25.0,
                                    central_latitude=55))
            ax2 = plt.subplot(2,2,2,projection=ccrs.AzimuthalEquidistant(
                                    central_longitude=-25.0,
                                    central_latitude=55))
            ax3= plt.subplot(2,2,3,projection=ccrs.AzimuthalEquidistant(
                                    central_longitude=-25.0,
                                    central_latitude=55))
            
            ax4= plt.subplot(2,2,4,projection=ccrs.AzimuthalEquidistant(
                                    central_longitude=-25.0,
                                    central_latitude=55))
         
            ax1.coastlines(resolution="50m")
            ax2.coastlines(resolution="50m")
            ax3.coastlines(resolution="50m")
            ax4.coastlines(resolution="50m")
            
        
            gl1=ax1.gridlines(draw_labels=True,dms=True,
                              x_inline=False,y_inline=False)
            gl2=ax2.gridlines(draw_labels=True,dms=True,
                              x_inline=False,y_inline=False)
            gl3=ax3.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
            gl4=ax4.gridlines(draw_labels=True,dms=True,
                          x_inline=False,y_inline=False)
        
        
            gl1.top_labels=True
            gl1.bottom_labels=False
            gl1.left_labels=True
            gl1.right_labels=False
            
            gl2.top_labels=True
            gl2.bottom_labels=False
            gl2.left_labels=False
            gl2.right_labels=False
            
            gl3.top_labels=False
            gl3.bottom_labels=False
            gl3.left_labels=True
            gl3.right_labels=False
            
            gl4.top_labels=False
            gl4.bottom_labels=False
            gl4.left_labels=False
            gl4.right_labels=False
            
            ticklabel_color="black"
            tick_size=16
            
            gl1.xlabel_style= {'size':tick_size,
                               'color':ticklabel_color}
            gl2.xlabel_style= {'size':tick_size,
                               'color':ticklabel_color}
            gl1.ylabel_style= {'size':tick_size,
                               'color':ticklabel_color}
            gl2.ylabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        else:
            map_fig=plt.figure(figsize=(14,12))
        
            ax1 = plt.subplot(2,1,1,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=-10.0,
                                central_latitude=55))
            ax2 = plt.subplot(2,1,2,projection=ccrs.AzimuthalEquidistant(
                                central_longitude=-10.0,
                                central_latitude=55))
            
            ax1.coastlines(resolution="50m")
            ax2.coastlines(resolution="50m")
            
            gl1=ax1.gridlines(draw_labels=True,dms=True,
                              x_inline=False,y_inline=False)
            gl2=ax2.gridlines(draw_labels=True,dms=True,
                              x_inline=False,y_inline=False)
        
            gl1.top_labels=True
            gl1.bottom_labels=False
            gl1.left_labels=True
            gl1.right_labels=False
            
            gl2.top_labels=True
            gl2.bottom_labels=False
            gl2.left_labels=False
            gl2.right_labels=False
            
            
            ticklabel_color="black"
            tick_size=16
            
            gl1.xlabel_style= {'size':tick_size,
                               'color':ticklabel_color}
            gl2.xlabel_style= {'size':tick_size,
                               'color':ticklabel_color}
            gl1.ylabel_style= {'size':tick_size,
                               'color':ticklabel_color}
            gl2.ylabel_style= {'size':tick_size,
                           'color':ticklabel_color}
        if cut_radar!=None and cut_radar:
            temp_halo_df=halo_df.truncate(
                                    before=cut_radar["Reflectivity"].index[0])
            temp_halo_df=temp_halo_df.truncate(
                                    after=cut_radar["Reflectivity"].index[-1])
        else:
            temp_halo_df=halo_df.copy()
        start_pos=temp_halo_df.iloc[0,:]
        end_pos=temp_halo_df.iloc[-1]
        
        lat_extension=1
        lon_extension=1
    
        deg_ratio=(start_pos["longitude"]-end_pos["longitude"])/\
            (start_pos["latitude"]-end_pos["latitude"])
        #print("Meridional-Zonal ratio: ",deg_ratio)
        resizing_done=False
        if abs((start_pos["longitude"]-end_pos["longitude"])/\
               (start_pos["latitude"]-end_pos["latitude"]))>1.5:
            lat_extension=1.0
            resizing_done=True
        elif abs((start_pos["latitude"]-end_pos["latitude"])/\
                 (start_pos["longitude"]-end_pos["longitude"]))>1.5:
            lon_extension=1.0
            resizing_done=True
        else:
            pass
        
        ax1.set_extent([np.min([start_pos["longitude"],
                                end_pos["longitude"]])-lon_extension*5,
                        np.max([start_pos["longitude"],
                                end_pos["longitude"]])+lon_extension*5,
                        np.min([start_pos["latitude"],
                                end_pos["latitude"]])-lat_extension*5,
                        np.max([start_pos["latitude"],
                                end_pos["latitude"]])+lat_extension*5,
                        ])
        
        ax2.set_extent([np.min([start_pos["longitude"],
                                end_pos["longitude"]])-lon_extension*5,
                        np.max([start_pos["longitude"],
                                end_pos["longitude"]])+lon_extension*5,
                        np.min([start_pos["latitude"],
                                end_pos["latitude"]])-lat_extension*5,
                        np.max([start_pos["latitude"],
                                end_pos["latitude"]])+lat_extension*5,
                        ])
        if not with_ICON:
            ax3=ax2
        else:
            ax3.set_extent([np.min([start_pos["longitude"],
                                end_pos["longitude"]])-lon_extension*5,
                        np.max([start_pos["longitude"],
                                end_pos["longitude"]])+lon_extension*5,
                        np.min([start_pos["latitude"],
                                end_pos["latitude"]])-lat_extension*5,
                        np.max([start_pos["latitude"],
                                end_pos["latitude"]])+lat_extension*5,
                        ])
        
            ax4.set_extent([np.min([start_pos["longitude"],
                                end_pos["longitude"]])-lon_extension*5,
                        np.max([start_pos["longitude"],
                                end_pos["longitude"]])+lon_extension*5,
                        np.min([start_pos["latitude"],
                                end_pos["latitude"]])-lat_extension*5,
                        np.max([start_pos["latitude"],
                                end_pos["latitude"]])+lat_extension*5,
                        ])
        print("Coastlines and Domain plotted")
        #Identify periods of strong radar reflectivity
        
        ax1.set_title("ERA-5")
        if with_ICON:
            ax2.set_title("ICON-NWP (2km)")
        x,y=np.meshgrid(ds["longitude"],ds["latitude"])
        norm=BoundaryNorm(
                met_var_dict["levels"]["LWP"],
                ncolors=plt.get_cmap(met_var_dict["colormap"]["LWP"]).N,
                clip=True)
        temp_ds=ds[met_var_dict["ERA_name"]["LWP"]][last_hour,:,:]*1000
        #temp_ds=temp_ds[temp_ds>50]
        C1=ax1.pcolormesh(
                x,y,temp_ds,
                norm=norm,transform=ccrs.PlateCarree(),
                cmap=met_var_dict["colormap"]["LWP"])
        print("Map 1 of 4 plotted")
        if with_ICON:
            C2=ax2.scatter(np.rad2deg(lwp_icon.clon),
                           np.rad2deg(lwp_icon.clat),
                           c=lwp_icon[0,:],
                           cmap=met_var_dict["colormap"]["LWP"],
                           s=0.5,vmin=50.0,vmax=upper_iwv,
                           transform=ccrs.PlateCarree())
        
            axins2=inset_axes(ax2,width="3%",
                          height="80%",
                          loc="center",
                          bbox_to_anchor=(0.55,0,1,1),
                          bbox_transform=ax2.transAxes,
                          borderpad=0)   
        
            cb2=map_fig.colorbar(C2,cax=axins2,extend="max")
            cb2.set_label("LWP"+" "+met_var_dict["units"]["LWP"])
            cb2.set_ticks([50,100,150,200,250,300,350,400,450,500])
            print("Map 2 of 4 plotted")
        
        norm=BoundaryNorm(
                met_var_dict["levels"]["IWP"],
                ncolors=plt.get_cmap(met_var_dict["colormap"]["LWP"]).N,clip=True)
        temp_ds=ds[met_var_dict["ERA_name"]["IWP"]][last_hour,:,:]*1000
        #temp_ds=temp_ds[temp_ds>50]
        C3=ax3.pcolormesh(
                    x,y,temp_ds,
                    norm=norm,transform=ccrs.PlateCarree(),
                    cmap=met_var_dict["colormap"]["IWP"])
        
        print("Map 3 of 4 plotted")
        if with_ICON:
            C4=ax4.scatter(np.rad2deg(lwp_icon.clon),
                           np.rad2deg(lwp_icon.clat),c=iwp_icon[0,:],
                           cmap=met_var_dict["colormap"]["IWP"],
                           s=0.5,vmin=50.0,vmax=upper_iwv,
                           transform=ccrs.PlateCarree())
        
            axins4=inset_axes(ax4,width="3%",
                          height="80%",
                          loc="center",
                          bbox_to_anchor=(0.55,0,1,1),
                          bbox_transform=ax4.transAxes,
                          borderpad=0)   
            cb4=map_fig.colorbar(C4,cax=axins4,extend="max")
            cb4.set_label("IWP"+" "+met_var_dict["units"]["IWP"])
            cb4.set_ticks([50,100,150,200,250,300,350,400,450,500])
            print("Map 4 of 4 plotted")
        if AR_in_Catalogue:
            print("Plot from Guan & Waliser")
        # # Plot AR Detection 
        # plt.rcParams.update({'hatch.color': 'lightgrey'})
        # if last_hour<6:
        #     AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
        #             AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
        #             hatches=['/'],cmap='bone',alpha=0.1,
        #             transform=ccrs.PlateCarree())
        # elif 6<=last_hour<12:
        #     AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
        #             AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
        #             hatches=[ '/'],cmap='bone',alpha=0.1,
        #             transform=ccrs.PlateCarree())
        # elif 12<=last_hour<18:
        #     AR_C=ax1.contourf(AR_era_ds.lon,AR_era_ds.lat,
        #             AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
        #             alpha=0.1,hatches=["/"],cmap=None,
        #             transform=ccrs.PlateCarree())
        # else:
        #     AR_C=ax1.contour(AR_era_ds.lon,AR_era_ds.lat,
        #             AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
        #             hatches=['/'],cmap='bone',alpha=0.1,color="white",
        #             transform=ccrs.PlateCarree())
        # if last_hour<6:
        #     ax2.contour(AR_era_ds.lon,AR_era_ds.lat,
        #             AR_era_ds.shape[0,AR_era_data["model_runs"].start,0,:,:],
        #             hatches=['/'],cmap='bone',alpha=0.1,
        #             transform=ccrs.PlateCarree())
        # elif 6<=last_hour<12:
        #     ax2.contour(AR_era_ds.lon,AR_era_ds.lat,
        #             AR_era_ds.shape[0,AR_era_data["model_runs"].start+1,0,:,:],
        #             hatches=['/'],cmap='bone',alpha=0.1,
        #             transform=ccrs.PlateCarree())
        # elif 12<=last_hour<18:
        #     ax2.contourf(AR_era_ds.lon,AR_era_ds.lat,
        #             AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
        #             alpha=0.1,hatches=['/'],cmap=None,
        #             transform=ccrs.PlateCarree())
        # else:
        #     ax2.contour(AR_era_ds.lon,AR_era_ds.lat,
        #             AR_era_ds.shape[0,AR_era_data["model_runs"].start+3,0,:,:],
        #             hatches=['/'],alpha=0.1,
        #             transform=ccrs.PlateCarree())
        # AR_C.set_label="AR (Guan & Waliser, 2019)"
        
        #C2=ax2.contourf(ds["longitude"],ds["latitude"],
        #                    ds[met_var_dict["ERA_name"]["IVT"]][last_hour,:,:],
        #                    levels=met_var_dict["levels"]["IVT"],
        #                    extend="max",transform=ccrs.PlateCarree(),
        #                    cmap=met_var_dict["colormap"]["IVT"],alpha=0.95)
        #cb2=map_fig.colorbar(C2,cax=axins2)
        
        
        #Plot HALO flight course
        ax1.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),
                  color="salmon",linestyle='--',linewidth=1.0,alpha=0.9)    
        if not self.synthetic_campaign:
            ax1.scatter(
                halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                         cut_radar["Reflectivity"].index[-1]],
                halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                        cut_radar["Reflectivity"].index[-1]],
                transform=ccrs.PlateCarree(),marker=".",s=3,color="red",
                alpha=0.95,zorder=1)    
        else:
            ax1.scatter(
                halo_df["longitude"],halo_df["latitude"],
                transform=ccrs.PlateCarree(),marker=".",s=3,color="red",
                alpha=0.95,zorder=1)    
        
        #---------------------------------------------------------------------#
        if with_ICON:
            ax2.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),color="salmon",
                     linestyle='--',linewidth=1.0,alpha=0.9)    
            if not self.synthetic_campaign:
                ax2.scatter(
                    halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                             cut_radar["Reflectivity"].index[-1]],
                    halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                            cut_radar["Reflectivity"].index[-1]],
                    transform=ccrs.PlateCarree(),marker='.',s=3,
                    color="red",alpha=0.95,zorder=1)
            else:
                ax2.scatter(halo_df["longitude"],halo_df["latitude"],
                            transform=ccrs.PlateCarree(),marker=".",
                            s=3,color="red",alpha=0.95,zorder=1)    
        
        #---------------------------------------------------------------------#
        ax3.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),color="salmon",
                  linestyle='--',linewidth=1.0,alpha=0.9)    
        if not self.synthetic_campaign:
            ax3.scatter(
                halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:
                                         cut_radar["Reflectivity"].index[-1]],
                halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                    cut_radar["Reflectivity"].index[-1]],
                transform=ccrs.PlateCarree(),marker='.',s=3,
                color="red",alpha=0.95,zorder=1)    
        else:
            ax3.scatter(halo_df["longitude"],halo_df["latitude"],
                        transform=ccrs.PlateCarree(),marker=".",
                        s=3,color="red",alpha=0.95,zorder=1)    
       
        #---------------------------------------------------------------------#
        if with_ICON:
            ax4.plot(halo_df["longitude"],halo_df["latitude"],
                  transform=ccrs.PlateCarree(),color="salmon",
                  linestyle='--',linewidth=1.0,alpha=0.9)    
            if not self.synthetic_campaign:
                ax4.scatter(
                    halo_df["longitude"].loc[cut_radar["Reflectivity"].index[0]:
                                             cut_radar["Reflectivity"].index[-1]],
                    halo_df["latitude"].loc[cut_radar["Reflectivity"].index[0]:\
                                            cut_radar["Reflectivity"].index[-1]],
                    transform=ccrs.PlateCarree(),marker='.',s=3,
                    color="red",alpha=0.95,zorder=1)    
            else:
                ax4.scatter(halo_df["longitude"],halo_df["latitude"],
                transform=ccrs.PlateCarree(),marker=".",s=3,color="red",
                alpha=0.95,zorder=1)    
        
            print("HALO flight path plotted")        
        #---------------------------------------------------------------------#
        
        #Plot high reflectivity values
        if not self.synthetic_campaign:
            if not cut_radar==None:
                high_dbZ_index=cut_radar["Reflectivity"]\
                                    [cut_radar["Reflectivity"]>15].any(axis=1)
                high_dbZ=cut_radar["Reflectivity"].loc[high_dbZ_index]
        
                ax1.scatter(halo_df["longitude"].reindex(high_dbZ.index),
                        halo_df["latitude"].reindex(high_dbZ.index),
                        s=30,color="white",marker="D",linewidths=0.5,
                        label="Radar dBZ > 15",edgecolor="k",
                        transform=ccrs.PlateCarree())
            
                ax2.scatter(halo_df["longitude"].reindex(high_dbZ.index),
                        halo_df["latitude"].reindex(high_dbZ.index),
                        s=30,color="white",marker="D",
                        linewidths=0.5,edgecolor="k",
                        transform=ccrs.PlateCarree())
            
                ax3.scatter(halo_df["longitude"].reindex(high_dbZ.index),
                        halo_df["latitude"].reindex(high_dbZ.index),
                        s=30,color="white",marker="D",linewidths=0.5,
                        label="Radar dBZ > 15",edgecolor="k",
                        transform=ccrs.PlateCarree())
                #-------------------------------------------------------------#
                if with_ICON:
                    if not self.synthetic_campaign:
                        ax4.scatter(halo_df["longitude"].reindex(high_dbZ.index),
                                    halo_df["latitude"].reindex(high_dbZ.index),
                                    s=30,color="white",marker="D",linewidths=0.5,
                                    label="Radar dBZ > 15",edgecolor="k",
                                    transform=ccrs.PlateCarree())
                    else:
                        pass
                            
            #-----------------------------------------------------------------#
            print("High Reflectivities plotted")
        
            #plot Dropsonde releases
            date=campaign_cls.year+campaign_cls.flight_month[flight_str]
            date=date+campaign_cls.flight_day[flight_str]
        
            if not Dropsondes=={}:
                # in some cases the Dropsondes variable can be a dataframe or
                # just a series, if only one sonde has been released
                if isinstance(Dropsondes["Lat"],pd.DataFrame):
                    dropsonde_releases=pd.DataFrame(index=pd.DatetimeIndex(
                                                        Dropsondes["LTS"].index))
                    dropsonde_releases["Lat"]=Dropsondes["Lat"]\
                                                .loc[:,"6000.0"].values
                    dropsonde_releases["Lon"]=Dropsondes["Lon"]\
                                                .loc[:,"6000.0"].values
            
                else:
                    index_var=Dropsondes["Time"].loc["6000.0"]
                    dropsonde_releases=pd.Series()
                    dropsonde_releases["Lat"]=np.array(
                                                Dropsondes["Lat"].loc["6000.0"])
                    dropsonde_releases["Lon"]=np.array(
                                                Dropsondes["Lon"].loc["6000.0"])
                    dropsonde_releases["Time"]=index_var
            
                if not self.flight=="RF08":
                    relevant_dropsondes=dropsonde_releases.loc[\
                                        cut_radar["Reflectivity"].index[0]:\
                                        cut_radar["Reflectivity"].index[-1]]
                
                if relevant_dropsondes.shape[0]>0:
                    ax1.scatter(relevant_dropsondes["Lon"],
                                relevant_dropsondes["Lat"],
                                s=100,marker="v",color="orange",
                                edgecolors="black",label="Dropsondes",
                                transform=ccrs.PlateCarree())
                    
                    ax2.scatter(relevant_dropsondes["Lon"],
                                relevant_dropsondes["Lat"],
                                s=100,marker="v",color="orange",
                                edgecolors="black",
                                transform=ccrs.PlateCarree())
                    ax3.scatter(relevant_dropsondes["Lon"],
                                relevant_dropsondes["Lat"],
                                s=100,marker="v",color="orange",
                                edgecolors="black",
                                transform=ccrs.PlateCarree())
                    if with_ICON:
                        ax4.scatter(relevant_dropsondes["Lon"],
                                relevant_dropsondes["Lat"],
                                s=100,marker="v",color="orange",
                                edgecolors="black",
                                transform=ccrs.PlateCarree())
            print("Dropsondes plotted")        
        #if not resizing_done:    
        map_fig.suptitle(
            campaign_cls.name+\
            ": Liquid Water Path (LWP), Ice Water Path (IWP) for \n"+\
            self.flight+": "+campaign_cls.year+"-"+\
            campaign_cls.flight_month[flight_str]+"-"+\
            campaign_cls.flight_day[flight_str]+" "+calc_time,y=0.98)
        #else:
            
        legend=ax3.legend(bbox_to_anchor=(0.65,-0.2,1.5,0),
                          facecolor='lightgrey',loc="lower center",
                          ncol=2,mode="expand")
        legend.get_frame().set_linewidth(2.0)
        legend.get_frame().set_edgecolor("k")
        
        #Save figure
        if with_ICON:
            fig_name=self.ar_of_day+"_"+self.flight+"_"+campaign_cls.name+\
                    '_LWP_IWP_MAP_ERA5_ICON_'+str(last_hour)+"UTC.png"
        else:
            fig_name=self.ar_of_day+"_"+self.flight+"_"+campaign_cls.name+\
                    '_LWP_IWP_MAP_ERA5_'+str(last_hour)+"UTC.png"
        if opt_plot_path==None:
            fig_path=self.plot_path#campaign_cls.plot_path+flight+"/"
            #if with_ICON:
            #    fig_path=campaign_cls.plot_path+flight+"ICON_2km/"
        else:
            fig_path=opt_plot_path
        
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        map_fig.savefig(fig_path+fig_name,bbox_inches="tight",dpi=250)
        print("Figure saved as:",fig_path+fig_name)
        return None
    def plot_ar_section_internal_leg_ICON(self,campaign_cls):
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        import matplotlib.path as mpath
    
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
       
        map_fig=plt.figure(figsize=(12,12))
        ax = plt.axes(projection=ccrs.LambertConformal(
                                central_longitude=-7,
                                central_latitude=70))
                      #projection=ccrs.AzimuthalEquidistant(
                      #              central_longitude=-10.0,
                      #              central_latitude=70))
        ax.coastlines(resolution="50m")
        #ax.set_extent([-35,10,60,75])
        
        
        #######################################################################
        ### Include ICON
        hydrometeor_icon_path=campaign_cls.campaign_path+"/data/ICON_LEM_2KM/"
        resolution=2000 # units m
        last_hour=14
        var="Hydrometeor"#
        
        import ICON
        icon=ICON.ICON_NWP(str(last_hour),resolution,
                               for_flight_campaign=True,
                               campaign="NAWDEX",research_flights=None,
                               icon_path=hydrometeor_icon_path)
        
        
        print("Open ICON-Simulations Start Hour")
        icon_file_name=var+"_ICON_"+self.flight+"_"+str(last_hour)+"UTC.nc"    
    
        icon_ds=icon.load_icon_data(icon_file_name)
        lwp_icon=icon_ds["tqc_dia"]*1000    # units g/m2
        iwp_icon=icon_ds["tqi_dia"]*1000    # units g/m2
        hmp_icon=lwp_icon+iwp_icon
        hmp_icon=hmp_icon.where(hmp_icon>50)
        C1=ax.scatter(np.rad2deg(hmp_icon.clon),
                           np.rad2deg(hmp_icon.clat),c=hmp_icon[0,:],
                           cmap="GnBu",
                           s=0.5,vmin=50.0,vmax=800,
                           transform=ccrs.PlateCarree())
        axins=inset_axes(ax,width="3%",
                          height="60%",
                          loc="center",
                          bbox_to_anchor=(0.4,0,1.3,1),
                          bbox_transform=ax.transAxes,
                          borderpad=0)
        cb=map_fig.colorbar(C1,cax=axins,extend="max")
        cb.set_label("CWP"+" "+"(g/$\mathrm{m}^2$)",fontsize=20)
        cb.set_ticks([50,200,400,600,800])
        cb.ax.tick_params(labelsize=20)    
        #######################################################################
        
        
        ax.plot(self.track_dict["inflow"]["longitude"],
                       self.track_dict["inflow"]["latitude"],
                       transform=ccrs.PlateCarree(),lw=3,ls="--",color="red",
                       label="shifted track (inflow)")

        ax.plot(self.track_dict["internal"]["longitude"],
                       self.track_dict["internal"]["latitude"],
                       transform=ccrs.PlateCarree(),lw=3,color="black",
                       label="synthetic track (internal)")

        ax.plot(self.track_dict["outflow"]["longitude"],
                       self.track_dict["outflow"]["latitude"],
                       transform=ccrs.PlateCarree(),lw=3,ls="--",
                       color="salmon",label="synthetic track (outflow)")
        ax.legend(fontsize=16,loc="center right", bbox_to_anchor=[0.9, 0.25, 0, 0])            
        ax.gridlines()
        xlim = [-31, 10]
        ylim = [61, 75]
        
        rect = mpath.Path([[xlim[0], ylim[0]],
                   [xlim[1], ylim[0]],
                   [xlim[1], ylim[1]],
                   [xlim[0], ylim[1]],
                   [xlim[0], ylim[0]],
                   ]).interpolated(20)

        proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
        rect_in_target = proj_to_data.transform_path(rect)
        lower_space = 2#45
        ax.set_boundary(rect_in_target)
        ax.set_extent([xlim[0], xlim[1], ylim[0] - lower_space, ylim[1]])
    
        fig_name="Internal_Leg_map.png"
        map_fig.savefig(self.plot_path+fig_name,
                        dpi=300,bbox_inches="tight")
        print("Figure saved as:",self.plot_path+fig_name)
        

###############################################################################
###############################################################################
def main():    
    base_path=os.getcwd()+"/../../../"
    path=base_path+"/Work/GIT_Repository/"
    name="data_config_file"
    config_file_exists=False
    #campaign_name="NAWDEX"
    campaign_name="Second_Synthetic_Study"#"HALO_AC3"#"NA_February_Run"##"NA_February_Run"#"Second_Synthetic_Study"#"NA_February_Run"#"HALO_AC3"#"NA_February_Run"    
    flights=["SRF08"]#["SRF07"]#["RF07"]#["SRF06"]
    met_variable="IVT"
    ar_of_day="SAR_internal"
    ###Switcher in order to specify maps plots to create
    should_plot_iop_map=False
    should_plot_era_map=True
    should_plot_RFs_AR_map=True
    
    # Check if config-File exists and if not create the relevant first one
    if data_config.check_if_config_file_exists(name):
        config_file=data_config.load_config_file(path,name)
    else:
        data_config.create_new_config_file(file_name=name+".ini")
        
    if sys.platform.startswith("win"):
        system_is_windows=True
    else:
        system_is_windows=False
        
    if system_is_windows:
        if not config_file["Data_Paths"]["system"]=="windows":
            windows_paths={
                "system":"windows",
                "campaign_path":os.getcwd()+"/"    
                    }
            windows_paths["save_path"]=windows_paths["campaign_path"]+"Save_path/"
            data_config.add_entries_to_config_object(name,windows_paths)
        
                        
    if campaign_name=="NAWDEX":     
        is_flight_campaign=True
        nawdex=NAWDEX(is_flight_campaign=True,
                          major_path=config_file["Data_Paths"]["campaign_path"],
                          aircraft="HALO",instruments=["radar","radiometer","sonde"])
        nawdex.specify_flights_of_interest(flights)
        nawdex.create_directory(directory_types=["data"])
        print("NAWDEX interested flights", nawdex.interested_flights)
        cmpgn_cls=nawdex
    
    elif campaign_name=="NA_February_Run":
        is_flight_campaign=True
        ar_na=North_Atlantic_February_Run(is_flight_campaign=True,
                          major_path=config_file["Data_Paths"]["campaign_path"],
                          aircraft="HALO",interested_flights=[flights[0]],
                          instruments=["radar","radiometer","sonde"])
        ar_na.specify_flights_of_interest(flights)
        ar_na.create_directory(directory_types=["data"])
        print("NA_February_Run interested flights", ar_na.interested_flights)
        cmpgn_cls=ar_na
    elif campaign_name=="Second_Synthetic_Study":
        is_flight_campaign=True
        ar_na=Second_Synthetic_Study(is_flight_campaign=True,
                          major_path=config_file["Data_Paths"]["campaign_path"],
                          aircraft="HALO",interested_flights=[flights[0]],
                          instruments=["radar","radiometer","sonde"])
        ar_na.specify_flights_of_interest(flights)
        ar_na.create_directory(directory_types=["data"])
        print(campaign_name+" interested flights", ar_na.interested_flights)
        cmpgn_cls=ar_na
    elif campaign_name=="HALO_AC3":
        is_flight_campaign=True
        ar_na=HALO_AC3(is_flight_campaign=True,
                          major_path=config_file["Data_Paths"]["campaign_path"],
                          aircraft="HALO",interested_flights=[flights[0]],
                          instruments=["radar","radiometer","sonde"])
        ar_na.specify_flights_of_interest(flights)
        ar_na.create_directory(directory_types=["data"])
        print("NA_February_Run interested flights", ar_na.interested_flights)
        cmpgn_cls=ar_na

    else:
        pass
    flight_maps=FlightMaps(cmpgn_cls.major_path,cmpgn_cls.campaign_path,
                           cmpgn_cls.aircraft,cmpgn_cls.instruments,
                           cmpgn_cls.interested_flights,
                           ar_of_day=ar_of_day, analysing_campaign=False,
                           synthetic_campaign=True)
    
    #%% Run plotting of desired maps
    
    # IOP MAP
    if should_plot_iop_map:
        flight_maps.flight_map_iop(nawdex)
    
    # RFs with present ARs
    if should_plot_RFs_AR_map:
        pass
        #rf_flights=["RF03","RF10","RF12"]
        #rf_colors={"red":"Reds",
        #           "green":"Greens",
        #           "orange":"Oranges"}
                   
                   #"blue":"Blues",
                   #"black":"Greys",
                   #"purple":"Purples",
                   #"green":"Greens",
                   #"red":"Reds",
                   #"blue":"Blues",
                   #"orange":"Oranges",
                   #"magenta":"spring_r",
                   #"darkgreen":"summer_r"}
        #flight_maps.plot_flight_map_complete_NAWDEX_with_ARs(nawdex,rf_flights,
        #                                                     rf_colors,
        #                                                     with_dropsondes=True,
        #                                                     include_ARs=True)
        
    if should_plot_era_map:
        #from Cloudnet import Cloudnet_Data
        #campaign_cloudnet=Cloudnet_Data(cmpgn_cls.campaign_path)
        #if not cmpgn_cls.is_synthetic_campaign:
        #    station_coords=campaign_cloudnet.get_cloudnet_station_coordinates(
        #                                        cmpgn_cls.campaign_path)
        station_coords={}
        flight_maps.plot_flight_map_era(cmpgn_cls,station_coords,flights[0],
                                        met_variable,show_AR_detection=True,
                                        show_supersites=False,use_era5_ARs=True)
    

if __name__=="__main__":
    main()




### Specifications:
##Data
    
# years=["2016"]#["2016"]
# months= ["10"]#,"02","02","02","02"]
# days=   ["09"]#,"03","04","05","06","07","08","09","10","11","12","13","14","15"]
# hours_time=[  '00:00', '01:00', '02:00',
#                             '03:00', '04:00', '05:00',
#                             '06:00', '07:00', '08:00',
#                             '09:00', '10:00', '11:00',
#                             '12:00', '13:00', '14:00',
#                             '15:00', '16:00', '17:00',
#                             '18:00', '19:00', '20:00',
#                             '21:00', '22:00', '23:00',]
# analysing_campaign=True
# #if len(sys.argv)==1:
# #    central_path="C:/Users/Henning/OneDrive/PhD/Work/"
# #else:
# central_path="/home/zmaw/u300737/PhD/Work/"
# halo_data_path="/scratch/uni/u237/users/hdorff/"
# campaign="NAWDEX"#"Storm_Case_2"#"NAWDEX"

# central_path="/scratch/uni/u237/users/hdorff/"+campaign+"_data/ERA-5/"
# flight="RF08"

# ##Plotting specifications
# try:
#     import typhon 
#     import cartopy.crs as ccrs
#     colormap="density"
# except:
#     import cartopy.crs as ccrs
#     print("No modules of Typhon are loaded, but Cartopy")
#     colormap="viridis"
# set_font=14
# matplotlib.rcParams.update({'font.size':set_font})
# levels=np.linspace(0,50,50)
# ####Run main script
# #check if data is already downladed
# file_name=central_path+"total_columns_"+years[0]+"_"+months[0]+"_"+days[0]+".nc"


# ##Plotting specifications
# try:
#     import typhon 
# except:
#     print("No Typhon module loaded")
#     pass


# ####Run main script
# #check if data is already downladed
# if not os.path.isfile(file_name):
#     print("Data is not yet downloaded!")
#     get_hourly_era5_total_columns(years,months,days,central_path,defined_area= [90, -80, 20,  50])
# else:
#     print("Data is already downloaded on storage! Start analysing")
# ## Load ERA-5 Dataset

# #Paths
# ds=xr.open_dataset(file_name)
# latitude=np.array(ds["latitude"])
# longitude=np.array(ds["longitude"])

# #Columns
# #hydrometeor_lvls_path=halo_data_path+campaign+"_data/ERA-5/"
# #hydrometeor_lvls_file="hydrometeors_pressure_levels_"+years[0]+months[0]+days[0]+".nc"
# #print("open hydrometeor_levels")
# #ds_lvls=xr.open_dataset(hydrometeor_lvls_path+hydrometeor_lvls_file)
# #sys.exit()
# #Load Cloudnet data
# cloudnet_path="/scratch/uni/u237/users/hdorff/Cloudnet_data/"
# print("Load CLoudnet Sites")
# try:
#     ny_alesund  = load_cloudnet(cloudnet_path,"Ny-Alesund",years[0]+months[0]+days[0],["LWC","IWC"],["lwc","iwc"])
# except:
#     pass
# try:
#     summit      = load_cloudnet(cloudnet_path,"Summit",years[0]+months[0]+days[0],["LWC","IWC"],["lwc","iwc"])
# except:
#     pass
# try:
#     mace_head   = load_cloudnet(cloudnet_path,"Mace-Head",years[0]+months[0]+days[0],["categorize","classification"],["categorize","target_classification"])
# except:
#     mace_head= {}

# home_path="/home/zmaw/u300737/PhD/Work/"
# #sys.exit()
# if analysing_campaign:
# # Load Halo Dataset
#     halo_df,campaign_path=load_aircraft_position(home_path,flight,campaign)
#     halo_df["Closest_Lat"]  = np.nan
#     halo_df["Closest_Lon"]  = np.nan
#     halo_df.index           = pd.DatetimeIndex(halo_df.index)
#     halo_df["Hour"]=halo_df.index.hour
#     print("HALO data is downloaded!")
#     Dropsondes=load_dropsonde_data(halo_data_path,campaign,years[0]+months[0]+days[0],print_arg="yes",dt="all",plotting="no")
#     print("Dropsondes done")
#     if isinstance(Dropsondes["Lat"],pd.DataFrame):
#         dropsonde_releases=pd.DataFrame(index=pd.DatetimeIndex(Dropsondes["LTS"].index))
#         dropsonde_releases["Lat"]=Dropsondes["Lat"].loc[:,"6000.0"].values
#         dropsonde_releases["Lon"]=Dropsondes["Lon"].loc[:,"6000.0"].values
    
#     else:
#         index_var=Dropsondes["Time"].loc["6000.0"]
#         dropsonde_releases=pd.Series()
#         dropsonde_releases["Lat"]=np.array(Dropsondes["Lat"].loc["6000.0"])
#         dropsonde_releases["Lon"]=np.array(Dropsondes["Lon"].loc["6000.0"])
#         dropsonde_releases["Time"]=index_var
    
#     #Dropsondes["IWV"]=Dropsondes["IWV"].loc[radar["Reflectivity"].index[0]:radar["Reflectivity"].index[-1]]
#     #sys.exit()
# else:
#     print("This none dataset of the flight campaign so no airborne datasets will be integrated.")
# print("Research Flight, ",flight)
# #print(halo_df[["latitude","longitude"]].describe())
# #halo_df["longitude"].describe()
# #sys.exit()
# # Calculate min degree distance time point
# #mace_head["latitude"]=mace_head["latitude"]+3.49

# #lon_dist=abs(halo_df["longitude"]-mace_head["longitude"]+360)
# #lat_dist=abs(halo_df["latitude"]-mace_head["latitude"])
# #deg_dist=np.sqrt(lon_dist**2+lat_dist**2)
# #deg_dist=deg_dist.sort_values(ascending=True)
# #closest_timepoint=deg_dist.index[0]
# #end_timepoint=closest_timepoint+pd.Timedelta("60min")
# #closest_halo_index=halo_df.index.get_loc(closest_timepoint,method="nearest")

# #cutted_mace_head={}#mace_head.copy()

# #cutted_mace_head["Radar-Reflectivity"]=mace_head["Radar-Reflectivity"].loc[closest_timepoint:end_timepoint]
# #cutted_mace_head["iwc"]=mace_head["iwc"].loc[closest_timepoint:end_timepoint]
# #cutted_mace_head["lwc"]=mace_head["lwc"].loc[closest_timepoint:end_timepoint]
# #cutted_mace_head["target_classification"]=mace_head["target_classification"].loc[closest_timepoint:end_timepoint]
# radar=load_hamp_data(halo_data_path,campaign,[flight],instrument="Halo")
# #cutted_halo_radar=radar["Reflectivity"].iloc[closest_halo_index-270:closest_halo_index]
# #sys.exit()

# pd.plotting.register_matplotlib_converters()
# plot_path=home_path+"ERA-5/"+flight+"/"
# """ Mapping flights
# """
# style_name="Typhon"
# try: 
#     with plt.style.context(styles(style_name)):
#         print("Create maps")
#         plot_flight_map(ds,halo_df,dropsonde_releases,mace_head,summit,ny_alesund,flight,plot_path)
# except:
#     plot_flight_map(ds,halo_df,dropsonde_releases,mace_head,summit,ny_alesund,flight,plot_path)
# """ Overpass analysis

# try:
#     with plt.style.context(styles(style_name)):
#         print("Plots created with Typhon")
#         plot_halo_super_site_overpass(cutted_halo_radar,cutted_mace_head,station,date)
# except:
#         plot_halo_super_site_overpass(cutted_halo_radar,cutted_mace_head,station,date)
# """

fig_path="empty_path"
fig_name="no_figure_name"    
