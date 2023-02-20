# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:49:14 2020

@author: u300737
"""
import sys
import os
import glob
import data_config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Atmospheric_Rivers():
    def __init__(self,reanalysis,use_era5=False):
        self.catalogue_path="C:/Users/u300737/Desktop/PhD_UHH_WIMI/"+\
                            "Overall_Data/AR_Catalogues/"
        self.era5_catalogue_path=self.catalogue_path+"ARcatalog_ERA5_march_1979-2022/"
        self.reanalysis=reanalysis
        #os.chdir(self.catalogue_path)
        self.AR_catalogue_nc=glob.glob(self.catalogue_path+"*ERA*.nc")
        self.AR_catalogue_nc_2020=glob.glob(self.catalogue_path+"*MERRA*.nc")
        self.use_era5=use_era5
        #self.AR_catalogue_era5=glob.glob
            
    def open_AR_catalogue(self,after_2019=False,
                          year="2022",month="03"):
        import xarray as xr
        if self.use_era5:
            AR_month=glob.glob(self.era5_catalogue_path+"*"+
                               year+month+".nc")
            AR_ds=xr.open_dataset(AR_month[0],engine="netcdf4")
        else:
            if not after_2019:
                AR_ds=xr.open_dataset(self.AR_catalogue_nc[0])
            else:
                AR_ds=xr.open_dataset(self.AR_catalogue_nc_2020[0])
            
        return AR_ds
    
    def specify_AR_data(self,AR_ds,flight_day):
        AR_era_data={}
        AR_era_data["time"]=pd.Series(AR_ds["time"])
        AR_era_data["model_runs"]=pd.DatetimeIndex(
                                        AR_era_data["time"]).get_loc(flight_day)
        AR_era_data["shape"]=AR_ds["shape"][:]
        if not self.use_era5:
            AR_era_data["kidmap"]=AR_ds["kidmap"][:]
        AR_era_data["lat"]=np.array(AR_ds["lat"])
        AR_era_data["lon"]=np.array(AR_ds["lon"])
        return AR_era_data
    
    def get_HALO_NAWDEX_AR_cross_sections(self):
        ARs={}
        ARs["RF01"]={}
        ARs["RF02"]={}
        ARs["RF03"]={}
        ARs["RF04"]={}
        ARs["RF05"]={}
        ARs["RF08"]={}
        ARs["RF09"]={}
        ARs["RF10"]={}
        ARs["RF12"]={}
        
        ARs["RF01"]["AR1"]={"start":"2016-09-17 10:30",
                            "end":"2016-09-17 12:15"}
        ARs["RF02"]["AR1"]={"start":"2016-09-21 14:40",
                            "end":"2016-09-21 15:40"}
        ARs["RF02"]["AR2"]={"start":"2016-09-21 16:45",
                            "end":"2016-09-21 18:10"}
        ARs["RF03"]["AR1"]={"start":"2016-09-23 11:00",
                            "end":"2016-09-23 11:59"} #12:15
        ARs["RF03"]["AR2"]={"start":"2016-09-23 12:20",
                            "end":"2016-09-23 13:05"}
        ARs["RF03"]["AR3"]={"start":"2016-09-23 14:00",
                            "end":"2016-09-23 15:00"}
        ARs["RF04"]["AR1"]={"start":"2016-09-26 11:35",
                            "end":"2016-09-26 12:10"}
        ARs["RF04"]["AR2"]={"start":"2016-09-26 13:20",
                            "end":"2016-09-26 13:55"}
        ARs["RF04"]["AR3"]={"start":"2016-09-26 15:55",
                            "end":"2016-09-26 16:45"}
        ARs["RF04"]["AR4"]={"start":"2016-09-26 17:40",
                            "end":"2016-09-26 18:05"}
        ARs["RF05"]["AR1"]={"start":"2016-09-27 14:00",
                            "end":"2016-09-27 15:15"}
        ARs["RF05"]["AR2"]={"start":"2016-09-27 17:00",
                            "end":"2016-09-27 17:45"}
        ARs["RF05"]["AR3"]={"start":"2016-09-27 18:00",
                            "end":"2016-09-27 19:00"}
        ARs["RF08"]["AR1"]={"start":"2016-10-09 16:00",
                            "end":"2016-10-09 17:00"}
        ARs["RF09"]["AR1"]={"start":"2016-10-10 14:00",
                            "end":"2016-10-10 14:50"}
        ARs["RF09"]["AR2"]={"start":"2016-10-10 15:00",
                            "end":"2016-10-10 15:50"}
        ARs["RF10"]["AR1"]={"start":"2016-10-13 09:15",
                            "end":"2016-10-13 10:15"}
        ARs["RF10"]["AR2"]={"start":"2016-10-13 10:45",
                            "end":"2016-10-13 11:15"}
        #Default
        ARs["RF10"]["AR3"]={"start":"2016-10-13 12:10",
                            "end":"2016-10-13 13:50"}
        
        #Default
        #ARs["RF05"]["AR99"]={"start":"2016-09-27 14:00",
        #                    "end":"2016-09-27 14:10"}
        
        #ARs["RF10"]["AR3"]={"start":"2016-10-13 12:06",
        #                    "end":"2016-10-13 14:00"}
        #ARs["RF12"]["AR1"]={"start":"2016-10-15 09:45",
        #                    "end":"2016-10-15 10:45"}
        #ARs["RF12"]["AR2"]={"start":"2016-10-15 14:45",
        #                    "end":"2016-10-15 15:50"}     
        #ARs["RF12"]["AR99"]={"start":"2016-10-15 10:00",
        #                    "end":"2016-10-15 10:20"}
        
        return ARs
    def get_ids_of_ARs_from_campaign_month(self,AR_ds,campaign_cls):
        pass
    def get_ids_of_ARs_from_HALO_cross_sections(self,AR_ds,campaign_cls,
                                                       config_file,
                                                       single_flight="",
                                                       single_ARs=""):
        store_AR_ds=AR_ds.copy()
        unique_AR=np.empty(1)
        from gridonhalo import ERA_on_HALO
        #era_on_halo=ERA_on_HALO()
        print("Use AR Catalogue from Guan & Waliser")
        
        AR_case_s     = pd.Series()
        AR_cross_start= pd.Series()
        AR_cross_end  = pd.Series()
        AR_clat_s     = pd.Series()
        AR_kid_s      = pd.Series()
        AR_kstatus_s  = pd.Series()
        AR_tivt_s     = pd.Series()                
        AR_mean_ivtx_s= pd.Series()
        AR_mean_ivty_s= pd.Series()
        AR_ivtdir_s   = pd.Series()
        AR_kstatus_s  = pd.Series()
        AR_klifetime_s= pd.Series()
        AR_knormage_s = pd.Series()
        AR_time_s     = pd.Series()
        AR_length_s   = pd.Series()
        AR_width_s    = pd.Series()            
        unique_AR_list=[]
        
        # Depending on flights to use
        if single_flight!="":
            flights_to_loop=[single_flight]
        else:
            try:
                flights_to_loop=campaign_cls.flights_of_interest
            except:
                flights_to_loop=campaign_cls.flights
        if campaign_cls.name=="NAWDEX":
            ARs_of_day=["AR3"]
            if single_ARs!="":
                ARs_of_day=[single_ARs]
            #get aircraft (HALO) position
            aircraft_position=campaign_cls.get_aircraft_position(
                flights_to_loop)
        elif (campaign_cls.name=="NA_February_Run") or \
            (campaign_cls.name=="Second_Synthetic_Study"):
            # Call the Flight Track Creator
            import flight_track_creator
            ARs_of_day=["SAR_internal"]
            aircraft_position={}
            
            for flight in flights_to_loop:
                if flight.startswith("S"):
                    print(flight)
                    Tracker=flight_track_creator.Flighttracker(campaign_cls,
                                                    flight,
                                                    ARs_of_day[0],
                                                    track_type="internal",
                                                    shifted_lat=0,
                                                    shifted_lon=0)
                    track_df,campaign_path=Tracker.get_synthetic_flight_track()
                    aircraft_dict=Tracker.make_dict_from_aircraft_df()
             
                    print("Synthetic flight track loaded")
                    # If halo df is a dict, then this may arise from the 
                    # internal leg created flight track. So merge it to a 
                    # single dataframe, or loc current leg from it
                    #if isinstance(halo_df,dict):
                    #    aircraft_dict=halo_df.copy()
                    #    halo_df,leg_dicts=Flight_Tracker.concat_track_dict_to_df()
                    
                    #track_df,cmpgn_path=Tracker.run_flight_track_creator()
                    aircraft_position[flight]=aircraft_dict["inflow"]
                else:
                    #continue
                    import Flight_Campaign
                    # should then be NAWDEX flight RF10
                    nawdex=Flight_Campaign.NAWDEX(is_flight_campaign=True,
                      major_path=config_file["Data_Paths"]["campaign_path"],
                      aircraft="HALO",instruments=["radar","radiometer","sonde"])
       
                    Tracker=flight_track_creator.Flighttracker(nawdex,
                                                    flight,
                                                    ARs_of_day[0],
                                                    track_type="internal",
                                                    shifted_lat=0,
                                                    shifted_lon=-12)
                    track_df,cmpgn_path=Tracker.run_flight_track_creator()
                    aircraft_position[flight]=track_df["inflow"]
                
        i=0
            
        for f in flights_to_loop:
            if campaign_cls.name!="NAWDEX":
                if f.startswith("R"):
                    continue
            flight_date=campaign_cls.years[f]+"-"+campaign_cls.flight_month[f]
            flight_date=flight_date+"-"+campaign_cls.flight_day[f]
            if int(flight_date[0:4])>2019:
                AR_ds=self.open_AR_catalogue(after_2019=\
                                                   int(flight_date[0:4])>2019)
            else:
                AR_ds=store_AR_ds.copy()
            AR_era_data=self.specify_AR_data(AR_ds,flight_date)
            j=0
            for ar in ARs_of_day:
                str_value=pd.Series(f+"_"+ar)
                AR_case_s=AR_case_s.append(str_value)
                try:
                    era_on_halo=ERA_on_HALO(aircraft_position[f],os.getcwd(),
                                    os.getcwd(),None,True,"NAWDEX",
                                    config_file["Data_Paths"]["campaign_path"],
                                    f,flight_date,config_file)
                    ARs=self.get_HALO_NAWDEX_AR_cross_sections()
                    ar_aircraft_position={}
                    ar_aircraft_position[f]=aircraft_position[f].loc[\
                                             ARs[f][ar]["start"]:\
                                             ARs[f][ar]["end"]]
                except:
                    ar_aircraft_position=aircraft_position
                    
                last_hour=ar_aircraft_position[f].index[-1].hour+1
                    
                if last_hour<6:
                    model_idx=0
                elif 6<=last_hour<12:
                    model_idx=1
                elif 12<=last_hour<18:
                    model_idx=2
                else:
                    model_idx=3
                #if f=="SRF03":
                #    model_idx=2
                era_lon=pd.Series(np.array(AR_era_data["lon"]))
                shifted_lon=era_lon.copy()
                shifted_lon[shifted_lon>180]=shifted_lon[shifted_lon>180]-360
                AR_id_field=pd.DataFrame(\
                                  np.array(AR_era_data["kidmap"][0,
                                    AR_era_data["model_runs"].start+model_idx,
                                    0,:,:]),index=AR_era_data["lat"],
                                  columns=shifted_lon) 
                    
                AR_id_field=AR_id_field.sort_index(axis=1)
                
                #cut AR_id field to Cross-section
                AR_id_field_cut = AR_id_field.loc[\
                            ar_aircraft_position[f]["latitude"].min()-1.25:
                            ar_aircraft_position[f]["latitude"].max()+1.25]
                AR_id_field_cut = AR_id_field_cut.loc[:,
                        ar_aircraft_position[f]["longitude"].min()-1.25:
                        ar_aircraft_position[f]["longitude"].max()+1.25]    
                AR_kid          = pd.Series(np.array(AR_ds["kid"]\
                            [0,AR_era_data["model_runs"].start+model_idx,
                             0,:]))
                kid_value   = np.unique(AR_id_field_cut)
                kid_value   = float(kid_value[~np.isnan(kid_value)])
                kid         = AR_kid[AR_kid==kid_value].index.tolist()
                AR_id_field_cut=AR_id_field_cut.replace(to_replace=np.nan,
                                                        value=-999)
                
                AR_cross_start=AR_cross_start.append(pd.Series(
                        pd.Timestamp(ar_aircraft_position[f].index[0])))
                    
                AR_cross_end=AR_cross_end.append(pd.Series(
                    pd.Timestamp(ar_aircraft_position[f].index[-1])))
                AR_time_s=AR_time_s.append(pd.Series(model_idx*6))
                    
                AR_kid_s   = AR_kid_s.append(AR_kid.iloc[kid])
                AR_clat_s  = AR_clat_s.append(pd.Series(np.array(AR_ds["clat"]\
                                        [0,AR_era_data["model_runs"].start+\
                                         model_idx,0,:])).iloc[kid])
                    
                AR_kstatus_s  = AR_kstatus_s.append(pd.Series(
                                    np.array(AR_ds["kstatus"]\
                                        [0,AR_era_data["model_runs"].start+\
                                         model_idx,0,:])).iloc[kid])
                    
                AR_tivt_s     = AR_tivt_s.append(pd.Series(
                                    np.array(AR_ds["tivt"]\
                                        [0,AR_era_data["model_runs"].start+\
                                         model_idx,0,:])).iloc[kid])                
                    
                AR_mean_ivtx_s= AR_mean_ivtx_s.append(pd.Series(np.array(
                                    AR_ds["ivtx"][0,
                                        AR_era_data["model_runs"].start+\
                                            model_idx,0,:])).iloc[kid])
                    
                AR_mean_ivty_s= AR_mean_ivty_s.append(pd.Series(
                                    np.array(AR_ds["ivty"]\
                                        [0,AR_era_data["model_runs"].start+\
                                         model_idx,0,:])).iloc[kid])
                    
                AR_ivtdir_s= AR_ivtdir_s.append(pd.Series(np.array(
                                AR_ds["ivtdir"]\
                                [0,AR_era_data["model_runs"].start+model_idx,
                                 0,:])).iloc[kid])
                    
                AR_length_s= AR_length_s.append(pd.Series(np.array(
                                AR_ds["length"][0,
                                    AR_era_data["model_runs"].start+model_idx,
                                    0,:])).iloc[kid])
                    
                AR_width_s= AR_width_s.append(pd.Series(np.array(AR_ds["width"]\
                        [0,AR_era_data["model_runs"].start+model_idx,0,:])).iloc[kid])
                    
                    
                AR_klifetime_s= AR_klifetime_s.append(pd.Series(np.array(AR_ds["klifetime"]\
                        [0,AR_era_data["model_runs"].start+model_idx,0,:])).iloc[kid])
                AR_knormage_s = AR_knormage_s.append(pd.Series(np.array(AR_ds["knormage"]\
                        [0,AR_era_data["model_runs"].start+model_idx,0,:])).iloc[kid])
#                except:
#                    AR_cross_start= AR_cross_start.append(pd.Series(np.nan))
#                    AR_cross_end  = AR_cross_end.append(pd.Series(np.nan))
#                    AR_clat_s     = AR_clat_s.append(pd.Series(np.nan))
#                    AR_time_s     = AR_time_s.append(pd.Series(np.nan))
#                    AR_kid_s      = AR_kid_s.append(pd.Series(np.nan))
#                    AR_kstatus_s  = AR_kstatus_s.append(pd.Series(np.nan))
#                    AR_tivt_s     = AR_tivt_s.append(pd.Series(np.nan))
#                    AR_mean_ivtx_s= AR_mean_ivtx_s.append(pd.Series(np.nan))
#                    AR_mean_ivty_s= AR_mean_ivty_s.append(pd.Series(np.nan))
#                    AR_ivtdir_s   = AR_ivtdir_s.append(pd.Series(np.nan))
#                    AR_klifetime_s= AR_klifetime_s.append(pd.Series(np.nan))
#                    AR_knormage_s = AR_knormage_s.append(pd.Series(np.nan))
#                    AR_width_s    = AR_width_s.append(pd.Series(np.nan))
#                    AR_length_s   = AR_length_s.append(pd.Series(np.nan))
            j+=1
                    
        unique_AR_df=pd.DataFrame(data=np.array(AR_cross_start),
                                      columns=["Cross_Start"],
                                      index=AR_case_s.values)
            
        unique_AR_df["Cross_End"]           = AR_cross_end.values
        unique_AR_df["clat"]                = AR_clat_s.values    
        unique_AR_df["width"]               = AR_width_s.values
        unique_AR_df["length"]              = AR_length_s.values
        unique_AR_df["ID"]                  = AR_kid_s.values
        unique_AR_df["ERAInterim_Time"]     = AR_time_s.values
        unique_AR_df["kStatus"]             = AR_kstatus_s.values
        unique_AR_df["TIVT"]                = AR_tivt_s.values
        unique_AR_df["IVT_x"]               = AR_mean_ivtx_s.values
        unique_AR_df["IVT_y"]               = AR_mean_ivty_s.values
        unique_AR_df["Lifetime"]            = AR_klifetime_s.values
        unique_AR_df["Norm_Age"]            = AR_knormage_s.values
        return unique_AR_df
    
    def plot_campaign_AR_statistics(self,AR_unique_df,config_file,only_IVT):
        import matplotlib
        import seaborn as sns
        if not only_IVT:
            AR_stat_fig=plt.figure(figsize=(16,14))
        else:
            AR_stat_fig=plt.figure(figsize=(9,10))
        matplotlib.rcParams.update({"font.size":25})
        uniques,indices=np.unique(AR_unique_df["ID"],return_index=True)
        unique_ID=np.array(AR_unique_df["ID"][np.sort(indices)])
        
        markers =["p","o","d","D","s","H"]
        #flight_colors={"RF01":,"RF02":"paleturquoise","RF03":"salmon",
        #         "RF04":"peru","RF05":"skyblue","RF06":"moccasin",
        #         "RF07":"slateblue","RF08":"bisque","RF09":"thistle",
        #         "RF10":"lightgreen","RF11":"lightpink","RF12":"gold",
        #         "RF13":"rosybrown"}
       
        mark_cls=["grey","salmon","skyblue","bisque","lightgreen","orange"]
        if not only_IVT:
            ax1=AR_stat_fig.add_subplot(221)
            ax2=AR_stat_fig.add_subplot(222)
            ax3=AR_stat_fig.add_subplot(223)#,sharex=ax1)
            ax4=AR_stat_fig.add_subplot(224)#,sharex=ax2)
            
            i=0
            marker_entries=[]
            for uni_ID in unique_ID:
                
                #IVT preprocessing
                IVT_x=AR_unique_df["IVT_x"][AR_unique_df["ID"]==uni_ID]
                IVT_y=AR_unique_df["IVT_y"][AR_unique_df["ID"]==uni_ID]
                IVT=np.sqrt(IVT_x**2+IVT_y**2)
                TIVT=AR_unique_df["TIVT"][AR_unique_df["ID"]==uni_ID]
                
                Length=AR_unique_df["length"][AR_unique_df["ID"]==uni_ID]
                Width=AR_unique_df["width"][AR_unique_df["ID"]==uni_ID]
                Aspect_Ratio=Length/Width
                
                Norm_Age=AR_unique_df["Norm_Age"][AR_unique_df["ID"]==uni_ID]
                x_number=np.repeat(i+1,IVT.shape[0])
                m1=ax1.scatter(x_number,
                            IVT,s=150,marker=markers[i],
                            color=mark_cls[i],edgecolor="k",
                            label=str([uni_ID]))
                marker_entries.append(m1)
                ax3.scatter(x_number,
                            Norm_Age,s=150,marker=markers[i],
                            color=mark_cls[i],edgecolor="k",label=str([uni_ID]))
                
                ax2.scatter(x_number,
                            Length/1000,s=150,marker=markers[i],
                            color=mark_cls[i],edgecolor="k",label=str([uni_ID]))
                ax4.scatter(x_number,
                            Width/1000,s=150,marker=markers[i],
                            color=mark_cls[i],edgecolor="k",label=str([uni_ID]))
                
                i+=1        
            
            ax1.set_xticks(np.linspace(1,6,6))
            ax1.set_ylim([100,600])
            ax2.set_ylim([2000,10000])
            ax4.set_ylim([200,1200])
            ax3.set_ylim([0,1.0])
            ax1.set_xlim([.9,6.1])
            ax2.set_xlim([.9,6.1])
            ax3.set_xlim([.9,6.1])
            ax4.set_xlim([.9,6.1])
            
            for axis in ["left","bottom"]:
                ax1.spines[axis].set_linewidth(2)
                ax2.spines[axis].set_linewidth(2)
                ax3.spines[axis].set_linewidth(2)
                ax4.spines[axis].set_linewidth(2)
            
            ax1.tick_params(length=10,width=3)
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
            ax3.set_xticklabels(["IOP1","IOP3","IOP5",
                                 "IOP9","IOP10","IOP12"])
            ax4.set_xticklabels(["IOP1","IOP3","IOP5",
                                 "IOP9","IOP10","IOP12"])
            
            ax2.tick_params(length=10,width=3)
            ax3.tick_params(length=10,width=3)
            ax4.tick_params(length=10,width=3)
            ax1.set_ylabel("AR-$\overline{\mathrm{IVT}}$ \
                           (kg$\mathrm{m}^{-1}\mathrm{s}^{-1})$")
            #ax2.set_ylabel("TIVT (1e8 kg$\mathrm{s}^{-1})$")
            ax2.set_ylabel("AR-Length (km)")
            
            ax4.set_ylabel("AR-Width (km)")
            ax3.set_ylabel("AR-Normalized Age")
            sns.despine(offset=10)
            plt.subplots_adjust(hspace=0.2)
            plt.subplots_adjust(wspace=0.4)
            AR_stat_fig.legend(marker_entries,
                               labels=["Ian","Vladiana","Walpurga",
                                       "Sanchez","Thor","TPV"],
                               ncol=6,loc="lower center",
                               facecolor="lightgrey",edgecolor="black",
                               bbox_to_anchor=(0.0,-0.005,0.9,1.0),
                               title="NAWDEX-AR Associated Synoptic Systems:")
            fplot_name="AR_NAWDEX_Major_Characteristics.png"
        elif only_IVT:
            ax1=AR_stat_fig.add_subplot(111)
            i=0
            
                
            marker_entries=[]
            for uni_ID in unique_ID:
                
                #IVT preprocessing
                IVT_x=AR_unique_df["IVT_x"][AR_unique_df["ID"]==uni_ID]
                IVT_y=AR_unique_df["IVT_y"][AR_unique_df["ID"]==uni_ID]
                IVT=np.sqrt(IVT_x**2+IVT_y**2)
                x_number=np.repeat(i+1,IVT.shape[0])
            
                m1=ax1.scatter(x_number,
                            IVT,s=250,marker=markers[i],
                            color=mark_cls[i],edgecolor="k",
                            label=str([uni_ID]))
                marker_entries.append(m1)
                i+=1        
            
            ax1.set_xticks(np.linspace(1,6,6))
            ax1.set_ylim([100,600])
            ax1.set_xlim([.9,6.1])
            
            for axis in ["left","bottom"]:
                ax1.spines[axis].set_linewidth(2)
            
            ax1.tick_params(length=10,width=3)
            ax1.set_xticklabels(["IOP1","IOP3","IOP5",
                                 "IOP9","IOP10","IOP12"])
            
            ax1.set_ylabel("AR-$\overline{\mathrm{IVT}}$ \
                           (kg$\mathrm{m}^{-1}\mathrm{s}^{-1})$")
            sns.despine(offset=10)
            AR_stat_fig.legend(marker_entries,
                               labels=["Ian","Vladiana","Walpurga",
                                       "Sanchez","Thor","TPV"],
                               ncol=3,loc="lower center",
                               facecolor="lightgrey",edgecolor="black",
                               #bbox_to_anchor=(0.5,1.0,0,-0.3),
                               title="AR Associated Synoptic Systems (NAWDEX):",
                               fontsize=18,title_fontsize=18)
            plt.subplots_adjust(bottom=0.22)
            fplot_name="AR_NAWDEX_IVT_Characteristics.png"
        plot_path=config_file["Data_Paths"]['campaign_path']+\
                    config_file["Data_Paths"]["campaign"]+"/plots/"
        AR_stat_fig.savefig(plot_path+fplot_name,
                                dpi=300,bbox_inches="tight")
    
    @staticmethod
    def look_up_synthetic_AR_cross_sections(campaign,invert_flight=False):
        if campaign=="NAWDEX":
            ARs={}
            ARs["RF10"]={}
            #ARs["RF10"]["AR1"]={"start":"2016-10-13 14:30",
            #                    "end":"2016-10-13 14:35"}
            ARs["RF10"]["AR1"]={"start":"2016-10-13 14:00",
                                "end":"2016-10-13 15:40"}
            
            ARs["RF10"]["AR2"]={"start":"2016-10-13 17:30",
                                "end":"2016-10-13 18:30"}
            ARs["RF10"]["AR3"]={"start":"2016-10-13 12:10",
                                "end":"2016-10-13 13:59"}
        return ARs
    def look_up_AR_cross_sections(campaign,invert_flight=False):
        if campaign=="NAWDEX":
            ARs={}
            ARs["RF01"]={}
            ARs["RF02"]={}
            ARs["RF03"]={}
            ARs["RF04"]={}
            ARs["RF05"]={}
            ARs["RF08"]={}
            ARs["RF09"]={}
            ARs["RF10"]={}
            ARs["RF12"]={}
            
            #ARs["RF01"]["AR1"]={"start":"2016-09-17 11:05",
            #                    "end":"2016-09-17 12:25"}
            #ARs["RF02"]["AR1"]={"start":"2016-09-21 16:30",
            #                    "end":"2016-09-21 17:30"}
            #ARs["RF02"]["AR2"]={"start":"2016-09-21 17:35",
            #                    "end":"2016-09-21 18:30"}
            ARs["RF03"]["AR1"]={"start":"2016-09-23 11:00",
                                "end":"2016-09-23 12:15"}
            ARs["RF03"]["AR2"]={"start":"2016-09-23 12:20",
                                "end":"2016-09-23 13:05"}
            ARs["RF03"]["AR3"]={"start":"2016-09-23 14:00", # Default 14:00
                                "end":"2016-09-23 15:00"}   # Defaul 15:00
            ##ARs["RF03"]["AR22"]={"start":"2016-09-23 13:05",
            ##                    "end":"2016-09-23 13:25"}
            ##ARs["RF03"]["AR32"]={"start":"2016-09-23 13:45",
            ##                    "end":"2016-09-23 14:15"}
            #ARs["RF04"]["AR1"]={"start":"2016-09-26 11:35",
            #                    "end":"2016-09-26 12:10"}
            #ARs["RF04"]["AR2"]={"start":"2016-09-26 13:00",
            #                    "end":"2016-09-26 13:55"}
            #ARs["RF04"]["AR3"]={"start":"2016-09-26 15:30",
            #                    "end":"2016-09-26 16:45"}
            ##ARs["RF04"]["AR31"]={"start":"2016-09-26 16:50",
            ##                    "end":"2016-09-26 17:25"}
            #ARs["RF04"]["AR4"]={"start":"2016-09-26 17:15",
            #                    "end":"2016-09-26 18:30"}
            #ARs["RF05"]["AR1"]={"start":"2016-09-27 13:50",
            #                    "end":"2016-09-27 15:15"}
            #ARs["RF05"]["AR2"]={"start":"2016-09-27 17:00",
            #                    "end":"2016-09-27 17:45"}
            ARs["RF05"]["AR3"]={"start":"2016-09-27 18:00",
                                "end":"2016-09-27 19:00"}
            ##ARs["RF05"]["AR99"]={"start":"2016-09-27 14:00",
            ##                    "end":"2016-09-27 14:10"}
            # Correct
            #ARs["RF08"]["AR1"]={"start":"2016-10-09 16:00",
            #                    "end":"2016-10-09 17:00"}
            
            #ARs["RF09"]["AR1"]={"start":"2016-10-10 14:00",
            #                    "end":"2016-10-10 14:50"}
            #ARs["RF09"]["AR2"]={"start":"2016-10-10 14:55",
            #                    "end":"2016-10-10 15:45"}
            ARs["RF10"]["AR1"]={"start":"2016-10-13 09:15",
                                "end":"2016-10-13 10:15"}
            ARs["RF10"]["AR2"]={"start":"2016-10-13 10:45",
                                "end":"2016-10-13 11:15"}
            ##Default
            ARs["RF10"]["AR3"]={"start":"2016-10-13 12:10",
                                "end":"2016-10-13 13:50"}
            ARs["RF10"]["AR99"]={"start":"2016-10-13 12:05",
                                 "end":"2016-10-13 12:50"}
            ##ARs["RF10"]["AR31"]={"start":"2016-10-13 12:10",
            ##                    "end":"2016-10-13 12:20"}
            #ARs["RF12"]["AR1"]={"start":"2016-10-15 09:45",
            #                    "end":"2016-10-15 10:45"}
            #ARs["RF12"]["AR2"]={"start":"2016-10-15 14:45",
            #                    "end":"2016-10-15 15:50"}     
        
        elif campaign=="HALO_AC3_Dry_Run":
            ARs["RF01"]={}
            ARs["RF02"]={}
            ARs["RF03"]={}
            ARs["RF04"]={}
            
            ARs["RF04"]["AR1"]={"start":"2021-03-26 10:00",
                                "end":"2021-03-26 10:25"}
            ARs["RF04"]["AR2"]={"start":"2021-03-26 11:13",
                                "end":"2021-03-26 11:43"}
            ARs["RF04"]["AR3"]={"start":"2021-03-26 12:29",
                                "end":"2021-03-26 13:04"}
            ARs["RF04"]["AR4"]={"start":"2021-03-26 13:55",
                                "end":"2021-03-26 14:30"}
            
            if invert_flight:
            
                ARs["RF01"]={}
                ARs["RF02"]={}
                ARs["RF03"]={}
                ARs["RF04"]={}
                
                ARs["RF04"]["AR1"]={"start":"2021-03-26 07:49",
                                    "end":"2021-03-26 08:26"}
                ARs["RF04"]["AR2"]={"start":"2021-03-26 09:18",
                                    "end":"2021-03-26 09:53"}
                ARs["RF04"]["AR3"]={"start":"2021-03-26 10:38",
                                    "end":"2021-03-26 11:11"}
                ARs["RF04"]["AR4"]={"start":"2021-03-26 11:55",
                                    "end":"2021-03-26 12:22"}

        elif campaign=="HALO_AC3":
            ARs={}
            ARs["RF02"]={}
            ARs["RF03"]={}
            ARs["RF04"]={}
            ARs["RF05"]={}
            ARs["RF06"]={}
            ARs["RF07"]={}
            ARs["RF08"]={}
            ARs["RF16"]={}
            ARs["RF05"]["AR_entire"]={"start":"2022-03-15 09:30",
                                      "end":"2022-03-15 17:15"}
            ARs["RF06"]["AR_entire"]={"start":"2022-03-16 09:30",
                                      "end":"2022-03-16 17:45"}
            ARs["RF02"]["AR1"]={"start":"2022-03-12 10:25",
                                "end":"2022-03-12 12:10"}
            ARs["RF02"]["AR2"]={"start":"2022-03-12 11:28",
                                "end":"2022-03-12 13:32"}
            ARs["RF03"]["AR1"]={"start":"2022-03-13 10:00", # temporary
                                "end":"2022-03-13 11:45"}   # temporary
            ARs["RF04"]["AR1"]={"start":"2022-03-14 16:00",
                                "end":"2022-03-14 16:45"}
            ARs["RF05"]["AR_entire"]={"start":"2022-03-15 10:11",
                                "end":"2022-03-15 13:15"}
            ARs["RF05"]["AR1"]={"start":"2022-03-15 10:11",
                                "end":"2022-03-15 11:05"}
            ARs["RF05"]["AR2"]={"start":"2022-03-15 11:15",
                                "end":"2022-03-15 12:15"}
            ARs["RF05"]["AR3"]={"start":"2022-03-15 12:15",
                                "end":"2022-03-15 13:10"}
            ARs["RF05"]["AR4"]={"start":"2022-03-15 14:20",
                                "end":"2022-03-15 15:20"}
            ARs["RF16"]["AR1"]={"start":"2022-04-10 10:30",
                                "end":"2022-04-10 12:22"}
            ARs["RF16"]["AR2"]={"start":"2022-04-10 11:45",
                                "end":"2022-04-10 13:45"}
            ARs["RF06"]["AR1"]={"start":"2022-03-16 10:45",
                                "end":"2022-03-16 12:55"}
            ARs["RF06"]["AR2"]={"start":"2022-03-16 12:12",
                                "end":"2022-03-16 14:18"}
            ARs["RF07"]["AR1"]={"start":"2022-03-20 15:22",
                                "end":"2022-03-20 16:24"}
            ARs["RF08"]["AR1"]={"start":"2022-03-21 09:20",
                                "end":"2022-03-21 10:25"}
        return ARs
    
    def locate_AR_cross_section_sectors(HALO_Dict,Hydrometeors,analysed_flight):
        # Get inflow and outflow
        inflow_flight_df  = HALO_Dict[analysed_flight]["inflow"]
        outflow_flight_df = HALO_Dict[analysed_flight]["outflow"]
    
        hmp_inflow  = Hydrometeors[analysed_flight]["AR_internal"].loc[\
                                                        inflow_flight_df.index]
        hmp_outflow = Hydrometeors[analysed_flight]["AR_internal"].loc[\
                                                        outflow_flight_df.index]
        
        # CARRA and ERA have different Interp_IVT names. This has to be defined here
        grid_name=Hydrometeors[analysed_flight]["AR_internal"].name
        if grid_name=="ERA5":
            
            ivt_var_arg="Interp_IVT"
        elif grid_name=="CARRA":
            ivt_var_arg="highres_Interp_IVT"
        else:
            raise Exception("Something went wrong in the grid naming")
        # Adapt the IVT max distance for both flow cross-sections
        hmp_inflow["IVT_max_distance"]=hmp_inflow["IVT_max_distance"]-hmp_inflow[\
                                        "IVT_max_distance"].iloc[\
                                            hmp_inflow[ivt_var_arg].argmax()]
        hmp_outflow["IVT_max_distance"]=hmp_outflow["IVT_max_distance"]-hmp_outflow[\
                                        "IVT_max_distance"].iloc[
                                            hmp_outflow[ivt_var_arg].argmax()]
        # Following Cobb et al. (2020), 
        # now declare the sectors (AR cold sector, warm sector, AR core)
        inflow_core=hmp_inflow.loc[\
                    hmp_inflow[ivt_var_arg]>0.8*hmp_inflow[ivt_var_arg].max()]
        outflow_core=hmp_outflow.loc[\
                    hmp_outflow[ivt_var_arg]>0.8*hmp_outflow[ivt_var_arg].max()]
        
        ar_inflow=hmp_inflow.loc[\
                    hmp_inflow[ivt_var_arg]>0.33*hmp_inflow[ivt_var_arg].max()]
        ar_outflow=hmp_outflow.loc[\
                    hmp_outflow[ivt_var_arg]>0.33*hmp_outflow[ivt_var_arg].max()]
    
        ar_inflow=ar_inflow[ar_inflow[ivt_var_arg]>100]
        ar_outflow=ar_outflow[ar_outflow[ivt_var_arg]>100]
        
        # Warm and Cold Sector Specification, 
        # is here possible as we are flying all the time E-W heading
        ar_inflow_warm_sector=ar_inflow.loc[ar_inflow.index[0]:\
                                            inflow_core.index[0]]
        ar_inflow_cold_sector=ar_inflow.loc[inflow_core.index[-1]:\
                                            ar_inflow.index[-1]]
        ar_outflow_warm_sector=ar_outflow.loc[ar_outflow.index[0]:\
                                              outflow_core.index[0]]
        ar_outflow_cold_sector=ar_outflow.loc[outflow_core.index[-1]:\
                                              ar_outflow.index[-1]]
        AR_inflow_dict={}
        AR_outflow_dict={}
        AR_inflow_dict["entire_inflow"]=hmp_inflow
        AR_inflow_dict["AR_inflow"]=ar_inflow
        AR_inflow_dict["AR_inflow_core"]=inflow_core
        AR_inflow_dict["AR_inflow_warm_sector"]=ar_inflow_warm_sector
        AR_inflow_dict["AR_inflow_cold_sector"]=ar_inflow_cold_sector
        AR_outflow_dict["entire_outflow"]=hmp_outflow        
        AR_outflow_dict["AR_outflow"]=ar_outflow
        AR_outflow_dict["AR_outflow_core"]=outflow_core
        AR_outflow_dict["AR_outflow_warm_sector"]=ar_outflow_warm_sector
        AR_outflow_dict["AR_outflow_cold_sector"]=ar_outflow_cold_sector
        
        return AR_inflow_dict,AR_outflow_dict
    
    def calc_TIVT_of_cross_sections_in_AR_sector(AR_inflow,AR_outflow,grid_name):
        if grid_name=="ERA5":
            ivt_var_arg="Interp_IVT"
        elif grid_name=="CARRA":
            ivt_var_arg="highres_Interp_IVT"
        else:
            print("Others also like ICON-2km are not yet included.")
        # Adapt the IVT max distance for both flow cross-sections
        AR_inflow["IVT_max_distance"]=AR_inflow["IVT_max_distance"]-AR_inflow[\
                                        "IVT_max_distance"].iloc[\
                                            AR_inflow[ivt_var_arg].argmax()]
        AR_outflow["IVT_max_distance"]=AR_outflow["IVT_max_distance"]-AR_outflow[\
                                        "IVT_max_distance"].iloc[
                                            AR_outflow[ivt_var_arg].argmax()]
        TIVT={}
        # Total AR
        TIVT["inflow"]=(AR_inflow[ivt_var_arg].rolling(2).mean()*\
                           AR_inflow["IVT_max_distance"].diff()).sum()
        TIVT["outflow"]=(AR_outflow[ivt_var_arg].rolling(2).mean()*\
                            AR_outflow["IVT_max_distance"].diff()).sum()
        return TIVT
    def calc_TIVT_of_sectors(AR_inflow_dict,AR_outflow_dict,grid_name):
        # Calc TIVT
        if grid_name=="ERA5":
            ivt_var_arg="Interp_IVT"
        elif grid_name=="CARRA":
            ivt_var_arg="highres_Interp_IVT"
        else:
            print("Others also like ICON-2km are not yet included.")
        
        ar_inflow=AR_inflow_dict["AR_inflow"]
        ar_outflow=AR_outflow_dict["AR_outflow"]
        
        inflow_core=AR_inflow_dict["AR_inflow_core"]
        outflow_core=AR_outflow_dict["AR_outflow_core"]
        
        # Total AR
        tivt_inflow_total=(ar_inflow[ivt_var_arg].rolling(2).mean()*\
                           ar_inflow["IVT_max_distance"].diff()).sum()
        tivt_outflow_total=(ar_outflow[ivt_var_arg].rolling(2).mean()*\
                            ar_outflow["IVT_max_distance"].diff()).sum()
        # AR core
        tivt_inflow_core=(inflow_core[ivt_var_arg].rolling(2).mean()*\
                          inflow_core["IVT_max_distance"].diff()).sum()
        tivt_outflow_core=(outflow_core[ivt_var_arg].rolling(2).mean()*\
                           outflow_core["IVT_max_distance"].diff()).sum()
        
        TIVT_inflow_dict={}
        TIVT_inflow_dict["total"] = tivt_inflow_total
        TIVT_inflow_dict["core"]  = tivt_inflow_core
        
        TIVT_outflow_dict={}
        TIVT_outflow_dict["total"] = tivt_outflow_total
        TIVT_outflow_dict["core"]  = tivt_outflow_core
        
        return TIVT_inflow_dict,TIVT_outflow_dict 
        
        
def main():
    import Flight_Campaign
    import data_config
    
    plot_AR_statistics=True
    
    #pass
    AR=Atmospheric_Rivers("ERA")
    AR_era_ds=AR.open_AR_catalogue()
    
    
    config_name="data_config_file"
    path=os.getcwd()
    campaign_name="North_Atlantic_February_Run"    
    
    # Check if config-File exists and if not create the relevant first one
    if data_config.check_if_config_file_exists(config_name):
        config_file=data_config.load_config_file(path,config_name)
    else:
        data_config.create_new_config_file(file_name=config_name+".ini")
        
    if sys.platform.startswith("win"):
        system_is_windows=True
    else:
        system_is_windows=False
        
    if system_is_windows:
        if not config_file["Data_Paths"]["system"]=="windows":
            windows_paths={
                "system":"windows",
                "campaign_path":os.getcwd()+"/"#+config_file_object["Data_Paths"]["campaign"]+"/"    
                    }
            windows_paths["save_path"]=windows_paths["campaign_path"]+"Save_path/"
            data_config.add_entries_to_config_object(config_name,windows_paths)
    
    if campaign_name=="NAWDEX":     
        is_flight_campaign=True
        #campaign_path='/scratch/uni/u237/users/hdorff/'    
        nawdex=Flight_Campaign.NAWDEX(is_flight_campaign=True,
                      major_path=config_file["Data_Paths"]["campaign_path"],
                      aircraft="HALO",instruments=["radar","radiometer","sonde"])
            
        AR_nawdex_cross_sections=AR.get_HALO_NAWDEX_AR_cross_sections()
        AR_unique_df            =AR.get_ids_of_ARs_from_NAWDEX_HALO_cross_sections(
                                                    AR_era_ds,nawdex,config_file,
                                                    AR_nawdex_cross_sections)
    elif campaign_name=="North_Atlantic_February_Run":
        flights=["SRF02","SRF03","SRF04","SRF05"]
        na_run=Flight_Campaign.North_Atlantic_February_Run(
                                    is_flight_campaign=True,
                                    major_path=config_file["Data_Paths"]\
                                                ["campaign_path"],aircraft="HALO",
                                    interested_flights=flights,
                                    instruments=["radar","radiometer","sonde"])
        na_run.specify_flights_of_interest(flights[0])
        na_run.create_directory(directory_types=["data"])
        AR_unique_df= AR.get_ids_of_ARs_from_HALO_cross_sections(
                                    AR_era_ds,na_run,config_file,AR_cross_sections,
                                    single_flight="",single_ARs="")
    else:
        raise Exception("No campaign with this name ",
                        campaign_name," is included")         
    AR_unique_df.dropna(subset=["Cross_Start"],inplace=True)
    fpath=config_file["Data_Paths"]["data"]
    fname=campaign_name+"_AR_Catalogue.csv"
    AR_unique_df.to_csv(path_or_buf=fpath+fname,index=True)
    print(campaign_name," AR Catalogue saved as: ",fpath+fname)
    print("ARs overpassed:", AR_unique_df.describe())
    if plot_AR_statistics:
        AR.plot_campaign_AR_statistics(AR_unique_df,config_file,only_IVT=True)

if __name__=="__main__":
    main()

#     flight_day="1996-12-31"
#     AR_era_data={}
#     AR_era_data["time"]=pd.Series(AR_era_ds["time"])
#     AR_era_data["model_runs"]=pd.DatetimeIndex(AR_era_data["time"]).get_loc(flight_day)
#     AR_era_data["shape"]=AR_era_ds["shape"][:]
#     AR_era_data["kidmap"]=AR_era_ds["kidmap"][:]
#     AR_era_data["numobj"]=AR_era_ds["numobj"]
#     AR_era_data["width"]=AR_era_ds["width"]
#     AR_era_data["lat"]=np.array(AR_era_ds["lat"])
#     AR_era_data["lon"]=np.array(AR_era_ds["lon"])
    
#     ar_case={}
#     #ar_case["shape"]=pd.DataFrame(data=np.array(AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:]))
#     ar_case["kidmap"]=pd.DataFrame(data=np.array(AR_era_ds.kidmap[0,AR_era_data["model_runs"].start+2,0,:,:]))
#     #ar_case["kcnt"]=pd.DataFrame(data=np.array(AR_era_ds.kcnt[0,AR_era_data["model_runs"].start+2,0,:,:]))
#     AR_era_data["numobj"]
#     #ar_case["width"]=pd.DataFrame(data=np.array(AR_era_ds.width[0,AR_era_data["model_runs"].start+2,0,:,:]))
#     plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
#                  AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
#                  hatches=['-', 'x','/', '\\', '//'],cmap='bone_r',alpha=0.3)
# #sys.exit()
    # import Flight_Campaign
    # import cartopy.crs as ccrs
    #  # Map plotting
    # from Cloudnet import Cloudnet_Data
            
    # # get station_locations
    # campaign_name="NAWDEX"
    # # Data specifications
    # flights=["RF03"]
    # main_flights=flights#["RF01","RF03","RF04","RF05","RF10"]
    # # Load config file
    # config_file=data_config.load_config_file(os.getcwd(),
    #                                          "data_config_file")
    # if campaign_name=="NAWDEX":
    #     # call the campaign class which is per default
    #     nawdex=Flight_Campaign.NAWDEX(is_flight_campaign=True,
    #                                   major_path=config_file["Data_Paths"]["campaign_path"],
    #                                   aircraft="HALO",instruments=["radar","radiometer","sonde"])
    # campaign_cloudnet=Cloudnet_Data(nawdex.campaign_path)
    # station_coords=campaign_cloudnet.get_cloudnet_station_coordinates(nawdex.campaign_path)
    
    # ar_map=plt.figure(figsize=(12,12))
    # ax = plt.axes(projection=ccrs.AzimuthalEquidistant(central_longitude=-25.0,
    #                                                            central_latitude=55))
    
    # ax.coastlines(resolution="50m")
    # ax.set_extent([-70,10,20,90])
    # ax.gridlines()
    # print(station_coords)
            
    #         #Summit
    # ax.text(station_coords["Summit"]["Lon"]-8,
    #         station_coords["Summit"]["Lat"]-1.5, 'Summit\n(Greenland)',
    #         horizontalalignment='left', transform=ccrs.Geodetic(),
    #         fontsize=10,color="blue")
    # plt.scatter(station_coords["Summit"]["Lon"],
    #             station_coords["Summit"]["Lat"],
    #             marker='x', transform=ccrs.PlateCarree(),color="blue")
    
    # plt.contourf(AR_era_ds.lon,AR_era_ds.lat,
    #              AR_era_ds.shape[0,AR_era_data["model_runs"].start+2,0,:,:],
    #              hatches=['-', 'x','/', '\\', '//'],cmap='bone_r',alpha=0.3,
    #              transform=ccrs.PlateCarree())
    # sys.exit()

