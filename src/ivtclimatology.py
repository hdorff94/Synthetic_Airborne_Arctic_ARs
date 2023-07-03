# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 08:03:50 2021

@author: u300737
"""
import os
import sys
import Performance 

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
sys.path.insert(1,os.getcwd()+"/../config/")
import flightcampaign
import data_config

def is_mar_or_apr(month):
    return (month >=3) & (month <= 4)
def is_sep_or_oct(month):
    return (month >= 9) & (month <= 10)

def season_selection(season=""):
    if season=="autumn":
        season_function=is_sep_or_oct
    elif season=="spring":
        season_function=is_mar_or_apr
    return season_function
def get_centered_halo_lat(campaign_name,flight):
    #if not "glob" in sys.modules:
    import glob
    base_data_path=os.getcwd()+"/../../../Work/GIT_Repository/"
    rf_path=base_data_path+campaign_name+"/data/"+"/aircraft_position/"
    flight_track_file=glob.glob(rf_path+flight+"*")[0]
    flight_track_df=pd.read_csv(flight_track_file)
    centered_lat=flight_track_df["latitude"].mean()
    return centered_lat
    
#def process_ERA5_seasonal_AR_clim()
def process_seasonal_AR_climatology(cfg_file,cmpgn_cls,AR_cls,AR_era_ds,
                                    season="spring",
                                    lower_lat=55,upper_lat=90,
                                    western_lon=-60,eastern_lon=40,
                                    add_single_flight="RF10"):
    """
    

    Parameters
    ----------
    cfg_file : config file
        config file giving specifications to use.
    cmpgn_cls : class
        class of Flight Campaign to evaluate.
    AR_cls : class
        class of ARs based on Guan & Waliser AR catalogue.
    AR_era_ds : xr.Dataset
        Dataset of ARs detected and quantified by Guan & Waliser (2019).
    season : str, optional
        Specification of season to handle. This argument will call another 
        module to load the respective seasonal data. The default is "spring".
    lower_lat : str, optional
        Lower lat boundary to consider. The default is 50.
    upper_lat : str, optional
        Upper lat boundary to consider. The default is 90.
    western_lon : str, optional
        Western longitude boundary to consider. The default is -60.
    eastern_lon : str, optional
        Eastern longitude boundary to consider. The default is 40.

    Returns
    -------
    regional_ARs : pd.DataFrame
        dataset containing the ARs for the region desired
    AR_campaign_df : pd.DataFrame
        concrete dataset of AR cases from campaign
    """
    # Check if config-File exists and if not create the relevant first one
    import gridonhalo
    AR_unique_df=pd.DataFrame()
    #-------------------------------------------------------------------------#
    # Geolocate
    # Cut latitude and season periods
    season_func=season_selection(season=season)
    season_AR_era_ds_all= AR_era_ds.sel(time=season_func(
                                        AR_era_ds['time.month']))
    season_AR_era_ds= season_AR_era_ds_all.sel(lat=slice(lower_lat,upper_lat))

    # Shift lon coordinates
    era_lon=pd.Series(np.array(season_AR_era_ds["lon"]))
    shifted_lon=era_lon.copy()
    shifted_lon[shifted_lon>180]=shifted_lon[shifted_lon>180]-360
    #-------------------------------------------------------------------------#
    # Allocation for loop
    values_list=[]
    performance=Performance.performance()
    # Loop over timesteps
    for t in range(season_AR_era_ds.time.shape[0]):
        #Add land mask to ignore ARs that are purely over land
        AR_id_field=pd.DataFrame(np.array(season_AR_era_ds["kidmap"][0,t,0,:,:])*\
                             np.invert(np.array(
                                 season_AR_era_ds["islnd"][:]).astype(bool))*\
                             np.invert(np.array(
                                 season_AR_era_ds["iscst"][:]).astype(bool)),
                             index=season_AR_era_ds["lat"],
                             columns=shifted_lon)              
    
        AR_id_field=AR_id_field.sort_index(axis=1)
        AR_id_field=AR_id_field.replace(to_replace=0,value=np.nan)
        
        #cut AR_id field to zonal region
        AR_id_field_cut=AR_id_field.loc[:,western_lon:eastern_lon]
        
        #get AR-ids inside this map region
        kid_values   = np.unique(AR_id_field_cut)
        if kid_values[~np.isnan(kid_values)].shape[0]>0:
            kid_values   = kid_values[~np.isnan(kid_values)]
            # for given timestep get list of all AR-ids from AR ds
            AR_kid      = pd.Series(np.array(season_AR_era_ds_all["kid"]\
                            [0,t,0,:]))
        
            time=pd.to_datetime(np.array(season_AR_era_ds.time[t]))
        
            AR_ivtx=pd.Series(np.array(season_AR_era_ds_all["ivtx"][0,t,0,:]))
            AR_ivty=pd.Series(np.array(season_AR_era_ds_all["ivty"][0,t,0,:]))
            AR_clon=pd.Series(np.array(season_AR_era_ds_all["clon"][0,t,0,:]))
            AR_clat=pd.Series(np.array(season_AR_era_ds_all["clat"][0,t,0,:]))
        
            for kid in kid_values:
                # Get index    
                kid_idx         = AR_kid[AR_kid==float(kid)].index.tolist()    
                values_dict={}
                values_dict["kid"]=float(AR_kid[kid_idx])
                values_dict["time"]=time
                values_dict["ivt_x"]=float(AR_ivtx.iloc[kid_idx])
                values_dict["ivt_y"]=float(AR_ivty.iloc[kid_idx])
                values_dict["clon"]=float(AR_clon.iloc[kid_idx])
                values_dict["clat"]=float(AR_clat.iloc[kid_idx])
                values_list.append(values_dict)
        performance.updt(season_AR_era_ds.time.shape[0],t)
        
    # Assign all quantities of listed regional ARs to dataframe
    regional_ARs=pd.DataFrame(data=values_list,columns=["kid","time",
                                                          "ivt_x","ivt_y",
                                                          "clat","clon"])

    regional_ARs["ivt"]=np.sqrt(regional_ARs["ivt_x"]**2+\
                                regional_ARs["ivt_y"]**2)
    if lower_lat>30:
        if (western_lon > -90) and (eastern_lon<=90):
            ar_file_name="North_Atlantic_ARs_Longterm"
    
    if season=="autumn":
        AR_nawdex_cross_sections=AR_cls.get_HALO_NAWDEX_AR_cross_sections()
        AR_campaign_df=AR_cls.get_ids_of_ARs_from_HALO_cross_sections(
                                    AR_era_ds,cmpgn_cls,cfg_file,
                                    AR_nawdex_cross_sections,
                                    single_flight="RF10")
    elif season=="spring":
        print("Currently no AR locator for spring is defined",
              " but will be added.")
        # If the NAWDEX case is included
        #AR_nawdex_cross_sections=AR_cls.get_HALO_NAWDEX_AR_cross_sections()
        #AR_campaign_cross_sections=Grid_on_HALO.cut_halo_to_AR_crossing(
        #                        "SAR_internal",campaign=cmpgn_cls[0],
        #                        device="halo")    
        if add_single_flight=="RF10":
            NAWDEX=flightcampaign.NAWDEX(is_flight_campaign=True,
                major_path=cfg_file["Data_Paths"]["campaign_path"],
                aircraft="HALO",instruments=["radar","radiometer","sonde"])
            
            #AR_nawdex_cross_sections=AR_cls.get_HALO_NAWDEX_AR_cross_sections()
            AR_nawdex_df            =AR_cls.get_ids_of_ARs_from_HALO_cross_sections(
                                                    AR_era_ds,NAWDEX,cfg_file,
                                                    single_flight="RF10")
        
        AR_campaign_df=AR_cls.get_ids_of_ARs_from_HALO_cross_sections(
                                         AR_era_ds,cmpgn_cls,cfg_file)
        if "AR_nawdex_df" in locals().keys():
            AR_campaign_df=pd.concat([AR_campaign_df,AR_nawdex_df])

    ar_file_name=ar_file_name+"_"+season+".csv"
    
    regional_ARs.to_csv(path_or_buf=cmpgn_cls.plot_path+ar_file_name)
    print("North Atlantic AR statistics saved as:",
          cmpgn_cls.plot_path+ar_file_name)
    return regional_ARs,AR_campaign_df

def plot_AR_IVT_climatology(north_atlantic_ARs,plot_path,
                            AR_unique_df=pd.DataFrame(),
                            season=" "):
    
    import seaborn as sns
    snsplot=sns.jointplot(data=north_atlantic_ARs,x="ivt",y="clat",
                      s=5,color="mediumseagreen",space=1.0)
    snsplot.plot_joint(sns.kdeplot, color="forestgreen", zorder=1, levels=3)
    snsplot.ax_marg_x.set_xticklabels("")
    snsplot.ax_marg_y.set_yticklabels("")
    snsplot.ax_joint.set_xlabel("$\overline{IVT}$ ($\mathrm{kgms}^{-1})$")
    if season=="autumn":
        snsplot.ax_joint.scatter(np.sqrt(AR_unique_df["IVT_x"]**2+\
                                     AR_unique_df["IVT_y"]**2),
                             AR_unique_df["clat"],color="red",marker='s',
                             s=20,edgecolor="k",label="NAWDEX_AR")
    snsplot.ax_joint.set_ylabel("AR Centre Latitude in $^{\circ}$N")
    snsplot.ax_joint.set_ylim([40,90])
    snsplot.ax_joint.set_xlim([100,800])
    snsplot.ax_joint.legend(loc="best")
    sns.despine(offset=2)
    if season=="autumn":
        fig_title="North Atlantic ARs (Sep/Oct, 1979-2018)"
    elif season=="spring":
        fig_title="North Atlantic ARs (Mar/Apr, 1979-2018)"
    snsplot.fig.suptitle(fig_title,y=1.0)
    
    fig_name=plot_path+"Climatology_North_Atlantic_ARs_"+season+".pdf"
    snsplot.savefig(fig_name,dpi=200,bbox_inches="tight")
    print("Figure saved as:",fig_name)
    return None

###############################################################################
"""
        MAIN FUNCTION to run
"""
###############################################################################
def plot_single_season_characteristics(season="autumn",take_both_campaigns="True"):
    """
    

    Parameters
    ----------
    season : str, optional
        Season from which the statistics should be calculated. 
        The default is "autumn".
    
    take_both_campaigns : str, optional
        specifies either one synthetic campaign like North_Atlantic_Run or
        Second_Synthetic_Run, or both which is given by True. The default is "True".

    Returns
    -------
    None.

    """
    config_name="data_config_file"
    path=os.getcwd()
    if season=="autumn":
        campaign_name="NAWDEX"    
    elif season=="spring":
        if take_both_campaigns=="True":
            campaign_name=["North_Atlantic_February_Run",
                           "Second_Synthetic_Study"]
        else:
            campaign_name=take_both_campaigns
    
    # Check if config-File exists and if not create the relevant first one
    if data_config.check_if_config_file_exists(config_name):
            config_file=data_config.load_config_file(path,config_name)    
            
    plot_AR_statistics=True
    
    import atmospheric_rivers as AR
    AR=AR.Atmospheric_Rivers("ERA")
    AR_era_ds=AR.open_AR_catalogue()
    AR_unique_df=pd.DataFrame()
    # Cut latitude and september october periods
    season_func=season_selection(season=season)
    season_AR_era_ds_all= AR_era_ds.sel(time=season_func(AR_era_ds['time.month']))
    season_AR_era_ds= season_AR_era_ds_all.sel(lat=slice(50,90))
    # Shift lon coordinates
    era_lon=pd.Series(np.array(season_AR_era_ds["lon"]))
    shifted_lon=era_lon.copy()
    shifted_lon[shifted_lon>180]=shifted_lon[shifted_lon>180]-360
    # Loop over timesteps
    values_list=[]
    performance=Performance.performance()
    for t in range(100):#season_AR_era_ds.time.shape[0]):
        #Add land mask to ignore ARs that are purely over land
        AR_id_field=pd.DataFrame(np.array(season_AR_era_ds["kidmap"][0,t,0,:,:])*\
                             np.invert(np.array(
                                 season_AR_era_ds["islnd"][:]).astype(bool))*\
                             np.invert(np.array(
                                 season_AR_era_ds["iscst"][:]).astype(bool)),
                             index=season_AR_era_ds["lat"],
                             columns=shifted_lon)              
    
        AR_id_field=AR_id_field.sort_index(axis=1)
        AR_id_field=AR_id_field.replace(to_replace=0,value=np.nan)
        #cut AR_id field to zonal region
        AR_id_field_cut=AR_id_field.loc[:,-60:40]
        #get AR-ids inside this map region
        kid_values   = np.unique(AR_id_field_cut)
        if kid_values[~np.isnan(kid_values)].shape[0]>0:
            kid_values   = kid_values[~np.isnan(kid_values)]
            # for given timestep get list of all AR-ids from AR ds
            AR_kid      = pd.Series(np.array(season_AR_era_ds_all["kid"]\
                            [0,t,0,:]))
        
            time=pd.to_datetime(np.array(season_AR_era_ds.time[t]))
        
            AR_ivtx=pd.Series(np.array(season_AR_era_ds_all["ivtx"][0,t,0,:]))
            AR_ivty=pd.Series(np.array(season_AR_era_ds_all["ivty"][0,t,0,:]))
            AR_clon=pd.Series(np.array(season_AR_era_ds_all["clon"][0,t,0,:]))
            AR_clat=pd.Series(np.array(season_AR_era_ds_all["clat"][0,t,0,:]))
        
            for kid in kid_values:
                # Get index    
                kid_idx         = AR_kid[AR_kid==float(kid)].index.tolist()    
                values_dict={}
                values_dict["kid"]=float(AR_kid[kid_idx])
                values_dict["time"]=time
                values_dict["ivt_x"]=float(AR_ivtx.iloc[kid_idx])
                values_dict["ivt_y"]=float(AR_ivty.iloc[kid_idx])
                values_dict["clon"]=float(AR_clon.iloc[kid_idx])
                values_dict["clat"]=float(AR_clat.iloc[kid_idx])
                values_list.append(values_dict)
        performance.updt(season_AR_era_ds.time.shape[0],t)
    
    north_atlantic_ARs=pd.DataFrame(data=values_list,columns=["kid","time",
                                                          "ivt_x","ivt_y",
                                                          "clat","clon"])

    north_atlantic_ARs["ivt"]=np.sqrt(north_atlantic_ARs["ivt_x"]**2+\
                                  north_atlantic_ARs["ivt_y"]**2)
    ar_file_name="North_Atlantic_ARs_Longterm_Spring.csv"
    output_path=os.getcwd()
    if season=="autumn":
        nawdex=Flight_Campaign.NAWDEX(is_flight_campaign=True,
                      major_path=config_file["Data_Paths"]["campaign_path"],
                      aircraft="HALO",instruments=["radar","radiometer","sonde"])    
        AR_nawdex_cross_sections=AR.get_HALO_NAWDEX_AR_cross_sections()
        AR_unique_df            =AR.get_ids_of_ARs_from_NAWDEX_HALO_cross_sections(
                                                    AR_era_ds,nawdex,config_file,
                                                    AR_nawdex_cross_sections)
        output_path=output_path+"/NAWDEX/"
        ar_file_name="North_Atlantic_ARs_Longterm_October.csv"
    elif season=="spring":
        if take_both_campaigns=="North_Atlantic_Run":
            na_synth_run=Flight_Campaign.North_Atlantic_February_Run(
                                    is_flight_campaign=True,
                                    major_path=config_file["Data_Paths"]\
                                            ["campaign_path"],aircraft="HALO",
                                    interested_flights=["SRF02","SRF03",
                                                        "SRF04","SRF05"],
                                    instruments=["radar","radiometer","sonde"])
        
            output_paths=[output_path+"/NA_February_Run/"]
            cmpgn_classes=[na_synth_run]
        elif take_both_campaigns=="Second_Synthetic_Study":
            na_synth_run=Flight_Campaign.Second_Synthetic_Study(
                                    is_flight_campaign=True,
                                    major_path=config_file["Data_Paths"]\
                                            ["campaign_path"],aircraft="HALO",
                                    interested_flights=["SRF02","SRF03",
                                                        "SRF04","SRF05"],
                                    instruments=["radar","radiometer","sonde"])
            
            cmpgn_classes=[na_synth_run]
        
        else: # take both cases
            na_run=Flight_Campaign.North_Atlantic_February_Run(
                                    is_flight_campaign=True,
                                    major_path=config_file["Data_Paths"]\
                                            ["campaign_path"],aircraft="HALO",
                                    interested_flights=["SRF02"],#["SRF02","SRF04",
                                                       # "SRF07","SRF08"],
                                    instruments=["radar","radiometer","sonde"])
            
            na_synth_run=Flight_Campaign.Second_Synthetic_Study(
                                    is_flight_campaign=True,
                                    major_path=config_file["Data_Paths"]\
                                            ["campaign_path"],aircraft="HALO",
                                    interested_flights=["SRF03"],#["SRF02","SRF03",
                                                       # "SRF08","SRF09","SRF12"],
                                    instruments=["radar","radiometer","sonde"])
            
            output_paths=[output_path+"/NA_February_Run/",
                          output_path+"/Second_Synthetic_Study/"]
            cmpgn_classes=[na_run,na_synth_run]
            AR_unique_na_df=AR.get_ids_of_ARs_from_HALO_cross_sections(AR_era_ds,
                                                    cmpgn_classes[0],
                                                    config_file,
                                                    single_flight="",
                                                    single_ARs="")
            AR_unique_snd_df=AR.get_ids_of_ARs_from_HALO_cross_sections(
                                                    AR_era_ds,cmpgn_classes[-1],
                                                    config_file,single_flight="",
                                                    single_ARs="")
            
            
    north_atlantic_ARs.to_csv(path_or_buf=output_path[-1]+ar_file_name)
    print("North Atlantic AR statistics saved as:",output_paths[-1]+ar_file_name)
    plot_AR_IVT_climatology(north_atlantic_ARs,output_path,
                            AR_unique_df=AR_unique_df,
                            season=season)
    return None

def plot_combined_AR_characteristics(spring_AR_df,autumn_AR_df):
        import seaborn as sns
        import matplotlib
        matplotlib.rcParams.update({"font.size":14})
        autumn_AR_df["season"]="autumn"
        
        combined_AR_df=pd.concat([spring_AR_df,autumn_AR_df])    
        #snsplot.plot_joint(sns.kdeplot, color="darkblue", zorder=1, levels=3)
        snsplot=sns.jointplot(data=combined_AR_df,x="ivt",y="clat",hue="season",
                          s=3,alpha=0.3,space=1.2,height=8)
        snsplot.plot_joint(sns.kdeplot, zorder=1, levels=3)
        
        ### Add NAWDEX ARs
        config_name="data_config_file"
        path=os.getcwd()
        campaign_name="NAWDEX"    
    
        # Check if config-File exists and if not create the relevant first one
        if data_config.check_if_config_file_exists(config_name):
                config_file=data_config.load_config_file(path,config_name)    
                
        plot_AR_statistics=True
        import AR
        AR=AR.Atmospheric_Rivers("ERA")
        AR_era_ds=AR.open_AR_catalogue()
        
        nawdex=Flight_Campaign.NAWDEX(is_flight_campaign=True,
                          major_path=config_file["Data_Paths"]["campaign_path"],
                          aircraft="HALO",
                          instruments=["radar","radiometer","sonde"])    
        AR_nawdex_cross_sections= AR.get_HALO_NAWDEX_AR_cross_sections()
        AR_unique_df            = AR.get_ids_of_ARs_from_HALO_cross_sections(
                                        AR_era_ds,nawdex,config_file,
                                        AR_nawdex_cross_sections)
        
        snsplot.ax_joint.scatter(np.sqrt(AR_unique_df["IVT_x"]**2+\
                                         AR_unique_df["IVT_y"]**2),
                                 AR_unique_df["clat"],color="red",marker='s',
                                 s=20,edgecolor="k",label="NAWDEX")
        
        snsplot.ax_joint.set_xlabel("$\overline{IVT}$"+\
                                    " ($\mathrm{kgm}^{-1}\mathrm{s}^{-1})$")
        snsplot.ax_joint.set_ylabel("AR Centre Latitude in $^{\circ}$N")
        snsplot.ax_joint.set_ylim([40,90])
        snsplot.ax_joint.set_xlim([100,800])
        snsplot.ax_joint.legend(loc="best")
        sns.despine(offset=2)
        output_path=os.getcwd()+"/"
        snsplot.savefig(output_path+"Seasonal_AR_statistics.png",
                        dpi=200,bbox_inches="tight")
        print("Statistics saved as:",output_path+"Seasonal_AR_statistics.png")

def plot_IVT_long_term_characteristics(cmpgn_cls,AR_df,AR_campaign_df,
                                       lower_lat=55,upper_lat=90,
                                       add_centered_halo_lat=True):
    """
    

    Parameters
    ----------
    lower_lat : str
         southern boundary of ARs to consider and ylim 
    upper_lat : str
         northern boundary of ARs to consider and ylim
         
    cmpgn_cls : class
        class of flight campaign.
    AR_df : list 
        List of AR dataframes from AR catalogue (Guan & Waliser, 2019).
    AR_campaign_df : pd.DataFrame
        dataframe of AR legs flown or synthetically created.
   
    Returns
    -------
    None.

    """
    import seaborn as sns
    import matplotlib
    # Allocation
    matplotlib.rcParams.update({"font.size":22})
        
    combined_AR_df=AR_df    
        
                
    plot_AR_statistics=True
  
    flight_dates={"2016-10-13":["NAWDEX","RF10"],
                  "2011-03-17":["Second_Synthetic_Study","SRF02"],
                  "2011-04-23":["Second_Synthetic_Study","SRF03"],
                  "2015-03-14":["Second_Synthetic_Study","SRF08"],
                  "2016-03-11":["Second_Synthetic_Study","SRF09"],
                  "2018-02-24":["NA_February_Run","SRF02"],
                  "2018-02-25":["Second_Synthetic_Study","SRF12"],
                  "2019-03-19":["NA_February_Run","SRF04"],
                  "2020-04-16":["NA_February_Run","SRF07"],
                  "2020-04-19":["NA_February_Run","SRF08"]}
    flight_colors={"2016-10-13":"aquamarine",
                   "2011-03-17":"purple",
                   "2011-04-23":"brown",
                   "2015-03-14":"gold",
                   "2016-03-11":"lightpink",
                   "2018-02-24":"coral",
                   "2018-02-25":"navy",
                   "2019-03-19":"mediumseagreen",
                   "2020-04-16":"darkgreen",
                   "2020-04-19":"grey"}        
    # Plotting
    snsplot=sns.jointplot(data=combined_AR_df,x="ivt",y="clat",
                          s=20,alpha=0.5,color="teal",
                          space=1.2,height=10)
        
    snsplot.plot_joint(sns.kdeplot, zorder=1, levels=[0.25,0.75],
                       color="teal")
    
    # get legend entries depending on available indices and flights
    legend_label=[str(AR_campaign_df.index[i]) \
                  for i in range(AR_campaign_df.shape[0])]    
    if AR_campaign_df.shape[0]>0:
        for i in range(AR_campaign_df.shape[0]):
            marker_type="s"
            legend_key=str(legend_label[i])
            if legend_label[i]=="2016-10-13":
                
                legend_label[i]=legend_label[i]+"\n(NAWDEX-RF10)"
                marker_type="v"
            snsplot.ax_joint.scatter(np.sqrt(AR_campaign_df["IVT_x"]**2+\
                                     AR_campaign_df["IVT_y"]**2)[i],
                                     AR_campaign_df["clat"][i],
                                     color=flight_colors[legend_key],
                                     marker=marker_type,s=120,edgecolor="k",
                                     label=legend_label[i]+" (AR"+str(i+1)+")")
            snsplot.ax_joint.spines["left"].set_linewidth(3.0)
            snsplot.ax_joint.spines["bottom"].set_linewidth(3.0)
            
            snsplot.ax_joint.xaxis.set_tick_params(width=2,length=6)
            snsplot.ax_joint.yaxis.set_tick_params(width=2,length=6)
            
            if add_centered_halo_lat:
                campaign_name=flight_dates[legend_label[i]][0]
                flight=flight_dates[legend_label[i]][1]
                centered_halo_lat=get_centered_halo_lat(campaign_name, flight)
                #centered_halo_lat=75+i
                snsplot.ax_joint.axhline(centered_halo_lat,xmin=0.95,
                                     xmax=0.99,lw=3,
                                     color=flight_colors[legend_label[i]])

    snsplot.ax_joint.set_xlabel("$\overline{IVT}$"+\
                                    " ($\mathrm{kgm}^{-1}\mathrm{s}^{-1})$")
    snsplot.ax_joint.set_ylabel("AR Centre Latitude in $^{\circ}$N")
    snsplot.ax_joint.set_ylim([lower_lat,upper_lat])
    snsplot.ax_joint.set_xlim([100,500])
    snsplot.ax_joint.set_xticks([100,200,300,400,500])
    snsplot.ax_joint.legend(loc="upper left",fontsize=15,ncol=3)
    sns.despine(offset=5)
    output_path=cmpgn_cls.plot_path
    snsplot.fig.set_figwidth(12)
    snsplot.savefig(output_path+"fig02_Seasonal_AR_statistics.png",
                        dpi=300,bbox_inches="tight")
    print("Statistics saved as:",output_path+"fig02_Seasonal_AR_statistics.png")
    return None    

def run_plot_IVT_long_term_stats(cmpgn_cls,HMPs,
                                 flights_to_analyse,
                                 upper_lat=90,lower_lat=50,
                                 western_lon=-30,eastern_lon=90,
                                 add_single_flight="RF10"):
    
    import AR
    config_name="data_config_file"
    #path=os.getcwd()
    campaign_name=cmpgn_cls.name    
    
    # Check if config-File exists and if not create the relevant first one
    if data_config.check_if_config_file_exists(config_name):
        config_file=data_config.load_config_file(cmpgn_cls.major_path,
                                                 config_name)
    AR=AR.Atmospheric_Rivers("ERA")
    AR_era_ds=AR.open_AR_catalogue()
    # Get all relevant AR characteristics
    
    #if add_single_flight!=None:
    #nawdex_flight=add_single_flight
    if "RF10" in flights_to_analyse:
        add_single_flight="RF10"
    else:
        add_single_flight=None
    regional_ARs,AR_campaign_df=process_seasonal_AR_climatology(
                                config_file,cmpgn_cls,AR,AR_era_ds,
                                season="spring",lower_lat=lower_lat,
                                upper_lat=upper_lat,western_lon=western_lon,
                                eastern_lon=eastern_lon,add_single_flight=add_single_flight)
    
    plot_IVT_long_term_characteristics(cmpgn_cls, regional_ARs, AR_campaign_df,
                                       lower_lat=lower_lat,upper_lat=upper_lat)

def run_plot_combined_campaign_IVT_long_term_stats(cmpgn_classes,
                                 upper_lat=90,lower_lat=55,
                                 western_lon=-30,eastern_lon=90,
                                 add_single_flight="RF10",other_plot_path=""):
    
    #from IVT_climatology import process_seasonal_AR_climatology
    import atmospheric_rivers as AR
    config_name="data_config_file"
    #path=os.getcwd()
    campaign_name=cmpgn_classes[-1].name    
    
    
    #### Loop over two campaigns and merge regional ARs afterwards
    # Check if config-File exists and if not create the relevant first one
    if data_config.check_if_config_file_exists(config_name):
        config_file=data_config.load_config_file(cmpgn_classes[-1].major_path,
                                                 config_name)
    AR=AR.Atmospheric_Rivers("ERA")
    AR_era_ds=AR.open_AR_catalogue()
    # Get all relevant AR characteristics
    
    # Add NA February_Run
    NA_regional_ARs,NA_AR_campaign_df=process_seasonal_AR_climatology(
                                config_file,cmpgn_classes[0],AR,AR_era_ds,
                                season="spring",lower_lat=lower_lat,
                                upper_lat=upper_lat,western_lon=western_lon,
                                eastern_lon=eastern_lon,
                                add_single_flight=add_single_flight)
    # Second_Synthetic_Run
    SND_regional_ARs,SND_AR_campaign_df=process_seasonal_AR_climatology(
                                config_file,cmpgn_classes[-1],AR,AR_era_ds,
                                season="spring",lower_lat=lower_lat,
                                upper_lat=upper_lat,western_lon=western_lon,
                                eastern_lon=eastern_lon,add_single_flight=None)
    NA_AR_campaign_df["flights"]=NA_AR_campaign_df.index.values
    SND_AR_campaign_df["flights"]=SND_AR_campaign_df.index.values
    NA_AR_campaign_df.index=pd.DatetimeIndex(
                        NA_AR_campaign_df["Cross_Start"].values).date
    SND_AR_campaign_df.index=pd.DatetimeIndex(
                        SND_AR_campaign_df["Cross_Start"].values).date
    
    AR_campaigns_df=pd.concat([NA_AR_campaign_df,SND_AR_campaign_df])
    AR_campaigns_df=AR_campaigns_df.sort_index()
    plot_IVT_long_term_characteristics(cmpgn_classes[-1],
                                       SND_regional_ARs, AR_campaigns_df,
                                       lower_lat=lower_lat,upper_lat=upper_lat)

###############################################################################
######## Main Function
def main():
    season="spring"
    plot_single=True
    
    if plot_single:
        plot_single_season_characteristics(season=season)
    else:
        spring_AR_df=pd.read_csv(os.getcwd()+"/HALO_AC3_Dry_Run/"+\
                             "North_Atlantic_ARs_Longterm_Spring.csv")
    spring_AR_df["season"]="spring"
    autumn_AR_df=pd.read_csv(os.getcwd()+"/NAWDEX/"+\
                             "North_Atlantic_ARs_Longterm_October.csv")
    plot_combined_AR_characteristics(spring_AR_df,autumn_AR_df)

if __name__=="__main__":
    main()    