# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:31:35 2021

@author: u300737
"""
import os
import Performance
import sys

import numpy as np
import pandas as pd
import xarray as xr

import metpy.calc as mpcalc
from metpy.units import units


if "ERA5" not in sys.modules:
    from reanalysis import ERA5 as ERA5
if "CARRA" not in sys.modules:
    from reanalysis import CARRA as CARRA
    #    import ERA as ERA_module
#if "ICON" not in sys.modules:
from ICON import ICON_NWP as ICON
###############################################################################
#### Arbitary Functions
###############################################################################
##############################################################################
def round_partial(data,resolution):
    """
    value may be a pandas series
    """
    partial_rounded=round(data/resolution)*resolution
    
    return partial_rounded
###############################################################################
def harvesine_distance(origin, destination):
    import math
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d

def vectorized_harvesine_distance(s_lat, s_lng, e_lat, e_lng):

   # approximate radius of earth in km
   R = 6373.0

   s_lat = s_lat*np.pi/180.0                      
   s_lng = np.deg2rad(s_lng)     
   e_lat = np.deg2rad(e_lat)                       
   e_lng = np.deg2rad(e_lng)  

   d = np.sin((e_lat - s_lat)/2)**2 +\
       np.cos(s_lat)*np.cos(e_lat) *\
       np.sin((e_lng - s_lng)/2)**2

   return 2 * R * np.arcsin(np.sqrt(d))

def load_ar_dropsonde_profiles(dropsond_path,flight,ar_of_day,
                               sounding_name="Sounding",use_ivt_vars=True):
    if use_ivt_vars:
        vars_to_use=["q","Wspeed","IVT","Pres"]
    
    sonde_dict={}
    for var in vars_to_use:
        sonde_dict[var]=pd.read_csv(dropsond_path+flight+"_"+ar_of_day+"_"+\
                                    var+"_"+sounding_name+".csv")
        sonde_dict[var].index=pd.DatetimeIndex(sonde_dict[var].iloc[:,0])
    #if synthetic_icon_lat==4:
    #    sonde_ivt=sonde_ivt.iloc[0:-1]
    #    sonde_p=sonde_p.iloc[0:-1]
    #    sonde_q=sonde_q.iloc[0:-1]
    #    sonde_wind=sonde_wind.iloc[0:-1]
            
            
    #del sonde_q[sonde_q.columns[0]]
    #del sonde_wind[sonde_wind.columns[0]]
    #del sonde_ivt[sonde_ivt.columns[0]]
    #del sonde_p[sonde_p.columns[0]]
    return sonde_dict

###############################################################################
###############################################################################
#### Data processing and interpolation
###############################################################################
"""
## ERA
"""
#%%
class ERA_on_HALO(ERA5):
    def __init__(self,halo_df,hydrometeor_lvls_path,
                 hydrometeor_lvls_file,interpolated_lvls_file,
                 is_flight_campaign,campaign,major_path,
                 flight,date,config_file,
                 ar_of_day=None,
                 last_index=None,
                 synthetic_flight=False,
                 HMPs=["IWV","E","Precip","LWP","IWP","IVT"],
                 HMCs=["IWC","LWC","PWC","Geopot_Z","q","theta_e","u","v"],
                 do_instantaneous=False):

             super().__init__(self,is_flight_campaign,
                              major_path,era_path=hydrometeor_lvls_path)
             self.campaign=campaign
             self.flight=flight[0]
             if do_instantaneous==True:
                 if not self.flight.endswith("instantan"):
                     self.flight==self.flight+"_instantan"
             self.interpolated_lvls_file=interpolated_lvls_file
             self.date=date
             self.halo_df=halo_df
             self.hydrometeor_lvls_path=hydrometeor_lvls_path
             self.hydrometeor_lvls_file=hydrometeor_lvls_file
             self.HMPs=HMPs
             self.HMCs=HMCs
             self.config=config_file
             self.synthetic_flight=synthetic_flight
             self.ar_of_day=ar_of_day
             if last_index==None:
                 self.last_index=len(self.halo_df.index)
    
    def update_halo_df(self,new_halo_df,change_last_index=True):
        self.halo_df=new_halo_df
        if change_last_index:
            self.last_index=len(self.halo_df)
    #%% HMCs    
    def interpolate_grid_on_halo(self,content_levels=False):
        
        """
        

        Parameters
        ----------
        halo_df : pd.DataFrame
            Aircraft dataframe containing aircraft position, attitude and speed.
        upsampled_data : TYPE
            DESCRIPTION.
        last_index : int
            last index to perform interpolation.
        content_levels : str, optional
            DESCRIPTION. The default is False. If True then vertical profiles 
                         will be interpolated. This way more expensive

        Returns
        -------
        halo_era : pd.DataFrame() / dict
            Hydrometeor variables interpolated onto the HALO Track. If 
            content_levels, this returns a dictionary as each variable within 
            a df has the height as second dimension.

        """
        upsampled_data=self.upsampled_intc
        performance=Performance.performance()
        #self.halo_df.index=pd.DatetimeIndex(self)
        if not content_levels:
            # Return data will be a dataframe
            self.halo_df.index=pd.DatetimeIndex(self.halo_df.index)
            self.halo_era_hmp=pd.DataFrame(index=self.halo_df.index)
            
            #First ERA minute of day
            first_minute=int(upsampled_data["IWV"].time[0].dt.hour*60+\
                            upsampled_data["IWV"].time[0].dt.minute)
            # The first information are given from the aircraft position
            # If for some randomn reason the minutes are not shifted for indexing
            if not "Minutesofday" in self.halo_era_hmp.columns:
                self.halo_era_hmp["Minutesofday"]=self.halo_df["Minutesofday"]
            if self.halo_df["Minutesofday"].iloc[0]!=0:
                self.halo_era_hmp["Minutesofday"]=self.halo_df["Minutesofday"]-\
                        first_minute
                  
            self.halo_era_hmp["Halo_Lat"]=self.halo_df["latitude"]
            self.halo_era_hmp["Halo_Lon"]=self.halo_df["longitude"]
            
            # ERA5 has a 0.25°x0.25° resolution
            self.halo_era_hmp["Closest_Era_Lat"]=round_partial(self.halo_df["latitude"],
                                                      0.25)
            self.halo_era_hmp["Closest_Era_Lon"]=round_partial(self.halo_df["longitude"],
                                                      0.25)
            print("The hydrometeorpaths will be interpolated")
            
            #Define pd.Series for each hydrometeor and water vapour
            hmp_nearest_point=pd.Series(index=self.halo_era_hmp.index[0:self.last_index],
                                        dtype=float)
            hmp_interp_point=pd.Series(index=self.halo_era_hmp.index[0:self.last_index],
                                       dtype=float)
            
            # Perform this interpolation only up to the last index defined, 
            # default is the length of HALO dataframe (so entire flight data)
            iterative_length=self.last_index
            
            # Start interpolation, progress bar will show up for each variable
            for hmp in self.HMPs:
                print("Interpolate "+hmp)
                
                for i in range(iterative_length):
                    hmp_interp_point.iloc[i]=upsampled_data[hmp]\
                    [self.halo_era_hmp["Minutesofday"].iloc[i],:,:]\
                    .interp(latitude=self.halo_era_hmp["Halo_Lat"].iloc[i],
                            longitude=self.halo_era_hmp["Halo_Lon"].iloc[i])                                                                              
                    performance.updt(iterative_length,i) 
            
                # write the interpolated data given as pd.Series onto
                # the dataframe to be returned
                interp_hmp_str="Interp_"+hmp
                
                self.halo_era_hmp[interp_hmp_str]=hmp_interp_point
        
        ### If vertical profiles are of interest    
        else:
            self.halo_df.index=pd.DatetimeIndex(self.halo_df.index)
            halo_era_t=pd.DataFrame(index=pd.DatetimeIndex(self.halo_df.index))
            halo_era_t["Minutesofday"]=self.halo_df["Minutesofday"]
            halo_era_t["Halo_Lat"]=self.halo_df["latitude"]
            halo_era_t["Halo_Lon"]=self.halo_df["longitude"]
            halo_era_t["Closest_Era_Lat"]=round_partial(
                                            self.halo_df["latitude"],0.25)
            halo_era_t["Closest_Era_Lon"]=round_partial(
                                            self.halo_df["longitude"],0.25)
            #halo_era_t["Minutesofday"]=halo_era_t["Minutesofday"]-360
            
            print("The hydrometeorcontents will be interpolated")
            iterative_length=self.last_index
            
            # Define dictionary that will contains the vertical profiles
            # interpolated onto the flight track.
            # First entry represents the aircraft position
            halo_era={}
            halo_era["Position"] = halo_era_t

            for var in self.lvl_var_dict.keys():#self.HMCs:
                # Due to limited ressources the ERA-data is not necessarily 
                # upsampled to 1min frequency. So this is checked here and
                # treaten in the loop for further geographical interpolation 
                # of ERA-data onto HALO.
                
                tenminutes_index=pd.DatetimeIndex(pd.Series(
                                        np.array(upsampled_data[var].time[:])))
                
                time_res_is_1min=pd.Series(tenminutes_index).diff().mean()==\
                                            pd.Timedelta("1min")
            
                #time_res=pd.Timedelta(pd.Series(tenminutes_index)).mean()
            
                print("Interpolate "+var)
                for i in range(iterative_length):
                    if not time_res_is_1min:
                        two_first_tenminutes_indexes=np.sort(np.argpartition(\
                            abs(tenminutes_index-halo_era_t.index[i]),0)[0:2])
                    
                        two_first_tenminutes_data=upsampled_data["IWC"]\
                                                [two_first_tenminutes_indexes]
                        
                        upsampled_tenminutes_data=two_first_tenminutes_data.\
                                    resample(time="1min").interpolate("linear")
                        
                        closest_minute=np.argpartition(abs(pd.DatetimeIndex(\
                                pd.Series(upsampled_tenminutes_data.time[:]))-\
                                halo_era_t.index[i]),0)[0]
                        
                        temp_data=upsampled_tenminutes_data[\
                            closest_minute,:,:,:].interp(\
                                    latitude=halo_era_t["Halo_Lat"].iloc[i],
                                    longitude=halo_era_t["Halo_Lon"].iloc[i])
                
                    else:
                        closest_minute=np.argpartition(abs(pd.DatetimeIndex(\
                                    pd.Series(upsampled_data[var].time[:]))-\
                                    halo_era_t.index[i]),0)[0]
                        temp_data=upsampled_data[var][closest_minute,:,:,:]\
                            .interp(latitude=halo_era_t["Halo_Lat"].iloc[i],
                                    longitude=halo_era_t["Halo_Lon"].iloc[i])
                    if i==0:
                        interp_array=np.array(temp_data.T)                                                                                                
                    else:
                        interp_array=np.vstack((interp_array,
                                                np.array(temp_data).T))
                        performance.updt(iterative_length,i)
            
                hmc_interp_point=pd.DataFrame(data=interp_array,
                                        index=halo_era_t.index[0:iterative_length],
                                        columns=upsampled_data[var]["level"],
                                        dtype=float)
                
                halo_era[var]         = hmc_interp_point
        
        #turn the interpolated data, which is either a dataframe if columns
        # are analysed or a dictionary in case of vertical profiles
        if content_levels:
            return halo_era
        else:
            return self.halo_era_hmp
    def upsample_and_save_hwc(self,cut_to_AR=False):
        #---------------------------------------------------------------------#
        ####### Preallocate and open ERA5 Hydrometeors
        self.lvl_var_dict={"IWC":"ciwc",
                           "LWC":"clwc",
                           "PWC":["crwc","cswc"],
                           "Geopot_Z":"z",
                           "q":"q",
                           "theta_e":"theta_e",
                           "u":"u",
                           "v":"v"}
        
        #Values are given in kg/kg
        ds_lvls=xr.open_dataset(self.hydrometeor_lvls_path+\
                                self.hydrometeor_lvls_file)
        # cut dataset to region and time period of interest
        lon_range=[self.halo_df["longitude"].min()-1,
                   self.halo_df["longitude"].max()+1]
        lat_range=[self.halo_df["latitude"].min()-1,
                   self.halo_df["latitude"].max()+1]
        
        self.upsampled_intc={}
        
        # Entire dataset, which is VERY time consuming,
        # reaching computation capacities            
        if not cut_to_AR:
            start=0
            end=-1
        else:
            # Consider flight hours around AR Cross-Section    
            start = pd.DatetimeIndex(self.halo_df.index).hour[0]
            end   = pd.DatetimeIndex(self.halo_df.index).hour[-1]+2
        #---------------------------------------------------------------------#
        ####### Select relevant time period and subregion to save computation time
        for var in self.lvl_var_dict.keys():                
                
            if var=="theta_e":
                ds_cutted=ds_lvls.isel(time=np.arange(start,end))
                var_era5    = ds_cutted.sel(longitude=slice(lon_range[0],lon_range[1]),
                                     latitude=slice(lat_range[1],lat_range[0]))\
                                    .astype("float16")
                var_era5=self.calculate_theta_e(var_era5)[var]
                
            elif var=="PWC":
                da1=ds_lvls[self.lvl_var_dict[var][0]][start:end,:,:,:]
                da2=ds_lvls[self.lvl_var_dict[var][1]][start:end,:,:,:]
                var_era5  = da1.sel(longitude=slice(lon_range[0],lon_range[1]),
                                    latitude=slice(lat_range[1],lat_range[0]))+\
                            da2.sel(longitude=slice(lon_range[0],lon_range[1]),
                                    latitude=slice(lat_range[1],lat_range[0]))
                var_era5  = var_era5.astype("float16")#*1000
           
            elif var=="Geopot_Z":
                da=ds_lvls[self.lvl_var_dict[var]][start:end,:,:,:]
                var_era5  = da.sel(longitude=slice(lon_range[0],lon_range[1]),
                                   latitude=slice(lat_range[1],lat_range[0]))\
                                    .astype("float32")
                var_era5 = var_era5/9.82
            
            else:
                da=ds_lvls[self.lvl_var_dict[var]][start:end,:,:,:]
                var_era5  = da.sel(longitude=slice(lon_range[0],lon_range[1]),
                                   latitude=slice(lat_range[1],lat_range[0]))\
                                    .astype("float16")
                
                if var=="q":
                  #Mixing ratio is required for the calculation.
                  ds_lvls["mixing_ratio"]=xr.DataArray(data=np.array(\
                                mpcalc.mixing_ratio_from_specific_humidity(
                                    ds_lvls["q"])),
                                coords=ds_lvls["q"].coords)
                  print("Mixing ratio calculated")
        
            upsampling_done=False 
            #-----------------------------------------------------------------#
            ####### Upsample vars to minutes
            #return upsampling_done 
            time_intv="1min"
            if not upsampling_done:
                if self.config["Data_Paths"]["system"]=="windows":
                    #        #having it as g/kg
                    print("Upsample hydrometeor contents to 1min resolution,",
                          " this will take a while")
                    print(var)
                if var in ["IWC","LWC","PWC"]:
                    self.upsampled_intc[var]=var_era5.resample(time=time_intv)\
                        .interpolate("linear").astype("float16")*1000
                else:
                    self.upsampled_intc[var]=var_era5.resample(time=time_intv)\
                        .interpolate("linear").astype("float16")
            print("Upsampling finished")
        #---------------------------------------------------------------------#        
        del ds_lvls
        # For upsampled data: run interpolation onto HALO flight track    
        halo_era5_hmc=self.interpolate_grid_on_halo(content_levels=True)
        del self.upsampled_intc
                
        if self.ar_of_day:
            string_AR=self.ar_of_day
        else:
            string_AR=""
        #Save individual variables as csv files: 
        #    (Row: Time Index, Columns= Heights)
        for var in self.lvl_var_dict.keys():
            if not self.synthetic_flight:                
                hmc_fname=self.hydrometeor_lvls_path+self.flight+"_"+\
                        string_AR+"_"+var+"_"+self.date+".csv"
            else:
                hmc_fname=self.hydrometeor_lvls_path+"Synthetic_"+self.flight+\
                            "_"+string_AR+"_"+var+"_"+self.date+".csv"                
            halo_era5_hmc[var].to_csv(hmc_fname)
            print(var, " file saved as: ", hmc_fname)
        return halo_era5_hmc     
            
    def load_hwc(self):
        if not os.path.isfile(self.hydrometeor_lvls_path+\
                              self.interpolated_lvls_file):
           print("HMCs are not yet interpolated onto HALO; will be done now")
           halo_era5_hmc=self.upsample_and_save_hwc(cut_to_AR=True)
           #halo_era5_hmc default return variable
           print("Upsampling of vertical ERA5 hydrometeor contents done. ")
            
        else:
            print("Hydrometeor Content Profiles are already interpolated and saved")
            print("So load the data")
            halo_era5_hmc={}
            
            for vertical_var in self.HMCs:
                print(vertical_var)
                # If files are only based on AR cross-sections
                if self.ar_of_day is not None:
                    self.hwc_file=self.flight+"_"+self.ar_of_day+"_"+\
                                    vertical_var+"_"+self.date+".csv"
                else:
                    self.hwc_file=self.flight+"_"+vertical_var+"_"+\
                                    self.date+".csv"
                if self.synthetic_flight:
                    self.hwc_file="Synthetic_"+self.hwc_file
                try:
                    halo_era5_hmc[vertical_var]=pd.read_csv(
                                            self.hydrometeor_lvls_path+self.hwc_file)
                    halo_era5_hmc[vertical_var].index=pd.DatetimeIndex(
                                        halo_era5_hmc[vertical_var].iloc[:,0])
                    halo_era5_hmc[vertical_var].drop(
                                    halo_era5_hmc[vertical_var].columns[[0]],
                                    axis=1,inplace=True)
                except:
                    raise ValueError(vertical_var,
                                     " has not yet been interpolated")            
        return halo_era5_hmc     

    #%% HMPs
    def upsample_hmp(self):
        # take total column datasets, unit is kg/m2 change this to g/m2 
        # for hydrometeors, water vapour in kg/m2 is correct
        self.upsampled_intc={}#pd.DataFrame(columns=)
        for hmp in self.HMPs:
            print(hmp)
            if hmp=="IWV":
                hmp_era5=self.ds["tcwv"]
                
            elif hmp=="IVT":
                # Newly add IVT which is THE relevant quantity to detect ARs.
                #IVT Processing
                self.ds["IVT_v"]=self.ds["p72.162"]
                self.ds["IVT_u"]=self.ds["p71.162"]
                hmp_era5=np.sqrt(self.ds["IVT_u"]**2+self.ds["IVT_v"]**2)
            elif hmp=="LWP":
                hmp_era5=self.ds["tclw"]*1000
            elif hmp=="IWP":
                hmp_era5=self.ds["tciw"]*1000
            elif hmp=="E":
                hmp_era5=self.ds["e"]*1000
            elif hmp=="Precip":
                hmp_era5=self.ds["tp"]*1000
            hour_start=int(self.halo_df.index.hour[0])
            hour_end=int(self.halo_df.index.hour[-1]+2)
            hmp_era5=hmp_era5[hour_start:hour_end,:,:]
            print("Interpolate the hourly total columns to minutely data")
            print("in order to better simulate the time variability")
            print("---this takes a while, as it is compute extensively----")
            self.upsampled_intc[hmp]=hmp_era5.resample(time="1min").\
                                        interpolate("linear")
    
    def update_interpolated_hmp_file(self,interpolated_hmp_file):
        self.interpolated_hmp_file=interpolated_hmp_file
    
    def save_interpolated_hmps(self):
        self.halo_era_hmp.to_csv(self.hydrometeor_lvls_path+\
                                  self.interpolated_hmp_file)
        print("Flight-track interpolated hydrometeor paths are saved as",
              self.hydrometeor_lvls_path+self.interpolated_hmp_file)
    
    def load_hmp(self,campaign_cls):
        
        # If interpolated files are not present
        if not os.path.isfile(self.hydrometeor_lvls_path+\
                              self.interpolated_hmp_file):
            if self.flight.endswith("instantan"):
                initial_flight_name=self.flight.split("_")[0]
                flight_name=initial_flight_name
            else:
                flight_name=self.flight
            file_name="total_columns_"+campaign_cls.years[flight_name]+"_"+\
                            campaign_cls.flight_month[flight_name]+"_"+\
                            campaign_cls.flight_day[flight_name]+".nc"    
            #era5=ERA5()
            # Load ERA5 data
            self.ds,self.era_path=self.load_era5_data(file_name)
            
            # Upsample ERA5 to 1min
            self.upsample_hmp()
            
            # Interpolate this upsampled dataset onto the HALO flight track 
            self.interpolate_grid_on_halo(content_levels=False)
            
            # and save it
            self.save_interpolated_hmps()
            self.upsampled_hmp=self.halo_era_hmp
        # Else if interpolated files are existent    
        else:
            print("Load calculated vertical total column interpolated data")
            #if ar_of_day:
            #    interpolated_hmp_file=ar_of_day+interpolated_hmp_file
            self.upsampled_hmp=pd.read_csv(self.hydrometeor_lvls_path+\
                                           self.interpolated_hmp_file)
            self.upsampled_hmp.index=pd.DatetimeIndex(
                                            self.upsampled_hmp.iloc[:,0])
        return self.upsampled_hmp
            
    #%% Others
    def cut_halo_to_AR_crossing(self,AR_of_day,flight,initial_halo,
                                initial_df,campaign="NAWDEX",device="radar",
                                invert_flight=False):
            
            # Create copies to assure that initial dataset is not touched and 
            # remain complete
            if not initial_df==None:
                df=initial_df.copy()
            else:
                df=None
            halo=initial_halo.copy()
            
            # Load AR class to access campaign AR cross-section look-up table
            #if "AR" not in sys.modules:
            import AR
            AR_class=AR.Atmospheric_Rivers
            if campaign=="NAWDEX":
                ARs=AR_class.look_up_AR_cross_sections(self.campaign)
                ARs_NAWDEX=ARs.copy()
                if "ARs_NAWDEX" in locals():
                    #merge both dictionairies
                    ARs=ARs | ARs_NAWDEX
                
            cut_start   = ARs[flight][AR_of_day]["start"]
            cut_end     = ARs[flight][AR_of_day]["end"]
            #if not halo in locals():
            #    import flight_track_creator
            #    Flight_Tracker=flight_track_creator.Flighttracker(
            #                                campaign,flight,AR_of_day,
            #                                track_type="internal",
            #                                shifted_lat=0,shifted_lon=0)
           # 
           #     halo_df,campaign_path=Flight_Tracker.run_flight_track_creator()
         
            if isinstance(halo,pd.DataFrame):
                halo=halo.loc[cut_start:cut_end]
            else:
                halo[flight]=halo[flight].loc[cut_start:cut_end]
            if device=="radar":
                df["Reflectivity"]=df["Reflectivity"].loc[cut_start:cut_end]
                df["LDR"]=df["LDR"].loc[cut_start:cut_end]
            elif device=="radiometer":
                df["T_b"]=df["T_b"].loc[cut_start:cut_end]
            elif device=="sondes":
                for var in df.keys():
                    if isinstance(df[var],pd.Series)\
                    or isinstance(df[var],pd.DataFrame):
                        df[var]=df[var].loc[cut_start:cut_end]
            elif device=="halo":
                pass
            else:
                Exception("Wrong device given")
            self.ARs=ARs
            self.AR_of_day=AR_of_day
            self.analyse_AR_core=True
            return halo,df,self.AR_of_day
            
    def apply_orographic_mask_to_era(self,era_df,threshold=4e-3,
                                         variable="LWC",short_test=False):
            """
            This function masks era values which do not take into account the 
            orography which leads to extrapolated values down to the surface.
            These values are set constant and are critical when integrating 
            vertically in order to get for example vertical columns. 
            
            This mask filters out values where the vertical gradient is zero
            as these are the extrapolated values.
        
            Parameters
            ----------
            era_df : pd.DataFrame
                ERA dataset to be masked.
            
            variable : str
                variable to analyse
                
            short_test : boolean
                Boolean specifying mask works properly by creating a quicklook.
                The default is False.
                
            Return
            -------
            masked_era_df : pd.DataFrame
                ERA dataset which is now masked for orography.
        
            """
            
            diff_era_q_sonde=era_df.diff(axis=1)
            mask_threshold=threshold
            mask_df=abs(diff_era_q_sonde)>mask_threshold
            mask_df.iloc[:,0]=True
            mask_df=mask_df*1
            mask_df=mask_df.rolling(2,axis=1,center=False).mean()
            mask_df.iloc[:,0:2]=1
            #mask_df=mask_df.replace(to_replace=np.nan,value=0)
            mask_df=mask_df.round()
            double_mask_df=mask_df.diff(axis=1)
            double_mask_df=double_mask_df.rolling(2,axis=1,center=False).mean()
            double_mask_df.iloc[:,0:1]=0
            
            double_mask_df=double_mask_df<0.05
            double_mask_df=double_mask_df*1
            mask_df=mask_df*double_mask_df
            del double_mask_df
            
            # Redo once again
            double_mask_df=mask_df.diff(axis=1)
            double_mask_df=double_mask_df.rolling(2,axis=1,center=False).mean()
            double_mask_df.iloc[:,0:2]=0
            double_mask_df=double_mask_df<0.05
            double_mask_df=double_mask_df*1
            
            # Change zero values in mask to nans as otherwise they would appear 
            # in plots
            mask_df=mask_df*double_mask_df
            mask_df=mask_df.replace(to_replace=0,value=np.nan)
            mask_df.iloc[:,0:15]=1
            self.mask_df=mask_df
            # Apply the mask
            masked_era_df=era_df*mask_df
            
            # Evaluate the mask if needed
            if short_test:
                import matplotlib.pyplot as plt
                plt.imshow(masked_era_df)
                print("A short masking evaluation has been plotted")
            else:
                pass
            
            print("ERA variable ",variable," is masked successfully")
            return masked_era_df,self.mask_df
    #%% Dropsondes 
    def save_ar_dropsonde_soundings(self,sondes,save_path,variables=[],
                                    research_flight="RF_None",
                                    ar_number="AR1",upsampled=False):
        """
        Parameters
        ----------
        sondes : dict
            Dictionary containing the dropsonde data.
        save_path : str
            Path to save the dropsonde data as csv.
        variables : list, optional
            List of strings containing the variables to save as csv. 
            The default is [].
        
        research_flight : str, optional
            Name of Research Flight from which data will be saved. The RF will 
            be included in the name of the csv-file.The default is "RF_None".
        ar_number : str, optional
            Number of AR-Cross-section of flight pattern. The default is AR1.
        upsampled : boolean, optional
            Specification if the dropsonde data to be saved is upsampled in 
            time or standard. The default is False (standard index of 
                                                    successive sonde releases).

        Returns
        -------
        None.
        """  
        print("Save the dropsonde data within the AR cross-sections.")
        print("Including variables: ", variables)
        time_index=sondes["IVT"].index
        
        AR_period=self.ARs[research_flight][ar_number]
        for var in variables:
            # iterate over the variables and save them
            file_major=save_path+research_flight+"_"+ar_number+"_"+var
            
            if not upsampled:
                file_name=file_major+"_Sounding.csv"
                try:
                    storage_Dropsondes=sondes[var]
                    storage_Dropsondes.index=time_index
                    storage_Dropsondes=storage_Dropsondes.loc[\
                                            AR_period["start"]:\
                                            AR_period["end"]]
                    storage_Dropsondes.to_csv(
                        path_or_buf=file_name,
                        header=True,index=True)
                except:
                    print("The variable ", var, "is not in the Dropsonde data")
            else:
                file_name=file_major+"_Upsampled_Sounding.csv"
                try:
                    storage_Dropsondes=sondes[var]
                    storage_Dropsondes.index=time_index
                    storage_Dropsondes.to_csv(
                        path_or_buf=file_name,
                        header=True,index=True)
                except:
                    print("The variable ", var, "is not in the Dropsonde data")
            print("Sounding Data saved as:", file_name)
        return None
    
    def load_upsampled_dropsondes(self,campaign_cls,sondes,
                                  cutted_to_AR=True):
        if not os.path.exists(campaign_cls.dropsonde_path):
            Dropsondes,Upsampled_Dropsondes=campaign_cls.load_ar_processed_dropsondes(
                                            with_upsampling=True)
            #sondes
            halo_df,Dropsondes,ar_of_day=self.cut_halo_to_AR_crossing(
                                                    self.AR_of_day, self.flight, 
                                                    self.halo_df,Dropsondes,
                                                    device="sondes")
    
            #upsampled sondes
            halo_df,Upsampled_Dropsondes,ar_of_day=self.cut_halo_to_AR_crossing(
                                                    self.AR_of_day, self.flight, 
                                                    self.halo_df,Upsampled_Dropsondes,
                                                    device="sondes")
            # save AR Dropsonde soundings
            self.save_ar_dropsonde_soundings(Dropsondes,
                                             variables=["IVT","IWV","q","Wdir",
                                                        "Wspeed","Pres"],
                                             save_path=campaign_cls.dropsonde_path,
                                             research_flight=self.flight,
                                             ar_number=self.AR_of_day,
                                             upsampled=False)
            # save Upsampled AR Dropsonde soundings
            self.save_ar_dropsonde_soundings(Upsampled_Dropsondes,
                                             save_path=campaign_cls.dropsonde_path,
                                             variables=["IVT","IWV","q","Wdir",
                                                        "Wspeed","Pres"],                                    
                                             research_flight=self.flight,
                                             ar_number=self.AR_of_day,
                                             upsampled=True)
###############################################################################
"""
## CARRA
"""
class CARRA_on_HALO(CARRA):
    def __init__(self,halo_df,carra_lvls_path,
                 is_flight_campaign,campaign,major_path,
                 flights,date,config_file,ar_of_day=None,last_index=None,
                 synthetic_flight=False,
                 HMPs=["IWV_clc","IVT","IVT_u","IVT_v"],
                 HMCs=["specific_humidity","u","v","z"],
                 do_instantaneous=False):

             super().__init__(self,is_flight_campaign,
                              major_path,carra_path=carra_lvls_path)
             self.campaign=campaign
             self.flight=flights[0]
             if do_instantaneous==True:
                 if not self.flight.endswith("instantan"):
                     self.flight=self.flight+"_instantan"
             self.date=date
             self.halo_df=halo_df
             self.carra_lvls_path=carra_lvls_path
             self.HMPs=HMPs
             self.HMCs=HMCs
             self.config=config_file
             self.synthetic_flight=synthetic_flight
             self.ar_of_day=ar_of_day
             if last_index==None:
                 self.last_index=len(self.halo_df.index)
             else:
                self.last_index=last_index
    
    def update_halo_df(self,new_halo_df,change_last_index=True):
        self.halo_df=new_halo_df
        if change_last_index:
            self.last_index=len(self.halo_df)

    def calc_specific_humidity_from_relative_humidity(self):
        """
        CARRA moisture data on pressure levels is only given as rel. humidity,
        for moisture budget, specific humidity is required

        Returns
        -------
        None.
        
        """
        print("Calculate q from RH")
        #Using metpy functions to calculate specific humidity from RH
        pressure=self.ds["isobaricInhPa"].data.astype(np.float32)
        if len(self.ds["t"].shape)>=3:
            pressure=pressure[np.newaxis,:]
            pressure=np.repeat(pressure,self.ds["t"].shape[0],axis=0)
            pressure=pressure[:,:,np.newaxis]
            pressure=np.repeat(pressure,self.ds["t"].shape[2],axis=2)
            pressure=pressure[:,:,:,np.newaxis]
            pressure=np.repeat(pressure,self.ds["t"].shape[3],axis=3)
            pressure=pressure * units.hPa
    
        rh=self.ds["r"].data/100
        rh=rh.astype(np.float32)
        temperature=self.ds["t"].data.astype(np.float32) * units.K
        mixing_ratio=mpcalc.mixing_ratio_from_relative_humidity(
                                        rh,temperature,pressure)
        print("mixing_ratio calculated")
        specific_humidity=xr.DataArray(np.array(
                                    mpcalc.specific_humidity_from_mixing_ratio(
                                        mixing_ratio)),
                                   dims=["time","isobaricInhPa","y","x"])
        print("specific humidity calculated")
        self.ds=self.ds.assign({"specific_humidity":specific_humidity})
    
    def calc_IVT_from_q(self):
        print("Calculate IVT from CARRA")
        g= 9.81
        list_timestamps=self.ds.time.values 
    
        if len(list_timestamps)==1:
            self.carra_ivt=xr.Dataset(coords={"longitude":self.ds.longitude,
                                              "latitude":self.ds.latitude})
            nan_array=np.empty((self.ds["r"].shape[1],
                        self.ds["r"].shape[2]))
    
        else:
            self.carra_ivt=xr.Dataset(coords={"time":self.ds.time,
                                 "longitude":self.ds.longitude,
                                 "latitude":self.ds.latitude})
        nan_array=np.empty((self.ds["r"].shape[0],self.ds["r"].shape[2],
                            self.ds["r"].shape[3]))
    
        nan_array[:]=np.nan
    
        self.carra_ivt["IVT"]    = xr.DataArray(data=nan_array,
                                                coords=self.carra_ivt.coords)
        self.carra_ivt["IVT_u"]  = xr.DataArray(data=nan_array.copy(),
                                                coords=self.carra_ivt.coords)
        self.carra_ivt["IVT_v"]  = xr.DataArray(data=nan_array.copy(),
                                                coords=self.carra_ivt.coords)
        self.carra_ivt["IWV_clc"]= xr.DataArray(data=nan_array.copy(),
                                                coords=self.carra_ivt.coords)
        for timestep in range(len(list_timestamps)):
        
            if len(list_timestamps)==1:
                q_loc = np.array(self.ds["specific_humidity"].values)
                u_loc = np.array(self.ds["u"].values)
                v_loc = np.array(self.ds["v"].values)
        
            else:
                q_loc = np.array(self.ds["specific_humidity"][\
                                                    timestep,:,:])
                u_loc = np.array(self.ds["u"][timestep,:,:])
                v_loc = np.array(self.ds["v"][timestep,:,:])
                
                qu=q_loc*u_loc
                qv=q_loc*v_loc
                
                pres_index=pd.Series(self.ds["isobaricInhPa"].values*100)
        
            iwv_temporary=-1/g*np.trapz(q_loc,axis=0,x=pres_index)
            ivt_u_temporary=-1/g*np.trapz(qu,axis=0,x=pres_index)
            ivt_v_temporary=-1/g*np.trapz(qv,axis=0,x=pres_index)
            ivt_temporary=np.sqrt(ivt_u_temporary**2+ivt_v_temporary**2)
            if len(list_timestamps)==1:
                self.carra_ivt["IVT"].values=ivt_temporary.T
                self.carra_ivt["IVT_u"].values=ivt_u_temporary.T
                self.carra_ivt["IVT_v"].values=ivt_v_temporary.T
                self.carra_ivt["IWV_clc"].values=iwv_temporary.T
            else:
                self.carra_ivt["IVT"].values[timestep,:,:]    = ivt_temporary
                self.carra_ivt["IVT_u"].values[timestep,:,:]  = ivt_u_temporary
                self.carra_ivt["IVT_v"].values[timestep,:,:]  = ivt_v_temporary
                self.carra_ivt["IWV_clc"].values[timestep,:,:]= iwv_temporary
            
    def ivt_quicklook(self,time_stamp=0):
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        from typhon.plots import styles
        from matplotlib import gridspec
        #self.carra_ivt["longitude"]=self.carra_ivt["longitude"].where(
        #    self.carra_ivt["longitude"]<180,self.carra_ivt["longitude"]-360)

        lat2=np.array(self.carra_ivt["latitude"][:])
        lon2=np.array(self.carra_ivt["longitude"][:])

        carra_ivt=xr.Dataset()
        carra_ivt["IVT"]=self.carra_ivt["IVT"][time_stamp,:,:]
        with plt.style.context(styles("typhon")):
                
            map_fig=plt.figure(figsize=(12,16))
            gs=gridspec.GridSpec(1,2,width_ratios=[1,1])
            ax=plt.subplot(gs[0],projection=ccrs.AzimuthalEquidistant(
                                        central_longitude=-50.0,central_latitude=70))
            ax.coastlines(resolution="50m")
            ax.gridlines()
            #ax.set_extent([-60,-10,60,90])
            carra_ivt["IVT"]=carra_ivt["IVT"].where(carra_ivt["IVT"]>50)
            C1=plt.contourf(lon2,lat2,carra_ivt["IVT"].values,
                            transform=ccrs.PlateCarree(),cmap=plt.get_cmap("speed"),
                    extend="both",levels=np.linspace(50,600,11))
            plt.plot(self.halo_df["longitude"],
                     self.halo_df["latitude"],color="r",transform=ccrs.PlateCarree())
            plt.colorbar(C1,ax=ax,shrink=0.3)
            fig_name=self.campaign+"_"+self.flight+"_"+self.date+".png"
            fig_path=os.getcwd()+"/"+self.campaign+"/"
            map_fig.savefig(fig_path+fig_name,bbox_inches="tight")
            print("Quicklook saved as:",fig_path+fig_name)
    def calc_ivt_from_origin_carra_ds(self,do_quicklook=False):
        self.calc_specific_humidity_from_relative_humidity()
        self.calc_IVT_from_q()
        if do_quicklook:
            self.ivt_quicklook()
            #sys.exit()
        
    def remove_times_before_after_flight(self):
        vld_t_series=pd.Series(self.ds.valid_time[:])
        vld_t_series = vld_t_series[vld_t_series.between(self.halo_df.index[0]-\
                                                         pd.Timedelta("1H"),
                                               self.halo_df.index[-1]+\
                                                   pd.Timedelta("1H"),
                                               inclusive=False)]
        self.ds=self.ds.isel({"step":vld_t_series.index})
        
        
    def merge_all_files_for_given_flight(self):
        import glob
        file_date_list=glob.glob(self.carra_path+"*vertical_levels*"+self.date+"*.nc")
        ds_list=[]
        step=0
        f=0
        for file in file_date_list:
            self.load_vertical_carra_data(self.date,
                                                  initial_time=file[-5:-3]+\
                                                      ":00")
            
            self.remove_times_before_after_flight()
            
            ds_item=self.ds
            if f==0:
                ds_item["step"]=ds_item["step"]+2
            else:
                ds_item["step"]=ds_item["step"]+step
            if not ds_item["r"].shape[0]==0:
                ds_list.append(ds_item)
            step+=3
            f+=1
        
        if not len(ds_list)==1:    
            self.ds=xr.concat(ds_list,dim="step")
        else:
            self.ds=ds_list[-1]
        self.ds=self.ds.rename_dims({"step":"time"})
        self.ds=self.ds.set_index(time="valid_time")
    
    def upsample_hmp(self):
        # take total column datasets, unit is kg/m2 change this to g/m2 
        # for hydrometeors, water vapour in kg/m2 is correct
        self.upsampled_intc={}
        print("Interpolate the hourly total columns to minutely data")
        print("in order to better simulate the time variability")
        print("---this takes a while, as it is compute extensively----")
            
        for hmp in self.HMPs:
            print(hmp)
            hmp_carra=self.carra_ivt[hmp].copy()
            hmp_carra=hmp_carra.where((hmp_carra.longitude > \
                                       self.halo_df["longitude"].min()-2) 
                                      & (hmp_carra.longitude < \
                                         self.halo_df["longitude"].max()+2)
                                      & (hmp_carra.latitude > \
                                       self.halo_df["latitude"].min()-2) 
                                      & (hmp_carra.latitude < \
                                         self.halo_df["latitude"].max()+2),
                                          drop=True)                              #data.lats < 30) & (data.lons > -80) & (data.lons < -75))
            
            test=hmp_carra.resample(time="1min").interpolate("linear")
            self.upsampled_intc[hmp]=test
    
    def get_data_from_closest_point(self,point,lat_grid,lon_grid,grid_ds):
        lat1, lon1 = point

        distance_array=vectorized_harvesine_distance(lat1,lon1,
                                                 lat_grid,
                                                 lon_grid)
        min_distance_indices=np.unravel_index(np.argmin(distance_array,
                                                        axis=None),
                                      distance_array.shape)
        
        return grid_ds.isel(x=min_distance_indices[1],y=min_distance_indices[0])

    def interpolate_hmp_on_halo(self):
        
        upsampled_data=self.upsampled_intc
        performance=Performance.performance()
        # Returned data will be a dataframe
        self.halo_df.index=pd.DatetimeIndex(self.halo_df.index)
        self.halo_carra_hmp=pd.DataFrame(index=self.halo_df.index)
            
        #First ERA minute of day
        first_minute=int(upsampled_data["IWV_clc"].time[0].dt.hour*60)
        #+\
                         #upsampled_data["IWV_clc"].time[0].dt.minute)
        # The first information are given from the aircraft position
        # If for some randomn reason the minutes are not shifted for indexing
        if not "Minutesofday" in self.halo_carra_hmp.columns:
            self.halo_carra_hmp["Minutesofday"]=self.halo_df["Minutesofday"]-\
                        first_minute
                  
        self.halo_carra_hmp["Halo_Lat"]=self.halo_df["latitude"]
        self.halo_carra_hmp["Halo_Lon"]=self.halo_df["longitude"]
        #### To be changed with nearest neighbour point --> see carra temporary script
        #Define pd.Series for each hydrometeor and water vapour
        hmp_nearest_point=pd.Series(index=self.halo_carra_hmp.index[\
                                                        0:self.last_index],
                                    dtype=float)
        hmp_interp_point=pd.Series(index=self.halo_carra_hmp.index[\
                                    0:self.last_index],
                                       dtype=float)
            
        # Perform this interpolation only up to the last index defined, 
        # default is the length of HALO dataframe (so entire flight data)
        iterative_length=self.last_index
        lat_grid=np.array(self.upsampled_intc["IWV_clc"].latitude)
        lon_grid=np.array(self.upsampled_intc["IWV_clc"].longitude)
        # Start interpolation, progress bar will show up for each variable
        for hmp in self.HMPs:
            print("Interpolate "+hmp)
            for i in range(iterative_length):
                point=(self.halo_carra_hmp["Halo_Lat"].iloc[i],
                       self.halo_carra_hmp["Halo_Lon"].iloc[i]) # core

                point_ds=self.get_data_from_closest_point(
                                                    point,lat_grid,lon_grid,
                                                    upsampled_data[hmp])
#                test_df=pd.DataFrame(data=np.array(upsampled_data["IWV_clc"]\
#                                                   [0,:,:]))
                hmp_interp_point.iloc[i]=float(point_ds[\
                                                self.halo_carra_hmp[\
                                                    "Minutesofday"].iloc[i]])
                # Linear Interpolation does not work yet
                #hmp_interp_point.iloc[i]=upsampled_data[hmp]\
                #    [self.halo_carra_hmp["Minutesofday"].iloc[i],:,:]\
                #    .interp(latitude=self.halo_carra_hmp["Halo_Lat"].iloc[i],
                #            longitude=self.halo_carra_hmp["Halo_Lon"].iloc[i])                                                                              
                performance.updt(iterative_length,i) 
            
            # write the interpolated data given as pd.Series onto
            # the dataframe to be returned
            interp_hmp_str="Interp_"+hmp
                
            self.halo_carra_hmp[interp_hmp_str]=hmp_interp_point
        
                                        
    def load_single_file_as_ds(self,file):
        self.ds=xr.open_dataset(self.carra_lvls_path+file,engine="netcdf4")  
    
    def check_if_hmp_data_is_interpolated(self):
        self.carra_is_interpolated=False
        if not self.ar_of_day==None:    
            self.interp_hmp_file=self.flight+"_"+self.ar_of_day+\
                                "_HMP_CARRA_HALO_"+self.date+".csv"
        else:
            self.interp_hmp_file="HMP_CARRA_HALO_"+self.date+".csv"
    
        if self.synthetic_flight:
            self.interp_hmp_file="Synthetic_"+self.interp_hmp_file
        
        if os.path.exists(self.carra_lvls_path+self.interp_hmp_file):
           self.carra_is_interpolated=True
        else:
            pass
    
    def load_interp_hmp_data(self):
        self.halo_carra_hmp=pd.read_csv(self.carra_lvls_path+\
                                              self.interp_hmp_file,index_col=0)
    def save_interpolated_hmps(self):
        self.halo_carra_hmp.to_csv(self.carra_lvls_path+\
                                  self.interp_hmp_file)
        print("Flight-track interpolated CARRA hydrometeor paths are saved as",
              self.carra_lvls_path+self.interp_hmp_file)
    
    def calc_interp_hmp_data(self):
        self.merge_all_files_for_given_flight()
        self.calc_ivt_from_origin_carra_ds()
        self.upsample_hmp()
        self.interpolate_hmp_on_halo()
        self.save_interpolated_hmps()
    def load_or_calc_interpolated_hmp_data(self):
        self.check_if_hmp_data_is_interpolated()
        if self.carra_is_interpolated:
            print("CARRA data is already interpolated")
            self.load_interp_hmp_data()
        else:
            print("CARRA data has not been interpolated")
            self.calc_interp_hmp_data()
    ###########################################################################
    #%%
    # HMC DATA    
    def check_if_hmc_data_is_interpolated(self):
        self.carra_hmc_is_interpolated=False
        interp_hmc_file=self.flight+"_"+self.ar_of_day+"_z_"+\
            self.date+".csv"
        if self.synthetic_flight:
                interp_hmc_file="Synthetic_"+interp_hmc_file
        if os.path.exists(self.carra_lvls_path+interp_hmc_file):
           self.carra_hmc_is_interpolated=True
        else:
            pass
    
    def load_interp_hmc_data(self):        
        self.carra_halo_hmc={}
        for hmc in self.HMCs:
            interp_hmc_file=self.flight+"_"+self.ar_of_day+"_"+hmc+"_"+\
                self.date+".csv"
            if self.synthetic_flight:
                interp_hmc_file="Synthetic_"+interp_hmc_file
            self.carra_halo_hmc[hmc]=pd.read_csv(self.carra_lvls_path+\
                                      interp_hmc_file,index_col=0)
            self.carra_halo_hmc[hmc].index=pd.DatetimeIndex(
                                        self.carra_halo_hmc[hmc].index)
        
    def upsample_hmc_var_data(self):
        
        self.upsampled_intc={}
        print("Interpolate vertical CARRA profiles onto the HALO track ", 
              "at given HALO resolution.")
        # cut dataset to region and time period of interest
        lon_range=[self.halo_df["longitude"].min()-1,
                   self.halo_df["longitude"].max()+1]
        lat_range=[self.halo_df["latitude"].min()-1,
                   self.halo_df["latitude"].max()+1]
        
        # Entire dataset, which is VERY time consuming,
        # reaching computation capacities            
        # Consider flight hours around AR Cross-Section    
        start = self.halo_df.index[0]-pd.Timedelta("1H")
        end   = self.halo_df.index[-1]+pd.Timedelta("1H")
        
        self.halo_df.index=pd.DatetimeIndex(self.halo_df.index)
        # Return data will be a dataframe
        self.halo_carra=pd.DataFrame(index=self.halo_df.index)
    
        # The first information are given from the aircraft position
        self.halo_carra["Minutesofday"]=self.halo_df["Minutesofday"].copy()
        self.halo_carra["Halo_Lat"]=self.halo_df["latitude"].copy()
        self.halo_carra["Halo_Lon"]=self.halo_df["longitude"].copy()
        
        # Perform this interpolation only up to the last index defined, 
        # default is the length of HALO dataframe (so entire flight data)
        iterative_length=self.halo_carra.shape[0]
            
            
        for var in self.HMCs:
            print("Upsample CARRA ",var)#,var," onto the Flight Track")
            print("Preprocessing")
            carra_q=self.ds[var]
            #q_interp_point=pd.DataFrame(index=halo_carra.index[0:self.last_index],
            #                        columns=np.arange(len(carra_q.level)),
            #                        dtype=float)
            da_cutted=carra_q.sel(time=slice(start,end))
            if var=="z":
                #da=ds_lvls[self.lvl_var_dict[var]][start:end,:,:,:]
                #var_era5  = da.sel(longitude=slice(lon_range[0],lon_range[1]),
                #                   latitude=slice(lat_range[1],lat_range[0]))\
                #                    .astype("float32")
                da_cutted = da_cutted/9.82
            
            cutted_var_carra=da_cutted.where((da_cutted.longitude > \
                                       self.halo_df["longitude"].min()-2) 
                                      & (da_cutted.longitude < \
                                         self.halo_df["longitude"].max()+2)
                                      & (da_cutted.latitude > \
                                       self.halo_df["latitude"].min()-2) 
                                      & (da_cutted.latitude < \
                                         self.halo_df["latitude"].max()+2),
                                          drop=True).astype("float16")                              #data.lats < 30) & (data.lons > -80) & (data.lons < -75))
            
            
            upsampling_done=False 
            #-----------------------------------------------------------------#
            ####### Upsample vars
            # Now upsampling is done for 10 minutes, but it is recommened to do 
            # that for 10 minutes and then upsample specific times
            self.upsample_time="10min"
            if not upsampling_done:
                if self.config["Data_Paths"]["system"]=="windows":
                    #        #having it as g/kg
                    print("Upsample hydrometeor contents to ",self.upsample_time,
                          " resolution, this will take a while")
                    print(var)
                self.upsampled_intc[var]=cutted_var_carra.astype("float16").\
                    resample(time=self.upsample_time).\
                        interpolate("linear").astype("float16")
                # Specific humidity given in kg/kg but g/kg output is desired
                if var=="q":
                    self.upsampled_intc[var]=self.upsampled_intc[var]*1000
                print("Upsampling finished")

    def idx_loc_desired_times(self,carra_var_data,
                              carra_time_index,
                              range_index,halo_time_index,int_idx):
        """
        Parameters
        ----------
        halo_time_index : TYPE
            DESCRIPTION.
        carra_time_index : TYPE
            DESCRIPTION.
        
        range_index : TYPE
            DESCRIPTION.
        int_idx : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        hour_range_idx=range_index[carra_time_index.hour==\
                                   halo_time_index.hour[int_idx]]
        if not hour_range_idx[-1]==range_index[-1]:
            hour_range_idx=np.append(hour_range_idx,hour_range_idx[-1]+1)
        self.hour_data=carra_var_data.isel({"time":hour_range_idx})
            
    def interpolate_hmc_on_halo(self):
        """
        Interpolate (and not more) the carra data onto grid if upsample time is
        1 minute. Otherwise an additional upsampling from e.g. 10 minutes to
        1 minute is envisioned but not yet implemented
        Returns
        -------
        """
        self.carra_halo_hmc={}
        performance=Performance.performance()
        lat_grid=np.array(self.upsampled_intc["z"].latitude)
        lon_grid=np.array(self.upsampled_intc["z"].longitude)
        # Start interpolation, progress bar will show up for each variable
        for var in self.HMCs:
            print("Interpolate CARRA",var, "onto HALO")
            i=0
            carra_time_index=pd.DatetimeIndex(self.upsampled_intc[var].time.values)
            carra_range_index=np.arange(0,carra_time_index.shape[0])
            self.idx_loc_desired_times(self.upsampled_intc[var],
                                  carra_time_index,carra_range_index,
                                  self.halo_df.index,i)
            print("Resample one hour data to minute frequency")
            self.hour_data=self.hour_data.resample(time="1min").interpolate("linear")
            hour_data_index=pd.DatetimeIndex(self.hour_data.time.values)
            hour_data_hours=hour_data_index.hour
            last_index=self.last_index
            for i in range(last_index):                
                if self.halo_df.index.hour[i] == hour_data_hours[0]:
                    pass
                else:
                    print("Resample one hour data to minute frequency")
            
                    self.idx_loc_desired_times(self.upsampled_intc[var],
                                  carra_time_index,carra_range_index,
                                  self.halo_df.index,i)
                    self.hour_data=self.hour_data.resample(time="1min").\
                        interpolate("linear")
                    hour_data_index=pd.DatetimeIndex(self.hour_data.time.values)
                    hour_data_hours=hour_data_index.hour
                    
                    #j=0
                
                # Find closest minute of day using np.argpartition
                closest_minute=np.argpartition(abs(hour_data_index-\
                                    self.halo_df.index[i]),0)[0]
                
                # Interpolate onto halo position
                temp_data=self.hour_data[closest_minute,:,:,:]
                point=(self.halo_carra["Halo_Lat"].iloc[i],
                       self.halo_carra["Halo_Lon"].iloc[i])
                point_ds=self.get_data_from_closest_point(point,
                                                          lat_grid,lon_grid,
                                                          temp_data)
                point_ds=np.array(point_ds)
                if i==0:
                    interp_array=np.array(point_ds)                                                                                                
                else:
                    interp_array=np.vstack((interp_array,point_ds.T))
                #j+=1
                performance.updt(last_index,i)
            
            hmc_interp_point=pd.DataFrame(
                                data=interp_array,
                                index=self.halo_carra.index[0:last_index],
                                columns=self.upsampled_intc[var]["isobaricInhPa"],
                                dtype=float)
                
            self.carra_halo_hmc[var]         = hmc_interp_point
            self.save_interpolated_hmcs(var)
    
    def upsample_and_calc_interpolated_hmc_data(self,cut_to_AR=True):
        self.upsample_hmc_var_data()
        self.interpolate_hmc_on_halo()
    def run_calc_interp_hmc_data(self):
        self.merge_all_files_for_given_flight()
        self.calc_specific_humidity_from_relative_humidity()
        self.upsample_and_calc_interpolated_hmc_data()
        #self.save_interpolated_hmcs()
            
    def load_or_calc_interpolated_hmc_data(self):
        self.check_if_hmc_data_is_interpolated()
        if self.carra_hmc_is_interpolated:
            print("print CARRA hmc data is already interpolated")
            self.load_interp_hmc_data()
        else:
            print("CARRA hmc data has not been interpolated")
            self.run_calc_interp_hmc_data()

    def save_interpolated_hmcs(self,hmc):
        #for hmc in self.HMCs:
        interp_hmc_file=self.flight+"_"+self.ar_of_day+"_"+hmc+"_"+\
                self.date+".csv"
        if self.synthetic_flight:
                interp_hmc_file="Synthetic_"+interp_hmc_file
        self.carra_halo_hmc[hmc].to_csv(self.carra_path+interp_hmc_file)
        print("Flight track interpolated CARRA ",hmc,"saved as:",
              self.carra_lvls_path+interp_hmc_file)
        

###############################################################################
"""
## ICON
"""
class ICON_on_HALO(ICON):
    def __init__(self,cmpgn_cls,icon_var_list,halo_df,flight,date,
                 interpolated_hmp_file=None,interpolated_hmc_file=None,
                 ar_of_day=None,synthetic_icon=False,synthetic_icon_lat=0,
                 HMPs=["IWV","LWP","IWP"],HMCs=[],
                 synthetic_flight=False):
        
        self.ar_of_day=ar_of_day
        self.hydrometeor_icon_path=cmpgn_cls.campaign_path+\
                                        "/data/ICON_LEM_2KM/"
        self.interpolated_hmp_file=interpolated_hmp_file
        self.interpolated_hmc_file=interpolated_hmc_file
        self.icon_var_list=icon_var_list
        self.halo_df=halo_df
        self.flight=flight[0]
        self.date=date
        self.synthetic_icon=synthetic_icon
        self.synthetic_icon_lat=synthetic_icon_lat
        self.last_index=self.halo_df.shape[0]
        self.HMPs=HMPs
        self.HMCs=HMCs
        self.synthetic_flight=synthetic_flight
        
    def update_ICON_hydrometeor_data_path(self,hydro_path):
        self.hydrometeor_icon_path=hydro_path
    #%% Hydrometeorpaths    
    def interpolate_icon_hmp(self,last_index):
        """
        This function spatially interpolates the ICON Triangle Grid Data on the 
        HALO flight path using the Triangulation methods provided by matplotlib
    
        Parameters
        ----------
        halo_df : pd.DataFrame()
            dataframe with halo position which is needed.
            
        last_index : TYPE
            DESCRIPTION.
    
        Returns
        -------
        halo_icon : pd.DataFrame()
            dataframe with interpolated HMP, 
    
        """
        import Performance
        from matplotlib.tri import Triangulation, LinearTriInterpolator
        performance=Performance.performance()
        
        print("Interpolate the ICON Hydrometeors onto the Flight Track")
        print("Preprocessing")
        # Return data will be a dataframe
        self.halo_df.index=pd.DatetimeIndex(self.halo_df.index)
        halo_icon=pd.DataFrame(index=self.halo_df.index)
    
        # The first information are given from the aircraft position
        halo_icon["Minutesofday"]=self.halo_df["Minutesofday"]
        halo_icon["Halo_Lat"]=self.halo_df["latitude"]
        halo_icon["Halo_Lon"]=self.halo_df["longitude"]
        
        # Perform this interpolation only up to the last index defined, 
        # default is the length of HALO dataframe (so entire flight data)
        iterative_length=last_index
        icon_time=pd.DataFrame()
        icon_time["Hour"]=pd.DatetimeIndex(np.array(
            self.icon_upsampled_hmp["IWV"].time)).hour
        icon_time["Minutes"]=pd.DatetimeIndex(np.array(
            self.icon_upsampled_hmp["IWV"].time)).minute
        icon_time["Minutesofday"]=icon_time["Hour"]*60+icon_time["Minutes"]
        icon_simulations_minute_of_day=icon_time["Minutesofday"]
        del icon_time
        
        #Define pd.Series for each hydrometeor and water vapour
        hmp_interp_point=pd.Series(index=halo_icon.index[0:last_index],
                                   dtype=float)
    
        print("The hydrometeorpaths will be interpolated")
        for hmp in self.HMPs:
            print("Interpolate ",hmp)
            for i in range(iterative_length):
                relevant_index=icon_simulations_minute_of_day[\
                           icon_simulations_minute_of_day==\
                           halo_icon["Minutesofday"].iloc[i]].index[0]
                icon_grid_values=self.icon_upsampled_hmp[hmp][relevant_index,:]
        
                lon=np.array(np.rad2deg(icon_grid_values.clon[:]))
                lat=np.array(np.rad2deg(icon_grid_values.clat[:]))
                #Old distance
                deg_dist=pd.Series(
                    np.sqrt((lon-halo_icon["Halo_Lon"].iloc[i])**2+\
                            (lat-halo_icon["Halo_Lat"].iloc[i])**2))
                #New spatial distances
                destination=(halo_icon["Halo_Lat"].iloc[i],
                             halo_icon["Halo_Lon"].iloc[i])
            
                idx=deg_dist.sort_values().index[:30]
                hav_dist=pd.Series(data=np.nan,index=idx)
                relevant_origins=pd.Series(zip(lat[idx],lon[idx]),
                                           index=idx)
                for pos in range(len(relevant_origins)):
                    origin=relevant_origins.iloc[pos]
                    hav_dist.iloc[pos]=harvesine_distance(origin, destination)
            
                min_hav_dist=hav_dist.sort_values()
                lon3=lon[min_hav_dist.index[:6]]
                lat3=lat[min_hav_dist.index[:6]]
    
                icon_triangle=np.array(icon_grid_values[min_hav_dist.index[:6]])
                triangle=Triangulation(lon3,lat3)
            
                fz=LinearTriInterpolator(triangle,icon_triangle)
                value_inside=fz(halo_icon["Halo_Lon"].iloc[i],
                                halo_icon["Halo_Lat"].iloc[i])
            
                #print(value_inside)
                hmp_interp_point.iloc[i]=value_inside
                performance.updt(iterative_length,i)
                """
                # There is still an issue with the nearest points as no 
                harvestine distance is calculated. This has to be solved by 
                an improved distance calculation.
                
                plt.close()
                print(value_inside)
                plt.close()
                if np.isnan(np.array(value_inside)):
                    #    value_inside=icon_grid_values[idx[0]]
                    if i<60:
                        plt.scatter(lon3,lat3)
                        plt.scatter(halo_icon["Halo_Lon"].iloc[i],
                                    halo_icon["Halo_Lat"].iloc[i],
                                    s=value_inside)
                        plt.show()
                else:
                    sys.exit()
            
                """
            # write the interpolated data given as pd.Series onto the dataframe
            # to be returned
            interp_hmp_str="Interp_"+hmp
            halo_icon[interp_hmp_str]=hmp_interp_point
            self.halo_icon_hmp=halo_icon
            
    def load_interpolated_hmp(self,lat_changed=True):
        if not os.path.isfile(self.hydrometeor_icon_path+\
                              self.interpolated_hmp_file):
            print("ICON HMP data for cross-section ",
                  self.ar_of_day," is not yet calculated.")
            self.load_and_upsample_icon("Hydrometeor")
            if self.synthetic_icon:
                if self.synthetic_icon_lat is not None:
                    if not lat_changed:
                        self.halo_df["latitude"]=self.halo_df["latitude"]+\
                                                    self.synthetic_icon_lat
                    print("Changed Latitude of HALO Aircraft",
                          " for Synthetic Observations")
            else:
                print("Synthetic Flight pattern, lat already changed:",
                      lat_changed)
            self.interpolate_icon_hmp(self.last_index)
            
            #---------------------------------------------------------#
            #print(halo_icon_hmp)
            self.halo_icon_hmp.to_csv(path_or_buf=self.hydrometeor_icon_path+\
                                              self.interpolated_hmp_file,
                                              index=True)
            print("ICON-HMPs saved as:",self.hydrometeor_icon_path+\
                                        self.interpolated_hmp_file)
        else:
            print(self.interpolated_hmp_file,
                  " is already calculated and will be opened")
            self.halo_icon_hmp=pd.read_csv(self.hydrometeor_icon_path+\
                                           self.interpolated_hmp_file)
            self.halo_icon_hmp.index=pd.DatetimeIndex(
                                            self.halo_icon_hmp["Unnamed: 0"])
        return self.halo_icon_hmp
    
    #%% Hydrometeorcontents (HMCs)
    def interpolate_icon_3D_data(self,icon_q,var,upsample_time=None,
                                 geo_interpolation_type="triangle",
                                 save_interpolation_df=False):
        """
        This function interpolates the given vertical columns onto the HALO 
        flight track using similar approaches as in the 2D Interpolation
    
        Parameters
        ----------
        icon_q : xr.DataArray
            DESCRIPTION.
        var : str
            Variable to interpolate the data to be saved will named by this.
        upsample_time : str
            str containing the resolution to consider. Default is None due to 
            computation ressources
        geo_interpolation_type : str
            str defining the interpolation method for the geolocation. 
            Default is triangle using Triangulation from matplotlib, 
            second possibility is the
        last_index : int
            index defining the length of the halo time frames to include.
        save_interpolation_df : bool
            boolean defining if the interpolated data should be saved as df.
            Default is False.
        Returns
        -------
        q_interpolated
        """
        import Performance
        performance=Performance.performance()
        print("Interpolate vertical ICON profiles onto the HALO track ", 
              "at given HALO resolution.")
        
        if var=="Specific_Humidity":
            varname="qv"
        elif var=="U_Wind":
            varname="u"
        elif var=="V_Wind":
            varname="v"
        elif var=="Pressure":
            varname="pres"
        elif var=="Z_Height":
            varname="z_mc"
        elif var=="Ice_Content":
            varname="qi"
        elif var=="Snow_Content":
            varname="qs"
        elif var=="Liquid_Content":
            varname="qc"
        elif var=="Rain_Content":
            varname="qr"
        else:
            raise Exception("This is a wrong var name given")
        
        icon_q=icon_q[varname]
        if not var=="Z_Height":
            if not (upsample_time is None):
                print("Interpolate ICON in time to ",upsample_time)
                icon_q=icon_q.resample(time=upsample_time).interpolate("linear")
            icon_initial_hour=self.icon_var_list[0]
        
            icon_q=icon_initial_hour.adapt_icon_time_index(icon_q,
                                                       self.date,
                                                       self.flight)
        print("Interpolate the ICON ",var," onto the Flight Track")
        print("Preprocessing")
        self.halo_df.index=pd.DatetimeIndex(self.halo_df.index)
        # Return data will be a dataframe
        halo_icon=pd.DataFrame(index=self.halo_df.index)
    
        # The first information are given from the aircraft position
        halo_icon["Minutesofday"]=self.halo_df["Minutesofday"]
        halo_icon["Halo_Lat"]=self.halo_df["latitude"]
        halo_icon["Halo_Lon"]=self.halo_df["longitude"]
       
        
        # Perform this interpolation only up to the last index defined, 
        # default is the length of HALO dataframe (so entire flight data)
        iterative_length=halo_icon.shape[0]
        if not var=="Z_Height":
            icon_time=pd.DataFrame()
            icon_time["Hour"]=pd.DatetimeIndex(np.array(
                icon_q.time)).hour
            icon_time["Minutes"]=pd.DatetimeIndex(np.array(
                                    icon_q.time)).minute
            icon_time["Minutesofday"]=icon_time["Hour"]*60+icon_time["Minutes"]
            icon_simulations_minute_of_day=icon_time["Minutesofday"]
            del icon_time
        
        
        #Define pd.Series for each hydrometeor and water vapour
        q_interp_point=pd.DataFrame(index=halo_icon.index[0:self.last_index],
                                    columns=np.arange(len(icon_q.height)),
                                    dtype=float)
        
        #temporary_q=temporary_q.to_dataarray()
        temp_clat=np.rad2deg(pd.Series(icon_q.clat))
                        
        cutted_clat=temp_clat[temp_clat.between(
                                        self.halo_df["latitude"].min()-2,
                                        self.halo_df["latitude"].max()+2)]
        temp_clon=np.rad2deg(pd.Series(icon_q.clon))
        temp_clon=temp_clon.loc[cutted_clat.index]
                        
        cutted_clon=temp_clon[temp_clon.between(
                                        self.halo_df["longitude"].min()-2,
                                        self.halo_df["longitude"].max()+2)]
                        
        icon_q=icon_q.isel(ncells=cutted_clon.index)
                        
        #if not var =="Z_Height":
        #                    
        #    cutted_q=temporary_q.copy()
        #    cutted_q=icon_hour.adapt_icon_time_index(\
        #                                         temporary_q,
        #                                         self.date, self.flight)#.astype("float32")

        print("Geo-interpolate")
        print("Computing ", var)
        if not var=="Z_Height":
            icon_q.data=icon_q.data.compute()
        print(var,"computed")
        
        for i in range(iterative_length):
            if not var=="Z_Height":
                closest_min=abs(icon_simulations_minute_of_day-\
                                halo_icon["Minutesofday"][i])
                relevant_index=closest_min.argsort()[0]
                icon_grid_values=icon_q[relevant_index,:,:]
            else:
                if len(icon_q.shape)==3:
                    icon_grid_values=icon_q[0,:,:]
                else:
                    icon_grid_values=icon_q[:,:]
            lon=np.array(np.rad2deg(icon_grid_values.clon[:]))
            lat=np.array(np.rad2deg(icon_grid_values.clat[:]))
            
            #Old distance
            deg_dist=pd.Series(np.sqrt((lon-halo_icon["Halo_Lon"].iloc[i])**2+\
                                       (lat-halo_icon["Halo_Lat"].iloc[i])**2))
            #New spatial distances
            destination=(halo_icon["Halo_Lat"].iloc[i],
                         halo_icon["Halo_Lon"].iloc[i])
            
            idx=deg_dist.sort_values().index[:20]
            hav_dist=pd.Series(data=np.nan,index=idx)
            relevant_origins=pd.Series(zip(lat[idx],lon[idx]),
                                   index=idx)
        
            for pos in range(len(relevant_origins)):
                origin=relevant_origins.iloc[pos]
                hav_dist.iloc[pos]=harvesine_distance(origin, destination)
    
            min_hav_dist=hav_dist.sort_values()
            lon3=lon[min_hav_dist.index[:6]]
            lat3=lat[min_hav_dist.index[:6]]
            
            
            if geo_interpolation_type=="triangle":
                from matplotlib.tri import Triangulation, LinearTriInterpolator
                icon_triangle=np.array(
                                    icon_grid_values[:,min_hav_dist.index[:6]])
                
                triangle=Triangulation(lon3,lat3)
                
                values_inside=np.zeros(icon_grid_values.shape[0])
                for height in range(icon_grid_values.shape[0]):    
                    
                    fz=LinearTriInterpolator(triangle,icon_triangle[height,:])
                    values_inside[height]=fz(halo_icon["Halo_Lon"].iloc[i],
                                             halo_icon["Halo_Lat"].iloc[i])
            
            q_interp_point.iloc[i,:]=values_inside
            performance.updt(iterative_length,i)
        
        # Save interpolated df
        q_file_name=self.flight+"_"+self.ar_of_day+"_"+\
                            var+"_interpolated_profile.csv"
        if self.synthetic_flight:
            q_file_name="Synthetic_"+q_file_name
        if save_interpolation_df:
            q_interp_point.to_csv(
                            path_or_buf=self.hydrometeor_icon_path+
                            q_file_name,
                            index=True)
            print("ICON Flight-Track Data saved as: ",
                  self.hydrometeor_icon_path+q_file_name)
        
        return q_interp_point
    
    def load_hwc(self,with_hydrometeors=False):
            interp_icon_q_file="Liquid_Content_interpolated_profile.csv"
            
            if self.ar_of_day is not None:
                interp_icon_q_file=self.flight+"_"+self.ar_of_day+\
                                    "_"+interp_icon_q_file
            if self.synthetic_flight:
                interp_icon_q_file="Synthetic_"+interp_icon_q_file
            
            if not with_hydrometeors:
                variables=[#"Pressure",]
                       "Specific_Humidity"]#,
                       #"U_Wind"]
                       #"V_Wind",
                       #"Z_Height"]
                halo_icon_keys=[#"pres",]
                                "qv"]#,
                                #"u",]
                                #"v",
                                #"Z_Height"]
                dataset_var=[#"pres",]#,
                             "qv"]#,
                             #"u"]
                             #"v",
                             #"z_mc"]
            else:
                variables=["Ice_Content","Snow_Content",
                           "Liquid_Content",
                           "Rain_Content"]#,
                           #"Pressure","Specific_Humidity",
                           #"U_Wind","V_Wind","Z_Height"]
                
                
                halo_icon_keys=["qi","qs",
                                "qc","qr"]
                #,"p","qv","u","v","Z_Height"]
                
                dataset_var=["qi","qs",
                             "qc","qr"]
                #,"pres","qv","u","v","z_mc"]
                
            #variables=["Specific_Humidity"]
            #halo_icon_keys=["q"]#"u","v","Z_Height"]
            
            #variables=["U_Wind","V_Wind","Z_Height"]
            #halo_icon_keys=["u","v","Z_Height"]
            
            #Preallocate
            self.halo_icon_hmc={}
            icon_da_list=[]
            print("File of interest:",self.hydrometeor_icon_path+\
                                  interp_icon_q_file)
            if not os.path.isfile(self.hydrometeor_icon_path+\
                                  interp_icon_q_file):
                
                print("Calculate and interpolate ICON-HMCs onto flight track")
                k=0
                for var in variables:
                    print(var)
                    hour=0
                    icon_q_files=var+"_ICON_"+self.flight+"_*UTC.nc"
                    if not var=="Z_Height":
                        icon_ds=xr.open_mfdataset(self.hydrometeor_icon_path+\
                                              icon_q_files, concat_dim="time")
                    else:
                        import glob
                        z_files_list=glob.glob(self.hydrometeor_icon_path+\
                                              icon_q_files)
                        icon_ds=xr.open_dataset(z_files_list[0])
                    """ old to be stored
                    for icon_hour in self.icon_var_list:
                        utc_time=str(int(icon_hour.start_hour))
                        icon_q_file=var+"_ICON_"+self.flight+"_"+\
                                        utc_time+"UTC.nc"
                        temporary_q=xr.open_dataset(
                                            self.hydrometeor_icon_path+\
                                            icon_q_file)
                        #temporary_q=temporary_q.to_dataarray()
                        temp_clat=np.rad2deg(pd.Series(temporary_q[\
                                                        dataset_var[k]].clat))
                        
                        cutted_clat=temp_clat[temp_clat.between(
                                        self.halo_df["latitude"].min()-2,
                                        self.halo_df["latitude"].max()+2)]
                        temp_clon=np.rad2deg(pd.Series(temporary_q[\
                                                        dataset_var[k]].clon))
                        temp_clon=temp_clon.loc[cutted_clat.index]
                        
                        cutted_clon=temp_clon[temp_clon.between(
                                        self.halo_df["longitude"].min()-2,
                                        self.halo_df["longitude"].max()+2)]
                        
                        temporary_q=temporary_q[dataset_var[k]].isel(
                                ncells=cutted_clon.index)
                        
                        if not var =="Z_Height":
                            
                            cutted_q=temporary_q.copy()
                            cutted_q=icon_hour.adapt_icon_time_index(\
                                                 temporary_q,
                                                 self.date, self.flight)#.astype("float32")

                            icon_da_list.append(cutted_q)
                        else:
                            icon_da_list=temporary_q.copy()
                        hour+=1
                       """ 
                    #del temporary_q
                    
                    #if len(icon_da_list)>1:
                    #    icon_q=xr.concat(icon_da_list,dim="time")
                    #else:
                    #    icon_q=icon_da_list[0]
                    #icon_q.values=icon_q.values.astype("float32")    
                    #if self.synthetic_icon:
                    #    if not self.lat_changed:
                    #        self.halo_df["latitude"]=self.halo_df["latitude"]+\
                    #                                    self.synthetic_icon_lat
                    #        print("Changed Latitude of HALO Aircraft",
                    #              "for Synthetic Observations")
                    #        self.lat_changed=True
                    interp_q_icon = self.interpolate_icon_3D_data(icon_ds,var,
                                            upsample_time=None,
                                            geo_interpolation_type="triangle",
                                            save_interpolation_df=True)
                    icon_var=halo_icon_keys[k]
                    if icon_var=="qv":
                        icon_var="q"
                    self.halo_icon_hmc[icon_var]=interp_q_icon
                    self.halo_icon_hmc[icon_var].index=pd.DatetimeIndex(
                                            self.halo_icon_hmc[icon_var].index)
                    k+=1
                
                self.interp_icon_ivt_file=self.flight+"_"+self.ar_of_day+"_"+\
                                        "ICON_Interpolated_IVT.csv"
                
                if not os.path.isfile(self.hydrometeor_icon_path+\
                                      self.interp_icon_ivt_file):
                    self.icon_ivt = self.calc_interp_IVT()
            
            #-----------------------------------------------------------------#        
            else:
                #if self.synthetic_icon:
                #    self.hydrometeor_icon_path=self.hydrometeor_icon_path+\
                #                "Latitude_"+str(self.synthetic_icon_lat)+"/"
                self.halo_df["latitude"]=self.halo_df["latitude"]+\
                                            self.synthetic_icon_lat
                print("Changed Latitude of HALO Aircraft for Synthetic Observations")
                #lat_changed=True
                if not with_hydrometeors:
                    halo_icon_keys=["p","q","u","v","Z_Height"]
                    csv_icon_keys=["Pressure","Specific_Humidity",
                               "U_Wind","V_Wind","Z_Height"]
                else:
                    halo_icon_keys=["p","q","u","v","Z_Height","qi","qs","qr","qc"]
                    csv_icon_keys=["Pressure","Specific_Humidity",
                               "U_Wind","V_Wind","Z_Height",
                               "Ice_Content","Snow_Content",
                               "Rain_Content","Liquid_Content"]
                k=0
                
                for halo_key in halo_icon_keys:
                    interp_icon_key_file=self.flight+"_"+self.ar_of_day+"_"+\
                        csv_icon_keys[k]+"_interpolated_profile.csv"
                    if self.synthetic_flight:
                        interp_icon_key_file="Synthetic_"+interp_icon_key_file
                    self.halo_icon_hmc[halo_key] = pd.read_csv(
                                                self.hydrometeor_icon_path+\
                                                interp_icon_key_file,
                                                index_col=0)
                    self.halo_icon_hmc[halo_key].index=pd.DatetimeIndex(
                                            self.halo_icon_hmc[halo_key].index)
                    k+=1
                
                # IVT    
                self.interp_icon_ivt_file=self.flight+"_"+self.ar_of_day+"_"+\
                                        "ICON_Interpolated_IVT.csv"
                if self.synthetic_flight:
                    self.interp_icon_ivt_file="Synthetic_"+self.interp_icon_ivt_file
                if not os.path.isfile(self.hydrometeor_icon_path+\
                                      self.interp_icon_ivt_file):
                    self.icon_ivt = self.calc_interp_IVT()
                else:
                    self.icon_ivt=pd.read_csv(self.hydrometeor_icon_path+\
                                              self.interp_icon_ivt_file)
                    self.icon_ivt.index=pd.DatetimeIndex(
                                            self.icon_ivt["Unnamed: 0"])
                    del self.icon_ivt["Unnamed: 0"]
            self.halo_icon_hmc["IVT"]=self.icon_ivt
            return self.halo_icon_hmc     
    
    def calc_interp_IVT(self,save_ivt=True):
        """
        
    
        Parameters
        ----------
        save_file : str
            string defining file path and file name
            
        Returns
        -------
        icon_ivt : pd.DataFrame()
            interpolated IVT values calculated from given variables.
        """
        import Performance
        performance=Performance.performance()
        # Start IVT Calculations
        icon_ivt = pd.DataFrame(data=np.nan, 
                                index=self.halo_icon_hmc["q"].index,
                                columns=["IWV_calc","IVT","IVT_u","IVT_v"])
        print("Calculate IVT from ICON")
        for ts in range(icon_ivt.shape[0]):
            q_loc= self.halo_icon_hmc["q"].iloc[ts,:].dropna()
            q_loc= q_loc#.sort_index()
            g= 9.81
            u_loc=self.halo_icon_hmc["u"].iloc[ts,:]
            u_loc=u_loc.loc[q_loc.index]
            v_loc=self.halo_icon_hmc["v"].iloc[ts,:]
            v_loc=v_loc.loc[q_loc.index]
            qu=q_loc*u_loc
            qv=q_loc*v_loc
            qu=qu.dropna()
            qv=qv.dropna()
            pres_index=self.halo_icon_hmc["p"].iloc[ts,:]
            pres_index=pres_index.loc[q_loc.index]
            qu.index=pres_index
            qv.index=pres_index
            try:
                icon_ivt["IWV_calc"].iloc[ts]= 1/g*np.trapz(q_loc,x=pres_index)
            except:
                icon_ivt["IWV_calc"].iloc[ts]=np.nan
            try:
                icon_ivt["IVT_u"].iloc[ts] = 1/g*np.trapz(qu,x=qu.index)
            except:
                icon_ivt["IVT_u"].iloc[ts]=np.nan
            
            try:
                icon_ivt["IVT_v"].iloc[ts] = 1/g*np.trapz(qv,x=qv.index)
            except:
                icon_ivt["IVT_v"].iloc[ts] = np.nan
            
            icon_ivt["IVT"].iloc[ts]   = np.sqrt(
                                                icon_ivt["IVT_u"].iloc[ts]**2+\
                                                icon_ivt["IVT_v"].iloc[ts]**2)
            
            performance.updt(icon_ivt.shape[0], ts)
        
        if save_ivt:
            save_file=self.flight+"_"+self.ar_of_day+"_"+\
                        "ICON_Interpolated_IVT.csv"
            if self.synthetic_flight:
                save_file="Synthetic_"+save_file
            icon_ivt.to_csv(path_or_buf=self.hydrometeor_icon_path+save_file,
                            index=True)
            print("ICON IVT saved as: ",self.hydrometeor_icon_path+save_file)
        return icon_ivt
