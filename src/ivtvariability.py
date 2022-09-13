# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:42:28 2021

@author: u300737
"""
import os
import sys
import logging 

import data_config

import xarray as xr
import pandas as pd
import numpy as np
import scipy.integrate as scint

import seaborn as sns
import matplotlib 

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch

import flightcampaign

import gridonhalo as grid_halo
import ICON

def calc_halo_delta_distances(halo_df):
        #if sounding_frequency=="standard":
            #    halo_df=halo_df.loc[sonde_dict["Pres"].index]
            #halo_df["Timedelta"]=pd.to_datetime(halo_df['Unnamed: 0']).diff()
        halo_df["Timedelta"]=halo_df.index.to_series().diff()
        #key_var=[*sonde_dict.keys()][0]
        #halo_df=halo_df.loc[sonde_dict["IVT"].index[0]:\
        #                                  sonde_dict["IVT"].index[-1]]
        halo_df["Seconds"]=halo_df["Timedelta"].dt.seconds
        halo_df["delta_distance"]=halo_df["Seconds"]*\
                                            halo_df["groundspeed"]
        halo_df["cumsum_distance"]=halo_df["delta_distance"].cumsum()
        return halo_df
    
def preprocess_halo_df(halo_path,flight,synthetic_icon_lat,synthetic_icon_lon,
                       sounding_frequency,sonde_dict,ar_of_day):
    halo_df=pd.read_csv(halo_path+"HALO_Aircraft_"+flight+".csv")
    if synthetic_icon_lat!=None:
        halo_df["latitude"]=halo_df["latitude"]+synthetic_icon_lat
        
        halo_df["longitude"]=halo_df["longitude"]+synthetic_icon_lon
        print("HALO lat position changed by:", 
              str(synthetic_icon_lat), " Degrees")
    
    halo_df.index=pd.DatetimeIndex(halo_df["Unnamed: 0"])
    # store a copy as back up
    
    # halo_df can be cutted to dropsondes releases, 
    # but not copy_halo representing the entire flight data
    copy_halo_df=halo_df.copy()
    import AR
    if synthetic_icon_lat==0:
        ARs=AR.Atmospheric_Rivers.look_up_AR_cross_sections(
                        campaign="NAWDEX")
        halo_df=halo_df.loc[ARs[flight][ar_of_day]["start"]:\
                               ARs[flight][ar_of_day]["end"]]
    
    else:
        if synthetic_icon_lat==-7:
            
            ARs=AR.Atmospheric_Rivers.look_up_AR_cross_sections(
                        campaign="NAWDEX")
            halo_df=halo_df.loc[ARs[flight]["AR3"]["start"]:\
                               ARs[flight]["AR3"]["end"]]
            halo_df=halo_df.loc["2016-10-13 12:10":"2016-10-13 13:30"]
        elif synthetic_icon_lat==1.8:
            ARs=AR.Atmospheric_Rivers.look_up_AR_cross_sections(
                        campaign="NAWDEX")
            halo_df=halo_df.loc[ARs[flight]["AR3"]["start"]:\
                               ARs[flight]["AR3"]["end"]]
            halo_df=halo_df.loc["2016-10-13 12:12":"2016-10-13 13:10"]
        
def create_synthetic_sondes(aircraft_df,aircraft_var_df,
                            sonde_dict,hmps_used=False,no_of_sondes=10):
    """
    

    Parameters
    ----------
    aircraft_df : pd.DataFrame
        aircraft data form specific section ("internal, inflow or outflow").
    aircraft_var_data : pd.DataFrame/dict
        the aircraft interpolated meteorological var data. Can be either dict
        or pd.DataFrame, depending on models used.
    sonde_dict : dict
        sonde data as dictionary, should be empty at the beginning.
    no_of_sondes : int, optional
        Number of sondes to place equidistantly along flight leg. Default is 10.

    Returns
    -------
    #halo_df : TYPE
    #    DESCRIPTION.
    copy_aircraft_df : TYPE
        DESCRIPTION.
    sonde_dict : TYPE
        DESCRIPTION.

    """
    aircraft_df=aircraft_df.groupby(level=0).first()
    aircraft_var_data=aircraft_var_df.copy()
                
    if not hmps_used:
        #aircraft_df=aircraft_df.drop_duplicates(keep="first")
        #aircraft_var_data=aircraft_var_df.copy()
        if not aircraft_var_data["name"]=="CARRA":
            aircraft_var_data["p"]=pd.DataFrame(data=np.tile(
                    np.array(aircraft_var_data["IWC"].columns[:].astype(float)),
                    (aircraft_var_data["IWC"].shape[0],1)),
                    columns=[aircraft_var_data["IWC"].columns[:]],
                    index=aircraft_var_data["IWC"].index)
        else:
            aircraft_var_data["p"]=pd.DataFrame(data=np.tile(
                    np.array(aircraft_var_data["u"].columns[:].astype(float)),
                    (aircraft_var_data["u"].shape[0],1)),
                    columns=[aircraft_var_data["u"].columns[:]],
                    index=aircraft_var_data["u"].index)
            
        try:
            aircraft_var_data["q"]=aircraft_var_data["q"].groupby(level=0).last()    
        except:
            aircraft_var_data["q"]=aircraft_var_data["specific_humidity"].groupby(level=0).last()
        aircraft_var_data["p"]=aircraft_var_data["p"].groupby(level=0).last()
        
        aircraft_var_data["u"]=aircraft_var_data["u"].groupby(level=0).last()
        
        if not "wind" in aircraft_var_data.keys():
            aircraft_var_data["wind"]=np.sqrt(aircraft_var_data["u"]**2+\
                                          aircraft_var_data["v"]**2)
        aircraft_var_data["wind"]=aircraft_var_data["wind"].groupby(level=0).last()
    
        sonde_index=pd.date_range(start=aircraft_df.index[0],
                                  end=aircraft_df.index[-1],
                                  periods=no_of_sondes)
        sonde_index=sonde_index.round("s")
        # Get moisture transport relevant variables
        sonde_dict["Pres"]=aircraft_var_data["p"].reindex(index=sonde_index).astype(float)    
        #sonde_dict["Pres"].index=sonde_dict["Pres"].index.round("s")
        
        sonde_dict["q"]=aircraft_var_data["q"].reindex(index=sonde_index)    
        #sonde_dict["q"].index=sonde_dict["q"].index.round("s")
    
        sonde_dict["Wspeed"]=aircraft_var_data["wind"].reindex(
                                    index=sonde_index)    
        #sonde_dict["Wspeed"].index=sonde_dict["Wspeed"].index.round("s")
        if "IVT" in aircraft_var_data.keys():    
            sonde_dict["IVT"]=aircraft_var_data["IVT"].reindex(index=sonde_index)
    else:
        #del aircraft_var_data["Unnamed: 0"]
        index_to_use=pd.date_range(start=aircraft_df.index[0],
                                   end=aircraft_df.index[-1],
                                   periods=no_of_sondes)
        index_to_use=pd.DatetimeIndex(index_to_use)
        index_to_use=index_to_use.round("s")
        aircraft_var_data = aircraft_var_data[\
                                ~aircraft_var_data.index.duplicated()]
        #aircraft_var_data=aircraft_var_data.drop_duplicates("index").keep("first")
        if not aircraft_var_df.name=="CARRA":
            sonde_dict["IVT"]=aircraft_var_data["Interp_IVT"].reindex(
                                index_to_use)
            sonde_dict["IWV"]=aircraft_var_data["Interp_IWV"].reindex(
                                index_to_use)
        else:
            sonde_dict["IVT"]=aircraft_var_data["highres_Interp_IVT"].reindex(
                                index_to_use)
            sonde_dict["IWV"]=aircraft_var_data["highres_Interp_IWV"].reindex(
                                index_to_use)
            
        ### needs to be calculated:
        #sonde_dict["IVT"].index=sonde_dict["IVT"].index.round("s")
    
    # Adapt flight data dataframe for further calculations
    copy_aircraft_df=aircraft_df.copy()
    #del copy_halo_df["Unnamed: 0"]
    return copy_aircraft_df,sonde_dict

    #del halo_df["Unnamed: 0"]

    #copy_halo_df=copy_halo_df.loc[sonde_dict["Pres"].index[0]:\
    #                              sonde_dict["Pres"].index[-1]]
    
    #copy_halo_df["Seconds"]=copy_halo_df["Timedelta"].dt.seconds
    #copy_halo_df["delta_Distance"]=copy_halo_df["Seconds"]*copy_halo_df["groundspeed"]
    #copy_halo_df["cumsum_distance"]=copy_halo_df["delta_Distance"].cumsum()
        
    
class ICON_IVT_Logger():
    def __init__(self,log_file_path=os.getcwd(),file_name="ICON_IVT_log_file.txt"):
        self.log_file_path=log_file_path
        self.file_name=log_file_path+file_name
        # Define plotting style
        # For now, this is typhon but  it can be adapted to an explicitly 
        # defined style sheet.

    def create_plot_logging_file(self):
        """
        Creates the logging file to list the performed steps as INFO.
        Parameters
        ----------
        
        Returns
        -------
        None.

        
        """
        try:
            os.remove(self.file_name)
        except OSError:
            pass
        self.icon_ivt_logger=logging.getLogger(__name__)    
        self.icon_ivt_logger.setLevel(logging.INFO)
        formatter=logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        log_handler=logging.FileHandler(self.file_name,'w')
        log_handler.setFormatter(formatter)
        self.icon_ivt_logger.addHandler(log_handler)
        self.icon_ivt_logger.propagate=False
    
    def close_plotting_logger(self,logger_obj):
        handlers=logger_obj.handlers[:]
        logger_obj.info("Close Logging File")
        for handler in handlers:
            handler.close()
            self.icon_ivt_logger.removeHandler(handler)

class IVT_variability():
    def __init__(self,grid_dict_hmp,grid_dict_hmc,sonde_dict,
                 grid_sounding_profiles,sounding_frequency,
                 halo,plot_path,ar_of_day,
                 flight,ivt_logger):
        if isinstance(grid_dict_hmp,pd.DataFrame):
            self.grid_data_name=grid_dict_hmp.name
            self.grid_dict=grid_dict_hmp
        else:
            if isinstance(grid_dict_hmc,pd.DataFrame) or isinstance(grid_dict_hmc,dict):
                self.grid_data_name=grid_dict_hmc["name"]
                self.grid_dict=grid_dict_hmc
        
        self.sonde_dict=sonde_dict
        
        self.grid_ivt={}
        
        self.sonde_ivt={}
        
        self.grid_sounding_profiles=grid_sounding_profiles
        self.halo=halo
        self.sounding_frequency=sounding_frequency
        self.plot_path=plot_path
        self.ar_of_day=ar_of_day
        self.flight=flight
        self.ivt_logger=ivt_logger

    def gaussian(self,x,amplitude,mean,sigma,offset):
            return amplitude*np.exp(-((x-mean)/(4*sigma))**2)+offset
    
    def skewed_gaussian(self,x,amplitude,mean,sigma,offset,gamma):
        from scipy import special
        return amplitude*np.exp(-((x-mean)/(4*sigma))**2)*\
            (1-special.erf(gamma*(x-mean)/np.sqrt(2)))+offset
    
    def gaussian_fit_IVT(self,x,data):
        from scipy import optimize,special
        gaussian_guess=np.array([data.max()-data.min(),x.iloc[data.argmax()]/1000,
                                        50,data.min()])
        skewed_gaussian_guess=np.array([data.max()-data.min(),
                                        x.iloc[data.argmax()]/1000,
                                        20,data.min(),-1.3])
        
        popt,_ = optimize.curve_fit(self.gaussian,x.values/1000,data.values,
                                    p0=gaussian_guess)
        popt_skewed,_= optimize.curve_fit(self.skewed_gaussian,x.values/1000,
                                          data.values,p0=skewed_gaussian_guess)
        return popt,popt_skewed
    
    def add_vertical_vars_to_ivt_dict(self,use_sondes=False):
        
        if not use_sondes:
            if not isinstance(self.grid_ivt,dict) or self.grid_ivt=={}:
               for var_key in self.grid_dict.keys():
                   if not var_key=="name": 
                       self.grid_ivt[var_key]=self.grid_dict[var_key].copy()
        else:
            if not isinstance(self.sonde_ivt,dict) or self.sonde_ivt=={}:
                if not self.grid_sounding_profiles:
                    for var_key in self.sonde_dict.keys():
                        self.sonde_ivt[var_key]=self.sonde_dict[var_key].copy().\
                                                    iloc[:,1:]
                else:
                    for var_key in self.sonde_dict.keys():
                        if not var_key=="name":
                            self.sonde_ivt[var_key]=self.sonde_dict[var_key].copy()
    
    def calc_single_tivt_from_ivt(self,grid_used=True,grid_sounding=False,
                                standard=True,sounding_frequency="standard"):
        
        self.halo["cumsum_distance"]=self.halo["cumsum_distance"].fillna(0)
        # Choose coherent distances indices depending on continuous or
        # sounding on grid field
        
        
        # Grid data is used to calculate TIVT
        if grid_used:
            # Use continuous sounding
            if not grid_sounding:
                grid_ivt_array=np.array(self.grid_ivt.rolling(2).mean().iloc[1:])
                delta_distance=np.array(self.halo["delta_distance"].reindex(
                                                    self.grid_ivt.index).iloc[1:])
                product_array=grid_ivt_array*delta_distance
                self.tivt_grid=product_array.sum()
            # Create synthetic soundings
            else: 
                grid_ivt_array=np.array(self.sonde_ivt.rolling(2).mean().iloc[1:])
                delta_distance=self.halo["cumsum_distance"].loc[\
                                            self.sonde_ivt.index].diff()[1:]
                self.tivt_sonde=(grid_ivt_array*delta_distance).sum()
        
        # Real soundings are used to calculate TIVT
        else:
            if sounding_frequency=="standard":
                sonde_ivt_array=np.array(
                            self.sonde_ivt.rolling(2).mean().iloc[1:,0])
                delta_distance=np.array(
                        self.halo["delta_Distance"].loc[\
                                                self.sonde_ivt.index].iloc[1:])
            elif sounding_frequency=="Upsampled":
                sonde_ivt_array=np.array(self.sonde_ivt.iloc[:,0])
                delta_distance=np.array(self.halo["delta_Distance"].loc[\
                                                            self.sonde_ivt.index])
            else:
                pass
            self.tivt_sonde=(sonde_ivt_array*delta_distance).sum()
    
    def calc_and_compare_TIVT_sondes_deleted_and_grid(self,number_delete_sondes):
        # Sondes_to_use is a dataframe containing the index iloc commands for the
        # sondes that will be used in the further analysis to calculate the
        # TIVT and its variability for different sounding positions.
        
        sondes_to_use=pd.DataFrame(data=-999,
                                   index=range(self.sonde_ivt.shape[0]-2),
                                   columns=range(self.sonde_ivt.shape[0]-\
                                                 number_delete_sondes))#np.array()
        index=0
        sondes_to_use.iloc[:,0]=0
        sondes_to_use.iloc[:,-1]=self.sonde_ivt.shape[0]-1
        print("Calculate TIVT and compare for deleted sondes and grid")
        #---------------------------------------------------------------------#
        # Fill sondes_to_use df with the index position values
        print("Sonde Bootstraping")
        for i in np.arange(1,self.sonde_ivt.shape[0]-1):
            if number_delete_sondes==1:
                used_sondes=np.arange(0,self.sonde_ivt.shape[0])
                used_sondes=np.delete(used_sondes,i)
                sondes_to_use.iloc[index,:]=used_sondes
            else:
                used_sondes=np.empty((1,self.sonde_ivt.shape[0]-\
                                      number_delete_sondes))
                used_sondes[0,1:-1]=-999
                used_sondes[0,0]=0
                used_sondes[0,-1]=self.sonde_ivt.shape[0]-1
                used_sondes[0,1:-1]=np.random.choice(
                                        np.arange(1,self.sonde_ivt.shape[0]-2),
                                        self.sonde_ivt.shape[0]-2-\
                                            number_delete_sondes, 
                                        replace=False)
                used_sondes=np.sort(used_sondes)
                sondes_to_use.iloc[index,1:-1]=used_sondes[0,1:-1]
                    
                while any(sondes_to_use.iloc[0:i,:].duplicated()):
                    used_sondes[0,1:-1]=np.random.choice(
                                        np.arange(1,self.sonde_ivt.shape[0]-1),
                                         self.sonde_ivt.shape[0]-2-\
                                             number_delete_sondes, 
                                         replace=False)
                    used_sondes=np.sort(used_sondes)
                    sondes_to_use.iloc[index,1:-1]=used_sondes[0,1:-1]
                    
            index+=1
        #---------------------------------------------------------------------#
        # index locator dataframe is created so now TIVT calculation can start
        sondes_to_use["TIVT"]=np.nan # added a new column comprising the therefrom
                                     # calculated TIVT values.
        # Check if halo has already distance information
        if not "cumsum_distance" in self.halo.columns:
            self.halo=calc_halo_delta_distances(self.halo)                             
        
        # Select sonde cases according to sondes_to_use dataframe values
        print("calc respective TIVT for samples")
        for i in range(sondes_to_use.shape[0]):
            
            #if i==0:
            selected_sonde_ivt=self.sonde_complete.iloc[\
                                sondes_to_use.iloc[i].values[:-1].astype(int).tolist()]
            #else:
            #    selected_sonde_ivt=self.sonde_complete.iloc[\
            #                    sondes_to_use.iloc[i].values[:-1].astype(int).tolist()]
            self.sonde_ivt=selected_sonde_ivt
            synthetic_sondes=False
            if self.flight.startswith("S"):
                synthetic_sondes=True
            self.calc_single_tivt_from_ivt(grid_used=synthetic_sondes,
                                           grid_sounding=True,
                                           standard=True,
                                           sounding_frequency="standard")
            sondes_to_use["TIVT"].iloc[i]=self.tivt_sonde
        return sondes_to_use      
    # def calc_and_compare_tivt_sondes_and_icon(self,do_icon=True,
    #                                                 delete_sondes=None,
    #                                                 synthetic=False):
    #     self.do_icon=do_icon
    #     if delete_sondes is not None:
    #         sondes_to_use=pd.DataFrame(data=-999,
    #                                    index=range(9),
    #                                    columns=range(self.sonde_ivt.shape[0]-\
    #                                                  delete_sondes))
    #     index=0
    #     sondes_to_use.iloc[:,0]=0
    #     sondes_to_use.iloc[:,-1]=self.sonde_ivt.shape[0]-1
    #     for i in np.arange(1,10):
    #         if delete_sondes==1:
    #             used_sondes=np.arange(0,11)
    #             used_sondes=np.delete(used_sondes,i)
    #             sondes_to_use.iloc[index,:]=used_sondes
    #         else:
    #             used_sondes=np.empty((1,self.sonde_ivt.shape[0]-delete_sondes))
    #             used_sondes[0,1:-1]=-999
    #             used_sondes[0,0]=0
    #             used_sondes[0,-1]=self.sonde_ivt.shape[0]-1
    #             used_sondes[0,1:-1]=np.random.choice(
    #                                     np.arange(1,
    #                                               self.sonde_ivt.shape[0]-1),
    #                                              self.sonde_ivt.shape[0]-2-\
    #                                                  delete_sondes, 
    #                                      replace=False)
    #             used_sondes=np.sort(used_sondes)
    #             sondes_to_use.iloc[index,1:-1]=used_sondes[0,1:-1]
                
    #             # Bootstrapping without taking same combinations
    #             while any(sondes_to_use.iloc[0:i,:].duplicated()):
    #                 used_sondes[0,1:-1]=np.random.choice(
    #                     np.arange(1,self.sonde_ivt.shape[0]-1),
    #                                 self.sonde_ivt.shape[0]-2-delete_sondes, 
    #                                 replace=False)
    #                 used_sondes=np.sort(used_sondes)
    #                 sondes_to_use.iloc[index,1:-1]=used_sondes[0,1:-1]
                    
    #         index+=1
    #     #sondes_to_use["TIVT"]=np.nan
        
    #     self.halo["cumsum_distance"]=self.halo["delta_Distance"].cumsum()
    #     self.halo["cumsum_distance"].iloc[0]=0.0
    
    #     #if (self.sounding_frequency=="Upsampled") and \
    #     #    not (self.grid_sounding_profiles): 
        
    #     ## Check if length of timeseries is the same
    #     if not self.icon_ivt.shape[0]==self.sonde_ivt.shape[0]:
    #         if self.icon_ivt.shape[0]>self.sonde_ivt.shape[0]:
    #             self.icon_ivt=self.icon_ivt.loc[self.sonde_ivt.index]
    #         elif self.icon_ivt.shape[0]<self.sonde_ivt.shape[0]: 
    #                 self.sonde_ivt=self.sonde_ivt.loc[self.icon_ivt.index]
    #         else:
    #             pass
            
    #         ## Calculate TIVT for both variables
    #         if self.do_icon:
    #             if self.grid_sounding_profiles:
    #                 self.ivt_logger.icon_ivt_logger.info(
    #                             "Calculate synthetic TIVT from ICON sounding")
                
    #                 icon_sounding=True
    #             else:
    #                 self.ivt_logger.icon_ivt_logger.info(
    #                             "Calculate TIVT from ICON")
    #                 icon_sounding=False
    #             self.calculate_tivt_from_ivt(icon_sounding=icon_sounding)
    #         else:    
    #             #if not synthetic:
    #             self.ivt_logger.icon_ivt_logger.info(
    #                                     "Calculate TIVT from Dropsondes")
    #             self.calculate_tivt_from_ivt(icon=False,
    #                                     sounding_frequency=self.sounding_frequency)
        
    #     elif (self.sounding_frequency=="standard") and \
    #             (self.grid_sounding_profiles):   
    #         ## Calculate TIVT for both variables
    #         self.ivt_logger.icon_ivt_logger.info(
    #             "Calculate TIVT from ICON sounding")

    def calc_BIAS_TIVT_sondes_grid(self,deleted_sondes_tivt):   
        grid_name=self.grid_data_name
        if not "highres_Interp_IVT" in self.grid_dict.keys():    
            self.grid_ivt=self.grid_dict["Interp_IVT"][self.halo.index]
        else:
            self.grid_ivt=self.grid_dict["highres_Interp_IVT"][self.halo.index]
            
        self.TIVT[grid_name]=(self.grid_ivt*self.halo["delta_distance"]).sum()
        self.sonde_ivt=self.sonde_complete.copy()
        self.calc_single_tivt_from_ivt(grid_sounding=True)
        self.TIVT["all_sondes"]=self.tivt_sonde
        self.TIVT["sondes_no"+str(deleted_sondes_tivt.shape[1]-1)]=deleted_sondes_tivt
        self.TIVT["sondes_no"+str(deleted_sondes_tivt.shape[1]-1)].name=\
            self.grid_data_name
        
        # Calc BIAS between all sondes and grid representation
        self.TIVT["Sondes_BIAS"]=(self.TIVT["all_sondes"]-self.TIVT[grid_name])/\
                                    self.TIVT[grid_name]
        # Calc BIAS for sub samples
        self.TIVT["sondes_no"+str(deleted_sondes_tivt.shape[1]-1)]["Rel_BIAS"]=\
                  (self.TIVT["sondes_no"+str(deleted_sondes_tivt.shape[1]-1)]["TIVT"]-\
                   self.TIVT[grid_name])/\
                                self.TIVT[grid_name]
            
        #print("TIVT; Sonde:",self.TIVT["al"],
        #          "-- ICON:",self.TIVT["ICON"])
       # 
       # else:
       #     self.TIVT["ICON_Sondes"]=np.nan
       #     print("TIVT; ICON-Synthetic:",self.TIVT["ICON_Sondes"],
       #           "-- ICON:",self.TIVT["ICON"])
       #     self.TIVT["Rel.Synth_Bias"]=(self.TIVT["ICON_Sondes"]-\
       #                                  self.TIVT["ICON"])/\
       #                                     self.TIVT["ICON"]
       ##     
       # self.TIVT["ICON_Sounding_Number"]=sonde_ivt_array.shape[0]
        #if not synthetic:
            #    TIVT["Sonde"]=tivt_sonde
            #    TIVT["Rel.Bias"]=(TIVT["Sonde"]-TIVT["ICON"])/TIVT["ICON"]
        #else:
            #    TIVT["Sonde"]=sondes_to_use
            #    TIVT["Sonde"]["Rel.Bias"]=(TIVT["Sonde"]["TIVT"]-\
                #TIVT["ICON"])/TIVT["ICON"]
            #return TIVT

       # return None
    
    def study_TIVT_sondes_grid_frequency_dependency(self,
                                            discrete_sounding_study=True,
                                            regular_resolution_study=False):
        """
        
        This functions creates data to further analyse how resolution or 
        number of sounds might affect the BIAS in TIVT which is important 
        for moisture budget understanding.

        Parameters
        ----------
        discrete_sounding_study : boolean, optional
            This specifies if the TIVT study should rely on neglecting given 
            number of sondes and then calculate TIVT BIAS. The default is True.
        regular_resolution_study : boolean, optional
            This specifies if the TIVT BIAS study should rely on comparison for 
            a list of idealized fixed resolutions in sounding release.
            The default is False.

        Returns
        -------
        None.

        """
        # Depending on the steps before, sonde_ivt does not have to be defined yet
        if not hasattr(self,"sonde_ivt"):
            self.sonde_ivt={}
        if isinstance(self.sonde_ivt,dict):
            if self.sonde_ivt=={}:
                sonde_dict={}
                sonde_dict["Pres"]=pd.DataFrame()
                sonde_dict["q"]=pd.DataFrame()
                sonde_dict["Wspeed"]=pd.DataFrame()
                sonde_dict["IVT"]=pd.DataFrame()
            
                ### HALO AR cross-sections
                #if analysed_flight.startswith("S"):
                    #    only_model_sounding_profiles=True
                ar_of_day="AR_internal"
                #else:
                    #    only_model_sounding_profiles=False

                if self.flight.startswith("S"):
                    ## apparently need to be defined 
                    # plot_path, ar_of_day,flight,
                    sonde_aircraft_df,sonde_dict=create_synthetic_sondes(
                                        self.halo,self.grid_dict,
                                        sonde_dict,hmps_used=True,
                                        no_of_sondes=10)
                    self.sonde_ivt=sonde_dict["IVT"]
            else:
                print("Sonde IVT is defined as dict. Is that desired?")
        
        self.sonde_complete=self.sonde_ivt.copy()
        
        self.TIVT={}
        if discrete_sounding_study:
            sonde_delete_numbers=[1,2,3,4,5,6]
            for delete_DS_number in sonde_delete_numbers:
                logger_string="Delete "+str(delete_DS_number)+" sonde"
                self.ivt_logger.icon_ivt_logger.info(logger_string)
                number_of_delete_sondes=delete_DS_number
                globals()["TIVT_synthetic_delete_"+str(number_of_delete_sondes)]=\
                self.calc_and_compare_TIVT_sondes_deleted_and_grid(
                                    delete_DS_number)
                TIVT_sonde_sample=globals()["TIVT_synthetic_delete_"+str(number_of_delete_sondes)]
                TIVT_sonde_sample.to_csv(
                        path_or_buf=self.plot_path+self.flight+"/"+\
                        self.ar_of_day+"_"+"TIVT_synthetic_delete_"+\
                        str(number_of_delete_sondes)+".csv",index=True)
                self.calc_BIAS_TIVT_sondes_grid(TIVT_sonde_sample)
                    
            #TIVT_synthetic_delete_3["Sonde"].to_csv(
            #    path_or_buf=plot_path+ar_of_day+"_"+flight+\
            #        "_TIVT_synthetic_delete_3.csv",index=True)
            #TIVT_synthetic_delete_5["Sonde"].to_csv(
            #    path_or_buf=plot_path+ar_of_day+"_"+flight+\
            #        "_TIVT_synthetic_delete_5.csv",index=True)
            #TIVT_synthetic_delete_7["Sonde"].to_csv(
            #    path_or_buf=plot_path+ar_of_day+"_"+flight+\
            #        "_TIVT_synthetic_delete_7.csv",index=True)
            #TIVT_synthetic_delete_8["Sonde"].to_csv(
            #    path_or_buf=plot_path+ar_of_day+"_"+flight+\
            #        "_TIVT_synthetic_delete_8.csv",index=True)
                    
            # Dirty Quicklook Comparison showing the soundes in ICON and 
            # the ones of the sondes
            #fig_icon_sonde_comparison=plt.figure(figsize=(16,9))
            #plt.plot(icon_ivt_synthetic,color="blue",label="Synthetic")
            #plt.plot(sonde_ivt,color="orange",label="Sondes")
            #relative_bias_icon_sonde=np.mean((sonde_ivt.values-icon_ivt_synthetic.values)/icon_ivt_synthetic.values)
            #plt.suptitle("Relative Mean Error Synthetic ICON-Sounding - Dropsondes: "+str(round(relative_bias_icon_sonde,3)))
            #sns.despine(offset=10)
            #plt.ylim([100,400])
            #        
            #fig_icon_sonde_comparison.savefig(plot_path+"Synthetic_ICON_Dropsondes.png",
            #                                  dpi=300,bbox_inches="tight")
        if regular_resolution_study:
            # Define file name to store TIVT from resolution study
            tivt_res_csv_file=self.plot_path+\
                                self.ar_of_day+"_"+self.flight+"_"+\
                                    "TIVT_Sonde_Resolution.csv"

            # Resolutions to analyse
            resolutions=["180s","240s","360s","480s","600s","720s",
                                 "840s","1200s","1800s","2700s","3000s","3600s"]
            resolutions_int=["180","240","360","480","600","720",
                                     "840","1200","1800","2700","3000","3600"]
            
            standard_icon_tivt=self.calculate_tivt_from_ivt()
            """
            # Old Format
            resampled_tivt=pd.DataFrame(columns=["Resolution","TIVT_Mean",
                                                 "TIVT_Std","Resampled_Distance"])
            """
            number_of_shifts=10
            df_resampled_tivt=pd.DataFrame(columns=resolutions,
                                           index=np.arange(0,number_of_shifts))
            df_resampled_tivt.index.name="Index_Shift_Factor"
            df_resampled_tivt.name=standard_icon_tivt
            i=0
            for res in resolutions:
                print("Current Resolution: ",res)
                resampled_grid_ivt=self.grid_ivt.asfreq(res)
                # 
                shift_factor=int(float(resolutions_int[i])/number_of_shifts)
                res_case_tivt=pd.Series(index=np.arange(0,number_of_shifts)*shift_factor)  
                        
                # Shift index to determine the spatial variability
                for shift in range(number_of_shifts):
                    shifter=shift_factor*shift
                    shifted_resampled_grid_ivt=self.grid_ivt.iloc[shifter:].asfreq(res)    
                    shifted_resampled_grid_ivt=shifted_resampled_grid_ivt.append(
                        self.grid_ivt.iloc[[0,-1]])
                    shifted_resampled_grid_ivt=shifted_resampled_grid_ivt.drop_duplicates(keep="first")
                    shifted_resampled_grid_ivt=shifted_resampled_grid_ivt.sort_index()
                    resampled_halo_df=pd.DataFrame(columns=["groundspeed",
                                "time_Difference","delta_Distance"],
                                index=shifted_resampled_grid_ivt.index)
                    for idx in range(len(shifted_resampled_grid_ivt.index)-1):
                        groundspeed=self.halo["groundspeed"].loc[\
                                        shifted_resampled_grid_ivt.index[idx]:\
                                            shifted_resampled_grid_ivt.index[idx+1]]
                        mean_groundspeed=groundspeed.mean()
                        resampled_halo_df["groundspeed"].iloc[idx+1]=mean_groundspeed
                                #copy_halo_df.set_index(np.arange(len(copy_halo_df))//time_frequency).mean(level=0)
                    resampled_halo_df.index=shifted_resampled_grid_ivt.index
                            
                    time_frequency=shifted_resampled_grid_ivt.index.to_series().diff()#.seconds
                    resampled_halo_df["time_Difference"]=time_frequency
                    resampled_halo_df["time_Difference"]=resampled_halo_df["time_Difference"].dt.total_seconds()
                    resampled_halo_df["delta_Distance"]=resampled_halo_df["groundspeed"]*resampled_halo_df["time_Difference"]
                    resampled_halo_df["delta_Distance"].iloc[0]=0.0
                    resampled_halo_df["cumsum_distance"]=resampled_halo_df["delta_Distance"].cumsum()
                    #print(res," ","Resampled_Distance:",
                    #      resampled_halo_df["cumsum_distance"][-1],
                    #      "Standard Distance",copy_halo_df["cumsum_distance"][-1])
                        
                    temp_tivt=self.calculate_tivt_from_ivt()
                    res_case_tivt.iloc[shift]=temp_tivt
                    i+=1
                            #df_resampled_tivt["cumsum_distance"].iloc[shift]=resampled_halo_df["cumsum_distance"][-1]
                    df_resampled_tivt.loc[:,res]=res_case_tivt.values    
                        
                    df_resampled_tivt["REAL-ICON-TIVT"]=float(df_resampled_tivt.name)    
                    df_resampled_tivt.to_csv(path_or_buf=tivt_res_csv_file,
                                             index=True)
                    print("Saved TIVT Sonde Resolution CSV File under:"+tivt_res_csv_file)
                    self.ivt_logger.icon_ivt_logger.info("Saved TIVT Sonde Resolution CSV File under:"+tivt_res_csv_file)
                    
                    """ OLD Without Shifting 
                        time_frequency=resampled_icon_ivt.index.to_series().diff().mean().seconds
                        
                        resampled_halo_df=copy_halo_df.set_index(np.arange(len(copy_halo_df))//time_frequency).mean(level=0)
                        resampled_halo_df.index=resampled_icon_ivt.index
                        resampled_halo_df["delta_Distance"]=resampled_halo_df["groundspeed"]*time_frequency
                        ### The last entry of delta distance has to be replaced due to different time_frequency last index in any case to maintain width of observed AR core!!
                        resampled_icon_ivt[icon_ivt.index[-1]]=icon_ivt.iloc[-1]
                        end_halo_df=copy_halo_df.loc[resampled_icon_ivt.index[-2]:icon_ivt.index[-1]]
                        ##resampled_halo_df[copy_halo_df.index[-1]]=copy_halo_df[-1,:]
                        ##resampled_halo_df.loc[copy_halo_df.index[-1]]=np.nan
                        ##resampled_halo_df["groundspeed"].loc[copy_halo_df.index[-1]]=end_halo_df["groundspeed"].mean() 
                        resampled_halo_df["delta_Distance"].iloc[-1]=end_halo_df["delta_Distance"].iloc[1:-1].sum() 
                        resampled_halo_df["cumsum_distance"]=resampled_halo_df["delta_Distance"].cumsum()
                        temp_tivt=calculate_tivt_from_icon_ivt(resampled_icon_ivt,
                                                               resampled_halo_df,
                                                               standard=False)
                        resampled_tivt["TIVT_Mean"].iloc[i]=temp_tivt
                        resampled_tivt["Resampled_Distance"]=resampled_halo_df["cumsum_distance"][-1]
                    
                        i+=1
                        """
                        #else:
            #    print("No TIVT comparison is done. Instead vertical separation of q and wind.")

    def calc_grid_moist_transport(self):
        if not "wind" in self.grid_ivt.keys():
           self.grid_ivt["wind"]=np.sqrt(self.grid_ivt["u"]**2+\
                                          self.grid_ivt["v"]**2)
        self.grid_ivt["wind"]=self.grid_ivt["wind"].groupby(level=0).last()
        try:
            self.grid_ivt["moist_transport"]=1/9.81*self.grid_ivt["wind"]*\
                                            self.grid_ivt["q"]
        except:
            self.grid_ivt["moist_transport"]=1/9.81*self.grid_ivt["wind"]*\
                                            self.grid_ivt["specific_humidity"]
    def calc_icon_moist_transport(self):
        #q should be in kg/kg initially
        self.icon_ivt["moist_transport"]=1/9.81*self.icon_ivt["wind"]*\
                                            self.icon_ivt["q"]
    def calc_sonde_moist_transport(self):
        if hasattr(self,"grid_sounding_profiles"):
            # this is already the "sounding" from icon
            sonde_q        = self.sonde_ivt["q"]
            sonde_p        = self.sonde_ivt["Pres"].values.astype(float)
            sonde_wind     = self.sonde_ivt["Wspeed"]
        else: 
            sonde_p        = self.sonde_ivt["Pres"]
            sonde_q        = self.sonde_ivt["q"].iloc[:,1:]
            sonde_wind     = self.sonde_ivt["Wspeed"].iloc[:,1:]
            
        self.sonde_ivt["moist_transport"]=1/9.81*sonde_wind*\
                                            sonde_q*1000
    def calc_grid_vars_mean_std(self):
        
        self.grid_mean={}
        if not "p" in self.grid_ivt.keys():
            # Probably the data is given on pressure levels
            try:
                self.grid_mean["p"]         = pd.Series(
                                    self.grid_ivt["q"].columns.astype(float))
            except:
                raise Exception("Something went wrong with your grid data. ",
                                "Apparently pressure p is neither a variable nor",
                                " it is well-defined in the columns.")
        else:
            self.grid_mean["p"]                 = self.grid_ivt["p"].mean(axis=0)
        
        self.grid_mean["wind"]              = self.grid_ivt["wind"].mean(axis=0)
        try:
            self.grid_mean["q"]             = self.grid_ivt["q"].mean(axis=0)
        except:
            self.grid_mean["q"]             = self.grid_ivt["specific_humidity"].mean(axis=0)
        self.grid_mean["moist_transport"]   = self.grid_ivt["moist_transport"].mean(axis=0)
        self.grid_mean["moist_transport"].dropna(inplace=True)
        try:
            self.grid_mean["z"]                 = self.grid_ivt["Geopot_Z"].mean(axis=0)
        except:
            self.grid_mean["z"]                 = self.grid_ivt["z"].mean(axis=0)
        self.grid_std={}
        self.grid_std["wind"]               = self.grid_ivt["wind"].std(axis=0)
        try:
            self.grid_std["q"]                  = self.grid_ivt["q"].std(axis=0)
        except:
            self.grid_std["q"]                  = self.grid_ivt["specific_humidity"].std(axis=0)
        self.grid_std["moist_transport"]    = self.grid_ivt["moist_transport"].\
                                                std(axis=0)
        self.grid_std["moist_transport"].dropna(inplace=True)
        
    def calc_icon_vars_mean_std(self):
        #%% Sonde IVT Processing and analysis
        
        # if grid_sounding_profiles are True, the profiles within the icon data
        # are used and compared with the overall icon AR cross-section
        print("ICON means and stds are calculated")
        #sonde_wind  = 
        #sonde_p     = self.icon_ivt["pres"]
            
        #    sonde_moist_transport=1/9.81*sonde_wind*sonde_q
        
        self.icon_mean={}
        self.icon_mean["p"]           = self.icon_ivt["p"].mean(axis=0)
        self.icon_mean["wind"]           = self.icon_ivt["wind"].mean(axis=0)
        self.icon_mean["q"]              = self.icon_ivt["q"].mean(axis=0)
        self.icon_mean["moist_transport"]= self.icon_ivt["moist_transport"].mean(axis=0)
        self.icon_mean["moist_transport"].dropna(inplace=True)
        self.icon_mean["z"]              = self.icon_ivt["Z_Height"].mean(axis=0)
        self.icon_std={}
        self.icon_std["wind"]              = self.icon_ivt["wind"].std(axis=0)
        self.icon_std["q"]                 = self.icon_ivt["q"].std(axis=0)
        self.icon_std["moist_transport"]   = self.icon_ivt["moist_transport"].\
                                                std(axis=0)
        self.icon_std["moist_transport"].dropna(inplace=True)

    def calc_sonde_vars_mean_std(self):
        #%% Sonde IVT Processing and analysis
        
        # if grid_sounding_profiles are True, the profiles within the icon data
        # are used and compared with the overall icon AR cross-section
        
        if self.grid_sounding_profiles:   
            print("Dropsondes are replaced by synthetic soundings")
       #     sonde_wind  = self.sonde_ivt["Wspeed"]
       #     sonde_q     = self.sonde_ivt["q"]
       #     sonde_p     = self.sonde_ivt["p"]
        else:
            print("Real dropsondes are used")
        sonde_wind  = self.sonde_ivt["Wspeed"]
        sonde_q     = self.sonde_ivt["q"]
        sonde_p     = self.sonde_ivt["Pres"]
            
             
        self.sonde_mean={}
        self.sonde_mean["Wind"]              = sonde_wind.mean(axis=0)
        self.sonde_mean["Pres"]              = sonde_p.mean(axis=0)   
        self.sonde_mean["q"]                 = sonde_q.mean(axis=0)
        self.sonde_mean["moist_transport"]   = self.sonde_ivt["moist_transport"].\
                                                mean(axis=0)
        self.sonde_mean["moist_transport"].dropna(inplace=True)
        
        self.sonde_std={}
        self.sonde_std["Wind"]              = sonde_wind.std(axis=0)
        self.sonde_std["Pres"]              = sonde_p.std(axis=0)
        self.sonde_std["q"]                 = sonde_q.std(axis=0)
        self.sonde_std["moist_transport"]   = self.sonde_ivt["moist_transport"].\
                                                std(axis=0)
        self.sonde_std["moist_transport"].dropna(inplace=True)
    
    def calc_grid_mean_ivt_cumsum(self):
        if self.grid_data_name=="ICON":
            self.icon_mean["cumsum"]=scint.cumtrapz(
                                    y=self.icon_mean["moist_transport"],
                                    x=self.icon_ivt["p"].mean(axis=0))
        else:
            self.grid_mean["cumsum"]=scint.cumtrapz(
                                    y=self.grid_mean["moist_transport"],
                                    x=self.grid_mean["p"])
            
        #icon_ivt_cumsum2=scint.cumtrapz(y=self.icon_mean["moist_transport"][::-1],
        #                                x=self.icon_ivt["Pres"].mean(axis=0)[::-1])
        #self.icon_ivt_cumsum
    def calc_sonde_mean_ivt_cumsum(self):
        #if not self.grid_sounding_profiles:
        rows_to_cut=self.sonde_ivt["moist_transport"].isna().sum()
        vertical_index=np.array(self.sonde_ivt["Pres"].loc[:,
                                    rows_to_cut[rows_to_cut<\
                                    self.sonde_ivt["moist_transport"].shape[0]/2].index]\
                                    .mean(axis=0)*100)
        
        not_nan_moist_transport=self.sonde_mean["moist_transport"].loc[\
                                    rows_to_cut[rows_to_cut<self.sonde_ivt[\
                                        "moist_transport"].shape[0]/2].index]
        self.not_nan_moist_trans_idx=not_nan_moist_transport.index
        self.sonde_mean["cumsum"]=scint.cumtrapz(
                                        y=not_nan_moist_transport,
                                        x=vertical_index)

    def calc_vertical_quantiles(self,use_grid=True,quantiles=["50","75",
                                                              "90","97","100"],
                                do_all_preps=False):
        
        # Grid data (ERA5,CARRA or ICON)
        if use_grid:
            if do_all_preps:
                # Add vertical vars to IVT dict
                self.add_vertical_vars_to_ivt_dict(use_sondes=False)
                if self.grid_data_name=="ICON-2km":    
                    self.calc_icon_moist_transport()
                    #self.calc_icon_moist_transport()
                    self.calc_icon_vars_mean_std()
                    #self.calc_icon_moist_transport()        
                    #ICON_IVT_CUMSUM Calculation
                    self.calc_icon_mean_ivt_cumsum()
            
                else:    
                    #%% ICON IVT Processing and analysis
                    self.calc_grid_moist_transport()
                    self.calc_grid_vars_mean_std()
                    #self.calc_icon_moist_transport()        
                    #ICON_IVT_CUMSUM Calculation
                    self.calc_grid_mean_ivt_cumsum()
            else:
                pass
            if self.grid_data_name=="ICON_2km":
                self.grid_ivt=self.icon_ivt.copy()
                self.grid_mean=self.icon_mean
                self.grid_mean["cumsum"]=self.icon_mean["cumsum"][10:]
            
            # relative vertical contribution
            rows_to_cut=self.grid_ivt["moist_transport"].isna().sum()
            if not "p" in self.grid_ivt.keys():
                pressure_array=np.expand_dims(self.grid_mean["p"].values,axis=0)
                pressure_array=np.repeat(pressure_array,self.grid_ivt["q"].shape[0],
                                         axis=0)
                self.grid_ivt["p"]=pd.DataFrame(data=pressure_array,
                                                columns=self.grid_ivt["q"].columns,
                                                index=self.grid_ivt["q"].index)
            vertical_index=np.array(self.grid_ivt["p"].loc[:,
                                    rows_to_cut[rows_to_cut<\
                                    self.grid_ivt["moist_transport"].shape[0]/2].index]\
                                    .mean(axis=0))
        
            #self.not_nan_moist_trans_idx=self.icon_mean["moist_transport"].loc[\
            #                            rows_to_cut[rows_to_cut<self.icon_ivt[\
            #                            "moist_transport"].shape[0]/2].index].\
            #                                index
            #print(self.icon_mean)
            #sys.exit()
            #vertical_index=self.not_nan_moist_trans_idx
            ivt_z_vertical_relative=pd.Series(
                                            data=(self.grid_mean["cumsum"][-1]-\
                                                  self.grid_mean["cumsum"])/\
                                                    self.grid_mean["cumsum"][-1],
                                            index=self.grid_mean["z"][1:])
            ivt_p_vertical_relative=pd.Series(
                                            data=(self.grid_mean["cumsum"][-1]-\
                                                  self.grid_mean["cumsum"])/\
                                                  self.grid_mean["cumsum"][-1],
                                            index=vertical_index[1:])
            
            self.grid_ivt_quant_z={}
            self.grid_ivt_quant_p={}
            for quantile in quantiles:
                self.grid_ivt_quant_z[quantile]=float(ivt_z_vertical_relative.index[\
                                        (ivt_z_vertical_relative-\
                                         float(quantile)/100).abs().\
                                            argmin()])
                self.grid_ivt_quant_p[quantile]=float(ivt_p_vertical_relative.index[\
                                        (ivt_z_vertical_relative-\
                                         float(quantile)/100).abs().\
                                            argmin()])/100
        else:
            #Dropsondes
            if do_all_preps:
                # Add vertical vars to IVT dict
                self.add_vertical_vars_to_ivt_dict(use_sondes=True)
                self.calc_sonde_moist_transport()
                self.calc_sonde_vars_mean_std()
                self.calc_sonde_mean_ivt_cumsum()
                
            self.sonde_ivt_quant_z={}
            self.sonde_ivt_quant_p={}
            #sonde_ivt=pd.Series(data=np.nan,index=sonde_moist_transport.index)
            #if synthetic_icon_lat==4:    
            #    loop_range=sonde_moist_transport.shape[0]-1
            #else:
            loop_range=self.sonde_ivt["moist_transport"].shape[0]
            if self.sonde_ivt["IVT"].shape[0]==0:
                self.sonde_ivt["IVT"]=pd.Series(data=np.nan,index=self.sonde_ivt["q"].index)
            for sond in range(loop_range):
                 moist_data=self.sonde_ivt["moist_transport"].iloc[sond,:].dropna()
                 #if not self.grid_sounding_profiles:
                 pressure=self.sonde_ivt["Pres"].iloc[sond,:].loc[moist_data.index]
                 #else:
                 #pressure=self.sonde_ivt["p"].iloc[sond,:].loc[moist_data.index]
                 self.sonde_ivt["IVT"].iloc[sond]=abs(scint.trapz(y=moist_data,
                                                              x=-pressure*100))
        
            # relative vertical contribution
            #if not self.grid_sounding_profiles:
            #    sonde_z_index=self.sonde_mean["Wspeed"].index[1:]
            #else:
            #sonde_z_index=self.sonde_mean["Wind"].index[1:]
            
            sonde_ivt_z_vertical_rel=pd.Series(
                                            data=(self.sonde_mean["cumsum"][-1]-\
                                                  self.sonde_mean["cumsum"])/\
                                                  self.sonde_mean["cumsum"][-1],
                                            index=self.not_nan_moist_trans_idx[:-1])
            sonde_ivt_p_vertical_rel=pd.Series(
                                            data=(self.sonde_mean["cumsum"][-1]-\
                                                  self.sonde_mean["cumsum"])/\
                                                self.sonde_mean["cumsum"][-1],
                                            index=self.sonde_mean["Pres"].\
                                                loc[self.not_nan_moist_trans_idx].values[:-1])
            for quantile in quantiles:
                self.sonde_ivt_quant_z[quantile]=float(sonde_ivt_z_vertical_rel.index[\
                                        (sonde_ivt_z_vertical_rel-\
                                         float(quantile)/100).\
                                            abs().argmin()])
                self.sonde_ivt_quant_p[quantile]=float(sonde_ivt_p_vertical_rel.index[\
                                        (sonde_ivt_p_vertical_rel-\
                                         float(quantile)/100).\
                                             abs().argmin()])
        print("Vertical quantiles are calculated")
    
    #if not synthetic:
        #    if not icon_ivt.shape[0]==sonde_ivt.shape[0]:
        #        if icon_ivt.shape[0]>sonde_ivt.shape[0]:
        #            icon_ivt=icon_ivt.loc[sonde_ivt.index]
        #        elif icon_ivt.shape[0]<sonde_ivt.shape[0]: 
        #            sonde_ivt=sonde_ivt.loc[icon_ivt.index]
        #        else:
        #            pass
        #    else:
        #        pass
        #    icon_ivt_array=np.array(icon_ivt.rolling(2).mean().iloc[1:])
        #    delta_distance=np.array(copy_halo_df["delta_Distance"].loc[sonde_ivt.index].iloc[1:])
        #else: 
        #    icon_ivt_array=np.array(icon_ivt)
        #    delta_distance=np.array(copy_halo_df["delta_Distance"].loc[icon_ivt.index])
        
        #tivt_icon=(icon_ivt_array*delta_distance).sum()
    
                
class IVT_Variability_Plotter(IVT_variability):
    def __init__(self,hmp_dict,hmc_dict,sonde_dict,model_sounding_profiles,
                 sounding_frequency,halo,plot_path,ar_of_day,
                 flight,ivt_logger):
        super().__init__(hmp_dict,hmc_dict,sonde_dict,model_sounding_profiles,
                 sounding_frequency,halo,plot_path,ar_of_day,
                 flight,ivt_logger)
        
        #if isinstance(grid_dict_hmp,pd.DataFrame):
        #    self.grid_data_name=grid_dict_hmp.name
        #    self.grid_dict=grid_dict_hmp
        # 
        # if isinstance(grid_dict_hmc,pd.DataFrame):
        #    self.grid_data_name=grid_dict_hmc.name
        #    self.grid_dict=grid_dict_hmc
        
        
        self.hmp_dict=hmp_dict
        self.hmc_dict=hmc_dict
        self.sonde_dict=sonde_dict
        self.icon_ivt={}
        self.sonde_ivt={}
        self.model_sounding_profiles=model_sounding_profiles
        self.halo=halo
        self.sounding_frequency=sounding_frequency
        self.plot_path=plot_path
        self.ar_of_day=ar_of_day
        self.flight=flight
        self.ivt_logger=ivt_logger
        self.TIVT={}
        
        #import matplotlib
        #import seaborn as sns
        #import matplotlib.dates as mdates
    
    def plot_model_sounding_frequency_comparison(self,name_of_grid_data="ERA5"):
        matplotlib.rcParams.update({"font.size":20})
        
        #self.grid_sounding_profiles=True
        if not self.flight.startswith("S"):
            if not self.model_sounding_profiles:
                self.ivt_logger.icon_ivt_logger.info("Load HALO-Dropsonde data")
    
            # NAWDEX cutted dropsondes can have erroneous soundings included once 
            # the synthetic flight track has been created for an existing flight.
            # The dropsonde data loader refers to time periods and might then 
            # include wrong sondes. So we load the actual dropsondes and then
            # loc them again for in- or outflow period.
            else:
                sonde_path=os.getcwd()
                ar_of_day="SAR_internal"#"AR3"
            import Grid_on_HALO
            if not self.flight=="RF10":
                sonde_dict=Grid_on_HALO.load_ar_dropsonde_profiles(sonde_path,
                                                self.flight,ar_of_day)
            else:
                sonde_dict=Grid_on_HALO.load_ar_dropsonde_profiles(sonde_path,
                                                self.flight,"AR3")
            for var in sonde_dict.keys():
                sonde_dict[var]=sonde_dict[var].loc[\
                                            self.halo.index[0]:\
                                            self.halo.index[-1]]
    
        else:
    
            # Flight Track where no sondes have been launched
            # use synthetic soundings
            # space them equidistantly along cross-section
            sonde_dict={}
            sonde_dict["Pres"]=pd.DataFrame()
            sonde_dict["q"]=pd.DataFrame()
            sonde_dict["Wspeed"]=pd.DataFrame()
            sonde_dict["IVT"]=pd.DataFrame()
            sonde_dict["IWV"]=pd.DataFrame()
            ### here a new function should be added that creates 
            ### synthetic soundings at certain intervals along 
            ### HALO AR cross-sections
            #sonde_dict["IVT"]
            # check sonde sensitivity
        sonde_representation_dict={}
        print("Create synthetic sondes")
        start=self.halo.index[0]
        end=self.halo.index[-1]
        
        if isinstance(self.hmp_dict,dict):
            if not self.hmp_dict[self.flight]["AR_internal"].name=="CARRA":
                ivt_continuous=self.hmp_dict[self.flight]["AR_internal"]["Interp_IVT"]
                iwv_continuous=self.hmp_dict[self.flight]["AR_internal"]["Interp_IWV"]
            else:
                ivt_continuous=self.hmp_dict[self.flight]["AR_internal"]["highres_Interp_IVT"]
                iwv_continuous=self.hmp_dict[self.flight]["AR_internal"]["highres_Interp_IWV"]
            ivt_continuous=ivt_continuous.loc[start:end]
            iwv_continuous=iwv_continuous.loc[start:end]
        elif isinstance(self.hmp_dict,pd.DataFrame):
            if not self.hmp_dict.name=="CARRA":
                ivt_continuous=self.hmp_dict["Interp_IVT"]
                iwv_continuous=self.hmp_dict["Interp_IWV"]
            else:
                ivt_continuous=self.hmp_dict["highres_Interp_IVT"]
                iwv_continuous=self.hmp_dict["highres_Interp_IWV"]
                        
            ivt_continuous=ivt_continuous.loc[start:end]
            iwv_continuous=iwv_continuous.loc[start:end]
        else:
            TypeError("HMP_dict is of wrong type. Recheck the class init")
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        release_fig=plt.figure(figsize=(12,12))
        ax1=release_fig.add_subplot(211)
        ax2=release_fig.add_subplot(212)
        sonde_no_list=[4,6,8,10]
        ivt_frequent_color=plt.get_cmap("Greens",len(sonde_no_list)+2)
        iwv_frequent_color=plt.get_cmap("Blues",len(sonde_no_list)+2)
        i=1
        # Calculate the distances for TIVT calculation
        self.halo=calc_halo_delta_distances(self.halo)
        # Calc TIVT for continuous representation
        self.grid_ivt=ivt_continuous
        self.calc_single_tivt_from_ivt(grid_used=True,grid_sounding=False,
                                standard=True,sounding_frequency="standard")
            
        
        for sonde_no in sonde_no_list:#,6,8,10,12]:
            print("Sonde no of: ",sonde_no)
            #aircraft_df,aircraft_var_df,
            #                sonde_dict,hmps_used=False,no_of_sondes=10):
            sonde_aircraft_df,sonde_dict=create_synthetic_sondes(self.halo,
                                        self.hmp_dict,sonde_dict,
                                        hmps_used=True,
                                        no_of_sondes=sonde_no)
            
            sonde_representation_dict[str(sonde_no)+"_met_vars"]=sonde_dict
            sonde_representation_dict[str(sonde_no)+"_aircraft"]=sonde_aircraft_df
            self.sonde_ivt=sonde_dict["IVT"]
            # Calc TIVT for synthetic soundings
            self.calc_single_tivt_from_ivt(grid_used=True,grid_sounding=True,
                                standard=True,sounding_frequency="standard")
            
            #print(self.tivt_sonde)
            ax1.plot(sonde_representation_dict[str(sonde_no)+"_met_vars"]["IVT"].index,
                    sonde_representation_dict[str(sonde_no)+"_met_vars"]["IVT"],
                    marker="v",c=ivt_frequent_color(i),ls="--",lw=2,
                    markersize=10,label=str(sonde_no)+" DS: TIVT="+\
                        str((self.tivt_sonde/1e6).round(1))+\
                            " x1e6 $\mathrm{kgs}^{-1}$")
            
            ax2.plot(sonde_representation_dict[str(sonde_no)+"_met_vars"]["IWV"].index,
                    sonde_representation_dict[str(sonde_no)+"_met_vars"]["IWV"],
                    marker="v",c=iwv_frequent_color(i),ls="--",lw=2,
                    markersize=10,label=str(sonde_no)+" DS")
        
            i+=1
        
        ax1.plot(ivt_continuous.index,
                 ivt_continuous.values,
                 color="darkgreen",lw=3,ls="-",
                 label="TIVT="+str((self.tivt_grid/1e6).round(1))+\
                     " x1e6 $\mathrm{kgs}^{-1}$")
        ax1.set_ylim([0,700])
        ax2.plot(iwv_continuous.index,
                 iwv_continuous.values,
                 color="darkblue",lw=3,ls="-")
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        ax1.legend(loc="upper left",fontsize=9.5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        ax2.set_xlabel("Time (UTC)")
        ax2.set_ylim([5,25])
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(3)
            ax2.spines[axis].set_linewidth(3)
        ax1.set_ylabel("IVT (kg $\mathrm{m}^{-1}\mathrm{s}^{-1})$")
        ax2.set_ylabel("IWV (kg $\mathrm{m}^{-2})$")
        
        ax1.tick_params(length=10,width=3)
        ax2.tick_params(length=10,width=3)
        sns.despine(offset=10)
        plot_path=self.plot_path+self.flight+"//"
        file_name=self.flight+"_"+str(name_of_grid_data)+"_TIVT_sounding_dependency.png"
        print("File name:",plot_path+file_name)            
        release_fig.savefig(plot_path+file_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ",file_name)
        
    def plot_distance_based_IVT(self,sondes_to_use,
                                synthetic_flight=False,
                                delete_sondes=None,name_of_grid_data="ERA5",
                                show_sondes=True):
        if hasattr(self,"sonde_ivt"):
            self.add_vertical_vars_to_ivt_dict(use_sondes=True)
        if hasattr(self,"grid_ivt"):
            self.add_vertical_vars_to_ivt_dict()
        
        delta_distance=self.halo["delta_distance"]
        if name_of_grid_data=="ERA5":
            cutted_ivt=self.grid_dict["Interp_IVT"]
        else:
            cutted_ivt=self.grid_dict["highres_Interp_IVT"]
            era5_cutted_ivt=self.grid_dict["Interp_IVT"]
            era5_cutted_ivt=era5_cutted_ivt.loc[self.halo.index[0]:self.halo.index[-1]]
            era5_cutted_ivt=era5_cutted_ivt[~era5_cutted_ivt.index.duplicated(keep='first')]
        
        cutted_ivt=cutted_ivt.loc[self.halo.index[0]:self.halo.index[-1]]
        cutted_ivt=cutted_ivt[~cutted_ivt.index.duplicated(keep='first')]
        #temporary_icon_ivt=delta_distance*self.icon_ivt["IVT"]
        
        matplotlib.rcParams.update({"font.size":28})
        sample_plot=plt.figure(figsize=(14,10))
        ax1=sample_plot.add_subplot(111)
        sns.despine(offset=20)
        ax1.plot(self.halo["cumsum_distance"]/1000,
                 cutted_ivt,
                 color="darkgreen",
                 linewidth=4.0,label=name_of_grid_data)
        if not name_of_grid_data=="ERA5":
            ax1.plot(self.halo["cumsum_distance"]/1000,
                 era5_cutted_ivt,
                 color="k",ls="-.",
                 linewidth=3.0,label="ERA5")
            
        #Start with synthetic observations
        #  if not synthetic_flight:
        #      ivt_logger.icon_ivt_logger.info("Calculate TIVT from Dropsondes")
        #  else:
            #  ivt_logger.icon_ivt_logger.info("Calculate synthetic TIVT from ICON")
        #---------------------------------------------------------------------#
        ### OLD parts for NAWDEX case
        """
        if self.sounding_frequency=="standard":
            # For TIVT calculation pay attention on start and end period
            if delete_sondes is not None:
                all_sonde_ivt=self.sonde_ivt.copy()
                for i in range(sondes_to_use.shape[0]):
                    sonde_ivt=all_sonde_ivt.iloc[sondes_to_use.iloc[i].\
                                                 values[:-1].astype(int).\
                                                     tolist()]
                    try:
                        sonde_ivt_array=np.array(sonde_ivt.\
                                                 rolling(2).mean().iloc[1:])
                    except:
                        sonde_ivt_array=np.array(sonde_ivt.\
                                                 rolling(2).mean().iloc[1:])
                    cum_sum_distance=self.halo["cumsum_distance"].loc[\
                                                        self.sonde_ivt.index]
                    
                    delta_distance=np.array(cum_sum_distance.diff().iloc[1:])
                    sondes_to_use["TIVT"].iloc[i]=(sonde_ivt_array*delta_distance).sum()
                    
                    #ax1.plot(sonde_ivt.index,sonde_ivt,color="grey",
                    #         linewidth=2.0,marker="v",markersize=20)
            else:
                try:
                    sonde_ivt_array=pd.Series(self.sonde_ivt["IVT"]["IVT"])
                except:
                    sonde_ivt_array=pd.Series(self.sonde_ivt["IVT"].iloc[:,0])#.rolling(2).\
                    #                        mean().iloc[1:]
            
                cum_sum_distance=self.halo["cumsum_distance"].loc[\
                                                        self.sonde_ivt["IVT"].index]
                        
                delta_distance=np.array(cum_sum_distance.diff().iloc[1:])
            #tivt_sonde=(sonde_ivt_array*delta_distance).sum()
    
        elif self.sounding_frequency=="Upsampled":
            sonde_ivt_array=np.array(self.sonde_ivt["IVT"].iloc[:,0])
            delta_distance=np.array(self.halo["delta_Distance"].loc[\
                                                        self.sonde_ivt.index])
            tivt_sonde=(sonde_ivt_array*delta_distance).sum()
    
        else:
            pass
        """
        if show_sondes:
            cum_sum_distance=self.halo["cumsum_distance"]
            sonde_ivt_array=self.sonde_ivt["IVT"]
            
        #---------------------------------------------------------------------#
            if not self.grid_sounding_profiles:
                sonde_label="Sondes"
                #ax1.plot(cum_sum_distance.loc[self.sonde_ivt.index]/1000,sonde_ivt_array,
                #         color="goldenrod",linewidth=1.0,marker="v",
                #         markeredgecolor="k",markersize=20,
                #     label=sonde_label)
            else:
                sonde_label="Synthetic DS"
        
            ax1.plot(cum_sum_distance.loc[sonde_ivt_array.index]/1000,
                     sonde_ivt_array,
                     color="lightgreen",linewidth=2.0,marker="v",
                     markeredgecolor="k",markersize=20,label=sonde_label,
                     zorder=2)
            #ax1.plot(delta_distance.cumsum()/1000,sonde_ivt_array
            
            # Values must not contain nan values
            sonde_ivt_array=sonde_ivt_array.dropna()
            distance_for_gaussian=self.halo["cumsum_distance"].loc[\
                                                        sonde_ivt_array.index]
        
            try:
                popt,popt_skewed=self.gaussian_fit_IVT(distance_for_gaussian, 
                                   sonde_ivt_array)
                #popt_skewed
                ax1.plot(self.halo["cumsum_distance"]/1000,
                 self.gaussian(self.halo["cumsum_distance"]/1000,*popt),
                 color="grey",ls="-",lw=2,label="Gaussian IVT Fit",zorder=3)
            except:
                print("No gaussian fit possible")
            #ax1.plot(distance_for_gaussian/1000,
            #         self.skewed_gaussian(distance_for_gaussian/1000,*popt_skewed),
            #         color="grey",ls="--",lw=3)
            
            #ax1.plot(sonde_ivt.index,sonde_ivt,
            #     ls='--',marker="v",color="grey",linewidth=4.0)
            ## Set time format and the interval of ticks (every 15 minutes)
            #xformatter = mdates.DateFormatter('%H:%M')
            #xlocator = mdates.MinuteLocator(interval = 15)
        
            ## Set xtick labels to appear every 15 minutes
            #ax1.xaxis.set_major_locator(xlocator)
            #ax1.set_xlabel("Time (UTC)")
            #for axis in ["left","bottom"]:
                #                    ax1.spines[axis].set_linewidth(2.0)
        #                    ax1.xaxis.set_tick_params(width=2,length=10)
        #                    ax1.yaxis.set_tick_params(width=2,length=10)
                    
        ax1.set_ylabel("IVT (kg $\mathrm{m}^{-1}\mathrm{s}^{-1})$")
        ymax=650
        #if sonde_ivt_array.max()>500:
        #    if sonde_ivt_array.max()< 650:
        #        ymax=650
        #    else:
        #        ymax=1200
        ax1.set_ylim([0,ymax])
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(3)
        ax1.tick_params(length=10,width=3)
        xmax=1000
        ax1.set_xlim([0,xmax])
        #if self.halo["cumsum_distance"].max()/1000>1100:
        #    ax1.set_xlim([0,1400])
        #else:
        #    if self.halo["cumsum_distance"].max()/1000>800:
        #        ax1.set_xlim([0,1000])
        #    else:
        #        ax1.set_xlim([0,800])
        ax1.set_xlabel("Lateral Distance $x$ (km)")
        ## Format xtick labels as HH:MM
        #plt.gcf().axes[0].xaxis.set_major_formatter(xformatter)
        ax1.legend(loc="upper left",fontsize=24)
        ax1.text(x=800,y=575,s=str(self.halo.index[0].date()),color="dimgray",
                 fontsize=28)
        file_name=self.plot_path+self.flight+"/"+\
                    self.flight+self.ar_of_day+\
                    "_"+name_of_grid_data+"_IVT_"
        if show_sondes:
            sondes_fig_str="Sounding_"+str(self.sonde_ivt.shape[0])+"_Sondes"
            file_name=file_name+sondes_fig_str
        file_end=".png"            
        file_name=file_name+file_end
        sample_plot.savefig(file_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ",file_name)
        return None

    def plot_IVT_icon_era5_synthetic_sondes(self,era5,sonde_ivt,icon_ivt,
                                            flight,AR_of_day,plot_path,
                                            save_figure=True):
        import matplotlib
        import seaborn as sns
        #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!############
        pd.plotting.register_matplotlib_converters()
        #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!############
            
        matplotlib.rcParams.update({"font.size":24})
        
        fig=plt.figure(figsize=(16,7))
        # ERA-5 IVT
        ax1=fig.add_subplot(111)
        ax1.set_ylabel("IVT (kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)")
        
        ax1.plot(era5["Interp_IVT"].index.time,
                 np.array(era5["Interp_IVT"]),ls='--',lw=3,
                 color="green",label="ERA-5")
        if not flight[0]=="RF08":
            # RF08 has only one or no (?) dropsonde which makes the plotting
            # more complicated
            ax1.plot(sonde_ivt.index.time,
                 np.array(sonde_ivt),
                 linestyle='',markersize=20,marker='^',color="lightgreen",
                 markeredgecolor="black",label="Synthetic Sondes")
        # ICON
        ax1.plot(icon_ivt.index.time,np.array(icon_ivt),
             lw=3,color="darkgreen",label="ICON")
        lower_lim=era5["Interp_IVT"].min()//50*50
        upper_lim=era5["Interp_IVT"].max()//50*50+100
        ax1.set_ylim([lower_lim,upper_lim])
        ax1.legend(loc="upper center",ncol=3,fontsize=18)
        ax1.set_xlabel('')
        ax1.set_xlabel('Time (UTC)')
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(3)
        ax1.tick_params(length=10,width=3)
        sns.despine(offset=10)
        figname=AR_of_day+"_"+flight+"_IVT_ERA5_Synthetic_Sondes_ICON"
        figname=figname+".png"    
        if save_figure:
            fig.savefig(plot_path+figname,dpi=300,bbox_inches="tight")
            print("Figure saved as: ",plot_path+figname)
        else:
            print(figname, "not saved as file")
        return None        


    def plot_sonde_quicklook(self,cmpgn_cls,
                         time_release=pd.Timestamp("2016-10-13 13:00")):
    
        date=time_release.date()

        nearest_sonde,sonde_no,exemplaric_sonde=cmpgn_cls.load_dropsonde_data(
                                                    date,print_arg="yes",
                                                    dt=time_release,plotting="no")

        sonde_quicklook=plt.figure(figsize=(8,11))
        matplotlib.rcParams.update({"font.size":20})
        ax1=sonde_quicklook.add_subplot(121)
        ax1.errorbar(y=exemplaric_sonde["Height"]/1000,
                     x=exemplaric_sonde["Wspeed_(m/s)"],
                     xerr=0.1,color="darkmagenta",
                     label="Accuracy:\n 0.1 m$\mathrm{s}^{-1}$")
        ax1.set_xlim([10,22])
        ax1.set_xlabel("Windspeed (m$\mathrm{s}^{-1}$)")
        ax1.legend(loc="upper right",fontsize=14)

        ax1.set_ylim([0,13])
        ax2=sonde_quicklook.add_subplot(122,sharey=ax1)
        ax2.errorbar(y=exemplaric_sonde["Height"]/1000,x=exemplaric_sonde["RH_(%)"],
                     xerr=2,color="blue",label="Accuracy: 2%")
        ax2.set_xlim([0,100])
        ax2.set_xlabel("Relative Humidity (%)")
        
        ax1.set_ylabel("Height (km)")
        ax2.yaxis.set_visible(False)                    
        ax2.legend(loc="upper right",fontsize=14)
        sns.despine(offset=20)
        ax2.spines["left"].set_visible(False)
        file_name=self.flight+"_Exemplaric_Sounding_No_"+str(sonde_no)+".png"
        sonde_quicklook.savefig(self.plot_path+file_name,dpi=300,bbox_inches="tight")
        print("Sounding saved as: ",self.plot_path+file_name)
        return None

    def plot_IVT_vertical_variability(self,subsample_day="",save_figure=True,
                                      undefault_path="default",
                                      manuscript_figure=True):
        # start plotting
        with_sondes=False
        #if not hasattr(self,"icon_p_quantile"):
        self.calc_vertical_quantiles(do_all_preps=True)
        if not self.sounding_frequency==None:
            self.calc_vertical_quantiles(use_grid=False,
                                         do_all_preps=True)
            pressure=self.sonde_mean["Pres"]
            if pressure.max()>1100:
                pressure=pressure/100 
            if self.sonde_mean["q"]<1:
               self.sonde_mean["q"]*=1000
            with_sondes=True
            
        fig=plt.figure(figsize=(18,12))
        matplotlib.rcParams.update({"font.size":22})
        model_pressure=self.grid_mean["p"]
        if model_pressure.max()>1100:
            model_pressure=model_pressure/100
        if model_pressure.max()<100:
            model_pressure=model_pressure*100
        if self.grid_mean["q"].max()<1:
            self.grid_mean["q"]*=1000
            self.grid_std["q"]*=1000
            self.grid_mean["moist_transport"]*=1000
            self.grid_std["moist_transport"]*=1000
        #Define axis width
        ax1=fig.add_subplot(131)
        # Specific Humidity
        ax2=fig.add_subplot(132,sharey=ax1)
        # Moisture transport
        ax3=fig.add_subplot(133,sharey=ax2)
        
        for axis in ["left","bottom"]:
                        ax1.spines[axis].set_linewidth(2.0)
                        ax1.xaxis.set_tick_params(width=2,length=10)
                        ax1.yaxis.set_tick_params(width=2,length=10)
                        ax2.spines[axis].set_linewidth(2.0)
                        ax2.xaxis.set_tick_params(width=2,length=10)
                        ax2.yaxis.set_tick_params(width=2,length=10)
                        ax3.spines[axis].set_linewidth(2.0)
                        ax3.xaxis.set_tick_params(width=2,length=10)
                        ax3.yaxis.set_tick_params(width=2,length=10)
        #---------------------------------------------------------------------#
        #Sondes
        # Wind
        if with_sondes:
            ax1.errorbar(self.sonde_mean["Wind"],pressure,
                     xerr=self.sonde_std["Wind"],
                     color="magenta",fmt='v',alpha=0.8)
            ax2.errorbar(self.sonde_mean["q"],pressure,
                     xerr=self.sonde_std["q"],color="blue",
                     fmt='x',alpha=0.8)
            ax3.errorbar(self.sonde_mean["moist_transport"],
                     pressure.loc[self.sonde_mean["moist_transport"].index],
                     xerr=self.sonde_std["moist_transport"],
                     color="black",fmt='o',alpha=0.8)
            self.sonde_mean["moist_transport"]=\
                self.sonde_mean["moist_transport"].loc[\
                                    self.sonde_std["moist_transport"].index]
            if not self.sounding_frequency=="Upsampled":
                sonde_label="Dropsondes"
            else: 
                sonde_label="Upsampled \n Sondes"
        
            if self.grid_sounding_profiles:
                sonde_label=" Synthetic\n"+sonde_label
        #Extreme day
        if subsample_day!="":
            sub_mean={}
            sub_mean["q"]=self.grid_ivt["q"].loc[subsample_day].mean(axis=0)
            sub_mean["wind"]=self.grid_ivt["wind"].loc[subsample_day].mean(axis=0)
            sub_mean["moist_transport"]=self.grid_ivt["moist_transport"].loc[\
                                                                subsample_day].\
                mean(axis=0)
            sub_std={}
            sub_std["q"]=self.grid_ivt["q"].loc[subsample_day].std(axis=0)
            sub_std["wind"]=self.grid_ivt["wind"].loc[subsample_day].std(axis=0)
            sub_std["moist_transport"]=self.grid_ivt["moist_transport"].loc[\
                                                                subsample_day].\
                std(axis=0)
            ax1.errorbar(sub_mean["wind"],model_pressure,
                     xerr=sub_std["wind"],
                     color="magenta",fmt='v',alpha=0.8)
            ax2.errorbar(sub_mean["q"],model_pressure,
                     xerr=sub_std["q"],color="blue",
                     fmt='x',alpha=0.8)
            ax3.errorbar(sub_mean["moist_transport"],
                     model_pressure,
                     xerr=sub_std["moist_transport"],
                     color="black",fmt='o',alpha=0.8)
            
        #---------------------------------------------------------------------#
        # Model grid
        ax1.fill_betweenx(y=model_pressure,
                  x1=self.grid_mean["wind"]-self.grid_std["wind"],
                  x2=self.grid_mean["wind"]+self.grid_std["wind"],
                  color="mistyrose")
        ax2.fill_betweenx(y=model_pressure,
                  x1=(self.grid_mean["q"]-self.grid_std["q"]),
                  x2=(self.grid_mean["q"]+self.grid_std["q"]),
                  color="lightsteelblue")
        ax3.fill_betweenx(y=model_pressure,
                  x1=(self.grid_mean["moist_transport"]-\
                      self.grid_std["moist_transport"]),
                  x2=(self.grid_mean["moist_transport"]+\
                      self.grid_std["moist_transport"]),
                  color="darkgrey")
        #---------------------------------------------------------------------#
        # Plot specifications
        # ax1
        xlim_wind_max=self.grid_mean["wind"].max()//10*10+10
        ax1.invert_yaxis()
        plt.yscale("log")
        ax1.set_xlabel(r'${v}_{\mathrm{h}}$ (m/s)')
        ax1.set_ylabel("Pressure (hPa)")
        ax1.set_yticks([400,500,600,700,850,925,1000])
        ax1.set_yticklabels(["400","500","600","700","850","925","1000"])
        ax1.set_ylim([1000,350])
        ax1.set_xlim([5,xlim_wind_max+5])
        
        # ax2
        xlim_q_max=self.grid_mean["q"].max()//6*6+6  
        ax2.set_xlabel("q (g/kg)")
        ax2.set_xlim([0,xlim_q_max])
        ax2.spines["left"].set_visible(False)
        ax2.yaxis.set_visible(False)
        
        # ax3 
        xlim_transport_max=self.grid_mean["moist_transport"].max()//5*5+10
        ax3.set_xlim([0,xlim_transport_max])
        ax3.set_xlabel(r'$\frac{1}{g}\cdot q\cdot{v}_{\mathrm{h}}$'+\
                       ' (g/kg$\cdot\mathrm{s}$)')
        ax3.spines["left"].set_visible(False)
        ax3.yaxis.set_visible(False)
        
        # Axis spines handling
        sns.despine(offset=15)
        
        if with_sondes:
            legend_elements=[mlines.Line2D([0],[0],color="magenta",lw=3,
                                marker="v",markersize=10,label=sonde_label),
            Patch(facecolor="mistyrose",edgecolor="salmon",
                           label="ICON")]
            ax1.legend(handles=legend_elements,loc="upper left")
        
        elif subsample_day!=" ":
            legend_elements=[mlines.Line2D([0],[0],color="magenta",lw=3,
                                marker="v",markersize=10,label=subsample_day)]
            #Patch(facecolor="mistyrose",edgecolor="salmon",
            #               label=subsample_day)]
            ax1.legend(handles=legend_elements,loc="upper left")
        else:
            pass
        ## Add height z
        h_pos=xlim_transport_max#+0.1*xlim_transport_max
        
        
        ivt_p=pd.Series(self.grid_ivt_quant_p)
        if ivt_p.values.max()<100:
            ivt_p=ivt_p*100
        #Vertical contribution to IVT    
        l1=mlines.Line2D([h_pos,h_pos],[1000,ivt_p["50"]],
                     label="50%",color="darkgreen",linestyle='-',lw=4)
        l1_hline=mlines.Line2D([h_pos-0.05*h_pos,h_pos+0.05*h_pos],[ivt_p["50"],
                                                      ivt_p["50"]],
                               color="black",lw=2)
        
        l2=mlines.Line2D([h_pos,h_pos],[ivt_p["50"],ivt_p["75"]],
                  label="25%",color="green",linestyle='--',lw=4)
        l2_hline=mlines.Line2D([h_pos-0.05*h_pos,h_pos+0.05*h_pos],[ivt_p["75"],
                                                      ivt_p["75"]],
                        color="black",lw=2)
        
        l3=mlines.Line2D([h_pos,h_pos],[ivt_p["75"],ivt_p["90"]],
                  label="15%",color="forestgreen",linestyle='-.',lw=4)
        l3_hline=mlines.Line2D([h_pos-0.05*h_pos,h_pos+0.05*h_pos],[ivt_p["90"],
                                                      ivt_p["90"]],
                        color="black",lw=2)
        
        l4=mlines.Line2D([h_pos,h_pos],[ivt_p["90"],ivt_p["97"]],
                  label="7%",color="olive",linestyle='-.',lw=4)
        l4_hline=mlines.Line2D([h_pos-h_pos*0.05,h_pos+h_pos*0.05],
                               [ivt_p["97"],ivt_p["97"]],
                                color="black",lw=2)
        
        l5=mlines.Line2D([h_pos,h_pos],[ivt_p["97"],self.grid_mean["p"].iloc[0]/100],
                          label="3%",color="yellowgreen",linestyle=':',lw=4)
        l5_hline=mlines.Line2D([h_pos-0.05*h_pos,h_pos+0.05*h_pos],
                                [self.grid_mean["p"].iloc[0]/100,
                                self.grid_mean["p"].iloc[0]/100],
                        color="black",lw=2)

        ax3.add_line(l1)
        ax3.add_line(l1_hline)
        ax3.add_line(l2)
        ax3.add_line(l2_hline)
        ax3.add_line(l3)
        ax3.add_line(l3_hline)
        ax3.add_line(l4)
        ax3.add_line(l4_hline)
        ax3.add_line(l5)
        ax3.add_line(l5_hline)
        handles, labels=ax3.get_legend_handles_labels()
        ax3.legend(handles[::-1],labels[::-1],title="Fraction \nof IVT",
           loc="upper center")
        plt.subplots_adjust(wspace=0.2)
        
        if save_figure:
            
            if not manuscript_figure:
                fig.suptitle(self.flight+": "+self.ar_of_day+" "+\
                         "lateral variability",
                             fontsize=22,y=0.95)
                major_name=self.ar_of_day+"_"+self.flight+\
                        "_IVT_lateral_variability_"
            else:
                major_name="Fig10_IVT_lateral_variability"
            file_format=".pdf"
            if self.sounding_frequency=="standard":
                if not self.grid_sounding_profiles:
                    major_name=major_name+"Dropsondes_"
                else:
                    pass
            elif self.sounding_frequency!=None:
                major_name=major_name+"Upsampled_Dropsondes_"
            else:
                pass
            if self.grid_sounding_profiles:
                major_name=major_name+"_Synthetic_"+\
                    self.grid_data_name+"_Sondes"
            if not subsample_day=="":
                major_name=major_name+"_"+subsample_day
            plot_file=major_name+file_format
            if not undefault_path=="default":
                plot_path=undefault_path
            else:
                plot_path=self.plot_path+self.flight+"/"
            fig.savefig(plot_path+plot_file,
                    dpi=300,bbox_inches="tight")
            print("Figure saved as: ",
                  plot_path+plot_file)
        return None
    
    def plot_TIVT_error_study(self,resolution_study=False):
        sonde_numbers=[key for key in self.TIVT.keys() if "sondes_no" in key]
        error_columns=[int(col[9:]) for col in sonde_numbers]
        error_columns=np.sort(error_columns)
        errors=pd.DataFrame(data=np.nan,index=self.TIVT["sondes_no9"].index,
                                    columns=[str(err) for err in error_columns])
        for num in errors.columns:
            errors[num]=self.TIVT["sondes_no"+num]["Rel_BIAS"]
                    
                    
                    #synthetic_TIVTs["Error_Trends"]=errors
        #Start plotting
        from matplotlib import ticker as tick
        import seaborn as sns
        #Start plotting
        sonde_freq_fig=plt.figure(figsize=(16,12), dpi= 300)
        matplotlib.rcParams.update({'font.size': 28})
        
        if not resolution_study:
            x_var = 'Number of Sondes \n AR Cross-Section'
            colors= {"3":"darkred","4":"red","5":"darkorange",
                     "6":"yellow","8":"greenyellow","9":"green"}
            ax1=sonde_freq_fig.add_subplot(111)
            ax1.axhline(0,color="grey",ls="--",lw=2,zorder=1)
            colors_taken           = [colors[sonde_no] for sonde_no in colors.keys()]
            #tips=sns.load_dataset("tips")
            sns.boxplot(data=errors.iloc[:,::-1],palette=colors_taken[::-1], 
                        notch=False,zorder=0)
            for patch in ax1.artists:
                r, g, b, a = patch.get_facecolor()
                patch.set_facecolor((r, g, b, .5))
            for axis in ["left","bottom"]:
                ax1.spines[axis].set_linewidth(3)
                ax1.xaxis.set_tick_params(width=3,length=10)
                ax1.yaxis.set_tick_params(width=3,length=10)
                
            ax1.set_ylabel("Rel. Bias in TIVT")
            sns.despine(offset=10)
            ax1.set_xlabel(x_var)
            ax1.set_ylim([-0.6,0.2])
            fig_name=self.ar_of_day+"_"+self.TIVT["grid_name"]+\
                        "_Synthetic_Sonde_Number_TIVT_BIAS.png"
            fig_path=self.plot_path+self.flight+"/"
            sonde_freq_fig.savefig(fig_path+fig_name,
                               dpi=300,bbox_inches="tight")
            print("Figure successfully saved as: ",fig_path+fig_name)
    
    @staticmethod
    def multiplot_inflow_outflow_IVT_sectors(cmpn_cls,HALO_dict,HMP_dict,
                                             grid_name,plot_path=""):
        import matplotlib
        matplotlib.rcParams.update({"font.size":16})
        import atmospheric_rivers as AR
        Flights_inflow_dict={}
        Flights_outflow_dict={}
        Flights_TIVT_inflow={}
        Flights_TIVT_outflow={}
        
        flights_of_dates={#"North_Atlantic_Run":
                      20180224:"SRF02",
                      20190319:"SRF04",
                      20200416:"SRF07",
                      20200419:"SRF08",
                      #Second Synthetic Study
                      20110317:"SRF02",
                      20110423:"SRF03",
                      20150314: "SRF08",
                      20160311: "SRF09",
                      20180225:"SRF12"}
        
        for flight in HMP_dict.keys():
            Flights_inflow_dict[flight], Flights_outflow_dict[flight]=\
                AR.Atmospheric_Rivers.locate_AR_cross_section_sectors(
                    HALO_dict,HMP_dict,flight)
            Flights_TIVT_inflow[flight],Flights_TIVT_outflow[flight]=\
                AR.Atmospheric_Rivers.calc_TIVT_of_sectors(
                    Flights_inflow_dict[flight],Flights_outflow_dict[flight],
                    grid_name)
        row_number=3
        col_number=int(len(HMP_dict.keys())/row_number)+len(HMP_dict.keys()) % row_number
        f,ax=plt.subplots(nrows=row_number,ncols=col_number,
                          figsize=(18,12),sharex=True,sharey=True)
        i=0
        print(ax)
        import seaborn as sns
        
        if not grid_name=="ERA5":
            ivt_var_arg="highres_Interp_IVT"
        else:
            ivt_var_arg="Interp_IVT"
            
        for flight in HMP_dict.keys():
            # Take relevant IVT dataframes from dict
            hmp_inflow   = Flights_inflow_dict[flight]["entire_inflow"]
            hmp_outflow  = Flights_outflow_dict[flight]["entire_outflow"]
        
            #ar_inflow    = Flights_inflow_dict[flight]["AR_inflow"]
            #ar_outflow   = Flights_outflow_dict[flight]["AR_outflow"]
            inflow_core  = Flights_inflow_dict[flight]["AR_inflow_core"]
            outflow_core = Flights_outflow_dict[flight]["AR_outflow_core"]
    
            ar_inflow_warm_sector  = Flights_inflow_dict[flight]["AR_inflow_warm_sector"]
            ar_inflow_cold_sector  = Flights_inflow_dict[flight]["AR_inflow_cold_sector"]
            ar_outflow_warm_sector = Flights_outflow_dict[flight]["AR_outflow_warm_sector"]
            ar_outflow_cold_sector = Flights_outflow_dict[flight]["AR_outflow_cold_sector"]
        
        
            # Take relevant TIVT values from dict
            TIVT_inflow_total=Flights_TIVT_inflow[flight]["total"]
            TIVT_outflow_total=Flights_TIVT_outflow[flight]["total"]
            TIVT_inflow_core=Flights_TIVT_inflow[flight]["core"]
            TIVT_outflow_core=Flights_TIVT_outflow[flight]["core"]
            
            if len(ax.shape)>=2:
                if i<col_number:
                    horizontal_field=i
                    plot_ax=ax[0,horizontal_field]
                elif i<2*col_number:
                    horizontal_field=i-col_number
                    plot_ax=ax[1,horizontal_field]
                    
                else:
                    horizontal_field=i-2*col_number
                    plot_ax=ax[2,horizontal_field]
                    plot_ax.set_xlabel("IVT max distance (km)")
                if horizontal_field==0:
                    plot_ax.set_ylabel("IVT ($\mathrm{kgm}^{-1}\mathrm{s}^{-1})$")
                
            else:
                horizontal_field=i
                plot_ax=ax[i]
                plot_ax.set_ylabel("IVT ($\mathrm{kgm}^{-1}\mathrm{s}^{-1})$")
                
            plot_ax.plot(hmp_inflow["IVT_max_distance"]/1000,hmp_inflow[ivt_var_arg],
             color="lightblue",lw=8,label="Total AR (in): TIVT=")
            line_core_in=plot_ax.plot(inflow_core["IVT_max_distance"]/1000,
                              inflow_core[ivt_var_arg],lw=2,color="darkblue",
                              label="AR core (in): TIVT="+\
                                          str((TIVT_inflow_core/1e6).round(1)))
    
            plot_ax.plot(hmp_outflow["IVT_max_distance"]/1000,
                 hmp_outflow[ivt_var_arg],color="orange",lw=8)
    
            line_core_out=plot_ax.plot(outflow_core["IVT_max_distance"]/1000,
                               outflow_core[ivt_var_arg],
                               lw=2,color="darkred",
                               label="AR core (out): TIVT="+\
                                   str((TIVT_outflow_core/1e6).round(1)))

        
            plot_ax.plot(ar_inflow_warm_sector["IVT_max_distance"]/1000,
                 ar_inflow_warm_sector[ivt_var_arg],
                 lw=3,ls=":",color="darkblue")
        
            plot_ax.plot(ar_inflow_cold_sector["IVT_max_distance"]/1000,
                 ar_inflow_cold_sector[ivt_var_arg],
                 lw=3,ls="-.",color="darkblue")
    
            plot_ax.plot(ar_outflow_warm_sector["IVT_max_distance"]/1000,
                 ar_outflow_warm_sector[ivt_var_arg],
                 lw=3,ls=":",color="darkred")
            plot_ax.plot(ar_outflow_cold_sector["IVT_max_distance"]/1000,
                 ar_outflow_cold_sector[ivt_var_arg],
                 lw=3,ls="-.",color="darkred")
            plot_ax.set_title(flight,fontsize=16,loc="left",y=0.9)
            plot_ax.set_xlim([-500,500])
            plot_ax.set_ylim([100,700])
            for axis in ["left","bottom"]:
                plot_ax.spines[axis].set_linewidth(2)
                plot_ax.tick_params(length=6,width=2)

            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D

            legend_patches = [Patch(facecolor='darkblue', edgecolor='k',
                                    label='TIVT (in)='+\
                                    str((TIVT_inflow_total/1e6).round(1))+\
                                    "$\cdot 1\mathrm{e}6\,\mathrm{kgs}^{-1}$"),
                              Patch(facecolor='darkred', edgecolor='k',
                                    label='TIVT (out)='+\
                                    str((TIVT_outflow_total/1e6).round(1))+\
                                    "$\cdot 1\mathrm{e}6\,\mathrm{kgs}^{-1}$")]
            legend_loc="upper right"
            ivt_max=hmp_inflow[ivt_var_arg].max()
            print(ivt_max)
            if ivt_max>450:
                legend_loc="lower center"
            #line_core_in[0],line_core_out[0],
            lgd = plot_ax.legend(handles=[\
                                      legend_patches[0],legend_patches[1]],
                             loc=legend_loc,fontsize=10,ncol=1)
                
            i+=1
        sns.despine(offset=10)
        fig_name="Fig12_"+grid_name+"_AR_TIVT_cases_overview.pdf"
        if plot_path=="":
            plt_path=cmpn_cls.plot_path
        else:
            plt_path=plot_path
        f.savefig(plt_path+fig_name,
                    dpi=60,bbox_inches="tight")
        print("Figure saved as:", cmpn_cls.plot_path+fig_name)
        return None
    def plot_inflow_outflow_IVT_sectors(cmpgn_cls,AR_inflow_dict,AR_outflow_dict,
                                        TIVT_inflow_dict,TIVT_outflow_dict,
                                        grid_name,analysed_flight):
        if not grid_name=="ERA5":
            ivt_var_arg="highres_Interp_IVT"
        else:
            ivt_var_arg="Interp_IVT"
        
        # Take relevant IVT dataframes from dict
        hmp_inflow   = AR_inflow_dict["entire_inflow"]
        hmp_outflow  = AR_outflow_dict["entire_outflow"]
    
        ar_inflow    = AR_inflow_dict["AR_inflow"]
        ar_outflow   = AR_outflow_dict["AR_outflow"]
        inflow_core  = AR_inflow_dict["AR_inflow_core"]
        outflow_core = AR_outflow_dict["AR_outflow_core"]
    
        ar_inflow_warm_sector  = AR_inflow_dict["AR_inflow_warm_sector"]
        ar_inflow_cold_sector  = AR_inflow_dict["AR_inflow_cold_sector"]
        ar_outflow_warm_sector = AR_outflow_dict["AR_outflow_warm_sector"]
        ar_outflow_cold_sector = AR_outflow_dict["AR_outflow_cold_sector"]
        
        
        # Take relevant TIVT values from dict
        TIVT_inflow_total=TIVT_inflow_dict["total"]
        TIVT_outflow_total=TIVT_outflow_dict["total"]
        TIVT_inflow_core=TIVT_inflow_dict["core"]
        TIVT_outflow_core=TIVT_outflow_dict["core"]
        
        import seaborn as sns
        ##############################################################################################################################
        fig=plt.figure(figsize=(12,9))
        ax1=fig.add_subplot(111)
    
        ax1.plot(hmp_inflow["IVT_max_distance"]/1000,hmp_inflow[ivt_var_arg],
             color="lightblue",lw=8,label="Total AR (in): TIVT=")
        line_core_in=ax1.plot(inflow_core["IVT_max_distance"]/1000,
                              inflow_core[ivt_var_arg],lw=2,color="darkblue",
                              label="AR core (in): TIVT="+\
                                          str((TIVT_inflow_core/1e6).round(1)))
    
        ax1.plot(hmp_outflow["IVT_max_distance"]/1000,
                 hmp_outflow[ivt_var_arg],color="orange",lw=8)
    
        line_core_out=ax1.plot(outflow_core["IVT_max_distance"]/1000,
                               outflow_core[ivt_var_arg],
                               lw=2,color="darkred",
                               label="AR core (out): TIVT="+\
                                   str((TIVT_outflow_core/1e6).round(1)))

        
        ax1.plot(ar_inflow_warm_sector["IVT_max_distance"]/1000,
                 ar_inflow_warm_sector[ivt_var_arg],
                 lw=3,ls=":",color="darkblue")
        
        ax1.plot(ar_inflow_cold_sector["IVT_max_distance"]/1000,
                 ar_inflow_cold_sector[ivt_var_arg],
                 lw=3,ls="-.",color="darkblue")
    
        ax1.plot(ar_outflow_warm_sector["IVT_max_distance"]/1000,
                 ar_outflow_warm_sector[ivt_var_arg],
                 lw=3,ls=":",color="darkred")
        ax1.plot(ar_outflow_cold_sector["IVT_max_distance"]/1000,
                 ar_outflow_cold_sector[ivt_var_arg],
                 lw=3,ls="-.",color="darkred")
        """
        ax1.plot(hmp_inflow["IVT_max_distance"]/1000,hmp_inflow["Interp_IVT"],color="lightblue")
        line_core_in=ax1.plot(inflow_core["IVT_max_distance"]/1000,inflow_core["Interp_IVT"],lw=2,color="darkblue",
            label="AR core (in): TIVT="+str((TIVT_inflow_core/1e6).round(1))+"$\cdot 1\mathrm{e}6\,\mathrm{kgs}^{-1}$")
        ax1.plot(hmp_outflow["IVT_max_distance"]/1000,hmp_outflow["Interp_IVT"],color="orange")

        line_core_out=ax1.plot(outflow_core["IVT_max_distance"]/1000,outflow_core["Interp_IVT"],lw=2,color="darkred",
            label="AR core (out): TIVT="+str((TIVT_outflow_core/1e6).round(1))+"$\cdot 1\mathrm{e}6\,\mathrm{kgs}^{-1}$")
    
        ax1.plot(ar_inflow_warm_sector["IVT_max_distance"]/1000,ar_inflow_warm_sector["Interp_IVT"],
             lw=3,ls=":",color="darkblue")
        ax1.plot(ar_inflow_cold_sector["IVT_max_distance"]/1000,ar_inflow_cold_sector["Interp_IVT"],
             lw=3,ls="-.",color="darkblue")
        ax1.plot(ar_outflow_warm_sector["IVT_max_distance"]/1000,ar_outflow_warm_sector["Interp_IVT"],
             lw=3,ls=":",color="darkred")
        ax1.plot(ar_outflow_cold_sector["IVT_max_distance"]/1000,ar_outflow_cold_sector["Interp_IVT"],
             lw=3,ls="-.",color="darkred")
        """
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_patches = [Patch(facecolor='darkblue', edgecolor='k',
                         label='AR total (in): TIVT='+\
                             str((TIVT_inflow_total/1e6).round(1))+\
                             "$\cdot 1\mathrm{e}6\,\mathrm{kgs}^{-1}$"),
                   Patch(facecolor='darkred', edgecolor='k',
                         label='AR total (out): TIVT='+\
                             str((TIVT_outflow_total/1e6).round(1))+\
                             "$\cdot 1\mathrm{e}6\,\mathrm{kgs}^{-1}$")]

        lgd = ax1.legend(handles=[line_core_in[0], 
                          line_core_out[0],
                          legend_patches[0],legend_patches[1]],
                 loc='lower center',fontsize=12,ncol=2)

        ax1.set_ylabel("IVT ($\mathrm{kgm}^{-1}\mathrm{s}^{-1})$")
        ax1.set_xlabel("IVT max distance (km)")
        ax1.set_ylim([0,600])
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(2)
            ax1.tick_params(length=8,width=2)

        sns.despine(offset=10)
        plot_path=cmpgn_cls.plot_path+"/budget/"
        fig_name=analysed_flight+"_"+grid_name+"_AR_TIVT_flow.png"
        fig.savefig(plot_path+fig_name,
                    dpi=300,bbox_inches="tight")

        print("Figure saved as:", cmpgn_cls.plot_path+fig_name)
        return None
        ## ----> has to be modified in the course of the week
        # if resolution_study:
        #     #check if data set exists
        #     if os.path.isfile(tivt_res_csv_file):
        #         df_tivt_res=pd.read_csv(tivt_res_csv_file)
        #         df_tivt_res.index=df_tivt_res["Index_Shift_Factor"]
        #         del df_tivt_res["Index_Shift_Factor"]
        #     else:
        #         raise FileNotFoundError("The File ",tivt_res_csv_file,
        #                                 " does not exist and has to be constructed.",
        #                                 "Set creating_data to True and plotting_data=False")
                        
        #     cmap=matplotlib.cm.get_cmap('RdYlGn_r')
        #     color_list=cmap([np.linspace(0,1,df_tivt_res.shape[1]-1)])[0]
        #     x_var = 'seconds between sonde releases \n in AR cross-section'
        #     #Start plotting
        #     ax1=sonde_res_fig.add_subplot(111)
        #     real_total_ivt=df_tivt_res["REAL-ICON-TIVT"].mean()
        #     del df_tivt_res["REAL-ICON-TIVT"]
        #     errors=(df_tivt_res-real_total_ivt)/real_total_ivt
        #     x_data=[int(df_tivt_res.columns[col][:-1]) for col in range(df_tivt_res.shape[1])]
        #     errors.columns=x_data
        #     sns.boxplot(data=errors,palette=color_list, notch=False)
        #     for axis in ["left","bottom"]:
        #         ax1.spines[axis].set_linewidth(2.0)
        #         ax1.xaxis.set_tick_params(width=2,length=10)
        #         ax1.yaxis.set_tick_params(width=2,length=10)
            
        #     ax1.set_ylabel("Rel. Error in TIVT")
        #     sns.despine(offset=10)
        #     ax1.set_xlabel(x_var)
        #     ax1.set_ylim([-0.26,0.06])
        #     fig_name=ar_of_day+"_"+flight+"_Synthetic_Sonde_Frequency_TIVT_BIAS.png"
        #     sonde_res_fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        #     print("Figure successfully saved as: ",plot_path+fig_name)
        #             # return None;
        #%%
        # Load era5 dataset
        # era_path="C://Users/u300737/Desktop/PhD_UHH_WIMI/Work/GIT_Repository/NAWDEX/data/ERA-5/"
        # if synthetic_icon_lat is not None:
        #     era_path=era_path+"Latitude_"+str(synthetic_icon_lat)+"/"
        # era_file=ar_of_day+"_HMP_ERA_HALO_"+flight+"_"+date+".csv"
        # era5_df=pd.read_csv(era_path+era_file)
        # era5_df.index=pd.DatetimeIndex(era5_df["Unnamed: 0"])
        # del era5_df["Unnamed: 0"]
        # ## Plot the synthetic part
        # plot_IVT_icon_era5_synthetic_sondes(era5_df,sonde_ivt,icon_ivt,
        #                                         flight,ar_of_day,plot_path,
        #                                         save_figure=True)
        # #calculate standard deviatio
        # sonde_std_q=sonde_q.std(axis=0)
        # sonde_std_wind=sonde_wind.std(axis=0)
        # sonde_std_moist_transport=sonde_moist_transport.std(axis=0)
        
        # icon_std_q=icon_q.std(axis=0)
        # icon_std_wind=icon_wind.std(axis=0)
        # icon_std_moist_transport=icon_moist_transport.std(axis=0)
        
                
        # """
        # return None;


        
    #def calculate_total_ivt(ivt_series,distance_arg,_2nd_arg_is_speed=False):
        #    total_ivt=np.nan
        #    if _2nd_arg_is_speed:
            #        time_delta=pd.Timedelta(distance_arg.index)
            #        distance=distance_arg*1
    
    
    #    return total_ivt

#%%
"""
#-----------------------------------------------------------------------------#
#%% Major configurations    
#------------------------------------------------------------------------------#
if synthetic_icon_lat!=None:
    halo_df["latitude"]=halo_df["latitude"]+synthetic_icon_lat
    print("HALO lat position changed by:", str(synthetic_icon_lat), " Degrees")
    
halo_df.index=pd.DatetimeIndex(halo_df["Unnamed: 0"])
copy_halo_df=halo_df.copy()
if sounding_frequency=="standard":
    halo_df=halo_df.loc[sonde_p.index]
halo_df["Timedelta"]=pd.to_datetime(halo_df['Unnamed: 0']).diff()
copy_halo_df["Timedelta"]=pd.to_datetime(copy_halo_df['Unnamed: 0']).diff()

halo_df=halo_df.loc[sonde_p.index[0]:sonde_p.index[-1]]
halo_df["Seconds"]=halo_df["Timedelta"].dt.seconds
halo_df["delta_Distance"]=halo_df["Seconds"]*halo_df["groundspeed"]
halo_df["cumsum_distance"]=halo_df["delta_Distance"].cumsum()
    
del halo_df["Unnamed: 0"]

copy_halo_df=copy_halo_df.loc[sonde_p.index[0]:sonde_p.index[-1]]
copy_halo_df["Seconds"]=copy_halo_df["Timedelta"].dt.seconds
copy_halo_df["delta_Distance"]=copy_halo_df["Seconds"]*copy_halo_df["groundspeed"]
copy_halo_df["cumsum_distance"]=copy_halo_df["delta_Distance"].cumsum()
    
del copy_halo_df["Unnamed: 0"]

ivt_logger.icon_ivt_logger.info("Cut Halo to Upsampled Sounding and IVT")

#sys.exit()

if only_grid_sounding_profiles:
    ivt_logger.icon_ivt_logger.info("Only consider synthetic sounding profiles in ICON")
    icon_p=icon_p.loc[sonde_p.index]
    icon_q=icon_q.loc[sonde_q.index]
    icon_wind=icon_wind.loc[sonde_wind.index]
    #icon_ivt=icon_ivt.loc[sonde_ivt.index]
    #icon_v=icon_v.loc[sonde_wind.index]


#icon_ivt_cumsum2=scint.cumtrapz(y=icon_mean_moist_transport[::-1],x=icon_p.mean(axis=0)[::-1])
icon_ivt=pd.Series(data=scint.trapz(y=icon_moist_transport,x=icon_p),index=icon_p.index)

#%% File names depending on flight and AR cross-section of the day
if synthetic_observations:
    plot_path=plot_path+"Synthetic_Sondes/"
if synthetic_icon_lat:
    plot_path=plot_path+"Latitude_"+str(synthetic_icon_lat)+"/"
if not os.path.exists(plot_path):
        os.makedirs(plot_path)
        
tivt_res_csv_file=plot_path+ar_of_day+"_"+flight+"_"+"TIVT_Sonde_Resolution.csv"
            
##############################################################################
# ----> this is to be done after Monday

#######
#%% TIVT Comparison Sondes and ICON
#if sounding_frequency=="Upsampled" and not only_grid_sounding_profiles:
#    TIVT=calculate_and_compare_tivt_real_sondes_and_icon(icon_ivt,sonde_ivt,
#                                                    sounding_frequency,
#                                                    only_grid_sounding_profiles)

#elif sounding_frequency=="standard" and only_grid_sounding_profiles:
#    TIVT=calculate_and_compare_tivt_real_sondes_and_icon(icon_ivt,sonde_ivt,
#                                                    sounding_frequency,
#                                                    only_grid_sounding_profiles,
#                                                    plot_path,ar_of_day,flight,
#                                                    synthetic=True)

    #elif sounding_frequency=="standard" and not only_grid_sounding_profiles:
if sounding_frequency=="standard" and not only_grid_sounding_profiles:
        if synthetic_observations:
            icon_ivt=icon_ivt.loc[copy_halo_df.index]
            icon_ivt_synthetic=icon_ivt.loc[sonde_ivt.index]
            icon_q_synthetic=icon_q.loc[sonde_ivt.index]
            icon_p_synthetic=icon_p.loc[sonde_ivt.index]
            icon_wind_synthetic=icon_wind.loc[sonde_ivt.index]
            
            if create_data:
                if discrete_real_sound:
                    ivt_logger.icon_ivt_logger.info("Delete 1 sonde")
                    TIVT_synthetic_delete_1=calculate_and_compare_tivt_real_sondes_and_icon(
                                                            icon_ivt,icon_ivt_synthetic,
                                                            sounding_frequency,True,
                                                            plot_path,ar_of_day,flight,
                                                            delete_sondes=1,synthetic=True)
                    print("Delete 2 sondes")
                    ivt_logger.icon_ivt_logger.info("Delete 2 sonde")
                    TIVT_synthetic_delete_3=calculate_and_compare_tivt_real_sondes_and_icon(
                                                            icon_ivt,icon_ivt_synthetic,
                                                            sounding_frequency,True,
                                                            plot_path,ar_of_day,flight,
                                                            delete_sondes=2,
                                                            synthetic=True)
                    print("Delete 5 sondes")
                    ivt_logger.icon_ivt_logger.info("Delete 5 sonde")
                    TIVT_synthetic_delete_5=calculate_and_compare_tivt_real_sondes_and_icon(
                                                            icon_ivt,icon_ivt_synthetic,
                                                            sounding_frequency,True,
                                                            plot_path,ar_of_day,flight,
                                                            delete_sondes=5,
                                                            synthetic=True)
                    print("Delete 7 sondes")
                    ivt_logger.icon_ivt_logger.info("Delete 7 sonde")
                    TIVT_synthetic_delete_7=calculate_and_compare_tivt_real_sondes_and_icon(
                                                            icon_ivt,icon_ivt_synthetic,
                                                            sounding_frequency,True,
                                                            plot_path,ar_of_day,flight,
                                                            delete_sondes=7,
                                                            synthetic=True)
                    print("Delete 8 sondes")
                    ivt_logger.icon_ivt_logger.info("Delete 8 sonde")
                    TIVT_synthetic_delete_8=calculate_and_compare_tivt_real_sondes_and_icon(
                                                             icon_ivt,icon_ivt_synthetic,
                                                             sounding_frequency,True,
                                                             plot_path,ar_of_day,flight,
                                                             delete_sondes=8,
                                                             synthetic=True)
                    
                    TIVT_synthetic_delete_1["Sonde"].to_csv(
                        path_or_buf=plot_path+ar_of_day+"_"+flight+"_TIVT_synthetic_delete_1.csv",index=True)
                    TIVT_synthetic_delete_3["Sonde"].to_csv(
                        path_or_buf=plot_path+ar_of_day+"_"+flight+"_TIVT_synthetic_delete_3.csv",index=True)
                    TIVT_synthetic_delete_5["Sonde"].to_csv(
                        path_or_buf=plot_path+ar_of_day+"_"+flight+"_TIVT_synthetic_delete_5.csv",index=True)
                    TIVT_synthetic_delete_7["Sonde"].to_csv(
                        path_or_buf=plot_path+ar_of_day+"_"+flight+"_TIVT_synthetic_delete_7.csv",index=True)
                    TIVT_synthetic_delete_8["Sonde"].to_csv(
                        path_or_buf=plot_path+ar_of_day+"_"+flight+"_TIVT_synthetic_delete_8.csv",index=True)
                    
                    # Dirty Quicklook Comparison showing the soundes in ICON and 
                    # the ones of the sondes
                    fig_icon_sonde_comparison=plt.figure(figsize=(16,9))
                    plt.plot(icon_ivt_synthetic,color="blue",label="Synthetic")
                    plt.plot(sonde_ivt,color="orange",label="Sondes")
                    relative_bias_icon_sonde=np.mean((sonde_ivt.values-icon_ivt_synthetic.values)/icon_ivt_synthetic.values)
                    plt.suptitle("Relative Mean Error Synthetic ICON-Sounding - Dropsondes: "+str(round(relative_bias_icon_sonde,3)))
                    sns.despine(offset=10)
                    plt.ylim([100,400])
                    
                    fig_icon_sonde_comparison.savefig(plot_path+"Synthetic_ICON_Dropsondes.png",
                                                  dpi=300,bbox_inches="tight")
                if resolution_study:
                    
                    
                    resolutions=["120s","240s","360s","480s","600s","720s",
                                 "840s","1200s","1800s","2700s","3000s","3600s"]
                    resolutions_int=["120","240","360","480","600","720",
                                     "840","1200","1800","2700","3000","3600"]
                    standard_icon_tivt=calculate_tivt_from_icon_ivt(icon_ivt,copy_halo_df)
                    
                    # Old Format
                    resampled_tivt=pd.DataFrame(columns=["Resolution","TIVT_Mean",
                                                         "TIVT_Std","Resampled_Distance"])
                    
                    number_of_shifts=10
                    df_resampled_tivt=pd.DataFrame(columns=resolutions,
                                                index=np.arange(0,number_of_shifts))
                    df_resampled_tivt.index.name="Index_Shift_Factor"
                    df_resampled_tivt.name=standard_icon_tivt
                    i=0
                    for res in resolutions:
                        print("Current Resolution: ",res)
                        resampled_icon_ivt=icon_ivt.asfreq(res)
                        # 
                        shift_factor=int(float(resolutions_int[i])/number_of_shifts)
                        res_case_tivt=pd.Series(index=np.arange(0,number_of_shifts)*shift_factor)  
                        
                        # Shift index to determine the spatial variability
                        for shift in range(number_of_shifts):
                            shifter=shift_factor*shift
                            shifted_resampled_icon_ivt=icon_ivt.iloc[shifter:].asfreq(res)    
                            shifted_resampled_icon_ivt=shifted_resampled_icon_ivt.append(icon_ivt.iloc[[0,-1]])
                            shifted_resampled_icon_ivt=shifted_resampled_icon_ivt.drop_duplicates(keep="first")
                            shifted_resampled_icon_ivt=shifted_resampled_icon_ivt.sort_index()
                            resampled_halo_df=pd.DataFrame(columns=["groundspeed",
                                                        "time_Difference",
                                                        "delta_Distance"],
                                                        index=shifted_resampled_icon_ivt.index)
                            for idx in range(len(shifted_resampled_icon_ivt.index)-1):
                                groundspeed=copy_halo_df["groundspeed"].loc[shifted_resampled_icon_ivt.index[idx]:shifted_resampled_icon_ivt.index[idx+1]]
                                mean_groundspeed=groundspeed.mean()
                                resampled_halo_df["groundspeed"].iloc[idx+1]=mean_groundspeed
                                #copy_halo_df.set_index(np.arange(len(copy_halo_df))//time_frequency).mean(level=0)
                            resampled_halo_df.index=shifted_resampled_icon_ivt.index
                            
                            time_frequency=shifted_resampled_icon_ivt.index.to_series().diff()#.seconds
                            resampled_halo_df["time_Difference"]=time_frequency
                            resampled_halo_df["time_Difference"]=resampled_halo_df["time_Difference"].dt.total_seconds()
                            resampled_halo_df["delta_Distance"]=resampled_halo_df["groundspeed"]*resampled_halo_df["time_Difference"]
                            resampled_halo_df["delta_Distance"].iloc[0]=0.0
                            resampled_halo_df["cumsum_distance"]=resampled_halo_df["delta_Distance"].cumsum()
                            #print(res," ","Resampled_Distance:",
                            #      resampled_halo_df["cumsum_distance"][-1],
                            #      "Standard Distance",copy_halo_df["cumsum_distance"][-1])
                        
                            temp_tivt=calculate_tivt_from_icon_ivt(shifted_resampled_icon_ivt,
                                                               resampled_halo_df,
                                                               standard=False)
                            res_case_tivt.iloc[shift]=temp_tivt
                        i+=1
                            #df_resampled_tivt["cumsum_distance"].iloc[shift]=resampled_halo_df["cumsum_distance"][-1]
                        df_resampled_tivt.loc[:,res]=res_case_tivt.values    
                        
                    df_resampled_tivt["REAL-ICON-TIVT"]=float(df_resampled_tivt.name)    
                    df_resampled_tivt.to_csv(path_or_buf=tivt_res_csv_file,
                                             index=True)
                    print("Saved TIVT Sonde Resolution CSV File under:"+tivt_res_csv_file)
                    ivt_logger.icon_ivt_logger.info("Saved TIVT Sonde Resolution CSV File under:"+tivt_res_csv_file)
                    #OLD Without Shifting 
                        #time_frequency=resampled_icon_ivt.index.to_series().diff().mean().seconds
                        
                        #resampled_halo_df=copy_halo_df.set_index(np.arange(len(copy_halo_df))//time_frequency).mean(level=0)
                        #resampled_halo_df.index=resampled_icon_ivt.index
                        #resampled_halo_df["delta_Distance"]=resampled_halo_df["groundspeed"]*time_frequency
                        ### The last entry of delta distance has to be replaced due to different time_frequency last index in any case to maintain width of observed AR core!!
                        #resampled_icon_ivt[icon_ivt.index[-1]]=icon_ivt.iloc[-1]
                        #end_halo_df=copy_halo_df.loc[resampled_icon_ivt.index[-2]:icon_ivt.index[-1]]
                        ##resampled_halo_df[copy_halo_df.index[-1]]=copy_halo_df[-1,:]
                        ##resampled_halo_df.loc[copy_halo_df.index[-1]]=np.nan
                        ##resampled_halo_df["groundspeed"].loc[copy_halo_df.index[-1]]=end_halo_df["groundspeed"].mean() 
                        #resampled_halo_df["delta_Distance"].iloc[-1]=end_halo_df["delta_Distance"].iloc[1:-1].sum() 
                        #resampled_halo_df["cumsum_distance"]=resampled_halo_df["delta_Distance"].cumsum()
                        #temp_tivt=calculate_tivt_from_icon_ivt(resampled_icon_ivt,
                        #                                       resampled_halo_df,
                        #                                       standard=False)
                        #resampled_tivt["TIVT_Mean"].iloc[i]=temp_tivt
                        #resampled_tivt["Resampled_Distance"]=resampled_halo_df["cumsum_distance"][-1]
                    
                        #i+=1
                        
                else:
                    print("No TIVT comparison is done. Instead vertical separation of q and wind.")
                        
            if plotting_data:
                if discrete_real_sound:
                    synthetic_TIVTs={}
                    synthetic_TIVTs["delete_1"]=pd.read_csv(
                        plot_path+ar_of_day+"_"+flight+"_TIVT_synthetic_delete_1.csv")
                    del synthetic_TIVTs["delete_1"]["Unnamed: 0"]
                    synthetic_TIVTs["delete_3"]=pd.read_csv(
                        plot_path+ar_of_day+"_"+flight+"_TIVT_synthetic_delete_3.csv")
                    del synthetic_TIVTs["delete_3"]["Unnamed: 0"]
                    
                    synthetic_TIVTs["delete_5"]=pd.read_csv(
                        plot_path+ar_of_day+"_"+flight+"_TIVT_synthetic_delete_5.csv")
                
                    del synthetic_TIVTs["delete_5"]["Unnamed: 0"]
                    
                    synthetic_TIVTs["delete_7"]=pd.read_csv(
                        plot_path+ar_of_day+"_"+flight+"_TIVT_synthetic_delete_7.csv")
                    del synthetic_TIVTs["delete_7"]["Unnamed: 0"]
                    
                    synthetic_TIVTs["delete_8"]=pd.read_csv(
                        plot_path+ar_of_day+"_"+flight+"_TIVT_synthetic_delete_8.csv")
                    del synthetic_TIVTs["delete_8"]["Unnamed: 0"]
                    errors=pd.DataFrame(data=np.nan,index=synthetic_TIVTs["delete_1"].index,
                                    columns=["10","8","6","4","3"])
                    errors["10"]=synthetic_TIVTs["delete_1"]["Rel.Bias"]
                    errors["8"]=synthetic_TIVTs["delete_3"]["Rel.Bias"]
                    errors["6"]=synthetic_TIVTs["delete_5"]["Rel.Bias"]
                    errors["4"]=synthetic_TIVTs["delete_7"]["Rel.Bias"]
                    errors["3"]=synthetic_TIVTs["delete_8"]["Rel.Bias"]
                    
                    synthetic_TIVTs["Error_Trends"]=errors
                    #Start plotting
                    from matplotlib import ticker as tick
                    import seaborn as sns
                    ## Prepare data
                    #config
                    x_var = 'Number of Sondes \n AR Cross-Section'#'Day of August 2016'
                    # groupby_var = nn_complete_all_days.index.day
                    # nn_agg = nn_complete_all_days.groupby(groupby_var)
                    # vals = [series.values.tolist() for i, series in nn_agg]
                    
                    # #Define bins
                    # log_bins=np.linspace(1,4,21)
                    # exp_bins=10**log_bins
                    
                    
                    #Start plotting
                    sonde_freq_fig=plt.figure(figsize=(16,12), dpi= 300)
                    matplotlib.rcParams.update({'font.size': 24})
                    colors= {"3":"darkred","4":"red","6":"orange","8":"greenyellow","10":"green"}
                    #research_flights={"10":"RF02","12":"RF03","15":"RF04","17":"RF05","19":"RF06","22":"RF07"}
                    ax1=sonde_freq_fig.add_subplot(111)
                    
                    colors_taken           = [colors[sonde_no] for sonde_no in colors.keys()]
                    #research_flights_taken = [research_flights[day] for day in relevant_days]
                    #n, bins, patches = ax1.hist(vals, bins=exp_bins, stacked=True, density=False,color=colors_taken)
                    tips=sns.load_dataset("tips")
                    #float(np.array(erros.columns))
                    #for i in range(errors.shape[1]):
                    sns.boxplot(data=errors,palette=colors_taken[::-1], notch=False)
                    for patch in ax1.artists:
                        r, g, b, a = patch.get_facecolor()
                        patch.set_facecolor((r, g, b, .5))
                    for axis in ["left","bottom"]:
                        ax1.spines[axis].set_linewidth(2.0)
                        ax1.xaxis.set_tick_params(width=2,length=10)
                        ax1.yaxis.set_tick_params(width=2,length=10)
                        #ax1.xaxis.spines(width=3)
                    ax1.set_ylabel("Rel. Bias in TIVT")
                    sns.despine(offset=10)
                    ax1.set_xlabel(x_var)
                    ax1.set_ylim([-0.6,0.2])
                
                    #ax1.set_xlim([])
                    fig_name=ar_of_day+"_"+flight+"_Synthetic_Sonde_Number_TIVT_BIAS.png"
                    sonde_freq_fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
                    print("Figure successfully saved as: ",plot_path+fig_name)
                    # return None;
                if resolution_study:
                    #check if data set exists
                    if os.path.isfile(tivt_res_csv_file):
                        df_tivt_res=pd.read_csv(tivt_res_csv_file)
                        df_tivt_res.index=df_tivt_res["Index_Shift_Factor"]
                        del df_tivt_res["Index_Shift_Factor"]
                    else:
                        raise FileNotFoundError("The File ",tivt_res_csv_file,
                                          " does not exist and has to be constructed.",
                                          "Set creating_data to True and plotting_data=False")
                        
                    cmap=matplotlib.cm.get_cmap('RdYlGn_r')
                    color_list=cmap([np.linspace(0,1,df_tivt_res.shape[1]-1)])[0]
                    #Start plotting
                    from matplotlib import ticker as tick
                    import seaborn as sns
                    ## Prepare data
                    x_var = 'seconds between sonde releases \n in AR cross-section'#'Day of August 2016'
                    
                    #Start plotting
                    sonde_res_fig=plt.figure(figsize=(20,12), dpi=300)
                    matplotlib.rcParams.update({'font.size': 24})
                    
                    #research_flights={"10":"RF02","12":"RF03","15":"RF04","17":"RF05","19":"RF06","22":"RF07"}
                    ax1=sonde_res_fig.add_subplot(111)
                    
                    #colors_taken           = [colors[sonde_no] for sonde_no in colors.keys()]
                    #research_flights_taken = [research_flights[day] for day in relevant_days]
                    #n, bins, patches = ax1.hist(vals, bins=exp_bins, stacked=True, density=False,color=colors_taken)
                    #tips=sns.load_dataset("tips")
                    #float(np.array(erros.columns))
                    #for i in range(errors.shape[1]):
                    real_total_ivt=df_tivt_res["REAL-ICON-TIVT"].mean()
                    del df_tivt_res["REAL-ICON-TIVT"]
                    errors=(df_tivt_res-real_total_ivt)/real_total_ivt
                    
                    x_data=[int(df_tivt_res.columns[col][:-1]) for col in range(df_tivt_res.shape[1])]
                    errors.columns=x_data
                    sns.boxplot(data=errors,palette=color_list, notch=False)
                    #for patch in ax1.artists:
                    #    r, g, b, a = patch.get_facecolor()
                    #    patch.set_facecolor((r, g, b, .5))
                    for axis in ["left","bottom"]:
                        ax1.spines[axis].set_linewidth(2.0)
                        ax1.xaxis.set_tick_params(width=2,length=10)
                        ax1.yaxis.set_tick_params(width=2,length=10)
                        #ax1.xaxis.spines(width=3)
                    ax1.set_ylabel("Rel. Error in TIVT")
                    sns.despine(offset=10)
                    ax1.set_xlabel(x_var)
                    ax1.set_ylim([-0.26,0.06])
                
                    #ax1.set_xlim([])
                    fig_name=ar_of_day+"_"+flight+"_Synthetic_Sonde_Frequency_TIVT_BIAS.png"
                    sonde_res_fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
                    print("Figure successfully saved as: ",plot_path+fig_name)
                    # return None;
#sys.exit()





#%%#Dropsonde quicklook
# Load config file
#sys.exit()




#     ax2.fill_betweenx(y=pressure,
#                       x1=(sonde_mean_q.loc[pressure.index]-sonde_std_q.loc[pressure.index])*1000,
#                       x2=(sonde_mean_q.loc[pressure.index]+sonde_std_q.loc[pressure.index])*1000,
#                       color="lavender")
#     ax3.fill_betweenx(y=pressure,
#                       x1=(sonde_mean_moist_transport.loc[pressure.index]-sonde_std_moist_transport.loc[pressure.index])*1000,
#                       x2=(sonde_mean_moist_transport.loc[pressure.index]+sonde_std_moist_transport.loc[pressure.index])*1000,
#                       color="gainsboro")

# ax1.errorbar(icon_mean_wind,icon_p.mean(axis=0)/100,xerr=icon_std_wind,
#              color="magenta",fmt='v')
# ax1.invert_yaxis()
# plt.yscale("log")
# ax1.set_xlabel(r'$\vec{v}_{\mathrm{h}}$ (m/s)')
# ax1.set_ylabel("Pressure (hPa)")
# ax1.set_yticks([400,500,600,700,850,925,1000])
# ax1.set_yticklabels(["400","500","600","700","850","925","1000"])
# ax1.set_ylim([1000,350])
# ax2=fig.add_subplot(132,sharey=ax1)
# ax2.errorbar(icon_mean_q*1000,icon_p.mean(axis=0)/100,xerr=icon_std_q*1000,
#              color="blue",fmt='x')
# ax2.set_xlabel("q (g/kg)")
# #ax2.invert_yaxis()
# ax3=fig.add_subplot(133,sharey=ax2)
# ax3.errorbar(icon_mean_moist_transport*1000,icon_p.mean(axis=0)/100,xerr=icon_std_moist_transport*1000,
#              color="black",fmt='o',alpha=0.8)
# ax3.set_xlabel(r'$\frac{1}{g}\cdot q\cdot\vec{v}_{\mathrm{h}}$')
# #ax3.invert_yaxis()
# sns.despine(offset=15)
# ax2.spines["left"].set_visible(False)
# ax2.yaxis.set_visible(False)
# ax3.spines["left"].set_visible(False)
# ax3.yaxis.set_visible(False)

# if with_sondes:
#     if synthetic_observations:
#         pressure=pressure/100
#     ax1.fill_betweenx(y=pressure,
#                       x1=sonde_mean_wind.loc[pressure.index]-sonde_std_wind.loc[pressure.index],
#                       x2=sonde_mean_wind.loc[pressure.index]+sonde_std_wind.loc[pressure.index],
#                       color="mistyrose")
#     ax2.fill_betweenx(y=pressure,
#                       x1=(sonde_mean_q.loc[pressure.index]-sonde_std_q.loc[pressure.index])*1000,
#                       x2=(sonde_mean_q.loc[pressure.index]+sonde_std_q.loc[pressure.index])*1000,
#                       color="lavender")
#     ax3.fill_betweenx(y=pressure,
#                       x1=(sonde_mean_moist_transport.loc[pressure.index]-sonde_std_moist_transport.loc[pressure.index])*1000,
#                       x2=(sonde_mean_moist_transport.loc[pressure.index]+sonde_std_moist_transport.loc[pressure.index])*1000,
#                       color="gainsboro")
#ax1.set_ylim([0,8000])

#%% Relative Lateral Variability defined as std

#del icon_q.iloc[0,:]

#%% #
#TEST Domain
matplotlib.rcParams.update({'font.size':24})
#plt.plot(icon_moist_transport,icon_p)
#icon_moist_transport=icon_moist_transport*icon_p
# icon_mean_wind=icon_wind.mean(axis=0)
# icon_mean_q=icon_q.mean(axis=0)
# icon_mean_moist_transport=icon_moist_transport.mean(axis=0)
# icon_ivt_cumsum=scint.cumtrapz(y=icon_mean_moist_transport,x=icon_p.mean(axis=0))
# #icon_ivt_cumsum2=scint.cumtrapz(y=icon_mean_moist_transport[::-1],x=icon_p.mean(axis=0))
# ivt=scint.trapz(y=icon_moist_transport,x=icon_p)
# ivt2=scint.trapz(y=icon_moist_transport,x=icon_p.iloc[:,::-1])
fig=plt.figure(figsize=(18,16))
ax1=fig.add_subplot(211)
#ax1.plot(icon_ivt_cumsum,icon_p.mean(axis=0)[1::],color="green")
#ax1.plot(-icon_ivt_cumsum2,icon_p.mean(axis=0)[1::],color="red")
#ax1.plot(icon_z[1:],(icon_ivt.mean()-icon_ivt_cumsum)/icon_ivt.mean(),color="green",label="ivt1")
#ax1.plot(((ivt2.mean()+icon_ivt_cumsum2))/ivt2.mean(),icon_z[1:],color="red",label="ivt2")
#ax1.invert_yaxis()
ax2=fig.add_subplot(212)
ax2.plot(icon_ivt,color="green",label=str(round(icon_ivt.mean(),1)))
#ax2.plot(-ivt2,color="red",label=str(round(ivt2.mean(),1)))

ax2.legend()
#sys.exit()
#"""
