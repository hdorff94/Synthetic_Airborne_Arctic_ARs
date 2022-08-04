#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 18:07:26 2020

@author: u300737
"""

import os
import pickle
import sys
import data_config
import Performance


import pandas as pd
import xarray as xr
import numpy as np

import metpy.calc as mpcalc

from metpy.units import units

import matplotlib.pyplot as plt

###############################################################################    
class Campaign:
    flight_day={}
    flight_month={}
    def __init__(self,is_flight_campaign=True,major_path=os.getcwd(),
                 aircraft=None,instruments=[],flights=[],
                 interested_flights="all"):
        
        self.is_flight_campaign=is_flight_campaign
        self.aircraft=aircraft
        self.instruments=instruments
        self.flight_day={}
        self.flight_month={}
        self.interested_flights="all"
        self.major_path=major_path
        self.is_synthetic_campaign=False
        
            
    #%% General
    def get_instrument_data(self):
        self.data={}
        self.data.keys=self.instruments
        if self.interested_flights=="all":
            rfs=[i + j for i, j in zip(list(self.flight_month.values()),
                                       list(self.flight_day.values()))]
            print(rfs)
    
    def specify_flights_of_interest(self,interested_flights):
        self.interested_flights=interested_flights
    def create_directory(self,directory_types):
        """
        

        Parameters
        ----------
        #overall_path : str
        #    specified path in which the subdirectories need to be integrated
            
        directory_types : list
            list of directory types to check if existent, can contain the 
            the entries ["data","result_data","plots"].

        Returns
        -------
        None.
        """
        
        #loop over list and check if path exist, if not create one:
        self.sub_paths={}
        added_paths={}
        for dir_type in directory_types:
            relevant_path=self.major_path+self.name+"/"+dir_type+"/"
            added_paths[dir_type]=relevant_path
            self.sub_paths[dir_type]=relevant_path
            if not os.path.exists(relevant_path):
                os.mkdir(relevant_path)
                print("Path ", relevant_path," has been created.")
            
        data_config.add_entries_to_config_object("data_config_file",
                                                 added_paths)    
    def merge_dfs_of_flights(self,data_dict,flights_of_interest,parameter):
        """
        Parameters
        ----------
        data_dict : dictionary
            dictionary containing the dataframes of measurement data for all
            flights.
        
        parameter: str
            meteorological variable to merge, e.g. reflectivity or whatever
        Returns
        -------
        merged_df : pd.DataFrame
            dataframe which concats all flight measurements into one dataset.

        """
        merged_df=pd.DataFrame()
        i=0
        print("Merge datasets")
        for flight in flights_of_interest:
           print(flight)
           if not bool(data_dict[flight]):
               print("Dictionary for ",flight,"is empty.")
               continue
           if i==0:
               merged_df=data_dict[flight][parameter]
           else:
               merged_df=merged_df.append(data_dict[flight][parameter])
           i+=1
        return merged_df
    def dataframe_to_csv(self,input_df,save_path):
        """
        This function saves the desired dataset (input_df) in the given path
        by assessing the dataframes name which is neccessarily required
        """
        
        # Check if path ending is correct, so that df is stored correctly
        if not save_path[-1]=="/":
            save_path=save_path+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Check several attributes of the df
        #   is input_df a real df
        #       has the df a name to refer on for saving df as csv
        if isinstance(input_df, pd.DataFrame):  
            try:
                file_name=input_df.name
                df_has_name=True
            except:
                df_has_name=False
                print("No name of the df was ")
        else:
            raise AssertionError("The given dataset is no pd.DataFrame")
            has_saved=False
        
        # if df has a name, the data is stored in a csv-file with the name of df 
        if df_has_name:
            input_df.to_csv(path_or_buf=save_path+file_name+".csv",index=True)
            has_saved=True
        if has_saved:
            print("DataFrame has saved successfully as:",
                  save_path+file_name+".csv")
    
    #%% Aircraft Data / Bahamas
    def get_aircraft_position(self,flights_of_interest,all_flights=False):
        """
        This function gets the lat/lon position of the aircraft via executing 
        load hamp data
        
        Input:
            list: flights_of_interest
            List of relevant flights to get data from
        Output:
            Dictionary: dict_position
            Dictionary with keys of interested flights each containing
            DataFrame with lat/lon as columns and UTC time as index
        """
        dict_position={}
        dict_halo={}  
        if all_flights:
            print("Get aircraft position data from ",self.campaign_name)
            dict_halo=self.load_hamp_data(self.name,
                                      flights_of_interest,instrument="Halo",
                                      flag_data=False)
            if "Position" in dict_halo.keys(): #dict_halo only describes one flight
                dict_position[flights_of_interest[0]]=dict_halo["Position"] 
        
        else: 
            self.interested_flights=flights_of_interest
            for flight in self.interested_flights:
                dict_position[flight],campaign_path=self.load_aircraft_position(
                                                    interested_flight=flight)
            #for flight in flights_of_interest:
            #        if flight.startswith("RF"): # height is also a key in dict_halo
            #            dict_position[flight]=dict_halo[flight]["Position"]
            #            dict_position[flight].name="Aircraft_Position_"+\
            #            self.campaign_name+"_"+flight
        return dict_position 
    
    def load_aircraft_position(self,interested_flight=""):
        campaign_path=self.major_path+self.campaign_name+"/"
        # if one is only interested in ohne flight
        if interested_flight == "":
            if len(self.interested_flights)==1:
                pos_df=pd.read_csv(campaign_path+"data/"+"Aircraft_Position_"+\
                               self.campaign_name+"_"+self.interested_flights[0]+".csv")
                pos_df.index=pos_df["Unnamed: 0"]
                del pos_df["Unnamed: 0"]
            # interested in more than one flight, return variable is a dictionary
            else:
                if isinstance(self.interested_flights,str):
                    pos_df=pd.read_csv(campaign_path+"data/BAHAMAS/"+\
                                       "HALO_Aircraft_"+\
                                   self.interested_flights+".csv")
                    pos_df.index=pos_df["Unnamed: 0"]
                    del pos_df["Unnamed: 0"]
        else:
            pos_df=pd.read_csv(campaign_path+"data/"+"Aircraft_Position_"+\
                               self.campaign_name+"_"+\
                                   interested_flight+".csv")
            pos_df.index=pos_df["Unnamed: 0"]
            del pos_df["Unnamed: 0"]
        pos_df.index=pd.DatetimeIndex(pos_df.index)
        return pos_df,campaign_path 
    
    def load_AC3_bahamas_ds(self,flights_of_interest):
        import glob
        device_data_path=self.major_path+self.campaign_name+"/"+"data/bahamas/"
        if isinstance(flights_of_interest,str):
            flight_to_read=flights_of_interest
        else:
            flight_to_read=flights_of_interest[0]
        fname      = "*"+str(flight_to_read)+"*.nc"
        bahamas_file=glob.glob(device_data_path+fname,
                               recursive=True)[0]
        #bahamas dataset
        self.bahamas_ds =   xr.open_dataset(bahamas_file)

    def load_attitude_data(self,campaign,flights_of_interest):
        """
        

        Parameters
        ----------
        campaign : str
            name of campaign, which is essential for data path defintion.
        flights_of_interest : list
            list of relevant flights (str) to load data from.
        instrument : str, optional
            DESCRIPTION. The default is "Halo".

        Raises
        ------
        Exception
            Raise an Error if wrong instrument is given which is not part of 
            the HALO Microwave Package.

        Returns
        -------
        att_dict : dict
            DESCRIPTION.

        """
        
        print("Load Flight Attitude Data from ", campaign)
        
        main_path=self.major_path+campaign+"/data/" # data path to load data.
        if campaign=="NAWDEX":
            pass
            
        # Depending on campaign, file name convention is different.    
        elif campaign=="NARVAL-II":
            ## adapt the functions from Master Script
            pass
        
        elif campaign=="EUREC4A":
            
            filestart="bahamas_"
            
            file_path=main_path+"BAHAMAS/"
            fileend="_v0.5.nc"
            files={"RF01":file_path+filestart+"20200119"+fileend,
                   "RF02":file_path+filestart+"20200122"+fileend,
                   "RF03":file_path+filestart+"20200124"+fileend,
                   "RF04":file_path+filestart+"20200126"+fileend,
                   "RF05":file_path+filestart+"20200128"+fileend,
                   "RF06":file_path+filestart+"20200130"+fileend,
                   "RF07":file_path+filestart+"20200202"+fileend,
                   "RF08":file_path+filestart+"20200205"+fileend,
                   "RF09":file_path+filestart+"20200207"+fileend,
                   "RF10":file_path+filestart+"20200209"+fileend,
                   "RF11":file_path+filestart+"20200211"+fileend,
                   "RF12":file_path+filestart+"20200213"+fileend,
                   "RF13":file_path+filestart+"20200218"+fileend}
            
        # Now load the data, and process relevant measurements to flight-
        # specific pd.DataFrames to store in meas_dict 
        att_dict={}
        for flight in flights_of_interest:
            if len(flights_of_interest)>1:
                att_dict[flight]=pd.DataFrame()
            if not campaign=="EUREC4A":
                #currently under development
                pass
            else:
                file=files[flight]
            
            dataset=xr.open_dataset(file)
            
            # Two levels of data quality exist, on CERA data is already 
            # calibrated and files containing "radar_" are unpublished and
            # uncalibrated datasets --> current status of EUREC4A data.
            
            if flight=="RF02":
                    continue
            
            df=pd.DataFrame(data=np.array(dataset["altitude"][:]),columns=["altitude"])
            df["latitude"]=np.array(dataset["lat"][:])
            df["longitude"]=np.array(dataset["lon"][:])
            df["roll"]=np.array(dataset["roll"][:])
            df["pitch"]=np.array(dataset["pitch"][:])
            #df["yaw"]=dataset["yaw"] Heike's Bahamas data contains no yaw, y?
            df.index=pd.DatetimeIndex(np.array(dataset.time))
                
            if len(flights_of_interest)==1:
                att_dict["Attitude"]=df
            else:
                att_dict[flight]=df
                
        return att_dict
    
    def restrict_roll_angles(self,meas_dict,att_dict,
                         roll_threshold=5,instrument="radar"):
        """
        Equivalent to restrict flight levels this function goes all
        over the directory with flight entries and restrict the data
        to periods where the roll angles are below a certain/given 
        threshold
        
        Parameters
        ----------
        meas_dict : dict
            measurement data given in dictionary, mostly HAMP data.
        att_dict : dict
            bahamas data including the flight attitude.
        roll_threshold : float
            threshold value of roll angle to restrict data on.
            default is 5 degrees.
        instrument : str
            The default is "radar". Simply the name of the instrument to cut
        Returns
        -------
        meas_dict_restricted : dict
            restricted measurement data as dict, fullfilling the 
            roll restrictions specified.

        """
        flights=self.interested_flights
        restr_roll_index={}
        meas_dict_restricted={}
        for flight in flights:
            meas_dict_restricted[flight]={}
            if not flight=="RF02":
                restr_roll_index[flight]=att_dict[flight][\
                            abs(att_dict[flight]["roll"])<roll_threshold].index
                if instrument=="radar":
                    meas_dict_restricted[flight]["Reflectivity"]=\
                    meas_dict[flight]["Reflectivity"].loc[\
                    meas_dict[flight]["Reflectivity"].index.intersection(\
                                                    restr_roll_index[flight])]
        print(instrument, "-data has been cutted for roll angle thresholds")
        return meas_dict_restricted
    
    
    #%% specMACS
    def load_specmacs_data(self,day,only_cloudmask=True):
        Specmacs={}
        
        path=self.major_path+self.name+"/data/specMACS/" # data path to load.
        
        print("Load SpecMacs Data File")
        print("Flight: 2016-08-",day)
        
        ds=xr.open_dataset(path+"specMACS_cloudmask_narval2_201608"+day+".nc")
        cloud_mask=pd.DataFrame(data=np.array(ds.cloud_mask.values),
                                index=np.array(ds.cloud_mask.time))
        cloud_mask.index=cloud_mask.index.round("ms")
        Specmacs["Specmacs_cloud_mask"]=cloud_mask
        
        if only_cloudmask:
            pass
        else:
            #VZA
            viewing_zenith_angle=pd.DataFrame(data=np.array(
                                    ds.viewing_zenith_angle.values),
                                    index=np.array(ds.viewing_zenith_angle.time))
            viewing_zenith_angle.index=viewing_zenith_angle.index.round("ms")
            Specmacs["VZA"]=viewing_zenith_angle
            
            #SWIR_1600nm
            swir_1600nm_radiance=pd.DataFrame(data=np.array(
                                    ds.swir_1600nm_radiance.values),
                                    index=np.array(ds.swir_1600nm_radiance.time))
            swir_1600nm_radiance.index=swir_1600nm_radiance.index.round("ms")
            
            Specmacs["SWIR_1600nm"]=swir_1600nm_radiance
            
            del viewing_zenith_angle,swir_1600nm_radiance

        print("Rounded Specmacs index on milliseconds")
        return Specmacs;
    
    def regrid_specmacs(self,Specmacs,I1,I2,
                        regridding_frequency="1s",
                        only_cloudmask=True):
        if Specmacs["Specmacs_cloud_mask"].shape[1]>=318:
            test=Specmacs["Specmacs_cloud_mask"].iloc[:,I1:I2]
            if not only_cloudmask:
                test_swir=Specmacs["SWIR_1600nm"].iloc[:,I1:I2]
                test_vza=Specmacs["VZA"].iloc[:,I1:I2]
        else:
            test=Specmacs["Specmacs_cloud_mask"]
            if not only_cloudmask:
                test_swir=Specmacs["SWIR_1600nm"]#.iloc[:,I1:I2]
                test_vza=Specmacs["VZA"]
        
        if regridding_frequency=="1s":
            print("Regrid Specmacs on HAMP resolution")
        else:
            print("Regrid Specmacs only in time resolution with time resolution of ",
                          regridding_frequency)
                   
        test_new=test.groupby(np.arange(len(test.columns))//6,axis=1).mean()
        cols=np.arange(I1,I2,6)
        cols=cols.astype("str")
        Regridded_Specmacs=pd.DataFrame(data=np.array(test_new),
                                        columns=cols,
                                        index=test.index)
        
        Regridded_Specmacs=Regridded_Specmacs.resample(regridding_frequency).mean()
        if not only_cloudmask:
             Regridded_SWIR1600=test_swir.resample(regridding_frequency).mean()
             Regridded_VZA=test_vza.resample(regridding_frequency).mean()    
        else:
             cols=test.columns
         
        Regridded_Specmacs=Regridded_Specmacs.round()
        Regridded_Specmacs=Regridded_Specmacs.dropna(how="all")
        print("NAN lines are dropped")
        print("Gridding performed successfully!")
        Regridded_specMACS={}
        
        Regridded_specMACS["Specmacs_cloud_mask"]=Regridded_Specmacs
        if not only_cloudmask:
            Regridded_specMACS["SWIR_1600nm"]   = Regridded_SWIR1600
            Regridded_specMACS["VZA"]           = Regridded_VZA
        
        return Regridded_specMACS    
    #%% HAMP Combined
    def calc_distance_to_IVT_max(self,halo_df,data_df):
        halo_df["distance"]=halo_df["groundspeed"].cumsum()-\
            halo_df["groundspeed"].iloc[0]
        max_ivt_idx=data_df["Interp_IVT"].argmax()
        data_df["IVT_max_distance"]=halo_df["distance"]-\
                                        halo_df["distance"].iloc[max_ivt_idx]
        return data_df
        
    def load_hamp_data(self,campaign,flights_of_interest,
                       instrument="Halo",flag_data=True,bahamas_desired=False):
        """
        

        Parameters
        ----------
        campaign : str
            name of campaign, which is essential for data path defintion.
        flights_of_interest : list
            list of relevant flights (str) to load data from.
        instrument : str, optional
            DESCRIPTION. The default is "Halo".

        Raises
        ------
        Exception
            Raise an Error if wrong instrument is given which is not part of 
            the HALO Microwave Package.

        Returns
        -------
        meas_dict : TYPE
            DESCRIPTION.

        """
        
        print("Load Data from ", campaign)
        
        main_path=self.major_path+campaign+"/data/" # data path to load data.
        if campaign=="NAWDEX":
            
            filestart="halo_nawd"
        
            if (instrument.lower()=="halo") or (instrument.lower()=="radar"):
                file_path=main_path+"/HAMP-Cloud_Radar/"
                filemid="_cr00_l1_any_v00_"
            elif instrument.lower()=="radiometer":
                file_path=main_path+"/HAMP-Radiometer/"
                filemid="_mwr00_l1_tb_v00_"
            else:
                raise Exception("Wrong instrument given, HAMP only contains HALO, radar,radiometer")
        
            # Due to unified grid fileend is same for both devices
            fileend= {"RF01":"20160917071743.nc","RF02":"20160921135538.nc",
                      "RF03":"20160923073648.nc","RF04":"20160926095704.nc",
                      "RF05":"20160927113204.nc","RF06":"20161001082245.nc",
                      "RF07":"20161006070243.nc","RF08":"20161009102459.nc",
                      "RF09":"20161010115842.nc","RF10":"20161013075815.nc",
                      "RF11":"20161014082339.nc","RF12":"20161015084101.nc",
                      "RF13":"20161018085121.nc"}
            
        # Depending on campaign, file name convention is different.    
        elif campaign=="NARVAL-II":
            filestart="halo_nar2"#
        
            if (instrument.lower()=="halo") or (instrument.lower()=="radar"):
                file_path=main_path+"HAMP-Cloud_Radar/"
                filemid="_cr00_l1_any_v00_"
                fileend={"RF01":"20160808081320.nc","RF02":"20160810115244.nc",
                         "RF03":"20160812114327.nc","RF04":"20160815114800.nc",
                         "RF05":"20160817144812.nc","RF06":"20160819122918.nc",
                         "RF07":"20160822131653.nc","RF08":"","RF09":"",
                         "RF10":"","RF11":"","RF12":"","RF13":""}
                         # Empty entries are defined for the seek of 
                         # simple/stabl algorithms for entire campaigns
            
            elif instrument.lower()=="radiometer":
                file_path=main_path+"Radiometer/"
                filestart="halo_nar2_mwr00_l1_tb_v00_"
                files =[file_path+filestart+"20160808081320.nc",
                        file_path+filestart+"20160810115244.nc",
                        file_path+filestart+"20160812114327.nc",
                        file_path+filestart+"20160815114800.nc",
                        file_path+filestart+"20160817144812.nc",
                        file_path+filestart+"20160819122918.nc",
                        file_path+filestart+"20160822131653.nc",
                        file_path+filestart+"20160824124347.nc",
                        file_path+filestart+"20160826134358.nc",
                        file_path+filestart+"20160830094244.nc"]
            
        elif campaign=="EUREC4A":
            
            filestart="radar_"
            
            if (instrument.lower()=="halo") or (instrument.lower()=="radar"):
                file_path=main_path+"HAMP-Cloud_Radar/"
                fileend="_v0.5.nc"
                files={"RF01":file_path+filestart+"20200119"+fileend,
                       "RF02":file_path+filestart+"20200122"+fileend,
                       "RF03":file_path+filestart+"20200124"+fileend,
                       "RF04":file_path+filestart+"20200126"+fileend,
                       "RF05":file_path+filestart+"20200128"+fileend,
                       "RF06":file_path+filestart+"20200130"+fileend,
                       "RF07":file_path+filestart+"20200202"+fileend,
                       "RF08":file_path+filestart+"20200205"+fileend,
                       "RF09":file_path+filestart+"20200207"+fileend,
                       "RF10":file_path+filestart+"20200209"+fileend,
                       "RF11":file_path+filestart+"20200211"+fileend,
                       "RF12":file_path+filestart+"20200213"+fileend,
                       "RF13":file_path+filestart+"20200218"+fileend,
                       }
            else:
                print("Radiometer for EURECA are not yet downloaded or used")

        # Now load the data, and process relevant measurements to flight-
        # specific pd.DataFrames to store in meas_dict 
        meas_dict={}
        for flight in flights_of_interest:
            if len(flights_of_interest)>1:
                meas_dict[flight]={}
            if campaign!="EUREC4A":
                file=file_path+filestart+filemid+fileend[flight]    
            else:
                file=files[flight]
            
            if campaign=="NARVAL-II":
                if not fileend[flight]:
                    print("No flight ",flight,
                          "or no measurements have been performed.", 
                          "Skip this flight and continue.")
                    continue
            
            
            dataset=xr.open_dataset(file)
            
            #-----------------------------------------------------------------#
            
            # Radar processing
            
            #-----------------------------------------------------------------#
            if instrument.lower()=="radar" or instrument.lower()=="halo":
                if "radar_" not in file:
                    df=pd.DataFrame(data=np.array(dataset.zsl),
                                columns=["altitude"])
                    
                    if bahamas_desired:
                        try:
                            date=fileend[flight][0:8]
                            bahamas=xr.open_dataset(
                                main_path+"BAHAMAS/HALO-DB_bahamas_"+date+"a_"+flight+"_v01.nc")
                        except:
                            raise FileNotFoundError("The Bahamas dataset has not yet been downloaded.")
                        airspeed=pd.Series(data=bahamas["IRS_GS"],
                                       index=pd.DatetimeIndex(np.array(bahamas["TIME"][:])))
                        airspeed=airspeed.resample("1s").mean()
                        df["groundspeed"]=np.array(airspeed)
                    else:
                        pass
                    
                    df["latitude"]=dataset["lat"][:]
                    df["longitude"]=dataset["lon"][:]
                    rf=dataset["dbz"][:]
                    reflectivity_factor=pd.DataFrame(data=np.array(rf),
                            columns=map(str,np.array(dataset["height"][:])))
                    reflectivity_factor.index=pd.DatetimeIndex(np.array(dataset.time))
                    ldr=dataset["ldr"][:]
                    ldr_factor=pd.DataFrame(data=np.array(ldr),
                            columns=map(str,np.array(dataset["height"][:])))
                    ldr_factor.index=pd.DatetimeIndex(np.array(dataset.time))
                else:
                    if flight=="RF02":
                        continue
                
                if flag_data:
                    if "radar_" not in file:
                        print("Flag the radar data")
                        radar_flag=pd.DataFrame(data=np.array(
                                dataset["radar_flag"][:]),
                                columns=map(str,np.array(dataset["height"][:])))
                        radar_flag.index=reflectivity_factor.index
                        radar_flag=radar_flag.replace(to_replace=[1,2,3,4],
                                                      value=np.nan)
                        reflectivity_factor=reflectivity_factor+radar_flag
                        ldr_factor=ldr_factor+radar_flag
                    else:
                        radar_flag=pd.DataFrame(data=np.array(
                                                    dataset["data_flag"][:]),
                                    columns=map(str,
                                            np.array(dataset["height"][:])))
                        radar_flag.index=reflectivity_factor.index
                        radar_flag=radar_flag.replace(to_replace=[1,2,3,4],
                                                      value=np.nan)
                        reflectivity_factor=reflectivity_factor+radar_flag
                        ldr_factor=ldr_factor+radar_flag
                    print("Radar data flagged")
                else:
                    print("no flagging of data is considered")
                
                df.index=reflectivity_factor.index
                # Two levels of data quality exist, on CERA data is already 
                # calibrated and files containing "radar_" are unpublished and
                # uncalibrated datasets --> current status of EUREC4A data.
                if len(flights_of_interest)==1:
                     meas_dict["Position"]=df
                     meas_dict["Reflectivity"]=reflectivity_factor
                     meas_dict["LDR"]=ldr_factor
                else:
                    meas_dict[flight]["Position"]=df
                    meas_dict[flight]["Reflectivity"]=reflectivity_factor
                    meas_dict[flight]["LDR"]=ldr_factor
                meas_dict["height"]=dataset["height"]
            #-----------------------------------------------------------------#
            
            # Radiometer processing
            
            #-----------------------------------------------------------------#
            elif instrument.lower()=="radiometer":
                time=np.array(dataset["time"][:])
                #time=pd.to_datetime(time,unit='s')
             
                frequencies=map(str,np.around(np.array(dataset["freq_sb"][:]),2))
                HAMP_Tb          =pd.DataFrame(data=np.array(dataset["tb"][:]),
                                               columns=frequencies,
                                               index=pd.DatetimeIndex(time))
             
                HAMP_Tb_intp_flag=pd.DataFrame(data=np.array(dataset["interpolate_flag"][:]),
                                               columns=HAMP_Tb.columns,
                                               index=pd.DatetimeIndex(time))
                if len(flights_of_interest)==1:
                     meas_dict["T_b"]=HAMP_Tb
                     meas_dict["T_b_interpolated_flag"]=HAMP_Tb_intp_flag
                else:
                    meas_dict[flight]["T_b"]=HAMP_Tb
                    meas_dict[flight]["T_b_interpolated_flag"]=HAMP_Tb_intp_flag
                if flag_data:
                    print("Until now no flagging for hamp is performed")
               
        # Return the dictionary comprising the measurement data of given device    
        return meas_dict
    
    #%% Radar Specific
    def restrict_data_to_flight_level(self,meas_data,flight_levels,
                                      instrument="radar",
                                      flights="all"):
        """
        

        Parameters
        ----------
        meas_data : Dictionary/pd.DataFrame
            DESCRIPTION.
        flight_levels : list
            DESCRIPTION. [lower upper] flight levels to constrain on.
        instrument : str, optional
            DESCRIPTION. The default is "radar".
        flights: str, optional
            DESCRIPTION. The default is "all". This specifies the flights to
            adapt the flight level restriction to. 
        Returns
        -------
        flight_level_index : dict
            DESCRIPTION. Dictionary of the 
        restricted_meas_data
        """
        
        if instrument=="radar":
            pass
        else:
            return None
        lower_level=flight_levels[0]
        upper_level=flight_levels[1]
        restricted_meas_data={}
        flight_level_index={}
        if isinstance(meas_data,dict):
            if flights=="all":
                for flight in self.interested_flights:
                    restricted_meas_data[flight]={}
                    try:
                        flight_level_index[flight]=meas_data[flight]["Position"]["altitude"].between(lower_level,upper_level)    
                        restricted_meas_data[flight]["Reflectivity"]=meas_data[flight]["Reflectivity"].loc[flight_level_index[flight]]
                    except:
                        pass
            print("Data is now restricted to flight level corridors")
        elif isinstance(meas_data,pd.DataFrame):
            pass
        else:
            raise AssertionError("The data given is of wrong type")            
        return flight_level_index,restricted_meas_data
        
    def calculate_cfad_radar_reflectivity(self,df,
                                    reflectivity_bins=np.linspace(-70,50,121)):
        
        """
        Parameters
        ----------
        df : pd.DataFrame
            dataframe of radar reflectivity measurements for given distance,
            ideally with height columns as provided in unified grid of HAMP.
        
        reflectivity_bins : numpy.array
            array of reflectivity bins to group data. Default is binwidth of 1
            for a Ka-Band typical reflectivity range (-60 to 30 dbZ)
            
        Returns
        -------
        cfad_hist : pd.DataFrame
            dataframe of the histogram for given settings columns are binmids

        """
        ## Create array to assign for dataframe afterwards
        #  Dimensions
        x_dim=len(df.columns)
        y_dim=len(reflectivity_bins)-1
        # Empty array at first
        cfad_array=np.empty((x_dim,y_dim))
        
        
        # if dataframe contain np.nans they should be replaced
        #df=df.replace(to_replace=np.nan, value=-reflectivity_bins[0]+0.1)
        cfad_hist=pd.DataFrame(data=cfad_array,index=df.columns,
                               columns=reflectivity_bins[:-1]+0.5)
        # Start looping
        print("Calculate CFAD for HALO Radar Reflectivity")
        i=0
        for height in df.columns:
            Performance.performance.updt(self,len(df.columns),i)
            
            # Start grouping by pd.cut and pd.value_counts
            bin_groups=pd.cut(df[height],reflectivity_bins)
            height_hist=pd.value_counts(bin_groups).sort_index()        
            
            #Assign counted bins to histogram dataframe
            cfad_hist.iloc[i,:]=height_hist.values
            i+=1
        
        # Finished, return histogram    
        return cfad_hist
    
    def plot_radar_AR_quicklook(self,hamp,ar_of_day,
                                flight,plot_path):
        
        import seaborn as sns
        import matplotlib.cm as cm
        fig,axs=plt.subplots(2,1,figsize=(20,16),sharex=True)
        
        y=hamp["height"][:]#/1000
        print("Start plotting HAMP Cloud Radar")
        axs[0].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[0].set_ylim([0,12000])
        axs[0].set_yticklabels(["0","2","4","6","8","10","12"])
        axs[0].set_ylabel("Altitude (km)")
        levels=np.arange(-30,30.0,1.0)
        ldr_levels=np.arange(-80,0,1.0)# 
        try:
            C1=axs[0].contourf(hamp["Reflectivity"].index.time,y,
                                     hamp["Reflectivity"].T,levels,
                                     cmap=cm.get_cmap(\
                                                "temperature",len(levels)-1),
                                             extend="both")
        except:
            C1=axs[0].contourf(hamp["Reflectivity"].index.time,y,
                                         hamp["Reflectivity"].T,levels,
                                         cmap=cm.get_cmap('viridis',
                                                          len(levels)-1),
                                         extend="both")
        axs[0].set_xlabel('')
            
        for label in axs[0].xaxis.get_ticklabels()[::8]:
            label.set_visible(False)
        cb = fig.colorbar(C1,ax=axs[0],shrink=0.6)
        cb.set_label('Reflectivity (dBZ)')
        labels = levels[::8]
        cb.set_ticks(labels)   
        labels_ldr=ldr_levels[::8]
        C2=axs[1].contourf(hamp["LDR"].index.time,y,
                           hamp["LDR"].T,ldr_levels,
                           cmap=cm.get_cmap("cubehelix_r",len(ldr_levels)-1),
                                             extend="min")
        cb2 = fig.colorbar(C2,ax=axs[1],shrink=0.6)
    
        cb2.set_label('LDR (dB)')
        cb2.set_ticks(labels_ldr)   
        axs[1].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[1].set_ylim([0,12000])
        axs[1].set_yticklabels(["0","2","4","6","8","10","12"])
    
        for label in axs[1].xaxis.get_ticklabels()[::8]:
            label.set_visible(False)
        axs[1].set_ylabel("Altitude (km)")
    
        axs[1].set_xlabel('Time (UTC)')
        sns.despine(offset=10)
        fig_name=flight+"_"+ar_of_day+"_radar_quicklook.png"
        fig.savefig(plot_path+fig_name,dpi=250,bbox_inches="tight")    
        print("Figure saved as:",plot_path+fig_name)
        return None
    
    def plot_cfad_2d_hist(self,cfad_df,plot_path,
                          flagged_data,ar_of_day=None,roll_threshold=10):
        #import matplotlib
        import matplotlib.pyplot as plt
        try: 
            import typhon
        except:
            print("Typhon module cannot be loaded")
        import seaborn as sns
        cfad_df=cfad_df.replace(to_replace=0, value=np.nan)
        #set_font=16
        #matplotlib.rcParams.update({'font.size':set_font})
        cfad_fig=plt.figure(figsize=(9,12))
        #cfad_df=cfad_df.replace()
        ax1=cfad_fig.add_subplot(111)
        ax1.set_xlabel("Reflectivity (dBZ)")
        ax1.set_ylabel("Height (km)")
        x_data=np.array(cfad_df.columns)[5:]
        y_data=np.array(cfad_df.index).astype("float")
        yy,xx=np.meshgrid(y_data,x_data)
        levels=np.linspace(0,20000,51)
        z=cfad_df.iloc[:,5:]
        C1=ax1.pcolormesh(xx,yy,z.T,cmap="cividis")
        ax1.set_yticks(np.linspace(0,13000,14))
        ax1.set_yticklabels(np.linspace(0,13,14).astype(int).astype(str))
        cb = plt.colorbar(C1,extend="max")
        cb.set_label('Absolute Counts')
        
        ax1.set_title(self.name+" Radar data flagged: "+\
                      str(flagged_data)+", roll angles < "+\
                          str(roll_threshold)+"deg")
        ax1.set_xlim([-60,40])
        ax1.set_ylim([0,12000])
        sns.despine(offset=10)
        if not flagged_data: 
            fig_name=self.name+"_CFAD_"+"no_flagging"+\
                str(roll_threshold)+".png"
            
            
        else:
            fig_name=self.name+"_CFAD_"+"flaged_radar"+\
                str(roll_threshold)+".png"
        if not ar_of_day==None:
           fig_name=ar_of_day+"_"+fig_name 
        cfad_fig.savefig(plot_path+fig_name,
                         dpi=300,bbox_inches="tight")
        print("Figure saved as: ",plot_path+fig_name)
        return cfad_df
    
    #%% Radiometer specific
    def plot_hamp_brightness_temperatures(self,Tb_df,flight,date,start,end,
                                          ar_of_day=None,
                                          plot_path=os.getcwd()):
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        print("Plotting ...")
        fig = plt.figure(figsize=(16,24))
        ax1=fig.add_subplot(411)
        ax1.plot(Tb_df["22.24"],label="22.24 GHz")
        ax1.plot(Tb_df["23.04"],label="23.04 GHz")
        ax1.plot(Tb_df["23.84"],label="23.84 GHz")
        ax1.plot(Tb_df["25.44"],label="25.44 GHz")
        ax1.plot(Tb_df["26.24"],label="26.24 GHz")
        ax1.plot(Tb_df["27.84"],label="27.84 GHz")
        ax1.plot(Tb_df["31.4"],label="31.40 GHz")
        ax1.set_ylabel("T$_{b}$ in K")
        ax1.grid()
        ax1.set_xlabel("Time in UTC")
        ax1.set_title(ar_of_day+" Brightness Temperatures "+date+"_"+flight[0])
        ax1.legend(loc="center left",bbox_to_anchor=(1.0,0.5))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ###########################################################################
        ax2=fig.add_subplot(412)
        ax2.plot(Tb_df["50.3"],label="50.3 GHz")
        ax2.plot(Tb_df["51.76"],label="51.76 GHz")
        ax2.plot(Tb_df["52.8"],label="52.8 GHz")
        ax2.plot(Tb_df["53.75"],label="53.75 GHz")
        ax2.plot(Tb_df["54.94"],label="54.94 GHz")
        ax2.plot(Tb_df["56.66"],label="56.66 GHz")
        ax2.plot(Tb_df["58.0"],label="58.00 GHz")
        ax2.set_ylabel("T$_{b}$ in K")
        ax2.grid()
        ax2.set_xlabel("Time in UTC")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.legend(loc="center left",bbox_to_anchor=(1.0,0.5))
        #######################################################################
        ax3= fig.add_subplot(413)
        ax3.plot(Tb_df["90.0"],label="90.0 GHz")
        ax3.plot(Tb_df["120.15"],label="(118.75 +/- 1.4) GHz")
        ax3.plot(Tb_df["121.05"],label="(118.75 +/- 2.3) GHz")
        ax3.plot(Tb_df["122.95"],label="(118.75 +/- 4.2) GHz")
        ax3.plot(Tb_df["127.25"],label="(118.75 +/- 8.5) GHz")
        ax3.set_ylabel("T$_{b}$ in K")
        ax3.grid()
        ax3.set_xlabel("Time in UTC")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax3.legend(loc="center left",bbox_to_anchor=(1.0,0.5))
        #######################################################################
        ax4= fig.add_subplot(414)
        ax4.plot(Tb_df["183.91"],label="(183.31 +/- 0.6) GHz")
        ax4.plot(Tb_df["184.81"],label="(183.31 +/- 1.5) GHz")
        ax4.plot(Tb_df["185.81"],label="(183.31 +/- 2.5) GHz")
        ax4.plot(Tb_df["186.81"],label="(183.31 +/- 3.5) GHz")
        ax4.plot(Tb_df["188.31"],label="(183.31 +/- 5.0) GHz")
        ax4.plot(Tb_df["190.81"],label="(183.31 +/- 7.5) GHz")
        ax4.plot(Tb_df["195.81"],label="(183.31 +/- 12.5) GHz")
        ax4.set_ylabel("T$_{b}$ in K")
        ax4.grid()
        ax4.set_xlabel("Time in UTC")
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax4.legend(loc="center left",bbox_to_anchor=(1.0,0.5))
        fig_name="HAMP_Tb_"+flight[0]+"_"+date+"_"+str(start)[-8:-6]+str(start)[-5:-3]+"_"+str(end)[-8:-6]+str(end)[-5:-3]+".png"
        if ar_of_day is not None:
                fig_name=ar_of_day+"_"+fig_name
        
        fig.savefig(plot_path+fig_name,dpi=150,bbox_inches="tight")
        print("Figure saved as : ",plot_path+fig_name)
        
        return None
    
    def load_radiometer_retrieval(self,campaign,flight="RF10",
                                  variables=["T","rho_v"],
                                  calculate_spec_hum=False,
                                  sonde_p=None,sondes_upsampled=True,
                                  cut_to_AR=True):
        """
        This loads the radiometer retrieval of A. Walbröl for abs. humidity and
        air temperature based on ECMWF-IFS simulations for training
        

        Parameters
        ----------
        campaign : str
            name of Campaign.
        flight : str, optional
            name of research flight. The default is "RF10".
        variables : list, optional
            variables to load retrieval data from. Default is ["T","rho_v"].
        calculate_spec_hum : Boolean, optional
            If specific humidity should be calculated. The default is False.
        sonde_p : pd.DataFrame, optional
            Dropsonde pressure data. The default is None.
        sondes_upsampled : Boolean, optional
            if sonde_p is already upsampled to 1-Hz data. The default is True.
        cut_to_AR : boolean, optional
            if output should be cutted to AR crossing period. That is defined
            at the beginning of the major routine.
            The default is True.

        Returns
        -------
        retrieval_dict : dict
            Retrieval data as dictionary containing pd.DataFrames for each var.

        """

        retrieval_path=self.major_path+campaign+"/data/HAMP-Retrieval/" # data path to load data.
        retrieval_dict={}
        if variables==["T","rho_v"]:
            do_both=True
        elif variables!=["T","rho_v"] and len(variables)==2:
            print("You inserted two variables, but their names are wrong.",
                  "Load T and rho_v nonetheless.")
            do_both=True
        elif len(variables)==1:
            do_both=False
        else:
            raise Exception("Wrong number and type of variables given")
        
        if do_both:
            t_retrieval_fname="T_prof_"+campaign+"_"+flight+"_regression.nc"
            abshum_retrieval_fname="rho_v_prof_"+campaign+"_"+flight+"_regression.nc"

            # Load data as xr.Dataset
            rho_v_ds=xr.open_dataset(retrieval_path+abshum_retrieval_fname)
            t_ds=xr.open_dataset(retrieval_path+t_retrieval_fname)
            # Transform to dataframe
            rho_v_df=pd.DataFrame(data=np.array(rho_v_ds["rho_v"][:]),
                      columns=np.array(rho_v_ds["height"][:]),
                      index=pd.DatetimeIndex(np.array(rho_v_ds["time"][:])).round("s"))

            air_t_df=pd.DataFrame(data=np.array(t_ds["T"][:]),
                     columns=np.array(t_ds["height"][:]),
                     index=pd.DatetimeIndex(np.array(t_ds["time"][:])).round("s"))
            # Insert in Dict
            retrieval_dict["Rho_v"]=rho_v_df
            retrieval_dict["T"]=air_t_df
            
            if calculate_spec_hum:
                
               R_l=287.1 #m²s-²K-1 
               print("Calculate specific humidity using dropsondes")
               if sonde_p is None:
                   print("Sonde Pressure should be given and is required ")
               else:
                   
                   if not sondes_upsampled:
                       print("Sonde is not upsampled and need to be resampled to seconds")
                       # This is condensed from upsample_dropsonde_data but not 
                       # using the function as this would be exaggerated
                       
                       #Preallocate
                       upsampled_sonde_p=pd.DataFrame(
                               data=np.nan,
                               columns=sonde_p.columns,
                               index=pd.date_range(start=sonde_p.index[0],
                                                   end=sonde_p.index[-1],
                                                   freq="s"))
                       upsampled_sonde_p.loc[sonde_p.index,:]=sonde_p
                       
                       # Interpolate
                       upsampled_sonde_p=upsampled_sonde_p.interpolate(
                                  method="time")
                       upsampled_sonde_p=upsampled_sonde_p.reset_index().drop_duplicates(
                                       subset='index', keep='last').set_index('index')
                       
                       intersect_index=rho_v_df.index.intersection(sonde_p.index)
                       retrieval_sonde_p=upsampled_sonde_p.loc[intersect_index,:]

                   else:
                       #Index settings for collocation
                       intersect_index=rho_v_df.index.intersection(sonde_p.index)
                       retrieval_sonde_p=sonde_p.loc[intersect_index,:]
                       retrieval_sonde_p.columns=pd.Float64Index(sonde_p.columns)
                       retrieval_sonde_p=retrieval_sonde_p.loc[:,rho_v_df.columns]
                       
                       #Calculate other variables
                       rho_d_df=retrieval_sonde_p*100/(R_l*air_t_df)
                       retrieval_dict["Rho_d"]= rho_d_df                   
                       retrieval_dict["q"]    = rho_v_df/(rho_v_df+rho_d_df)

        else:
            retrieval_var=variables[0]
            retrieval_fname=retrieval_var+"_prof"+campaign+"_"+flight+"_regression.nc"
            retrieval_ds=xr.open_dataset(retrieval_path+retrieval_fname)
            retrieval_df=pd.DataFrame(
                data=np.array(retrieval_ds[retrieval_var][:]),
                columns=np.array(retrieval_ds["height"][:]),
                index=np.array(retrieval_ds.time[:]))
            retrieval_dict[retrieval_var]=retrieval_df
        
        if cut_to_AR:
            for var in retrieval_dict.keys():
                retrieval_dict[var]=retrieval_dict[var].loc[intersect_index,:]
        
        return retrieval_dict

    def retrieval_humidity_plotting(self,halo_icon,retrieval_dict,dropsondes,
                                upsampled_dropsondes,date,
                                flight,path,start,end,icon_data_path,
                                plot_path=os.getcwd(),with_ivt=False,
                                do_masking=False,save_figure=True, 
                                low_level=False, ar_of_day=None):    
  
        import matplotlib
        import matplotlib.dates as mdates
        from matplotlib import gridspec
        #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!############
        pd.plotting.register_matplotlib_converters()
        #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!############
        
        set_font=18
        matplotlib.rcParams.update({'font.size':set_font})
        #plt.rc('text',usetex=True)
        plt.rc('axes',linewidth=1.5)
        #from matplotlib.colors import Boundarynorm
        hours = mdates.MinuteLocator(byminute=[0,10,20,30,40,50,60],
                                     interval = 1)
        h_fmt = mdates.DateFormatter('%H:%M')
    
        #theta_colormap      = "jet"
        humidity_colormap   = "terrain_r"
        #Then tick and format with matplotlib:
        if with_ivt:
            fig=plt.figure(figsize=(16,24))
            gs= gridspec.GridSpec(4,2,height_ratios=[1,1,2,2],width_ratios=[1,0.14])
            ax1=plt.subplot(gs[0,0])
            # Retrieval Dict IVT
            ax1.set_ylabel("HAMP-Retrieval, \n ICON, Sondes:\n IVT (kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)")
    
            ax1.plot(retrieval_dict["IVT"].index.time,
                     retrieval_dict["IVT"],ls='--',lw=2,
                     color="green",label="HAMP-Retrieval")
            
            # ICON IVT
            ivt_icon_file=flight+"_"+ar_of_day+"_Interpolated_IVT.csv"
            if os.path.isfile(icon_data_path+ivt_icon_file):
                print("ICON IVT exists and will be included")    
                icon_ivt=pd.read_csv(icon_data_path+ivt_icon_file,index_col=0)
                icon_ivt.index=pd.DatetimeIndex(icon_ivt.index)
                ax1.plot(icon_ivt["IVT"].index.time,
                icon_ivt["IVT"],ls='-',lw=3,color="darkgreen",label="ICON-IVT")
        
            # Dropsondes IVT
            if not flight[0]=="RF08":
                dropsondes["IVT"]=dropsondes["IVT"].loc[halo_icon["qv"].index[0]:halo_icon["qv"].index[-1]]
            # RF08 has only one or no (?) dropsonde which makes the plotting
            # more complicated
            ax1.plot(dropsondes["IVT"].index.time,
                     np.array(dropsondes["IVT"]),
                     linestyle='',markersize=15,marker='v',color="lightgreen",
                     markeredgecolor="black",label="Dropsondes")
            if dropsondes["IVT"].max()<500:
                ax1.set_ylim([50,550])
            elif 500<dropsondes["IVT"].max()<750:
                ax1.set_ylim([50,800])
            else:
                ax1.set_ylim([50,1400])
            ax1.legend(loc="upper center",ncol=3)
            ax1.set_xlabel('')
            ax1.set_xlim([halo_icon["qv"].index.time[0],
                          halo_icon["qv"].index.time[-1]])
            ax1.tick_params("both",length=5,width=1.5,which="major")
            
            ax1.xaxis.set_major_locator(hours)
            ax1.xaxis.set_major_formatter(h_fmt)
            ax1_2=plt.subplot(gs[1,0])
            ax1_2.plot(icon_ivt["IWV_calc"].index.time,icon_ivt["IWV_calc"],
                       ls="-",color="brown",label="ICON-IWV")
            ax1_2.plot(dropsondes["IWV"].index.time,np.array(dropsondes["IWV"]),
                       linestyle='',markersize=15,marker='v',color="orange",
                       markeredgecolor="black",label="Dropsondes")
            ax1_2.plot(retrieval_dict["IWV"].index.time,retrieval_dict["IWV"],
                       ls="--",color="saddlebrown",label="HAMP-Retrieval")
            ax1_2.set_xlim([halo_icon["qv"].index.time[0],
                            halo_icon["qv"].index.time[-1]])
            ax1_2.tick_params("both",length=5,width=1.5,which="major")
            ax1_2.set_ylabel(" IWV (kg$\mathrm{m}^{-2}$)")
    
            ax1_2.set_ylim([0,40])
            ax1_2.legend(loc="upper center")
            ax1_2.set_xlabel('')
            ax1_2.set_xlim([halo_icon["qv"].index.time[0],halo_icon["qv"].index.time[-1]])
            ax1_2.tick_params("both",length=5,width=1.5,which="major")
            
            ax1_2.xaxis.set_major_locator(hours)
            ax1_2.xaxis.set_major_formatter(h_fmt)    
            ax2=plt.subplot(gs[2,:])
            
        else:
            fig=plt.figure(figsize=(16,14))
            ax2=fig.add_subplot(211)
        
        #Specific humidity
        q_min=0
        q_max=10
        
        if low_level:
            q_min=0
            q_max=10
        
        # Add wind 
        halo_icon["wind"]=np.sqrt(halo_icon["u"]**2+halo_icon["v"]**2)
        x_temp=halo_icon["qv"].loc[start:end].index#.time
        y_temp=halo_icon["qv"].loc[start:end].columns
        y,x=np.meshgrid(y_temp,x_temp)
        #levels=np.linspace(q_min,q_max,50)
        y=np.array(halo_icon["Z_Height"].loc[start:end])
        
        #Create new Hydrometeor content as a sum of all water contents
        cutted_hmc=halo_icon["qv"].loc[start:end]
        #if do_masking:
        #    cutted_hmc=cutted_hmc*era5_on_halo.mask_df
        
        C2=ax2.pcolormesh(x,y,cutted_hmc*1000,
                          vmin=q_min,vmax=q_max,
                          cmap=humidity_colormap)
        mixing_ratio=cutted_hmc#mpcalc.mixing_ratio_from_specific_humidity(halo_era5_hmc["q"])
        
        moisture_levels=[5,10,20,30,40]
        wind_levels=[10,20,40,60]
        if low_level:
            moisture_levels=[5,10,20,30,35]
            wind_levels=[10,20,30]
        wind=halo_icon["wind"]
        #if do_masking:
        #    wind=wind*era5_on_halo.mask_df
        
        wv_flux=mixing_ratio*wind #halo_era5_hmc["wind"]
        moisture_flux=1/9.82*wv_flux*1000
        
        CS=ax2.contour(x,y,moisture_flux.loc[start:end],
                       levels=moisture_levels,colors="k",
                       linestyles="-",linewidths=1.0)
        
        CS2=ax2.contour(x,y,halo_icon["wind"].loc[start:end],
                       levels=wind_levels,colors="magenta",
                       linestyles="--",linewidths=1.0)
        ax2.scatter(dropsondes["IVT"].index,
                     np.ones(dropsondes["IVT"].shape[0])*5500,
                     s=50,marker='v',color="lightgreen",
                     edgecolor="black",label="Dropsondes")
        for sonde_index in dropsondes["IVT"].index:
            ax2.axvline(x=sonde_index,ymin=0,ymax=5450,
                        color="black",ls="--",alpha=0.6)
        #ax2.axvline(dropsondes["IVT"].index,5500,color="lightgreen",ls='--')
        # Contour lines and Colorbar specifications
        ax2.clabel(CS,fontsize=16,fmt='%1.1f',inline=1)
        ax2.clabel(CS2,fontsize=16,fmt='%1.1f',inline=1)
        
        cb = plt.colorbar(C2,extend="max")
        cb.set_label('ICON (2km): \n Specific Humidity in g/kg')
        
        if low_level:
            ax2.set_ylim(0,6000)
            ax2.set_yticks([0,500, 1000,1500,2000,2500, 3000, 
                            3500, 4000, 4500, 5000,5500,6000])
            ax2.set_yticklabels([0,500, 1000, 1500, 2000,2500,3000, 3500,
                                 4000, 4500, 5000,5500,6000])
        
            fig_name="Specific_Humidity_Retrieval_ICON_low_level_"+flight+"_"+date+"_"+str(start)[-8:-6]+str(start)[-5:-3]+"_"+str(end)[-8:-6]+str(end)[-5:-3]+".png"
            
        else:
            ax2.set_ylim(0,12000)
            ax2.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax2.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
            fig_name="Specific_Humidity_Retrieval_ICON_"+flight+"_"+date+"_"+str(start)[-8:-6]+str(start)[-5:-3]+"_"+str(end)[-8:-6]+str(end)[-5:-3]+".png"
        for label in ax2.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        #if with_ivt:
        #    xticks=ax2.xaxis.get_ticklabels()
        #    ax1.set_xticklabels(xticks)
        ax2.set_xlabel("Time (UTC)")
        ax2.set_ylabel('Altitude in m ')
        ax2.set_xlabel('')
        ax2.xaxis.set_major_locator(hours)
        ax2.xaxis.set_major_formatter(h_fmt)
        ax2.tick_params("both",length=5,width=1.5,which="major")
        
        if not with_ivt:
            ax3=fig.add_subplot(212)
        else:
            ax3=plt.subplot(gs[3,0])
        #-------------------------------------------------------------------------#
        # Specific Humidity Retrieval 
        
        x_temp=retrieval_dict["q"].loc[start:end].index#.time
        y_temp=retrieval_dict["q"].loc[start:end].columns
        y,x=np.meshgrid(y_temp,x_temp)
        
        #Create new Hydrometeor content as a sum of all water contents
        cutted_retrieval_hmc=retrieval_dict["q"].loc[start:end]
        
        C3=ax3.pcolormesh(x,y,cutted_retrieval_hmc*1000,
                          vmin=q_min,vmax=q_max,
                          cmap=humidity_colormap)
        
        moisture_levels=[10,20,30,40]
        wind_levels=[10,20,40,60]
        if low_level:
            moisture_levels=[5,10,20,30,35]
            wind_levels=[10,20,30]
        upsampled_dropsondes["Wspeed"].columns=pd.Float64Index(dropsondes["Wspeed"].columns)
        wind_retrieval=upsampled_dropsondes["Wspeed"][retrieval_dict["q"].columns]
        wind_retrieval=wind_retrieval.loc[start:end]
        retrieval_wv_flux=retrieval_dict["q"]*wind_retrieval 
        retrieval_moisture_flux=1/9.82*retrieval_wv_flux*1000
        
        CS3=ax3.contour(x,y,retrieval_moisture_flux.loc[start:end],
                       levels=moisture_levels,colors="k",
                       linestyles="-",linewidths=1.0)
        
        CS4=ax3.contour(x,y,wind_retrieval,
                       levels=wind_levels,colors="magenta",
                       linestyles="--",linewidths=1.0)
        
        ax3.scatter(dropsondes["IVT"].index,
                     np.ones(dropsondes["IVT"].shape[0])*5500,
                     s=50,marker='v',color="lightgreen",
                     edgecolor="black",label="Dropsondes")
        
        for sonde_index in dropsondes["IVT"].index:
            ax3.axvline(x=sonde_index,ymin=0,ymax=5450,
                        color="black",ls="--",alpha=0.6)
        
        # Contour lines and Colorbar specifications
        ax3.clabel(CS3,fontsize=16,fmt='%1.1f',inline=1)
        ax2.clabel(CS4,fontsize=16,fmt='%1.1f',inline=1)
        
        cb = plt.colorbar(C3,extend="max")
        cb.set_label('ICON (2km): \n Specific Humidity in g/kg')
        
        if low_level:
            ax3.set_ylim(0,6000)
            ax3.set_yticks([0,500, 1000,1500,2000,2500, 3000, 
                            3500, 4000, 4500, 5000,5500,6000])
            ax3.set_yticklabels([0,500, 1000, 1500, 2000,2500,3000, 3500,
                                 4000, 4500, 5000,5500,6000])
        
            fig_name="Specific_Humidity_Retrieval_ICON_low_level_"+flight+"_"+date+"_"+str(start)[-8:-6]+str(start)[-5:-3]+"_"+str(end)[-8:-6]+str(end)[-5:-3]+".png"
            
        else:
            ax3.set_ylim(0,12000)
            ax3.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax3.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
            fig_name="Specific_Humidity_Retrieval_ICON_"+flight+"_"+date+"_"+str(start)[-8:-6]+str(start)[-5:-3]+"_"+str(end)[-8:-6]+str(end)[-5:-3]+".png"
        for label in ax3.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        
        ax3.set_xlabel("Time (UTC)")
        ax3.set_ylabel('Altitude in m ')
        ax3.set_xlabel('')
        ax3.xaxis.set_major_locator(hours)
        ax3.xaxis.set_major_formatter(h_fmt)
        ax3.tick_params("both",length=5,width=1.5,which="major")
        
        if ar_of_day is not None:
                fig_name=ar_of_day+"_"+fig_name
        
        fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", plot_path+fig_name)
        return None

    
    def vertical_integral_retrieval(self,retrieval_dict,upsampled_sondes):
       g= 9.81
       
       retrieval_dict["IWV"]=pd.Series(data=np.nan,
                                       index=retrieval_dict["q"].index)
       retrieval_dict["IVT_u"]=pd.Series(data=np.nan,
                                       index=retrieval_dict["q"].index)
       retrieval_dict["IVT_v"]=pd.Series(data=np.nan,
                                       index=retrieval_dict["q"].index)
       retrieval_dict["IVT"]=pd.Series(data=np.nan,
                                       index=retrieval_dict["q"].index)
       
       # First cut dropsondes to collocated period for both wind variables
       upsampled_sondes["Wspeed"]=upsampled_sondes["Wspeed"].loc[retrieval_dict["q"].index]
       upsampled_sondes["Wdir"]=upsampled_sondes["Wdir"].loc[retrieval_dict["q"].index]
       upsampled_sondes["Pres"]=upsampled_sondes["Pres"].loc[retrieval_dict["q"].index]
       
       upsampled_sondes["RH"].columns=pd.Float64Index(
                                           upsampled_sondes["RH"].columns)
       upsampled_sondes["AirT"].columns=pd.Float64Index(
                                           upsampled_sondes["AirT"].columns)
       
       upsampled_sondes["Wspeed"].columns=pd.Float64Index(
                                           upsampled_sondes["Wspeed"].columns)
       upsampled_sondes["Wdir"].columns=pd.Float64Index(
                                           upsampled_sondes["Wdir"].columns)
       upsampled_sondes["Pres"].columns=pd.Float64Index(
                                           upsampled_sondes["Pres"].columns)
       
       
       # Loop and calculate vertical integrals
       int_idx=0
       print("Calculate IWV and IVT from upsampled retrieval profiles")
       for idx in upsampled_sondes["Wspeed"].index:
                            
           #IWV
           vertical_idx=retrieval_dict["T"].loc[idx,:].dropna().index
           
           airRH    = upsampled_sondes["RH"].loc[idx,vertical_idx].values * units.percent
           airT     = np.array(retrieval_dict["T"].loc[idx,vertical_idx]) * units.K
           airTdew  = mpcalc.dewpoint_from_relative_humidity(airT,airRH)
          
           airP=np.array(upsampled_sondes["Pres"].loc[idx,vertical_idx]) * units.hPa
           pw_value= mpcalc.precipitable_water(airTdew,airP)
           retrieval_dict["IWV"].loc[idx]=np.array(pw_value)/0.99
                        
           #IVT
           q=pd.Series(data=retrieval_dict["q"].loc[idx,:].values,
                       index=upsampled_sondes["Pres"].loc[idx,vertical_idx]*100)
           # prepare variables for metpy --> set units
           wspeed=upsampled_sondes["Wspeed"].loc[idx,vertical_idx].values * units.meter / units.second
           wdir= np.deg2rad(upsampled_sondes["Wdir"].loc[idx,vertical_idx].values)
           u_metpy,v_metpy= mpcalc.wind_components(wspeed,wdir)
       
           u=pd.Series(data=u_metpy,
                       index=upsampled_sondes["Pres"].loc[idx,vertical_idx]*100)
           v=pd.Series(data=v_metpy,
                       index=upsampled_sondes["Pres"].loc[idx,vertical_idx]*100)
           
           u=u.loc[q.index]
           v=v.loc[q.index]
           qu=q*u
           qv=q*v
           qu=qu.dropna()
           qv=qv.dropna()        
           
           retrieval_dict["IVT_u"].loc[idx] = 1/g*np.trapz(qu,x=qu.index)
           retrieval_dict["IVT_v"].loc[idx] = 1/g*np.trapz(qv,x=qv.index)
           retrieval_dict["IVT"].loc[idx]   = np.sqrt(
                                          retrieval_dict["IVT_u"].loc[idx]**2+
                                          retrieval_dict["IVT_v"].loc[idx]**2)
           Performance.performance.updt(self,upsampled_sondes["Wspeed"].shape[0],int_idx)
           int_idx+=1 
       return retrieval_dict
    #%% Dropsondes
    def load_dropsonde_data(self,date,print_arg="yes",dt="all",plotting="no"):
        R_L= 287  #J/(kg*K)
        c_p= 1005 #J(kg*K)
        file_path=self.dropsonde_path
        if self.name=="NARVAL-II":
            filestart="halo_nar2_sonde_l1_any_v00_"
            files =[file_path+filestart+"20160808113814.nc",
                file_path+filestart+"20160810132921.nc",
                file_path+filestart+"20160812130653.nc",
                file_path+filestart+"20160815123601.nc",
                file_path+filestart+"20160817161747.nc",
                file_path+filestart+"20160819134616.nc",
                file_path+filestart+"20160822151159.nc",
                file_path+filestart+"20160822151159.nc",
                file_path+filestart+"20160824135058.nc",
                file_path+filestart+"20160826142839.nc",
                file_path+filestart+"20160830102328.nc"]
        elif self.name=="NAWDEX":
            filestart="halo_nawd_sonde_l1_any_v00_"
            files =[file_path+filestart+"20160917100146.nc",
                    file_path+filestart+"20160921143445.nc",
                    file_path+filestart+"20160923081938.nc",
                    file_path+filestart+"20160926151144.nc",
                    file_path+filestart+"20160927122340.nc",
                    file_path+filestart+"20161006091741.nc",
                    file_path+filestart+"20161009105807.nc",
                    file_path+filestart+"20161010150549.nc",
                    file_path+filestart+"20161013082804.nc",
                    file_path+filestart+"20161014093129.nc",
                    file_path+filestart+"20161015091005.nc",
                    file_path+filestart+'20161018091859.nc']
        
        print("Load the Dropsonde Profiles from "+date+" in file:")    
        for file_loop in files:
            if date in file_loop:
                file=file_loop
                print(file)
                break
                
        dataset=xr.open_dataset(file)#netCDF4.Dataset(file)
        
        sonde_number=np.array(dataset["sonde_number"][:])
        launch_time=pd.DatetimeIndex(np.array(dataset["base_time"][:]))
        sonde_time=pd.DatetimeIndex(np.array(dataset["sonde_time"][:]))
        #time=dataset["sonde_time"][:]
        #launch_time=pd.to_datetime(launch_time,unit='s')
        info_df = pd.DataFrame(data=sonde_number, index= launch_time,columns=['Sonde_number'])
        if dt=="all":
            sonde_number = [ int(x) for x in sonde_number ]
            if print_arg=="yes":
                print('No specific Time chosen, dataset will cover profiles of all dropsondes during flight')
                print('So, all dropsondes profiles will be merged in a dictionary')
                print(sonde_number)
                #print(launch_time)
            else:
                pass
            
            #Specify Dropsonde Dict with Sonde number and heights as columns for pd.DataFrame
            sonde_dict={}
            for i in sonde_number:
                sonde_dict["Sonde{0}".format(i)]=launch_time[i-1].round('1s')
            
            df_heights=np.array(dataset["height"][:])
            df_heights= [ str(x) for x in df_heights ]
            #df_heights=list(df_heights)
            if len(sonde_number)>1:
                # Write data in dfs
                lon             = pd.DataFrame(data=np.array(dataset["lon"][:]),
                                               columns=df_heights)
                time            = pd.DataFrame(data=np.array(dataset["sonde_time"][:]),
                                               columns=df_heights)
                lat             = pd.DataFrame(data=np.array(dataset["lat"][:]),
                                               columns=df_heights)
                lon             = pd.DataFrame(data=np.array(dataset["lon"][:]),
                                               columns=df_heights)
                #MR              = pd.DataFrame(data=dataset["mr"][:],columns=df_heights)
                #MR.index        = sonde_dict.keys()
                RH              = pd.DataFrame(data=np.array(dataset["hur"][:]),
                                               columns=df_heights)
                TempAir         = pd.DataFrame(data=np.array(dataset["ta"][:]),
                                               columns=df_heights)
                Pres            = pd.DataFrame(data=np.array(dataset["pa"][:]),
                                               columns=df_heights)
                Wdir            = pd.DataFrame(data=np.array(dataset["wdir"][:]),
                                               columns=df_heights)
                Wspeed          = pd.DataFrame(data=np.array(dataset["wspeed"][:]),
                                               columns=df_heights)
                Theta           = TempAir*(1000/Pres)**(R_L/c_p)
                
                lat.index       = sonde_dict.keys()
                time.index      = sonde_dict.keys()
                lon.index       = sonde_dict.keys()
                RH.index        = sonde_dict.keys()
                TempAir.index   = sonde_dict.keys()
                Pres.index      = sonde_dict.keys()
                Wdir.index      = sonde_dict.keys()
                Wspeed.index    = sonde_dict.keys()
                Theta.index     = sonde_dict.keys()
                
                if self.name=="NARVAL-II":
                    #Cut Dropsondes which are generally more easterly then -45°E
                    time            = time.loc[lon.max(axis=1)<-40,:]
                    lat             = lat.loc[lon.max(axis=1)<-40,:]
                    RH              = RH.loc[lon.max(axis=1)<-40,:]
                    TempAir         = TempAir.loc[lon.max(axis=1)<-40,:]
                    Pres            = Pres.loc[lon.max(axis=1)<-40,:]
                    Wdir            = Wdir.loc[lon.max(axis=1)<-40,:]
                    Wspeed          = Wspeed.loc[lon.max(axis=1)<-40,:]
                    Theta           = Theta.loc[lon.max(axis=1)<-40,:]
                    lon             = lon.loc[lon.max(axis=1)<-40,:]
                
                # Additionally calculate the lower tropospheric stability
                columns=[]
                LTS=pd.DataFrame(index=Pres.index,columns=["Time","Sonde","LTS_[K]"])
                i=0
                #LTS is defined as the Theta-difference between 700 hPa and surface
                for idx in Pres.index:
                    pres_series=Pres.loc[idx,:]
                    
                    try:
                        columns.append(pres_series[pres_series==pres_series[(pres_series-700).abs().argsort()][0]].index[0])
                        LTS["LTS_[K]"].loc[idx]=Theta.loc[idx,columns[i]]-Theta.loc[idx,"0.0"]
                    except:
                        j=0
                        for column in Theta.columns[0:5]:
                            try:
                                columns.append(pres_series[pres_series==pres_series[(pres_series-700).abs().argsort()][j]].index[0])
                                LTS["LTS_[K]"].loc[idx]=Theta.loc[idx,columns[i]]-Theta.loc[idx,"30.0"]
                                break
                            except:
                                pass
                            j+=1
                    i=i+1
                LTS["Time"]=sonde_dict.values()
                LTS["Sonde"]=sonde_dict.keys()
                LTS.index=LTS["Time"]
                del LTS["Time"]
                  
                Dropsondes                  = {}
                Dropsondes["Sonde_Infos"]   = sonde_dict
                Dropsondes["Height"]        = dataset["height"][:]
                Dropsondes["Time"]          = time
                Dropsondes["Lat"]           = lat
                Dropsondes["Lon"]           = lon
                Dropsondes["AirT"]          = TempAir -273.16
                Dropsondes["Theta"]         = Theta
                Dropsondes["RH"]            = RH
                Dropsondes["Pres"]          = Pres
                Dropsondes["Wspeed"]        = Wspeed
                Dropsondes["Wdir"]          = Wdir
                Dropsondes["LTS"]           = LTS
                
                ### Extra calculated variables
                print("Calculate additional meteorological parameters")
                T     = Dropsondes['AirT'].values * units.degC
                RH    = Dropsondes["RH"].values * units.percent
                P     = Dropsondes["Pres"].values * units.hPa
                Tdew  = mpcalc.dewpoint_from_relative_humidity(T,RH)
                MR    = mpcalc.mixing_ratio_from_relative_humidity(RH, T, P)
                #print(MR)
                Dropsondes["MR"] = pd.DataFrame(data=np.array(MR),index=Dropsondes["AirT"].index)
                Dropsondes["Dewpoint"]=pd.DataFrame(data=np.array(Tdew),index=Dropsondes["AirT"].index)
                PW= pd.Series(data=np.nan,index=LTS.index)
                for sonde in range(Dropsondes["Dewpoint"].shape[0]):
                    
                    pw_value= mpcalc.precipitable_water(Tdew[sonde,:],P[sonde,:])
                    PW.iloc[sonde]=np.array(pw_value)
                Dropsondes["IWV"]=PW/0.99
                #print(test)
                #Not accessible variables
                #Dropsondes["Theta_e"]       = Theta_e
                #Dropsondes["MR"]            = MR
                return Dropsondes;
            else:
                lon             = pd.Series(data=np.array(dataset["lon"][:]),index=df_heights)
                time            = pd.Series(data=np.array(dataset["sonde_time"][:]),index=df_heights)
                lat             = pd.Series(data=np.array(dataset["lat"][:]),index=df_heights)
                lon             = pd.Series(data=np.array(dataset["lon"][:]),index=df_heights)
                Dropsondes                  = {}
                Dropsondes["Sonde_Infos"]   = sonde_dict
                Dropsondes["Height"]        = dataset["height"][:]
                Dropsondes["Time"]          = time
                Dropsondes["Lat"]           = lat
                Dropsondes["Lon"]           = lon
            return Dropsondes;
        else:
            print('Take only dropsonde launched around: ', dt)
            idx = info_df.index[info_df.index.get_loc(dt, method='pad')]
            print("Nearest Launch Time before time desired: ",idx)
            needed_sonde_no=int(info_df["Sonde_number"].loc[idx])
            df=pd.DataFrame(columns=["Sonde_Time","Height"])
            df["Height"]=dataset["height"][:]
            df["Sonde_Time"]=sonde_time[needed_sonde_no-1,:]
            df["Sonde_Time"]=pd.to_datetime(df["Sonde_Time"],unit='s')
            df.index=df["Sonde_Time"]
            #del df["Sonde_Time"]
            df["Lat_Sonde"]=dataset["lat"][needed_sonde_no-1,:]
            df["Lon_Sonde"]=dataset["lon"][needed_sonde_no-1,:]
            #df["MR_(g/kg)"]=dataset["mr"][needed_sonde_no-1,:]
            df["TempAir_(degC)"]=dataset["ta"][needed_sonde_no-1,:]
            df["P_(hPa)"]=dataset["pa"][needed_sonde_no-1,:]
            df["RH_(%)"]=dataset["hur"][needed_sonde_no-1,:]
            #mpcalc.relative_humidity_from_specific_humidity(df[], kwargs)
            #except:
            #    df["RH_(%)"]=
            
            df["Wdir_(deg)"]=dataset["wdir"][needed_sonde_no-1,:]
            df["Wspeed_(m/s)"]=dataset["wspeed"][needed_sonde_no-1,:]
    #        df["Theta_(K)"]=dataset["theta"][needed_sonde_no-1,:]
    #        df["Theta_e_(K)"]=dataset["theta_e"][needed_sonde_no-1,:]
            df=df.sort_index()
            df[df.Lat_Sonde==-999]=np.nan              # remove all NaT values
            df=df.dropna()#(["Sonde_Time"], axis=1, inplace=True)
            del df["Sonde_Time"]
            return idx,needed_sonde_no,df;
        
    def upsample_dropsonde_data(self,Dropsondes,dropsonde_var_list,halo_data,
                                interpolation_method="bfill",
                                dataset="Dropsonde"):
        """
        Parameters
        ----------
        Dropsondes : dict
            Dictionary containing the dropsonde data.
        dropsond_var_list : list
            variables to upsample.
        halo_data : TYPE
            DESCRIPTION.
        interpolation_method : TYPE, optional
            DESCRIPTION. The default is "bfill".
        dataset : TYPE, optional
            DESCRIPTION. The default is "Dropsonde".

        Returns
        -------
        Upsampled_Dropsondes : dict
            DESCRIPTION.

        """
        print("Upsample ",dataset," Dataset")
        Upsampled_Dropsondes={}
        for var in dropsonde_var_list:
            print("Interpolate Variable ",var)
            upsampled_dropsond_var_df=Dropsondes[var].copy()
            if not isinstance(upsampled_dropsond_var_df.index,pd.DatetimeIndex):
                sonde_index=pd.DatetimeIndex(Dropsondes["Sonde_Infos"].values())
                upsampled_dropsond_var_df.index=sonde_index
            if isinstance(upsampled_dropsond_var_df,pd.DataFrame):
                 new_df=pd.DataFrame(data=np.nan,
                                     columns=upsampled_dropsond_var_df.columns,
                                     index=pd.DatetimeIndex(halo_data.index))
            elif var=="IWV" or var=="IVT":
                new_df=pd.Series(data=np.nan,
                                 index=pd.DatetimeIndex(halo_data.index))
            else:
                # elif isinstance(upsampled_dropsond_var_df,pd.Series) 
                # and if var is not "IWV" or "IVT":
                print("Only 1 Dropsonde is included, no interpolation possible")
                AssertionError("The given variable has a wrong type,",
                                " choose series or DataFrame")
            
            # The new dataframe in the desired resolution is added to the current
            # dropsonde dataframe
            upsampled_dropsond_var_df=upsampled_dropsond_var_df.append(new_df)
            upsampled_dropsond_var_df=upsampled_dropsond_var_df.sort_index() 
            if (interpolation_method=="bfill") or (interpolation_method=="ffill"):
                upsampled_dropsond_var_df=upsampled_dropsond_var_df.fillna(
                    method="ffill").fillna(method="bfill")
            else:
                upsampled_dropsond_var_df=upsampled_dropsond_var_df.interpolate(
                    method=interpolation_method)
            upsampled_dropsond_var_df=upsampled_dropsond_var_df.loc[pd.DatetimeIndex(
                halo_data.index)]
            try:
                upsampled_dropsond_var_df=upsampled_dropsond_var_df.reset_index().\
                                        drop_duplicates(subset='index', 
                                                        keep='last').\
                                            set_index('index')
            except:
                upsampled_dropsond_var_df=upsampled_dropsond_var_df.reset_index().\
                                        drop_duplicates(subset='Unnamed: 0', 
                                                        keep='last').\
                                            set_index('Unnamed: 0')
                
            Upsampled_Dropsondes[var]=upsampled_dropsond_var_df
        return Upsampled_Dropsondes

    def calculate_dropsonde_ivt(self,Dropsondes,date,ar_of_day,
                                flight,save_sounding=True):
        """
        
    
        Parameters
        ----------
        Dropsondes : dict
            The Dropsondes dictionary containing all data on unified grid.
    
        Returns
        -------
        Dropsondes : dict
            Updated Dropsondes dictionary with IVT-keys on unified grid.
    
        """
        if not "q" in list(Dropsondes.keys()):
            # if q is not already calculated, it has to be computed
            Dropsondes["q"]=mpcalc.specific_humidity_from_mixing_ratio(Dropsondes["MR"])
            Dropsondes["q"].columns=Dropsondes["Wspeed"].columns
        
        # Start IVT Calculations
        Dropsondes["IVT_u"] = pd.Series(data=np.nan,index=Dropsondes["IWV"].index)
        Dropsondes["IVT_v"] = pd.Series(data=np.nan,index=Dropsondes["IWV"].index)
        Dropsondes["IVT"]   = pd.Series(data=np.nan,index=Dropsondes["IWV"].index)
        print("Calculate IVT from dropsondes")
        for sounding in range(Dropsondes["q"].shape[0]):
            #print("Sounding No. ", sounding)
            q     = pd.Series(data=Dropsondes["q"].iloc[sounding,:].values,
                              index=Dropsondes["Pres"].iloc[sounding,:].values*100)
            q= q.dropna()
            q=q.sort_index()
            g= 9.81
            wspeed=Dropsondes["Wspeed"].iloc[sounding,:].values * units.meter / units.second
            wdir= np.deg2rad(Dropsondes["Wdir"].iloc[sounding,:].values)
            u_metpy,v_metpy= mpcalc.wind_components(wspeed,wdir)
            u=pd.Series(data=u_metpy,index=Dropsondes["Pres"].iloc[sounding,:]*100)
            v=pd.Series(data=v_metpy,index=Dropsondes["Pres"].iloc[sounding,:]*100)
            u=u.loc[q.index]
            v=v.loc[q.index]
            qu=q*u
            qv=q*v
            qu=qu.dropna()
            qv=qv.dropna()        
            Dropsondes["IVT_u"].iloc[sounding] = 1/g*np.trapz(qu,x=qu.index)
            Dropsondes["IVT_v"].iloc[sounding] = 1/g*np.trapz(qv,x=qv.index)
            Dropsondes["IVT"].iloc[sounding]   = np.sqrt(Dropsondes["IVT_u"].iloc[sounding]**2+
                                                         Dropsondes["IVT_v"].iloc[sounding]**2)
        # if save_sounding:
        #     dropsond_data_path=self.major_path+self.name+"/data/HALO-Dropsonden/"
        #     print("Save AR Dropsonde Soundings under: ", dropsond_data_path)
        #     storage_Dropsondes={}
        #     storage_Dropsondes["IVT"]=Dropsondes["IVT"]
        #     storage_Dropsondes["q"]=Dropsondes["q"]
        #     storage_Dropsondes["Wdir"]=Dropsondes["Wdir"]#
        #     storage_Dropsondes["Wspeed"]=Dropsondes["Wspeed"]
        #     storage_Dropsondes["q"].index=Dropsondes["IVT"].index
        #     storage_Dropsondes["Wdir"].index=Dropsondes["IVT"].index
        #     storage_Dropsondes["Wspeed"].index=Dropsondes["IVT"].index
        #     storage_Dropsondes["IVT"].to_csv(path_or_buf=dropsond_data_path+flight+"_"+ar_of_day+"_IVT_Dropsondes.csv",
        #                                      header=True,index=True)
        #     storage_Dropsondes["q"].to_csv(path_or_buf=dropsond_data_path+flight+"_"+ar_of_day+"_Q_Dropsondes.csv",
        #                                    header=True,index=True)
        #     storage_Dropsondes["Wdir"].to_csv(path_or_buf=dropsond_data_path+flight+"_"+ar_of_day+"_WD_Dropsondes.csv",
        #                                       header=True,index=True)
        #     storage_Dropsondes["Wspeed"].to_csv(path_or_buf=dropsond_data_path+flight+"_"+ar_of_day+"_WS_Dropsondes.csv",
        #                                         index=True,header=True)
        
        return Dropsondes
    
    def load_ar_processed_dropsondes(self,grid_class,date,radar,halo_df,flight,
                                     with_upsampling=False,
                                     ar_of_day=None):
        file_path=self.major_path+self.name+"/data/HALO-Dropsonden/"
        self.dropsonde_path=file_path
        if ar_of_day:
            if not os.path.exists(self.dropsonde_path+flight[0]+"_"+ar_of_day+\
                                  "_Upsampled_Dropsondes.npy"):
                Dropsondes=self.load_dropsonde_data(date,print_arg="yes",
                                          dt="all",plotting="no")
    
                #print("Dropsondes done")
                #try:
                #    Dropsondes["IWV"]=Dropsondes["IWV"].loc[\
                #                                radar["Reflectivity"].index[0]:\
                #                                radar["Reflectivity"].index[-1]]
                #except:
                #    print("Just one Dropsonde so no further IWV Verification")
                Dropsondes["q"]=mpcalc.specific_humidity_from_mixing_ratio(
                                            Dropsondes["MR"])
                Dropsondes["q"].columns=Dropsondes["Wspeed"].columns
                #--------------------------------------------------------------#
                ## Calculate Theta_e for plots
                T     = Dropsondes['AirT'].values * units.degC
                P     = Dropsondes["Pres"].values * units.hPa
                Tdew  = Dropsondes["Dewpoint"].values * units.degC
    
                sonde_Theta_e=mpcalc.equivalent_potential_temperature(P,T,Tdew)
                del T
                del P
                del Tdew
                Dropsondes["Theta_e"] = pd.DataFrame(
                                            data=np.array(sonde_Theta_e),
                                            index=Dropsondes["AirT"].index,
                                            columns=Dropsondes["AirT"].columns)
                #--------------------------------------------------------------#            
                ## Calculate IVT from Dropsondes
                Dropsondes=self.calculate_dropsonde_ivt(Dropsondes,
                                                  date,ar_of_day,flight)
                dropsonde_vars_to_upsample=["Wspeed","Wdir","q","RH",
                                "AirT","Theta_e","IVT","IWV","Pres"]
                
                if with_upsampling:
                    try:
                        Upsampled_Dropsondes=self.upsample_dropsonde_data(
                                    Dropsondes,dropsonde_vars_to_upsample,
                                    halo_df,interpolation_method="time",
                                    dataset="Dropsonde")
                        Upsampled_Dropsondes=self.upsample_dropsonde_data(
                                            Upsampled_Dropsondes,
                                            dropsonde_vars_to_upsample,halo_df,
                                            interpolation_method="bfill",
                                            dataset="Dropsonde")        
                        if ar_of_day:
                            #upsampled sondes
                            halo_df,Upsampled_Dropsondes,ar_of_day=\
                                        grid_class.cut_halo_to_AR_crossing(
                                        ar_of_day, flight[0], 
                                        halo_df,Upsampled_Dropsondes,
                                        device="sondes")
                    except:
                        print("Dropsondes are not upsampled.")
                    
                    # save as pickle
                    # Dropsonde:
                    file_pickle=self.dropsonde_path+flight[0]+\
                            "_Dropsondes.npy"
                    
                    with open(file_pickle,'wb') as file:
                        pickle.dump({"Dropsondes":Dropsondes},file,protocol=-1)
                        print('Dropsonde:',file_pickle)
                    
                    #Upsampled Dropsondes
                    if not ar_of_day:
                        file_pickle=self.dropsonde_path+flight[0]+\
                                "_Upsampled_Dropsondes.npy"
                    else:
                        file_pickle=self.dropsonde_path+flight[0]+"_"+\
                            ar_of_day+"_Upsampled_Dropsondes.npy"
                    
                    with open(file_pickle,'wb') as file:
                        pickle.dump({"Upsampled_Dropsondes":\
                                     Upsampled_Dropsondes},file,protocol=-1)
                        print('Upsampled Dropsondes:',file_pickle)
                    
                    # save as csv
                    grid_class.save_ar_dropsonde_soundings(Dropsondes,
                                             variables=["IVT","IWV","q","Wdir",
                                                        "Wspeed","Pres"],
                                             save_path=self.dropsonde_path,
                                             research_flight=flight[0],
                                             ar_number=ar_of_day,
                                             upsampled=False)
                    # save Upsampled AR Dropsonde soundings
                    grid_class.save_ar_dropsonde_soundings(Upsampled_Dropsondes,
                                             save_path=self.dropsonde_path,
                                             variables=["IVT","IWV","q","Wdir",
                                                        "Wspeed","Pres"],                                    
                                             research_flight=flight[0],
                                             ar_number=ar_of_day,upsampled=True)
            else:
                
                fname_pickle=self.dropsonde_path+flight[0]+\
                        "_Dropsondes.npy"
                
                with open(fname_pickle,'rb') as file:
                    pickle_dict=pickle.load(file)
                    Dropsondes=pickle_dict["Dropsondes"]
                    del pickle_dict
                
                if ar_of_day:
                    fname_pickle=self.dropsonde_path+flight[0]+"_"+ar_of_day+\
                                    "_Upsampled_Dropsondes.npy"
                else:
                    fname_pickle=self.dropsonde_path+flight[0]+\
                        "_Upsampled_Dropsondes.npy"
                
                with open(fname_pickle,'rb') as file:
                    pickle_dict=pickle.load(file)
                    Upsampled_Dropsondes=pickle_dict["Upsampled_Dropsondes"]
            
            return Dropsondes,Upsampled_Dropsondes
        else:
            return Dropsondes
    
    def plot_AR_sonde_thermodynamics(self,upsampled_sondes,radar,date,
                                        flight,path,start,end,
                                        plot_path=os.getcwd(),
                                        save_figure=True,low_level=False,
                                        ar_of_day=None):    
        
        import matplotlib.dates as mdates
        hours = mdates.MinuteLocator(byminute=[0,10,20,30,40,50,60],interval = 1)
        h_fmt = mdates.DateFormatter('%H:%M')
    
        theta_colormap      = "RdYlBu_r"
        humidity_colormap   = "terrain_r"
        #Then tick and format with matplotlib:
        fig=plt.figure(figsize=(16,12))
        
        
        # Get high reflectivities
        high_dbZ_index=radar["Reflectivity"][radar["Reflectivity"]>15].any(axis=1)
        high_dbZ=radar["Reflectivity"].loc[high_dbZ_index]
        #Liquid water
        ax1=fig.add_subplot(211)
        x_temp=upsampled_sondes["Theta_e"].loc[start:end].index#.time
        y_temp=upsampled_sondes["Theta_e"].columns.astype(float)
        y,x=np.meshgrid(y_temp,x_temp)
        
        levels=np.linspace(285,350,66)
        if flight=="RF10":
            levels=np.linspace(275,330,56)
        if low_level:
            levels=np.linspace(285,320,36)
            if flight=="RF10":
                levels=np.linspace(275,310,26)
        #Create new Hydrometeor content as a sum of all water contents
        cutted_theta=upsampled_sondes["Theta_e"].loc[start:end]
        
        C1=ax1.pcolormesh(x,y,cutted_theta,cmap=theta_colormap,
                          vmin=levels[0],vmax=levels[-1])
        cb = plt.colorbar(C1,extend="both")
        cb.set_label('Interpolated Dropsondes: $\Theta_{e}$ in K')
        
        if low_level:
            ax1.set_ylim(0,6000)
            ax1.set_yticks([0,500, 1000,1500,2000,2500, 3000, 
                            3500, 4000, 4500, 5000,5500,6000])
            ax1.set_yticklabels([0,500, 1000, 1500, 2000,2500,3000, 3500,
                                 4000, 4500, 5000,5500,6000])
            marker_pos=np.ones(len(high_dbZ))*5800
        else:
            ax1.set_ylim(0,12000)
            ax1.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax1.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
            marker_pos=np.ones(len(high_dbZ))*11000
        
        ax1.scatter(high_dbZ.index,marker_pos,s=35,color="white",marker="D",
                    linewidths=0.2,edgecolor="k")
        
        
        for label in ax1.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        ax1.set_ylabel('Altitude in m ')
        ax1.set_xlabel('')
        ax1.xaxis.set_major_locator(hours)
        ax1.xaxis.set_major_formatter(h_fmt)
        
        #Specific humidity
        q_min=0
        q_max=10
        
        if low_level:
            q_min=0
            q_max=10
        
        
        ax2=fig.add_subplot(212)
        levels=np.linspace(q_min,q_max,50)
        
        #Create new Hydrometeor content as a sum of all water contents
        cutted_hmc=upsampled_sondes["q"].loc[start:end]
        
        C2=ax2.pcolormesh(x,y,cutted_hmc*1000,
                          vmin=q_min,vmax=q_max,
                          cmap=humidity_colormap)
        #mpcalc.mixing_ratio_from_specific_humidity(halo_era5_hmc["q"])
        mixing_ratio=cutted_hmc
        
        moisture_levels=[5,10,20,30,40]
        wind_levels=[10,20,40,60]
        if low_level:
            moisture_levels=[5,10,20,30,35]
            wind_levels=[10,20,30]
        wind=upsampled_sondes["Wspeed"]
        
        wv_flux=mixing_ratio*wind #halo_era5_hmc["wind"]
        moisture_flux=1/9.82*wv_flux*1000
        print("Contour lines take long")
        
        CS=ax2.contour(x,y,moisture_flux.loc[start:end],
                       levels=moisture_levels,colors="k",
                       linestyles="-",linewidths=1.0)
        print("Contour Moisture Flux done")
        CS2=ax2.contour(x,y,wind.loc[start:end],
                       levels=wind_levels,colors="magenta",
                       linestyles="--",linewidths=1.0)
        print("Contour Wind done")
        # Contour lines and Colorbar specifications
        ax2.clabel(CS,fontsize=16,fmt='%1.1f',inline=1)
        ax2.clabel(CS2,fontsize=16,fmt='%1.1f',inline=1)
        
        cb = plt.colorbar(C2,extend="max")
        cb.set_label('Interpolated Dropsondes: \n Specific Humidity in g/kg')
        
        if low_level:
            ax2.set_ylim(0,6000)
            ax2.set_yticks([0,500, 1000,1500,2000,2500, 3000, 
                            3500, 4000, 4500, 5000,5500,6000])
            ax2.set_yticklabels([0,500, 1000, 1500, 2000,2500,3000, 3500,
                                 4000, 4500, 5000,5500,6000])
        
            fig_name="Thermodynamics_HALO_Sondes_low_level_"+flight+"_"+\
                date+"_"+str(start)[-8:-6]+str(start)[-5:-3]+"_"+\
                str(end)[-8:-6]+str(end)[-5:-3]+".png"
            
        else:
            ax2.set_ylim(0,12000)
            ax2.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax2.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
            fig_name="Thermodynamics_HALO_Sondes_"+flight+"_"+date+"_"+\
                str(start)[-8:-6]+str(start)[-5:-3]+"_"+str(end)[-8:-6]+\
                    str(end)[-5:-3]+".png"
            
        for label in ax2.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        
        ax2.set_ylabel('Altitude in m ')
        ax2.set_xlabel('')
        ax2.xaxis.set_major_locator(hours)
        ax2.xaxis.set_major_formatter(h_fmt)
        if ar_of_day is not None:
                fig_name=ar_of_day+"_"+fig_name
        
        fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", path+fig_name)
        return None
            
            


#-----------------------------------------------------------------------------#
# Different Campaigns that are subclasses of Campaign                        
class NAWDEX(Campaign):
        def __init__(self,is_flight_campaign,major_path,aircraft,instruments,
                     interested_flights="all"):
            #number_of_flights,first_flight,last_flight,region):
            super().__init__(is_flight_campaign,major_path,aircraft,instruments,
                             interested_flights="all")
            self.name="NAWDEX"
            self.campaign_name=self.name
            self.flight_day={"RF01":"17","RF02":"21","RF03":"23","RF04":"26",
                             "RF05":"27","RF06":"01","RF07":"06","RF08":"09",
                             "RF09":"10","RF10":"13","RF11":"14","RF12":"15",
                             "RF13":"18"}
            self.flight_month={"RF01":"09","RF02":"09","RF03":"09","RF04":"09",
                               "RF05":"09","RF06":"10","RF07":"10","RF08":"10",
                               "RF09":"10","RF10":"10","RF11":"10","RF12":"10",
                               "RF13":"10"}
            self.years={"RF01":"2016","RF02":"2016","RF03":"2016",
                        "RF04":"2016","RF05":"2016","RF06":"2016",
                        "RF07":"2016","RF08":"2016","RF09":"2016",
                        "RF10":"2016","RF11":"2016","RF12":"2016",
                        "RF13":"2016"}
            
            self.flights=self.flight_day.keys()
            self.year="2016"
            
            self.campaign_path=self.major_path+"/"+self.name
            self.data_path=self.campaign_path+"/data/"
            self.campaign_data_path=self.campaign_path+"/data/"
            
            self.plot_path=self.campaign_path+"/plots/"
            if not os.path.exists(self.campaign_path):
                os.makedirs(self.campaign_path)
                print("Path of Campaign ",self.name," is created under: ",
                      self.campaign_path)
            else:
                print("Overall directory of campaign work is: ",
                      self.campaign_path)

class NARVALII(Campaign):
        def __init__(self,is_flight_campaign,major_path,aircraft,instruments,
                     interested_flights="all"):
            #number_of_flights,first_flight,last_flight,region):
            super().__init__(is_flight_campaign,major_path,aircraft,instruments,
                             interested_flights="all")
            self.name="NARVAL-II"
            self.flight_day={"RF01":"08","RF02":"10","RF03":"12","RF04":"15",
                             "RF05":"17","RF06":"19","RF07":"22","RF08":"24",
                             "RF09":"25"}
            self.flight_month={"RF01":"08","RF02":"08","RF03":"08","RF04":"08",
                               "RF05":"08","RF06":"08","RF07":"08","RF08":"08",
                               "RF09":"08"}
            self.year="2016"
            self.campaign_path=self.major_path+"/"+self.name
            
            if not os.path.exists(self.campaign_path):
                os.makedirs(self.campaign_path)
                print("Path of Campaign ",self.name," is created under: ",
                      self.campaign_path)
            else:
                print("Overall directory of campaign work is: ",
                      self.campaign_path)
                        
class EUREC4A(Campaign):
        def __init__(self,is_flight_campaign,major_path,aircraft,instruments,
                     interested_flights="all"):
            #number_of_flights,first_flight,last_flight,region):
            super().__init__(is_flight_campaign,major_path,aircraft,instruments,
                             interested_flights="all")
            self.name="EUREC4A"
            self.flight_day={"RF01":"17","RF02":"21","RF03":"23","RF04":"26",
                             "RF05":"27","RF06":"01","RF07":"06","RF08":"09",
                             "RF09":"10","RF10":"13","RF11":"14","RF12":"15",
                             "RF13":"18"}
            self.flight_month={"RF01":"09","RF02":"09","RF03":"09","RF04":"09",
                               "RF05":"09","RF06":"10","RF07":"10","RF08":"10",
                               "RF09":"10","RF10":"10","RF11":"10","RF12":"10",
                               "RF13":"10"}
            #self.flights=self.flights.keys()
            self.year="2020"
            self.campaign_path=self.major_path+"/"+self.name
            
            if not os.path.exists(self.campaign_path):
                os.makedirs(self.campaign_path)
                print("Path of Campaign ",self.name," is created under: ",
                      self.campaign_path)
            else:
                print("Overall directory of campaign work is: ",
                      self.campaign_path)
        
class HALO_AC3(Campaign):
    def __init__(self,is_flight_campaign,major_path,aircraft,instruments,
                 interested_flights="all"):
        super().__init__(is_flight_campaign,major_path,aircraft,instruments,
                         interested_flights="all")
        self.name="HALO_AC3"
        self.campaign_name=self.name
            
        self.flight_day={"RF01":"11",
                         "RF02":"12",
                         "RF03":"13",
                         "RF04":"14",
                         "RF05":"15",
                         "RF06":"16",
                         "RF07":"20",
                         "RF08":"21",
                         "RF09":"28",
                         "RF10":"29",
                         "RF11":"30",
                         "RF12":"01",
                         "RF13":"04",
                         "RF14":"07",
                         "RF15":"08",
                         "RF16":"10",
                         "RF17":"11",
                         "RF18":"12"}
        
        self.flight_month={"RF01":"03",
                           "RF02":"03",
                           "RF03":"03",
                           "RF04":"03",
                           "RF05":"03",
                           "RF06":"03",
                           "RF07":"03",
                           "RF08":"03",
                           "RF09":"03",
                           "RF10":"03",
                           "RF11":"03",
                           "RF12":"04",
                           "RF13":"04",
                           "RF14":"04",
                           "RF15":"04",
                           "RF16":"04",
                           "RF17":"04",
                           "RF18":"04"}
        
        #self.flights=self.flights.keys()
        self.year="2022"
        self.campaign_path=self.major_path+"/"+self.name
        if not os.path.exists(self.campaign_path):
            os.makedirs(self.campaign_path)
            print("Path of Campaign",self.name,"is created under:",
                  self.campaign_path)
        else:
            print("Overall directory of campaign is: ",
                  self.campaign_path)
###############################################################################
#%% Synthetic Campaign Class
class Synthetic_Campaign(Campaign):
    flight_day={}
    flight_month={}
    def __init__(self,is_flight_campaign=True,major_path=os.getcwd(),
                 aircraft=None,instruments=[],flights=[],
                 interested_flights="all"):
        
        self.is_flight_campaign=is_flight_campaign
        self.aircraft=aircraft
        self.instruments=instruments
        self.flight_day={}
        self.flight_month={}
        self.interested_flights="all"
        self.major_path=major_path
        self.is_synthetic_campaign=True
            
    #%% General
    def specify_flights_of_interest(self,interested_flights):
        if interested_flights==["all"]:
            self.interested_flights=self.flight_days.keys()
        else:
            self.interested_flights=interested_flights
    def create_directory(self,directory_types):
        """
        

        Parameters
        ----------
        overall_path : str
            specified path in which the subdirectories need to be integrated
            
        directory_types : list
            list of directory types to check if existent, can contain the 
            the entries ["data","result_data","plots"].

        Returns
        -------
        None.
        """
        
        #loop over list and check if path exist, if not create one:
        self.sub_paths={}
        added_paths={}
        for dir_type in directory_types:
            relevant_path=self.major_path+self.name+"/"+dir_type+"/"
            added_paths[dir_type]=relevant_path
            self.sub_paths[dir_type]=relevant_path
            if not os.path.exists(relevant_path):
                os.mkdir(relevant_path)
                print("Path ", relevant_path," has been created.")
            
        data_config.add_entries_to_config_object("data_config_file",
                                                 added_paths)
    def merge_dfs_of_flights(self,data_dict,flights_of_interest,parameter):
        """
        Parameters
        ----------
        data_dict : dictionary
            dictionary containing the dataframes of measurement data for all
            flights.
        
        parameter: str
            meteorological variable to merge, e.g. reflectivity or whatever
        Returns
        -------
        merged_df : pd.DataFrame
            dataframe which concats all flight measurements into one dataset.

        """
        merged_df=pd.DataFrame()
        i=0
        print("Merge datasets")
        for flight in flights_of_interest:
           print(flight)
           if not bool(data_dict[flight]):
               print("Dictionary for ",flight,"is empty.")
               continue
           if i==0:
               merged_df=data_dict[flight][parameter]
           else:
               merged_df=merged_df.append(data_dict[flight][parameter])
           i+=1
        return merged_df

    
    def dataframe_to_csv(self,input_df,save_path):
        """
        This function saves the desired dataset (input_df) in the given path
        by assessing the dataframes name which is neccessarily required
        """
        
        # Check if path ending is correct, so that df is stored correctly
        if not save_path[-1]=="/":
            save_path=save_path+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Check several attributes of the df
        # is input_df a real df?
        # Does have the df a name to refer on for saving df as csv?
        if isinstance(input_df, pd.DataFrame):  
            try:
                file_name=input_df.name
                df_has_name=True
            except:
                df_has_name=False
                print("No name of the df was ")
        else:
            raise AssertionError("The given dataset is no pd.DataFrame")
            has_saved=False
        
        # if df has a name, the data is stored in a csv-file with name of df 
        if df_has_name:
            input_df.to_csv(path_or_buf=save_path+file_name+".csv",index=True)
            has_saved=True
        if has_saved:
            print("DataFrame has saved successfully as:",
                  save_path+file_name+".csv")
    
    #%% Aircraft Data 
    def get_aircraft_waypoints(self,
                               filetype=".ftml"):
        """
        This function gets the waypoints (lat/lon) of aircraft via reading 
        the files according to file formats
        
        Input:
            list: flights_of_interest
            List of relevant flights to get data from
        Output:
            Dictionary: dict_position
            Dictionary with keys of interested flights each containing
            DataFrame with lat/lon as columns and UTC time as index
        """
        print("Get aircraft Waypoint data from ",self.name)
        self.waypoint_path=self.campaign_data_path+"Waypoints/"
        #dict_halo={}# --> to be changed
        dict_waypoints={}
        #if "Position" in dict_halo.keys(): #dict_halo only describes one flight
        #   dict_position[flights_of_interest[0]]=dict_halo["Position"] 
        
        # Read waypoints according to file format
        if filetype==".ftml":
            print("This will call the .ftml to .csv converter in the future")
            pass
        elif filetype==".csv":
            for flight in self.interested_flights:
                self.flight_date=self.year+self.flight_month[flight]+\
                    self.flight_day[flight]
                self.waypoint_fname="waypoints_"+self.flight_date+".csv"
                df=pd.read_csv(self.waypoint_path+self.waypoint_fname,
                                      sep=";",skiprows=1)
                df.index=pd.DatetimeIndex(df["Time (UTC)"])
                df.rename(columns={"Lat (+-90)": "Lat", "Lon (+-180)": "Lon"},
                          inplace=True)
                del df["Comments"]
                del df["Time (UTC)"]
                dict_waypoints[flight]=df
        else:
            raise TypeError("The format of the given waypoint file",
                            filetype," is wrong")
            
        return dict_waypoints     
    
    def invert_flight_from_waypoints(self,waypoints,flights):
        print("This function inverts the orientation of the flight track (cw&ccw)")
        print("So far, nothing is inverted, but will be added")
        
        inverted_waypoints={}
        inverted_waypoints[flights[0]]=waypoints[flights[0]].copy()
        inverted_waypoints[flights[0]]["Lon"]=inverted_waypoints[flights[0]]["Lon"]
        inverted_waypoints[flights[0]]["Lat"]=inverted_waypoints[flights[0]]["Lat"]
        inverted_waypoints[flights[0]]["Cum. dist. (km)"]=abs(\
                inverted_waypoints[flights[0]]["Cum. dist. (km)"]\
                -inverted_waypoints[flights[0]]["Cum. dist. (km)"].iloc[-1])
        inverted_waypoints[flights[0]]["timedelta"]=\
        inverted_waypoints[flights[0]].index.to_series().diff()
        inverted_waypoints[flights[0]]["timedelta"]=\
        inverted_waypoints[flights[0]]["timedelta"].shift(periods=-1,
                                                               fill_value=0)
        inverted_waypoints[flights[0]]=inverted_waypoints[flights[0]]\
                .reindex(index=inverted_waypoints[flights[0]].index[::-1])
        inverted_waypoints[flights[0]]["time_cumsum"]=\
            inverted_waypoints[flights[0]]["timedelta"].cumsum()
        inverted_waypoints[flights[0]]["inverted_time"]=\
            inverted_waypoints[flights[0]].index[-1]+\
                inverted_waypoints[flights[0]]["time_cumsum"]
        inverted_waypoints[flights[0]].index=\
                            inverted_waypoints[flights[0]]["inverted_time"]
    
        return inverted_waypoints
    def load_aircraft_position_from_csv(self,campaign,file_name_begin):
        campaign_path=self.major_path+campaign+"/"
        data_path=campaign_path+"data/"
        # if one is only interested in one flight
        if len(self.interested_flights)==1:
            #file_start is equivalently defined but depends on interested flights
            file_start=file_name_begin+campaign+self.interested_flights[0]
                
            pos_df=pd.read_csv(data_path+file_start+".csv")
            pos_df.index=pos_df["Unnamed: 0"]
            del pos_df["Unnamed: 0"]
            return pos_df,campaign_path
        
        # if interested in more than one flight, return variable is a dict
        else:
            for flight in self.interested_flights:
                file_start=file_name_begin+campaign+flight
                data_path=campaign_path+"data/"            
                pos_df=pd.read_csv(campaign_path+"data/BAHAMAS/"+\
                                   "HALO_Aircraft_"+flight+".csv")
                pos_df.index=pos_df["Unnamed: 0"]
                del pos_df["Unnamed: 0"]
            return pos_df,campaign_path        
    
    def interpolate_flight_from_waypoints(self,wp_df,resolution="1s"):
        """
        interpolate between the way points in time to simulate a continous 
        flight dataset in the resolution desired 

        Parameters
        ----------
        wp_df : pd.DataFrame
            Dataframe containing the waypoints (lat/lon) 
            specified with timestamps.

        Returns
        -------
        flight_df : pd. DataFrame
            Dataframe with the continuous flight data (lat/lon) 
            in desired resolution 

        """
        # Troubleshooting
        if not set(['lat', 'lon']).issubset(set([col.lower() \
                                                 for col in wp_df.columns])):
            raise NameError("Positional data is not in dataframe",
                            " or at least not given as lat/lon")
        try:
            wp_df.index=pd.DatetimeIndex(wp_df.index)
        except:
            raise TypeError("The index of input df cannot be provided as",
                            "pd.DatetimeIndex")
        #finally:
        #    print("func will be leaved")
        #    return None
        # Interpolate
        flight_df=pd.DataFrame(data=np.nan,
                               index=pd.date_range(start=wp_df.index[0],
                                                   end=wp_df.index[-1],
                                                   freq="1s"),
                               columns=wp_df.columns)
        flight_df.loc[wp_df.index,:]=wp_df.values
        flight_df=flight_df[["Lat","Lon","Flightlevel",
                             "Pressure (hPa)",
                             "Cum. dist. (km)"]]
        flight_df["Groundspeed"]=flight_df["Cum. dist. (km)"].diff()
        print("Interpolate waypoints in time")
        flight_df = flight_df.interpolate(method="time")
        flight_df = flight_df.reset_index().drop_duplicates(
                subset='index', keep='last').set_index('index')
        return flight_df
    
    def get_aircraft_position(self,ar_of_day,waypoints_existent=False):
        # Aircraft Position
        flight=self.flights_of_interest[0]
        
        if waypoints_existent:
            wp_df=self.get_aircraft_waypoints()
            flight_df=self.interpolate_flight_from_waypoints(wp_df)
        
        else:
            # Use the flight track
            from flight_track_creator import Flighttracker
            Tracker=Flighttracker(self,flight,ar_of_day,
                                  shifted_lat=0,shifted_lon=-12,
                                  track_type="internal")
            
            flight_df,cmpgn_path=Tracker.run_flight_track_creator()
        
        return flight_df
    
    #def create_synthetic_flight(self,ar_of_day):
    #    import flight_track_creator
    #    Flight_Tracker=flight_track_creator.Flighttracker(
    #                        self,flight=self.interested_flight[0],
    #                        shifted_lat=65,shifted_lon=-10,
    #                        track_type="internal")   
    # 
    #    track_df,cmpgn_path=Tracker.run_flight_track_creator()
                
    
#%%
# Synthetic Campaign Class HALO_AC3_Dry
class HALO_AC3_Dry_Run(Synthetic_Campaign):
    def __init__(self,is_flight_campaign,major_path,aircraft,
                 instruments,is_synthetic_campaign,interested_flights=[]):
            super().__init__(is_flight_campaign,major_path,aircraft,
                             instruments,is_synthetic_campaign,
                             interested_flights=[])
            self.name="HALO_AC3_Dry_Run"
            self.flight_day={"RF01":"23","RF02":"24","RF03":"25","RF04":"26"}
            self.flight_month={"RF01":"03","RF02":"03","RF03":"03","RF04":"03"}
            self.year="2021"
            self.campaign_path=self.major_path+self.name+"/"
            self.campaign_data_path=self.campaign_path+"data/"
            #self.is_synthetic_campaign=True
            if not os.path.exists(self.campaign_path):
                os.makedirs(self.campaign_path)
                print("Path of Campaign ",self.name," is created under: ",
                      self.campaign_path)
            else:
                print("Overall directory of campaign work is: ",
                      self.campaign_path)
        
class North_Atlantic_February_Run(Synthetic_Campaign):
    # This is the AR case study from the North Atlantic Pathway    
    def __init__(self,is_flight_campaign,major_path,aircraft,
                 instruments,is_synthetic_campaign=True,interested_flights=[]):
            super().__init__(is_flight_campaign,major_path,aircraft,
                             instruments,is_synthetic_campaign,
                             interested_flights=[])
            self.name="NA_February_Run"
            self.campaign_name=self.name
            self.flights_of_interest=interested_flights
            #self.flights=["SRF01","SRF02","SRF03","SRF04","SRF05"]
            #self.dates=["20190329","20180224",
            #            "20190319","20190416",
            #            "20190420"]
    
            self.flight_day={"SRF01":"27","SRF02":"24","SRF03":"26",
                             "SRF04":"19","SRF05":"20","SRF06":"23",
                             "SRF07":"16","SRF08":"19"}
            self.flight_month={"SRF01":"04","SRF02":"02","SRF03":"02",
                               "SRF04":"03","SRF05":"04","SRF06":"04",
                               "SRF07":"04","SRF08":"04"}
            self.years={"SRF01":"2018","SRF02":"2018","SRF03":"2018",
                        "SRF04":"2019","SRF05":"2019","SRF06":"2016",
                        "SRF07":"2020","SRF08":"2020"}
            self.flights=self.flight_day.keys()
            self.campaign_path=self.major_path+self.name+"/"
            self.campaign_data_path=self.campaign_path+"data/"
            self.plot_path=self.campaign_path+"plots/"
            if hasattr(self,"years"):
                try:
                    self.year=self.years[self.flights_of_interest[0]]
                except:
                    self.year="2016"
            #self.is_synthetic_campaign=True
            if not os.path.exists(self.campaign_path):
                os.makedirs(self.campaign_path)
                print("Path of Campaign ",self.name," is created under: ",
                      self.campaign_path)
            else:
                print("Overall directory of campaign work is: ",
                      self.campaign_path)

class Second_Synthetic_Study(Synthetic_Campaign):
    # This is the AR case study from the North Atlantic Pathway    
    def __init__(self,is_flight_campaign,major_path,aircraft,
                 instruments,is_synthetic_campaign=True,interested_flights=[]):
            super().__init__(is_flight_campaign,major_path,aircraft,
                             instruments,is_synthetic_campaign,
                             interested_flights=[])
            self.name="Second_Synthetic_Study"
            self.campaign_name=self.name
            self.flights_of_interest=interested_flights
            #self.flights=["SRF01","SRF02","SRF03","SRF04","SRF05"]
            #self.dates=["20190329","20180224",
            #            "20190319","20190416",
            #            "20190420"]
            # New dates
            # SRF01 2011-03-15
            # SRF02 2011-03-17
            # SRF03 2011-04-23
            
            # SRF04 2012-03-03
            # SRF05 2012-04-25
            
            # SRF06 2014-03-25
            
            # SRF07 2015-03-07
            # SRF08 2015-03-14
            
            
            # SRF09 2016-03-11
            # SRF10 2016-03-12
            # SRF11 2016-04-28
    
            # SRF12 2018-02-25
    
            # SRF13 2020-04-13
            
            # SRF01 2011-03-15
        
            self.flight_day={"SRF01":"15","SRF02":"17","SRF03":"23",
                             "SRF04":"03","SRF05":"25","SRF06":"25",
                             "SRF07":"07","SRF08":"14","SRF09":"11",
                             "SRF10":"12","SRF11":"28","SRF12":"25",
                             "SRF13":"13"}
            
            self.flight_month={"SRF01":"03","SRF02":"03","SRF03":"04",
                               "SRF04":"03","SRF05":"04","SRF06":"03",
                               "SRF07":"03","SRF08":"03","SRF09":"03",
                               "SRF10":"03","SRF11":"04","SRF12":"02",
                               "SRF13":"04"}
            
            self.years={"SRF01":"2011","SRF02":"2011","SRF03":"2011",
                        "SRF04":"2012","SRF05":"2012","SRF06":"2014",
                        "SRF07":"2015","SRF08":"2015","SRF09":"2016",
                        "SRF10":"2016","SRF11":"2016","SRF12":"2018",
                        "SRF13":"2020"}
            self.flights=self.flight_day.keys()
            self.campaign_path=self.major_path+self.name+"/"
            self.campaign_data_path=self.campaign_path+"data/"
            self.plot_path=self.campaign_path+"plots/"
            if hasattr(self,"years"):
                try:
                    self.year=self.years[self.flights_of_interest[0]]
                except:
                    self.year="2016"
            #self.is_synthetic_campaign=True
            if not os.path.exists(self.campaign_path):
                os.makedirs(self.campaign_path)
                print("Path of Campaign ",self.name," is created under: ",
                      self.campaign_path)
            else:
                print("Overall directory of campaign work is: ",
                      self.campaign_path)
                
    #def load_aircraft_position_from_csv(self, campaign, file_name_begin)
###############################################################################


###############################################################################
#%% Main function
def main(campaign_name="NAWDEX",flights=["RF03"]):
    #Default flights for NAWDEX:
    campaign_name="HALO_AC3_Dry_Run"
    flights=["RF04"]#["RF01":"RF13"]
    path=os.getcwd()
    name="data_config_file"
    config_file_exists=False
        

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
        
        #["RF01","RFO2","RF07","RF10"])#elif campaign_name=="NARVAL-II":
        nawdex.specify_flights_of_interest(flights)
        nawdex.create_directory(directory_types=["data"])
        bahamas=nawdex.load_hamp_data("NAWDEX",flights,instrument="Halo",
                                      bahamas_desired=True)
        aircraft_position=nawdex.get_aircraft_position(flights,"NAWDEX")
        
        for flight in flights:
            bahamas["Position"].name="HALO_Aircraft_"+flight
            nawdex.dataframe_to_csv(bahamas["Position"],nawdex.sub_paths["data"])
    
    elif campaign_name=="HALO_AC3_Dry_Run":
        
        dry_run=HALO_AC3_Dry_Run(is_flight_campaign=False,
                     major_path=config_file["Data_Paths"]["campaign_path"],
                     aircraft="HALO",instruments=[])
        dry_run.specify_flights_of_interest(flights)
        halo_waypoints=dry_run.get_aircraft_waypoints(filetype=".csv")
        halo_position={}
        for flight in dry_run.interested_flights:
            halo_position[flight]=dry_run.interpolate_flight_from_waypoints(\
                                                        halo_waypoints[flight])
            #--> to test
            #plt.scatter(halo_position[flight]["Lon"],
            #             halo_position[flight]["Lat"])
            coords_station=[]
            #Load Flight map class
            from Flight_Mapping import FlightMaps 
            
            Flightmap=FlightMaps(dry_run.major_path,dry_run.campaign_path,
                         dry_run.aircraft,dry_run.instruments,
                         dry_run.interested_flights,analysing_campaign=False)
            Flightmap.plot_flight_map_era(dry_run,coords_station,
                          flight,["IVT"],show_AR_detection=False,
                          show_supersites=False)
      
    else:
        pass
if __name__=="__main__":
    main()     
