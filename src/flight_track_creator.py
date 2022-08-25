# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 12:31:51 2021

@author: u300737
"""
import math
import numpy as np
import os
import pandas as pd 

import geopy
from   geopy.distance import geodesic

class Flighttracker():
    def __init__(self,cmpgn_cls,flight,ar,track_type="linear",shifted_lat=0,
                 shifted_lon=0,load_save_instantan=False):
        self.ar=ar
        self.flight=flight
        self.cmpgn_cls=cmpgn_cls
        self.shifted_lat=shifted_lat
        self.shifted_lon=shifted_lon,
        self.track_type=track_type
        self.ar_legs=["inflow","internal","outflow"]
        
        self.aircraft_data_path=self.cmpgn_cls.campaign_path+"/data/aircraft_position/"
        self.load_save_instantan=load_save_instantan
        if not os.path.exists(self.aircraft_data_path):
            os.makedirs(self.aircraft_data_path)
        

    def get_bearing(self,lat1, long1, lat2, long2):
        dLon = (long2 - long1)
        x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
        y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
            math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
                math.cos(math.radians(dLon))
        brng = np.arctan2(x,y)
        brng = np.degrees(brng)

        return brng

    def look_up_flight_track(self):
        self.flight_track_cfg={}

            ###################################################################
        if self.ar=="SAR1":
            if self.flight=="RF10":
        
                #flight_track_cfg["start_lat"]=56
                self.flight_track_cfg["start_lat"]=56
                self.flight_track_cfg["start_lon"]=-17.0
                self.flight_track_cfg["bearing"]=6
                self.flight_track_cfg["groundspeed"]=500
                self.flight_track_cfg["start_date"]="2016-10-13 14:00"
                self.flight_track_cfg["end_date"]="2016-10-13 15:40"
                self.flight_track_cfg["turn"]="2016-10-13 14:30"
                self.flight_track_cfg["before_turn_bearing"]=347
                self.flight_track_cfg["after_turn_bearing"]=8
                #flight_track_cfg["start_date"]="2016-10-13 14:30"
                #flight_track_cfg["end_date"]="2016-10-13 14:35"
                self.flight_track_cfg["altitude"]=10000
            ###################################################################
        if self.ar.startswith("SAR3_"):
            if self.flight=="RF10":    
                halo_df,campaign_path=self.cmpgn_cls.load_aircraft_position(
                                            self.cmpgn_cls.name)
                import AR
                ARs=AR.Atmospheric_Rivers.look_up_synthetic_AR_cross_sections(
                        campaign="NAWDEX")
                halo_df=halo_df.loc[ARs[self.flight]["AR3"]["start"]:\
                               ARs[self.flight]["AR3"]["end"]]
                if self.shifted_lat>=1.5:
                    halo_df=halo_df.iloc[0:3600,:]
                    self.flight_track_cfg["bearing"]=290
                    self.flight_track_cfg["groundspeed"]=300
                    self.flight_track_cfg["start_date"]=str(halo_df.index[0])
                    self.flight_track_cfg["end_date"]=str(halo_df.index[-1])
                    self.flight_track_cfg["altitude"]=10000
                elif self.shifted_lat==-7:
                    halo_df=halo_df.iloc[0:4500,:]
                    self.flight_track_cfg["bearing"]=280
                    self.flight_track_cfg["groundspeed"]=300
                    self.flight_track_cfg["start_date"]=str(halo_df.index[0])
                    self.flight_track_cfg["end_date"]=str(halo_df.index[-1])
                    self.flight_track_cfg["altitude"]=10000
                    #self.track_type="linear"
                self.flight_track_cfg["halo_df"]=halo_df    
            ###################################################################
        elif self.ar.startswith("SAR_internal"):
            if self.cmpgn_cls.name=="NAWDEX":
                if self.flight=="RF10":
                    import AR
                    ARs=AR.Atmospheric_Rivers.look_up_synthetic_AR_cross_sections(
                        campaign="NAWDEX")
                    halo_df,campaign_path=self.cmpgn_cls.load_aircraft_position()
                
                    halo_df=halo_df.loc["2016-10-13 12:20":"2016-10-13 13:45"]
                    halo_df["latitude"]=halo_df["latitude"]#-0.9
                    shifted_halo=halo_df.copy()
                    shifted_halo["latitude"]=shifted_halo["latitude"]+4.0
                    shifted_halo["longitude"]=shifted_halo["longitude"]+4.5
            
            if self.cmpgn_cls.name=="NA_February_Run":
                if (self.flight=="SRF06") or (self.flight=="SRF02")\
                or (self.flight=="SRF03") or (self.flight=="SRF04")\
                    or (self.flight=="SRF05") or (self.flight=="SRF07") or \
                        (self.flight=="SRF08"):
                
                    self.flight_track_cfg["bearing"]=275
                    self.flight_track_cfg["groundspeed"]=250
                    if self.flight=="SRF01":
                        self.flight_track_cfg["start_date"]="2018-02-24 13:00"
                        self.flight_track_cfg["end_date"]="2018-02-24 13:45"
                    
                    elif self.flight=="SRF02":
                        self.flight_track_cfg["start_date"]="2018-02-24 14:35"
                        self.flight_track_cfg["end_date"]="2018-02-24 15:25"
                    elif self.flight=="SRF03":
                        if self.ar.endswith("l"):
                            self.flight_track_cfg["start_date"]="2018-02-26 06:15"
                            self.flight_track_cfg["end_date"]="2018-02-26 07:10"
                        else:
                            self.flight_track_cfg["start_date"]="2018-02-26 09:00"
                            self.flight_track_cfg["end_date"]="2018-02-26 09:45"
                            
                    elif self.flight=="SRF04":
                        self.flight_track_cfg["start_date"]="2019-03-19 16:30"
                        self.flight_track_cfg["end_date"]="2019-03-19 17:20"
                    elif self.flight=="SRF05":
                        self.flight_track_cfg["start_date"]="2019-04-20 09:30"
                        self.flight_track_cfg["end_date"]="2019-04-20 10:15"
                    elif self.flight=="SRF06":
                        self.flight_track_cfg["start_date"]="2016-04-23 14:00"
                        self.flight_track_cfg["end_date"]="2016-04-23 15:05"
                    elif self.flight=="SRF07":
                        self.flight_track_cfg["start_date"]="2020-04-16 06:05"
                        self.flight_track_cfg["end_date"]="2020-04-16 06:50"
                        time_minutes=50
                    elif self.flight=="SRF08":
                        self.flight_track_cfg["start_date"]="2020-04-19 05:00"
                        self.flight_track_cfg["end_date"]="2020-04-19 05:52"
                    #elif self.flight=="SRF08":
                    #    self.flight_track_cfg["start_date"]="2020-04-19 05:00"
                    #    self.flight_track_cfg["end_date"]="2020-04-19 05:15"
                        
                    self.flight_track_cfg["altitude"]=10000
                    
                    if self.flight=="SRF02":
                        self.flight_track_cfg["start_lat"]=68.25
                        self.flight_track_cfg["start_lon"]=-5.
                        self.flight_track_cfg["bearing"]=266
                    elif self.flight=="SRF03":
                        if self.ar.endswith("internal"):
                            self.flight_track_cfg["start_lat"]=59
                            self.flight_track_cfg["start_lon"]=-16
                            self.flight_track_cfg["bearing"]=270
                        elif self.ar.endswith("internal2"):
                            self.flight_track_cfg["start_lat"]=66
                            self.flight_track_cfg["start_lon"]=-12
                            
                            self.flight_track_cfg["bearing"]=310
                    elif self.flight=="SRF04":
                        self.flight_track_cfg["start_lat"]=74.25
                        self.flight_track_cfg["start_lon"]=15
                        self.flight_track_cfg["bearing"]=275
                    elif self.flight=="SRF05":    
                        self.flight_track_cfg["start_lat"]=67.5
                        self.flight_track_cfg["start_lon"]=13  
                        self.flight_track_cfg["bearing"]=285
                    elif self.flight=="SRF06":
                        self.flight_track_cfg["start_lat"]=74.0
                        self.flight_track_cfg["start_lon"]=90
                        self.flight_track_cfg["bearing"]=275
                    elif self.flight=="SRF07":
                        self.flight_track_cfg["start_lat"]=81.5
                        self.flight_track_cfg["start_lon"]=54.5
                        self.flight_track_cfg["bearing"]=282
                    elif self.flight=="SRF08":
                        self.flight_track_cfg["start_lat"]=73.75
                        self.flight_track_cfg["start_lon"]=11
                        self.flight_track_cfg["bearing"]=295
                    
                    
                    halo_df,cmpgn_path=self.create_synthetic_linear_flight_track()
                    shifted_halo=halo_df.copy()
                    shifted_halo["latitude"]=shifted_halo["latitude"]+4.0
                
                    if self.flight=="SRF02":
                        shifted_halo["longitude"]=shifted_halo["longitude"]+1.5
                        shifted_halo["latitude"]=shifted_halo["latitude"]-1.65
                       
                    if self.flight=="SRF03":
                        shifted_halo["latitude"]=shifted_halo["latitude"]-1.0
                        if self.ar.endswith("l2"):
                            shifted_halo["latitude"]=shifted_halo["latitude"]-1.5
                    
                            shifted_halo["longitude"]=shifted_halo["longitude"]+5
                    
                    if self.flight=="SRF04":
                        shifted_halo["latitude"]=shifted_halo["latitude"]-1.75
                        shifted_halo["longitude"]=shifted_halo["longitude"]-1.75
                    
                    if self.flight=="SRF05":
                        shifted_halo["longitude"]=shifted_halo["longitude"]+4
                        shifted_halo["latitude"]=shifted_halo["latitude"]-1.5
                    
                    if self.flight=="SRF06":
                        shifted_halo["latitude"]=shifted_halo["latitude"]+0.8
                        shifted_halo["longitude"]=shifted_halo["longitude"]+6.5
                    
                    if self.flight=="SRF07":
                        # create extended shifted flight track
                        shifted_halo["latitude"]=shifted_halo["latitude"]-1.5
                        shifted_halo["longitude"]=shifted_halo["longitude"]+4.75
            
                    if self.flight=="SRF08":
                        shifted_halo["latitude"]=shifted_halo["latitude"]-1.5
                        shifted_halo["longitude"]=shifted_halo["longitude"]+2.5
            
                    self.flight_track_cfg["start_lat"]=float(shifted_halo["latitude"][0])
                    self.flight_track_cfg["start_lon"]=float(shifted_halo["longitude"][0])
                        
                    temporary_halo_df,cmpgn_path=self.create_synthetic_linear_flight_track()
                    shifted_halo=temporary_halo_df.copy()
                    
                    #if self.flight=="SRF07":
                    
            if self.cmpgn_cls.name=="Second_Synthetic_Study":
                self.flight_track_cfg["start_lat"]=73.5
                self.flight_track_cfg["start_lon"]=12
                self.flight_track_cfg["bearing"]=280
                self.flight_track_cfg["groundspeed"]=250
                self.flight_track_cfg["altitude"]=10000
                    
                date=self.cmpgn_cls.years[self.flight]+"-"+\
                     self.cmpgn_cls.flight_month[self.flight]+"-"+\
                     self.cmpgn_cls.flight_day[self.flight]
                self.flight_track_cfg["start_date"]=date+" 10:00"
                self.flight_track_cfg["end_date"]=date+" 11:00"
                
                if self.flight=="SRF01":
                   self.flight_track_cfg["bearing"]=282
                   self.flight_track_cfg["start_lat"]=73
                   self.flight_track_cfg["start_lon"]=10
                   self.flight_track_cfg["start_date"]=date+" 16:30"
                   self.flight_track_cfg["end_date"]=date+" 17:20"
                if self.flight=="SRF02":
                   self.flight_track_cfg["bearing"]=280
                   self.flight_track_cfg["start_lat"]=73.25
                   self.flight_track_cfg["start_lon"]=25
                   self.flight_track_cfg["start_date"]=date+" 11:30"
                   self.flight_track_cfg["end_date"]=date+" 12:30"
                if self.flight=="SRF03":
                   self.flight_track_cfg["bearing"]=283
                   self.flight_track_cfg["start_lat"]=75.5
                   self.flight_track_cfg["start_lon"]=12.5
                   #self.flight_track_cfg["start_date"]=date+" 16:30"
                   #self.flight_track_cfg["end_date"]=date+" 16:45"
                   self.flight_track_cfg["start_date"]=date+" 16:05"
                   self.flight_track_cfg["end_date"]=date+" 16:57"
                if self.flight=="SRF06":
                   self.flight_track_cfg["bearing"]=270
                   self.flight_track_cfg["start_lat"]=70.5
                   self.flight_track_cfg["start_lon"]=-2
                   self.flight_track_cfg["start_date"]=date+" 14:30"
                   self.flight_track_cfg["end_date"]=date+" 15:20"
                if self.flight=="SRF07":
                   self.flight_track_cfg["bearing"]=260
                   self.flight_track_cfg["start_lat"]=74.0
                   self.flight_track_cfg["start_lon"]=35
                   self.flight_track_cfg["start_date"]=date+" 13:30"
                   self.flight_track_cfg["end_date"]=date+" 14:20"
                
                if self.flight=="SRF08":
                   self.flight_track_cfg["bearing"]=280
                   self.flight_track_cfg["start_lat"]=71.5
                   self.flight_track_cfg["start_lon"]=4
                   self.flight_track_cfg["start_date"]=date+" 16:30"
                   self.flight_track_cfg["end_date"]=date+" 17:25"
                if self.flight=="SRF09":
                   self.flight_track_cfg["bearing"]=270
                   self.flight_track_cfg["start_lat"]=80.75
                   self.flight_track_cfg["start_lon"]=19
                   self.flight_track_cfg["start_date"]=date+" 14:25"
                   self.flight_track_cfg["end_date"]=date+" 15:10"
                if self.flight=="SRF12":
                   self.flight_track_cfg["start_lat"]=80.25
                   self.flight_track_cfg["start_lon"]=16.25
                   self.flight_track_cfg["bearing"]=285
                   self.flight_track_cfg["groundspeed"]=250
                   self.flight_track_cfg["altitude"]=10000
                   self.flight_track_cfg["start_date"]=date+" 07:05"
                   self.flight_track_cfg["end_date"]=date+" 07:44"
                
                    
                
                halo_df,cmpgn_path=self.create_synthetic_linear_flight_track()
                shifted_halo=halo_df.copy()
                shifted_halo["latitude"]=shifted_halo["latitude"]+2.5
                
                #if self.flight=="SRF12":
                    # create extended shifted flight track
                    #self.flight_track_cfg["end_date"]=date+" 08:00"
                #    temporary_halo_df,cmpgn_path=self.create_synthetic_linear_flight_track()
                    #self.flight_track_cfg["end_date"]=date+" 07:45"
                #    shifted_halo=temporary_halo_df.copy()
                #else:
                #if not self.flight=="SRF01":
                if self.flight=="SRF02":
                    shifted_halo["longitude"]=shifted_halo["longitude"]+4
                if self.flight=="SRF03":
                    self.flight_track_cfg["end_date"]=date+" 16:52"
                if self.flight=="SRF08":
                    self.flight_track_cfg["end_date"]=date+" 17:15"
                    shifted_halo["latitude"]=shifted_halo["latitude"]-0.1
                if self.flight=="SRF09":
                    shifted_halo["longitude"]=shifted_halo["longitude"]+5.5
                    shifted_halo["latitude"]=shifted_halo["latitude"]-0.5
                    self.flight_track_cfg["end_date"]=date+" 15:04"
                if self.flight=="SRF12":
                    shifted_halo["longitude"]=shifted_halo["longitude"]+7.5
                
                self.flight_track_cfg["start_lat"]=float(shifted_halo["latitude"][0])
                self.flight_track_cfg["start_lon"]=float(shifted_halo["longitude"][0])
                        
                temporary_halo_df,cmpgn_path=self.create_synthetic_linear_flight_track()
                shifted_halo=temporary_halo_df.copy()
                        
            lat1=float(halo_df["latitude"][-1])
            lon1=float(halo_df["longitude"][-1])
            lat2=float(shifted_halo["latitude"][0])
            lon2=float(shifted_halo["longitude"][0])
            self.flight_track_cfg["bearing"]=self.get_bearing(lat1, lon1,
                                                              lat2, lon2)
            # Define origin coordinates as geopy.Point
            origin = geopy.Point(lat1, lon1)
            distance_leg=geodesic((lat2,lon2), 
                                  (lat1,lon1)).km
            
            # Assuming 300 m/s, then every distance step is
            #track_df=pd.DataFrame()
            
            print("Create flight track")
            cumulated_distance=0
            track_df_latitude=[]
            track_df_longitude=[]
            b=self.flight_track_cfg["bearing"]
            
            # Assuming 250 m/s
            self.flight_track_cfg["groundspeed"]=250
            while cumulated_distance < distance_leg:    
                cumulated_distance=cumulated_distance+\
                    self.flight_track_cfg["groundspeed"]/1000
                destination_internal = geodesic(kilometers=cumulated_distance).\
                                    destination(origin, b)
                track_df_latitude.append(destination_internal.latitude)
                track_df_longitude.append(destination_internal.longitude)
            
            second_integer_array=np.arange(len(track_df_latitude))+1
            internal_time_index=pd.DatetimeIndex(pd.DatetimeIndex(
                                   halo_df["longitude"].index)[-1] +\
                                       second_integer_array *\
                                           pd.offsets.Second())
            internal_track_df=pd.DataFrame(
                                columns=["altitude","longitude","latitude"],
                                index=internal_time_index)
            internal_track_df["latitude"]   = track_df_latitude
            internal_track_df["longitude"]  = track_df_longitude
            internal_track_df["groundspeed"]    = 250
            internal_track_df["altitude"]   = 10000
            # Add Time shift to end section for continuous measurements
            shifted_halo.index=pd.DatetimeIndex(shifted_halo.index)
            time_diff_end=internal_track_df.index[-1]-shifted_halo.index[0]
            shifted_halo.index=shifted_halo.index+time_diff_end
            self.aircraft_dict={}
            self.aircraft_dict["inflow"]=halo_df
            self.aircraft_dict["internal"]=internal_track_df
            self.aircraft_dict["outflow"]=shifted_halo
            return self.aircraft_dict
    def concat_track_dict_to_df(self,merge_all=True,pick_legs=[]):
        """
        Parameters
        ----------
        halo_dict : dict
            this is segmented dict of flight legs having specific names.
        merge_all : boolean
            specifies if entire dict should be concat. Default is True
        pick_legs : list
            list of dict keys (flight legs) to be merged. Is passed if merge_all
            is True. 
        Returns
        -------
        halo_df : pd.DataFrame
            concatted dataframe
        leg_times_df : pd.DataFrame
            defining time periods for all legs

        """
        
        leg_list=[]
        leg_times_df=pd.DataFrame(columns=["start_time","end_time"])
        if merge_all:
            try:
                pick_list=self.leg_dict.keys()
            except:
                pick_list=self.ar_legs
        else:
            pick_list=pick_legs
        for key in pick_list:
            self.aircraft_dict[key]["leg_type"]=key
            leg_list.append(self.aircraft_dict[key])
            leg_times_df.loc[key,:]=[self.aircraft_dict[key].index[0],
                                 self.aircraft_dict[key].index[-1]]
        
        self.aircraft_df=pd.concat(leg_list)
        self.aircraft_df=self.aircraft_df.loc[\
                        self.aircraft_df.index.drop_duplicates(keep="first")]
        return self.aircraft_df,leg_times_df
    
    def save_created_flight_track_as_csv(self):
        # Merge self.leg_dict to dataframe with column defining leg_type
        if not hasattr(self,"aircraft_df"):
            self.aircraft_df,leg_times_df=self.concat_track_dict_to_df(merge_all=True,
                                                              pick_legs=[])
        # Save aircraft_df as csv 
        self.flight_name=self.flight
        if self.load_save_instantan:
            self.flight_name=self.flight_name+"_instantan"
        file_name=self.flight_name+"_aircraft_position.csv"
        
        self.aircraft_df.to_csv(path_or_buf=self.aircraft_data_path+file_name,index=True)
        print("FLight position saved as:",self.aircraft_data_path+file_name)
    def load_existing_flight_track(self):
        self.flight_name=self.flight
        if self.load_save_instantan:
            self.flight_name=self.flight_name+"_instantan"
        file_name=self.flight_name+"_aircraft_position.csv"
        self.aircraft_df=pd.read_csv(self.aircraft_data_path+file_name)
        self.aircraft_df.index=pd.DatetimeIndex(self.aircraft_df["Unnamed: 0"])
        campaign_path=None
        #aircraft_dict={}
        #for leg in ar_legs:
        #    aircraft_dict[leg]=aircraft_df[aircraft_df["leg_type"]==leg]
        return self.aircraft_df,campaign_path#,aircraft_dict

    def make_flight_instantaneous(self):
         self.flight_name=self.flight+"_instantan"
         if not hasattr(self,"aircraft_dict"):
             self.make_dict_from_aircraft_df()
         
         srf_date=self.aircraft_dict["internal"].index.date[0]
         central_hour=self.aircraft_dict["internal"].index.hour[
                     int(len(self.aircraft_dict["internal"].index)/2)]
         central_timestamp=pd.to_datetime(central_hour*3600,unit="s",
                                                  origin=srf_date)
         s=0
         for ar_sector in self.ar_legs:
             if s==0:
                 start_time=central_timestamp
             else:
                 start_time=self.aircraft_dict[[\
                                    *self.aircraft_dict.keys()][s-1]].index[-1]                    
             end_time=start_time+pd.Timedelta(5,unit="min")
                     
             self.aircraft_dict[ar_sector]["old_index"]=\
                             self.aircraft_dict[ar_sector].index.values    
             self.aircraft_dict[ar_sector].index=pd.date_range(start_time,end_time,
                                periods=self.aircraft_dict[ar_sector].shape[0])
             s+=1
         self.concat_track_dict_to_df()
         return self.aircraft_dict,self.aircraft_df
         
    def check_if_synthetic_tracks_already_exist(self):
    #    # Save aircraft_df as csv 
        file_exists=False    
        file_name=self.flight+"_aircraft_position.csv"
        if os.path.exists(self.aircraft_data_path+file_name):
            file_exists=True
        return file_exists
        
    def get_synthetic_flight_track(self,as_dict=False):
        #self.check_if_synthetic_tracks_already_exist()
        self.flight_name=self.flight
        if self.load_save_instantan:
            self.flight_name=self.flight_name+"_instantan"
        file_name=self.flight_name+"_aircraft_position.csv"
        if os.path.exists(self.aircraft_data_path+file_name):
            aircraft_df,campaign_path=self.load_existing_flight_track()
        else:
            # this one is always not instantaneous
            aircraft_df,campaign_path=self.run_flight_track_creator()
        
            if self.load_save_instantan:
                self.make_flight_instantaneous()
                print("Synthetic flight track was done instantaneous")    
                #    halo_df,time_legs_df=Flight_Tracker.concat_track_dict_to_df(
                #                                merge_all=merge_all_legs,
                #                                pick_legs=pick_legs)
            self.save_created_flight_track_as_csv()
        if as_dict:
            aircraft_df=self.make_dict_from_aircraft_df()
        return aircraft_df,campaign_path 
    
    def get_all_synthetic_flights(self,cmpgn_cls_dict):
        """
        

        Parameters
        ----------
        cmpgn_cls_dict : dictionary
            Specification of campaign cls (dict.values) for all given dates
            (dict.keys).

        Returns 
        -------
        flight_dict : dictionary
            Dictionary with all flight tracks and sorted along date

        """
        flight_dict={}
        for date in cmpgn_cls_dict.keys():
            self.cmpgn_cls=cmpgn_cls_dict[date][0]
            self.flight=cmpgn_cls_dict[date][1]
            self.aircraft_data_path=self.cmpgn_cls.campaign_path+\
                "/data/aircraft_position/"
            flight_dict[date],_temp=self.get_synthetic_flight_track()
        return flight_dict
    def make_dict_from_aircraft_df(self):
        
        self.aircraft_dict={}
        for leg in self.ar_legs:
            self.aircraft_dict[leg]=self.aircraft_df[\
                                        self.aircraft_df["leg_type"]==leg]
        return self.aircraft_dict
    
    def shift_existing_flight_track(self):
        campaign_path=os.getcwd()+"/NAWDEX/"
        #self.look_up_flight_track()
        if "halo_df" in self.flight_track_cfg.keys():
            track_df=self.flight_track_cfg["halo_df"]
            #if not 0.0<self.shifted_lat>=1.5:
            #    track_df["latitude"]=track_df["latitude"]+self.shifted_lat
            #    track_df["longitude"]=track_df["longitude"]+self.shifted_lon
            #else:
            self.flight_track_cfg["start_lat"]=self.flight_track_cfg[\
                                                    "halo_df"]["latitude"]\
                                                    .iloc[0]+self.shifted_lat
            self.flight_track_cfg["start_lon"]=self.flight_track_cfg[\
                                                    "halo_df"]["longitude"]\
                                                    .iloc[0]+self.shifted_lon
            track_df,campaign_path=self.create_synthetic_linear_flight_track()
            self.track_df=track_df
            self.campaign_path=campaign_path
            return track_df,campaign_path
        # given: lat1, lon1, b = bearing in degrees, d = distance in kilometers
    
    def create_synthetic_linear_flight_track(self):
        import Performance
        performance=Performance.performance()
        start_lat=self.flight_track_cfg["start_lat"]
        start_lon=self.flight_track_cfg["start_lon"]
        b=self.flight_track_cfg["bearing"]    
        groundspeed=self.flight_track_cfg["groundspeed"]
        start_date=self.flight_track_cfg["start_date"]
        end_date=self.flight_track_cfg["end_date"]
        altitude=self.flight_track_cfg["altitude"]
        # Define origin coordinates as geopy.Point
        origin = geopy.Point(start_lat, start_lon)
            
        time_idx=pd.date_range(start_date,end_date,freq="s")

        track_df=pd.DataFrame(data=groundspeed*np.arange(time_idx.shape[0]),
             columns=["distance"],index=time_idx)
        track_df["latitude"]=np.nan
        track_df["longitude"]=np.nan
        track_df["altitude"]=altitude
        track_df["groundspeed"]=groundspeed
        d=0
        print("Create flight track")
        for dist in track_df["distance"][:]:
            destination = geodesic(kilometers=dist/1000).\
                                        destination(origin, b)
            track_df["latitude"].iloc[d]=destination.latitude
            track_df["longitude"].iloc[d]=destination.longitude
            performance.updt(track_df.shape[0],d)
            d+=1
        
        campaign_path=os.getcwd()+"/NAWDEX/"
        
        self.campaign_path=campaign_path
        self.track_df=track_df
        return track_df,campaign_path

    def create_combined_linear_flight_track(self):
        first_flight_track_cfg=self.flight_track_cfg.copy()
        first_flight_track_cfg["bearing"]=self.flight_track_cfg["before_turn_bearing"]
        first_flight_track_cfg["end_date"]=self.flight_track_cfg["turn"]
        track_before_turn_df,cmpgn_path=self.create_synthetic_linear_flight_track(
                                                            first_flight_track_cfg)
        second_flight_track_cfg=self.flight_track_cfg.copy()
        second_flight_track_cfg["start_date"]=str(track_before_turn_df.index[-1])
        second_flight_track_cfg["bearing"]=self.flight_track_cfg["after_turn_bearing"]
        second_flight_track_cfg["start_lon"]=track_before_turn_df["longitude"].iloc[-1]
        second_flight_track_cfg["start_lat"]=track_before_turn_df["latitude"].iloc[-1]
        track_after_turn_df,cmpgn_path=self.create_synthetic_linear_flight_track(
            second_flight_track_cfg)
        track_df=pd.concat([track_before_turn_df,track_after_turn_df])
        self.track_df=track_df
        return track_df,cmpgn_path

    def run_flight_track_creator(self):
        if not self.ar.startswith("SAR_internal"):
            self.look_up_flight_track()
        else:
            self.aircraft_dict=self.look_up_flight_track()
            self.track_df=None
            cmpgn_path=None
            
        if self.track_type=="linear":
            track_df,cmpgn_path=self.create_synthetic_linear_flight_track()
        elif self.track_type=="combined":
            track_df,cmpgn_path=self.create_combined_linear_flight_track()
        elif self.track_type=="shifted":
            track_df,cmpgn_path=self.shift_existing_flight_track()
        elif self.track_type=="internal":
            return self.aircraft_dict,cmpgn_path
        else:
            raise Exception("Wrong track type chosen,",
                        "no synthetic flight track created")
        self.aircraft_df=track_df
        return track_df,cmpgn_path
    
    def plot_synthetic_flight_track(self):
        self.concat_track_dict_to_df()
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        map_fig=plt.figure(figsize=(12,12))
        if self.flight=="SRF06":
            ax = plt.axes(projection=ccrs.AzimuthalEquidistant(
                                    central_longitude=40.0,
                                    central_latitude=70))
            ax.coastlines(resolution="50m")
            ax.set_extent([-10,70,50,90])    
        else:
            ax = plt.axes(projection=ccrs.AzimuthalEquidistant(
                                    central_longitude=-10.0,
                                    central_latitude=70))
            ax.coastlines(resolution="50m")
            ax.set_extent([-40,30,55,90])
        
        if not self.track_df==None:    
            ax.scatter(self.track_df["longitude"],self.track_df["latitude"],
                       transform=ccrs.PlateCarree(),s=3,color="red")
        else:
            ax.plot(self.aircraft_dict["inflow"]["longitude"],
                       self.aircraft_dict["inflow"]["latitude"],
                       transform=ccrs.PlateCarree(),lw=3,color="red",
                       label="real flight track (inflow)")

            ax.plot(self.aircraft_dict["internal"]["longitude"],
                       self.aircraft_dict["internal"]["latitude"],
                       transform=ccrs.PlateCarree(),lw=3,color="black",
                       label="synthetic track (internal)")

            ax.plot(self.aircraft_dict["outflow"]["longitude"],
                       self.aircraft_dict["outflow"]["latitude"],
                       transform=ccrs.PlateCarree(),lw=2,ls="--",
                       color="salmon",label="synthetic track (outflow)")
            ax.legend(fontsize=15)
            
        ax.gridlines()
    
    #@staticmethod
    

def main(campaign="NA_February_Run",flight="SRF04",ar_of_day="SAR_internal",
         shifted_lat=0,shifted_lon=-12):

    import Flight_Campaign
    import data_config
    config_file=data_config.load_config_file(os.getcwd(),"data_config_file")

    if campaign=="NAWDEX":
        
        print("Analyse given flight: ",flight)
        cmpgn_cls=Flight_Campaign.NAWDEX(is_flight_campaign=True,
                major_path=config_file["Data_Paths"]["campaign_path"],
                aircraft="HALO",instruments=["radar","radiometer","sonde"])
        
        cmpgn_cls.specify_flights_of_interest(flight)
        cmpgn_cls.create_directory(directory_types=["data"])
    
    elif campaign=="NA_February_Run":
        
        print("Analyse given flight: ",flight)
        cmpgn_cls=Flight_Campaign.North_Atlantic_February_Run(
                    interested_flights=[flight],
                        is_flight_campaign=True,
                        major_path=config_file["Data_Paths"]["campaign_path"],
                        aircraft="HALO",instruments=[])
        
        cmpgn_cls.specify_flights_of_interest(flight)
        cmpgn_cls.create_directory(directory_types=["data"])
    elif campaign=="Second_Synthetic_Study":
        cmpgn_cls=Flight_Campaign.Second_Synthetic_Study(
                    interested_flights=[flight],
                        is_flight_campaign=True,
                        major_path=config_file["Data_Paths"]["campaign_path"],
                        aircraft="HALO",instruments=[])
        
        cmpgn_cls.specify_flights_of_interest(flight)
        cmpgn_cls.create_directory(directory_types=["data"])
    
    if not shifted_lat==0:
        ar_of_day=ar_of_day+"_"+str(shifted_lat)
        track_type="shifted"
    else:
        track_type="internal"
    Tracker=Flighttracker(cmpgn_cls,flight,ar_of_day,
                          shifted_lat=shifted_lat,
                          shifted_lon=shifted_lon,
                          track_type=track_type)   
    track_df,cmpgn_path=Tracker.run_flight_track_creator()
    Tracker.plot_synthetic_flight_track()        
        
if __name__=="__main__":
    #main() # used for NA_February_Run default pattern
    #main(campaign="NA_February_Run",flight="SRF07")
    main(campaign="Second_Synthetic_Study",flight="SRF08")