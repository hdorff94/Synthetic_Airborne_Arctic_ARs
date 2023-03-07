# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 13:39:46 2021

@author: u300737
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

class ICON_NWP():
    def __init__(self,start_hour,resolution,for_flight_campaign=True,
                 upsample_time="10min",
                 campaign="NAWDEX",flight="RF02",research_flights=None,
                 icon_path=os.getcwd(),synthetic_flight=False):
        prefix='C:\\Users\\u300737\\Desktop\\PhD_UHH_WIMI\\Work\\GIT_Repository/'
        
        self.for_flight_campaign=for_flight_campaign
        self.major_data_path=prefix+campaign+"/data/ICON_LEM_2km/"
        self.data_path=icon_path
        self.resolution=resolution
        if resolution>1000:
            self.simulation_type="NWP"
        elif resolution<1000:
            self.simulation_type="LES"
        if for_flight_campaign:
            self.campaign_name=campaign
            self.flight=flight
        self.upsample_time=upsample_time
        self.start_hour=start_hour
        self.minutes_time=['00:00', '00:10', '00:20',
                         '00:30', '00:40', '00:50']
        self.synthetic_flight=synthetic_flight
    
    def load_icon_data(self,file_name,extra_path=""):
        """

        Parameters
        ----------
        file_name : str
            ICON-Simulation file name.

        Returns
        -------
        ds : xarray dictionary
            ICON-Simulation Data.
        era_path : str
            path of the ICON-Simulation file.

        """
        ## Load ICON Dataset
        if extra_path=="":
            ds=xr.open_dataset(self.major_data_path+file_name)
        else:
            ds=xr.open_dataset(extra_path+file_name)
        return ds
    
    def adapt_icon_time_index(self,da,date,flights):
        """ 
        The ICON time coordinates are given in ordinal format, 
        this function changes the index into the typical format 
        %YYYY-%MM-%DD %HH:%MM (so rounded to minutes). This is required for the
        precedent interpolation in time to minutely frequencies.
        
        da : xr.DataArray
            ICON-Variable as DataArray (xarray).
        date : str
            str with 8 letters specifying the date, format: YYYYMMDD
        flights : list
            list of given flight
        Returns
        -------
        da : xr.DataArray
            ICON-Variable as DataArray (xarray) with updated Time Coordinates.
        """
        #Create index, fixed for 10-minute data
        if "-" in date:
            start_date=date[0:4]+date[5:7]+date[8:10]
        else:
            start_date=date#years[flight[0]]+months[flight[0]]+days[flight[0]]
        new_time_index=pd.to_datetime(abs(int(start_date)-np.array(da.time)),
                                  unit="d",origin=start_date)
        new_time_index=new_time_index.round("min")
        
        #Assign this index to the DataArray
        da=da.assign_coords(time=new_time_index)
        print("Index (Time Coordinate) of DataArray changed")
        return da
    def get_indexes_for_given_area(self,ds,lat_range,lon_range):
    
        """
        Find the index to consider for desired spatial domain
        
        Input
        -----
        ds        : xr.Dataset
            Icon Simulation Dataset
        lat_range : list
            list of shape two, including lower and upper latitude boundary
        lon_range : list
            list of shape two, including lower and upper longitude boundary
            """
    
        clon_s=pd.Series(np.rad2deg(ds.ncells.clon))
        clat_s=pd.Series(np.rad2deg(ds.ncells.clat))

        # Cut to defined lon domain
        clon_cutted=clon_s.loc[clon_s.between(lon_range[0],lon_range[1])]
        print(clat_s.shape)
        # adapt this to the lat domain
        clat_s=clat_s.loc[clon_cutted.index]
        print(clat_s.shape)
        # Cut to defined lat domain
        clat_cutted=clat_s.loc[clat_s.between(lat_range[0],lat_range[1])]
        # Finally cut lon to this array
        clon_cutted=clon_cutted.loc[clat_cutted.index]
        print(clon_cutted.shape,clat_cutted.shape)
        if not clon_cutted.index.all()==clat_cutted.index.all():
            raise Exception("The indexes are not equivalent so something",
                            " went wrong and no index list can be returned ")
        return clon_cutted.index
    
    def load_and_upsample_icon(self,var,extra_path=""):
        var="Hydrometeor"#
        #Open first hour
        print("Open ICON-Simulations Start Hour")
        i=0
        
        for icon_h in self.icon_var_list:
            padding = 2
            hour_arg=str(int(icon_h.start_hour)).zfill(padding)
            #hour_arg=str(int(start_hour)+i)
            dict_arg=hour_arg+"UTC"
            if not self.campaign_name=="HALO_AC3":
                file_name=var+"_ICON_"+self.flight+"_"+hour_arg+"UTC.nc"    
            else:
                file_name=var+"_ICON_"+self.flight+"_"+self.ar_of_day+"_"+\
                    hour_arg+"UTC.nc"
            print("Consider ICON Hour: "+dict_arg)
            
            icon_ds=icon_h.load_icon_data(file_name,extra_path=extra_path)
                
            
            
            temporary_iwv=icon_ds["tqv_dia"]
            temporary_iwp=icon_ds["tqi_dia"]*1000
            temporary_lwp=icon_ds["tqc_dia"]*1000
            
            # Interpolate the 10-min total columns to minutely data
            if not self.campaign_name=="HALO_AC3":
                temporary_iwv=icon_h.adapt_icon_time_index(temporary_iwv, 
                                                       self.date, self.flight)
                temporary_lwp=icon_h.adapt_icon_time_index(temporary_lwp, 
                                                       self.date, self.flight)
                temporary_iwp=icon_h.adapt_icon_time_index(temporary_iwp, 
                                                       self.date, self.flight)
            if i==0:
                iwv_icon=temporary_iwv#
                lwp_icon=temporary_lwp 
                iwp_icon=temporary_iwp 
            else:
                iwv_icon=xr.concat([iwv_icon,temporary_iwv],dim="time").\
                    drop_duplicates(dim="time")
                iwp_icon=xr.concat([iwp_icon,temporary_iwp],dim="time").\
                    drop_duplicates(dim="time")
                lwp_icon=xr.concat([lwp_icon,temporary_lwp],dim="time").\
                    drop_duplicates(dim="time")
                
            del temporary_iwv
            del temporary_lwp
            del temporary_iwp
            i+=1
            
        print("in order to better simulate the time variability")
        print("---this takes a while, as it is compute extensively----")
        icon_upsampled_hmp={}
        icon_upsampled_hmp["IWV"]=iwv_icon.resample(time=self.upsample_time).\
                                        interpolate("linear")
        icon_upsampled_hmp["LWP"]=lwp_icon.resample(time=self.upsample_time).\
                                        interpolate("linear")
        icon_upsampled_hmp["IWP"]=iwp_icon.resample(time=self.upsample_time).\
                                        interpolate("linear")
        self.icon_upsampled_hmp=icon_upsampled_hmp
    
    @staticmethod
    def lookup_ICON_AR_period_data(campaign,flight,ar_of_day,resolution,
                                   hydrometeor_icon_path,synthetic=False):
        if not synthetic:
            if campaign=="NAWDEX":
                if flight[0]=="RF10":
                    if ar_of_day=="AR3":
                        start_hour="12"
                        icon12=ICON_NWP(start_hour,resolution,
                                for_flight_campaign=True,campaign="NAWDEX",
                                research_flights=None,
                                icon_path=hydrometeor_icon_path)
                        icon13=ICON_NWP("13",resolution,
                                for_flight_campaign=True,campaign="NAWDEX",
                                research_flights=None,
                                icon_path=hydrometeor_icon_path)
                        icon14=ICON_NWP("14",resolution,for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon12,icon13,icon14]
                    elif ar_of_day=="AR99":
                        start_hour="12"
                        icon12=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon12]
                
                    elif ar_of_day=="AR31":
                        start_hour="12"
                        icon12=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon12]
                    elif ar_of_day=="AR1":
                        start_hour="09"
                        icon09=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon10=ICON_NWP("10",resolution,for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon09,icon10]
                    
                    elif ar_of_day=="AR2":     
                        start_hour="10"
                        icon10=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon11=ICON_NWP("11",resolution,for_flight_campaign=True,
                            campaign="NAWDEX",research_flights=None,
                            icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon10,icon11]
        
                elif flight[0]=="RF03":
                    if ar_of_day=="AR1":
                        start_hour="11"
                        icon11=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon12=ICON_NWP("12",resolution,for_flight_campaign=True,
                            campaign="NAWDEX",research_flights=None,
                            icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon11,icon12]
                    elif ar_of_day=="AR2":
                        start_hour="12"
                        icon12=ICON_NWP(start_hour,resolution,
                                        for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon13=ICON_NWP("13",resolution,for_flight_campaign=True,
                            campaign="NAWDEX",research_flights=None,
                            icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon12,icon13]
                    elif ar_of_day=="AR3":
                        start_hour="14"
                        icon14=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon15=ICON_NWP("15",resolution,for_flight_campaign=True,
                            campaign="NAWDEX",research_flights=None,
                            icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon14,icon15]
            
                    elif ar_of_day=="AR22":
                        start_hour="13"
                        icon13=ICON_NWP("13",resolution,for_flight_campaign=True,
                            campaign="NAWDEX",research_flights=None,
                            icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon13]
                    else:
                        raise Exception(
                                "This AR cross-section is not yet defined")
            elif campaign=="HALO_AC3":
                if flight[0]=="RF02":
                    if ar_of_day=="AR1":
                        start_hour="10"
                        icon10=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon11=ICON_NWP("11",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon12=ICON_NWP("12",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon10,icon11,icon12]

                    elif ar_of_day=="AR2":
                        start_hour="11"
                        icon11=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon12=ICON_NWP("12",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        
                        icon13=ICON_NWP("13",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon14=ICON_NWP("14",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        
                        icon_var_list=[icon11,icon12,icon13,icon14]
                if flight[0]=="RF03":
                        start_hour="10"
                        icon10=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon11=ICON_NWP("11",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon12=ICON_NWP("12",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon10,icon11,icon12]

                if flight[0]=="RF04":
                    start_hour="16"
                    icon16=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                    icon17=ICON_NWP("17",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                    icon_var_list=[icon16,icon17]
                if flight[0]=="RF05":
                    start_hour="10"
                    
                    if ar_of_day=="AR1":
                        icon10=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon11=ICON_NWP("11",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon10,icon11]
                    elif ar_of_day=="AR_entire_1":
                        icon10=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon11=ICON_NWP("11",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon12=ICON_NWP("12",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon13=ICON_NWP("13",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon10,icon11,icon12,icon13]
                    elif ar_of_day=="AR_entire_2":
                        start_hour="12"
                        icon12=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon13=ICON_NWP("13",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon14=ICON_NWP("14",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon15=ICON_NWP("15",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon12,icon13,icon14,icon15]
                        
                    elif ar_of_day=="AR2":    
                        icon11=ICON_NWP("11",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                   
                        icon12=ICON_NWP("12",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon11,icon12]
                    elif ar_of_day=="AR3":    
                        icon12=ICON_NWP("12",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                    
                        icon13=ICON_NWP("13",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon12,icon13]
                    elif ar_of_day=="AR4":
                        icon14=ICON_NWP("14",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                    
                        icon15=ICON_NWP("15",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        
                        icon_var_list=[icon14,icon15]
                if flight[0]=="RF06":
                    if ar_of_day=="AR1" or ar_of_day=="AR_entire_1":
                        start_hour="10"
                        icon10=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon11=ICON_NWP("11",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        
                        icon12=ICON_NWP("12",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon13=ICON_NWP("13",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon_var_list=[icon10,icon11,icon12,icon13]
                    if ar_of_day=="AR2" or ar_of_day=="AR_entire_2":
                        start_hour="12"
                        icon12=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        
                        icon13=ICON_NWP("13",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon14=ICON_NWP("14",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon15=ICON_NWP("15",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        
                        icon_var_list=[icon12,icon13,icon14,icon15]
                    
                if flight[0]=="RF07":
                    start_hour="14"
                    icon14=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                    icon15=ICON_NWP("15",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        
                    icon16=ICON_NWP("16",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        
                    icon_var_list=[icon14,icon15,icon16]
                if flight[0]=="RF08":
                    start_hour="09"
                    icon09=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                    icon10=ICON_NWP("10",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        
                    icon11=ICON_NWP("11",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                    icon_var_list=[icon09,icon10,icon11]
                if flight[0]=="RF16":
                    if ar_of_day=="AR1":
                        start_hour="10"
                        icon10=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon11=ICON_NWP("11",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon12=ICON_NWP("12",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon13=ICON_NWP("13",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        
                        icon_var_list=[icon10,icon11,icon12,icon13]
                    if ar_of_day=="AR2":
                        start_hour="10"
                        icon10=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon11=ICON_NWP("11",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon12=ICON_NWP("12",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon13=ICON_NWP("13",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        icon14=ICON_NWP("14",resolution,
                                    for_flight_campaign=True,
                                    campaign=campaign,research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                        
                        icon_var_list=[icon10,icon11,icon12,icon13,icon14]
                        
        else: # if flight track is synthetic
            if flight[0]=="RF10":
                if ar_of_day=="SAR1":
                    start_hour="14"
                    icon14=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                    icon15=ICON_NWP("15",resolution,for_flight_campaign=True,
                            campaign="NAWDEX",research_flights=None,
                            icon_path=hydrometeor_icon_path)
                    icon_var_list=[icon14,icon15]
                    
                if ar_of_day=="SAR_internal":
                    #     start_hour="12"
                    #     icon12=ICON_NWP(start_hour,resolution,
                    #                     for_flight_campaign=True,
                    #                     campaign="NAWDEX",research_flights=None,
                    #                     icon_path=hydrometeor_icon_path)
                    #     icon13=ICON_NWP("13",resolution,for_flight_campaign=True,
                    #                     campaign="NAWDEX",research_flights=None,
                    #                     icon_path=hydrometeor_icon_path)
                    #     icon14=ICON_NWP("14",resolution,for_flight_campaign=True,
                    #                     campaign="NAWDEX",research_flights=None,
                    #                     icon_path=hydrometeor_icon_path)
                    #     icon15=ICON_NWP("15",resolution,for_flight_campaign=True,
                    #                     campaign="NAWDEX",research_flights=None,
                    #                     icon_path=hydrometeor_icon_path)
                        
                    #     icon_var_list=[icon12,icon13,icon14,icon15]
                    
                    
                    # else:
                    start_hour="12"
                    icon12=ICON_NWP(start_hour,resolution,
                                    for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                    icon13=ICON_NWP("13",resolution,for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                    icon14=ICON_NWP("14",resolution,for_flight_campaign=True,
                                    campaign="NAWDEX",research_flights=None,
                                    icon_path=hydrometeor_icon_path)
                    icon15=ICON_NWP("15",resolution,for_flight_campaign=True,
                            campaign="NAWDEX",research_flights=None,
                            icon_path=hydrometeor_icon_path)
                    icon16=ICON_NWP("16",resolution,for_flight_campaign=True,
                            campaign="NAWDEX",research_flights=None,
                            icon_path=hydrometeor_icon_path)
                    
                    icon_var_list=[icon12,icon13,icon14,icon15,icon16]
            
                    #icon_var_list=[icon14]
            else:
                    raise Exception("This AR cross-section is not yet defined")
        return icon_var_list

    # def calculate_theta_e(self,ds):
    #     if not [i in list(ds.keys()) for i in ["r","t"]]:
    #         raise Exception("Some variables are not included in the dataset.",
    #                         "Re-download the correct Reanalysis data")
    #         return None
    #     else:
            
    #         print("Start calculating the different temperatures")
    #         T_profile=ds["t"]#[:,:,50,50]
    #         RH_profile=ds["r"]#[:,:,50,50]
            
    #         print("Replicate p-levels to ndarray")
    #         p_hPa=np.tile(ds["level"],(ds["t"].shape[0],1))
    #         p_hPa=np.expand_dims(p_hPa,axis=2)
    #         p_hPa=np.repeat(p_hPa, ds["t"].shape[2],axis=2)
    #         p_hPa=np.expand_dims(p_hPa,axis=3)
    #         p_hPa=np.repeat(p_hPa, ds["t"].shape[3],axis=3)
    #         p_hPa=p_hPa * units.hPa
            
    #         print("Calculate Potential Temperature")
    #         Theta_profile=mpcalc.potential_temperature(p_hPa, T_profile)
    #         ds["theta"]=xr.DataArray(np.array(Theta_profile),
    #                                  coords=ds["t"].coords,
    #                                  attrs={'units': "K",
    #                                         'long_name':"Potential Temperature"})
            
    #         print("Calculate Dewpoint")
    #         Td_profile=mpcalc.dewpoint_from_relative_humidity(T_profile,
    #                                                           RH_profile)
            
    #         print("Calculate Equivalent Potential Temperature -- takes a while")
    #         Te_profile=mpcalc.equivalent_potential_temperature(p_hPa,
    #                                                            T_profile,
    #                                                            Td_profile)
    #         ds["theta_e"]=xr.DataArray(np.array(Theta_profile),
    #                                  coords=ds["t"].coords,
    #                                  attrs={'units': "K",
    #                                 'long_name':"Equivalent Potential Temperature"})
            
    #         print("Equivalent Potential Temperature calculated")
    #     print("ERA dataset now contains new temperature parameters:",
    #           "theta_e, theta")            
    #     return ds
