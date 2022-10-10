# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 18:16:27 2021

@author: u300737
"""
import os
import sys


import numpy as np
import pandas as pd
import xarray as xr
sys.path.insert(1,os.getcwd()+"/../config/")
import data_config

import metpy.calc as mpcalc
import metpy.constants as mpconsts

from metpy.units import units

class ERA5():
    def __init__(self,for_flight_campaign=True,
                 campaign="NAWDEX",research_flights=None,
                 era_path=os.getcwd()):
        self.for_flight_campaign=for_flight_campaign
        self.data_path=era_path
        if for_flight_campaign:
            self.campaign_name=campaign
            self.flight=""
        self.hours=[  '00:00', '01:00', '02:00',
                         '03:00', '04:00', '05:00',
                         '06:00', '07:00', '08:00',
                         '09:00', '10:00', '11:00',
                         '12:00', '13:00', '14:00',
                         '15:00', '16:00', '17:00',
                         '18:00', '19:00', '20:00',
                         '21:00', '22:00', '23:00',]
        self.hours_time=self.hours
    
    def load_era5_data(self,file_name):
        """

        Parameters
        ----------
        file_name : str
            ERA5 file name.

        Returns
        -------
        ds : xarray dictionary
            ERA-5 Data.
        era_path : str
            path of the ERA file.

        """
        ## Load ERA-5 Dataset
        ds=xr.open_dataset(self.data_path+file_name)
        return ds,self.data_path
    
    def calculate_theta_e(self,ds):
        if not [i in list(ds.keys()) for i in ["r","t"]]:
            raise Exception("Some variables are not included in the dataset.",
                            "Re-download the correct Reanalysis data")
            return None
        else:
            
            print("Start calculating the different temperatures")
            T_profile=ds["t"]#[:,:,50,50]
            RH_profile=ds["r"]#[:,:,50,50]
            
            print("Replicate p-levels to ndarray")
            if not hasattr(ds, "name"):    
                p_hPa=np.tile(ds["level"],(ds["t"].shape[0],1))
            else:
                p_hPa=np.tile(float(str(ds["name"].values)[:-3]),
                              (ds["t"].shape[0],1))
            if not len(p_hPa.shape)==3:
                p_hPa=np.expand_dims(p_hPa,axis=2)
            #p_hPa=np.repeat(p_hPa,ds["t"].shape[1],axis=1)
            p_hPa=np.repeat(p_hPa, ds["t"].shape[2],axis=2)
            if len(ds["t"].shape)==4:
                p_hPa=np.expand_dims(p_hPa,axis=3)
                p_hPa=np.repeat(p_hPa, ds["t"].shape[3],axis=3)
            p_hPa=p_hPa * units.hPa
            
            print("Calculate Potential Temperature")
            Theta_profile=mpcalc.potential_temperature(p_hPa, T_profile)
            ds["theta"]=xr.DataArray(
                            np.array(Theta_profile),coords=ds["t"].coords,
                            attrs={'units': "K",
                                   'long_name':"Potential Temperature"})
            
            print("Calculate Dewpoint")
            Td_profile=mpcalc.dewpoint_from_relative_humidity(T_profile,
                                                              RH_profile)
            
            print("Calculate Equivalent Potential Temperature -- takes a while")
            Te_profile=mpcalc.equivalent_potential_temperature(p_hPa,
                                                               T_profile,
                                                               Td_profile)
            ds["theta_e"]=xr.DataArray(
                            np.array(Theta_profile),coords=ds["t"].coords,
                            attrs={'units': "K",
                               'long_name':"Equivalent Potential Temperature"})
            
            print("Theta_e calculated")
        print("ERA dataset now contains new temperature parameters:",
              "theta_e, theta")            
        return ds
    
    def apply_orographic_mask_to_era(self,era_df,threshold=4e-3,
                                     variable="LWC",short_test=False):
        """
        This function masks the era values which do not take into account the 
        orography which leads to extrapolated values down to the surface.
        These values are set constant and are critical when integrating 
        vertically in order to get for example vertical columns. 
        
        So this mask filters out the values where the vertical gradient is zero
        as these are the extrapolated values.
    
        Parameters
        ----------
        era_df : pd.DataFrame
            ERA dataset to be masked.
        
        variable : str
            variable to analyse
            
        short_test : boolean
            Boolean specifying the mask works properly by creating a quicklook.
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
        return masked_era_df

class ERA5_Downloader(ERA5):
    def __init__(self,dates,defined_area,for_flight_campaign=True,
                 campaign="NAWDEX",research_flights=None,
                 era_path=os.getcwd()):
        
        super().__init__(for_flight_campaign,campaign,research_flights,era_path)
        self.campaign=campaign
        self.dates=dates
        self.missing_dates=[]
        self.missing_months=[] # needed for monthly functions
        self.defined_area=defined_area

    ###########################################################################
    ## Major functions
    ###########################################################################    
    def get_hourly_era5_hydrometeors_pressure_lvls(self):
        import cdsapi
        c = cdsapi.Client()
        if self.defined_area==[]:
            self.defined_area= [90, -80, 20,  35]
        if not self.hours:
            time_list= [
                                '00:00', '01:00', '02:00',
                                '03:00', '04:00', '05:00',
                                '06:00', '07:00', '08:00',
                                '09:00', '10:00', '11:00',
                                '12:00', '13:00', '14:00',
                                '15:00', '16:00', '17:00',
                                '18:00', '19:00', '20:00',
                                '21:00', '22:00', '23:00',]
        else:
            time_list=self.hours
        for date in self.dates:
            year=date[0:4]
            month=date[4:6]
            day=date[6:8]
            c.retrieve(
                            'reanalysis-era5-pressure-levels',
                            {
                                    'product_type': 'reanalysis',
                                    'format': 'netcdf',
                                    'variable': [
                                            'geopotential','temperature',
                                            'relative_humidity','specific_humidity', 
                                            'specific_cloud_ice_water_content', 
                                            'specific_cloud_liquid_water_content',
                                            'specific_rain_water_content', 
                                            'specific_snow_water_content',
                                            'u_component_of_wind',
                                            'v_component_of_wind'
                                            ],
                                    'pressure_level': [
                                            '150', '175', '200',
                                            '225', '250', '300',
                                            '350', '400', '450',
                                            '500', '550', '600',
                                            '650', '700', '750',
                                            '775', '800', '825',
                                            '850', '875', '900',
                                            '925', '950', '975',
                                            '1000',
                                            ],
                                    'month': month,
                                    'day': day,
                                    'year': year,
                                    'time': time_list,
                                    'area': self.defined_area,
                                    },
                                    self.data_path+\
                                    'hydrometeors_pressure_levels_'+\
                                    year+month+day+'.nc')
                
            print('hydrometeors_pressure_levels'+year+month+day+'.nc',
                  ' successfully donwloaded')
    
    def get_hourly_era5_total_columns(self):
        import cdsapi
        
        c = cdsapi.Client()
        
        if not self.defined_area:
            data_area= [65, -30, 52,  0]
            #           N ,   S,  W,   E
        else:
            data_area=self.defined_area
        if not self.hours:
            time_list= [
                                '00:00', '01:00', '02:00',
                                '03:00', '04:00', '05:00',
                                '06:00', '07:00', '08:00',
                                '09:00', '10:00', '11:00',
                                '12:00', '13:00', '14:00',
                                '15:00', '16:00', '17:00',
                                '18:00', '19:00', '20:00',
                                '21:00', '22:00', '23:00',]
        else:
            time_list=self.hours
            
        for date in self.missing_dates:
            year=date[0:4]
            month=date[4:6]
            day=date[6:8]
            c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'format': 'netcdf',
                            'variable': [
                                'mean_sea_level_pressure',
                                'sea_ice_cover',
                                'total_column_cloud_ice_water',
                                'total_column_cloud_liquid_water', 
                                'total_column_water_vapour',
                                'vertical_integral_of_eastward_water_vapour_flux',
                                'vertical_integral_of_northward_water_vapour_flux',
                                '2m_temperature',
                                'vertical_integral_of_divergence_of_moisture_flux',
                                'vertically_integrated_moisture_divergence',
                                'evaporation',
                                'total_precipitation'
                            ],
                            'year': year,
                            'month': month,
                            'day': day,
                            'area': data_area,
                            'time': time_list,
                        },
                        self.data_path+'total_columns_'+year+'_'+\
                            month+'_'+day+'.nc')
    
    def get_monthly_averages_total_columns(self):
        import cdsapi
        
        c = cdsapi.Client()
        
        if not self.defined_area:
            data_area= [65, -30, 52,  0]
            #           N ,   S,  W,   E
        else:
            data_area=self.defined_area
        
        time_list= ['00:00']
        
        self.average_months=[]
        for month in self.missing_months:
            self.average_months.append(month[4:6])
        self.average_months=[*set(self.average_months)]
        
        m=0    
        for month in self.average_months:
            
            c.retrieve(
                        'reanalysis-era5-single-levels-monthly-means',
                        {   'format': 'netcdf',
                            'product_type': 'monthly_averaged_reanalysis',
                            'variable': ['mean_sea_level_pressure',
                                'total_column_cloud_ice_water',
                                'total_column_cloud_liquid_water',
                                'total_column_water_vapour',
                                'vertical_integral_of_eastward_water_vapour_flux',
                                'vertical_integral_of_northward_water_vapour_flux',
                                '2m_temperature',
                                'evaporation',
                                'total_precipitation'],
                            'year': [
                                '1979', '1980', '1981',
                                '1982', '1983', '1984',
                                '1985', '1986', '1987',
                                '1988', '1989', '1990',
                                '1991', '1992', '1993',
                                '1994', '1995', '1996',
                                '1997', '1998', '1999',
                                '2000', '2001', '2002',
                                '2003', '2004', '2005',
                                '2006', '2007', '2008',
                                '2009', '2010', '2011',
                                '2012', '2013', '2014',
                                '2015', '2016', '2017',
                                '2018', '2019','2020','2021','2022'],
                            'month': month,
                            'area': data_area,
                            'time': time_list,
                        },
                        self.data_path+'total_columns_monthly_average_'+ \
                            self.missing_months[m][0:4]+"_"+month+'.nc')
            m+=1
    ###########################################################################
    ### Specific functions
    ###########################################################################
    def get_hourly_era5_temp_850hPa(self,daily_average=False,add_theta_e=True,
                                    single_date=True):
        import cdsapi
        c = cdsapi.Client()
        if self.defined_area==[]:
            self.defined_area= [90, -80, 20,  35]
        if not self.hours:
            time_list= [
                                '00:00', '01:00', '02:00',
                                '03:00', '04:00', '05:00',
                                '06:00', '07:00', '08:00',
                                '09:00', '10:00', '11:00',
                                '12:00', '13:00', '14:00',
                                '15:00', '16:00', '17:00',
                                '18:00', '19:00', '20:00',
                                '21:00', '22:00', '23:00',]
        else:
            time_list=self.hours
        fname='temp850hPa_'
        for date in self.dates:
            year=date[0:4]
            month=date[4:6]
            if not single_date:
                if month=="03":
                    days=[str(dd) for dd in np.arange(8,32)]
                elif month=="04":
                    days=[str(dd) for dd in np.arange(1,18)]
            else:
                days=[date[6:8]]
            c.retrieve(
                            'reanalysis-era5-pressure-levels',
                            {
                                    'product_type': 'reanalysis',
                                    'format': 'netcdf',
                                    'variable': ['relative_humidity', 
                                                 'specific_humidity',
                                                 'temperature'],
                                    'pressure_level': ['850'],
                                    'month': month,
                                    'day': days,
                                    'year': year,
                                    'time': time_list,
                                    'area': self.defined_area,
                                    },
                            self.data_path+fname+year+month+days[0]+'.nc')
            
            print(fname+year+month+days[0]+'.nc',
                  ' successfully donwloaded')
            
            ds=xr.open_dataset(self.data_path+fname+year+month+days[0]+'.nc')
            nc_compression=dict(zlib=True,complevel=2,dtype=np.float64)
            nc_encoding={var:nc_compression for var in ds.variables}
            ds["name"]="850hPa"
            #import pandas as pd
            #print(pd.DatetimeIndex(ds.time))
            #sys.exit()
            if add_theta_e:
                ds=self.calculate_theta_e(ds)
                del ds["name"]
                ds.to_netcdf(self.data_path+fname+year+month+days[0]+'.nc',
                             mode="w")
            if daily_average:
                print("Resample to Daily Average")
                ds_copy=ds.copy()
                del ds
                ds_daily=ds_copy.resample(time='D').mean()
                print("save file as daily mean to keep storage")
            
                ds_daily.to_netcdf(self.data_path+fname+year+month+'.nc',mode="w",
                               engine="netcdf4",format="NETCDF4",
                               encoding=nc_encoding)
    
    def get_daily_avg_single_levels(self):
        import cdsapi
        
        c = cdsapi.Client()
        if not self.defined_area:
            data_area= [65, -30, 52,  0]
            #           N ,   S,  W,   E
        else:
            data_area=self.defined_area
        if self.campaign=="HALO_AC3_Dry_Run":
            fname='dry_run_integrals_'
        else:
            fname="single_level_daily_average_"
            
        for date in self.missing_dates:
            year=date[0:4]
            month=date[4:6]
            if month=="03":
                days=[str(dd) for dd in np.arange(5,32)]
            elif month=="04":
                days=[str(dd) for dd in np.arange(1,18)]
                
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                'sea_ice_cover', 'total_column_water_vapour',
                'vertical_integral_of_eastward_water_vapour_flux',
                'vertical_integral_of_northward_water_vapour_flux'],
                'year': year,
                'month': month,
                'day': days,
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                    ],
                    'area': data_area,
                        },  
                self.data_path+fname+year+month+'.nc')
            ds=xr.open_dataset(self.data_path+fname+year+month+'.nc')
            print("Resample to Daily Average")
            ds_copy=ds.copy()
            del ds
            ds_daily=ds_copy.resample(time='D').mean()
            print("save file as daily mean to keep storage")
            nc_compression=dict(zlib=True,complevel=9,dtype=np.float32)
            nc_encoding={var:nc_compression for var in ds_daily.variables}
    
            ds_daily.to_netcdf(self.data_path+fname+year+month+'.nc',mode="w",
                               engine="netcdf4",format="NETCDF4",
                               encoding=nc_encoding)
    
            
    def download_satellite_sea_ice(self):
        import cdsapi
        
        c = cdsapi.Client()
        self.average_months=[]
        self.average_years=[]
        days=['01', '02', '03','04', '05', '06','07', '08', '09',
              '10', '11', '12','13', '14', '15','16', '17', '18',
              '19', '20', '21','22', '23', '24','25', '26', '27',
              '28', '29', '30','31']
        
        for date in self.dates:
            self.average_years=date[0:4]
            self.average_months=date[4:6]
            if self.campaign=="HALO_AC3_Dry_Run" or self.campaign=="HALO_AC3":
                if self.average_months=="03":
                    days=[str(dd) for dd in np.arange(5,32)]
                elif self.average_months=="04":
                    days=[str(dd) for dd in np.arange(1,17)]
                else:
                    pass
            if int(self.average_years)<2016:
                cdr_type="cdr"
            else:
                cdr_type="icdr"
            c.retrieve(
                    'satellite-sea-ice-edge-type',
                    {
                        'version': '1_0',
                        'variable': 'sea_ice_edge',
                        'region': 'northern_hemisphere',
                        'cdr_type': cdr_type,
                        'format': 'zip',
                        'year': self.average_years,
                        'month': self.average_months,
                        'day': days,
                        },
                    self.data_path+"/monthly_sea_ice_"+self.average_months+\
                                "_"+self.average_years+'.zip')
    
    
    def process_satellite_sea_ice(self,sea_ice_path,
                             test_plot=True,
                             save_merged_nc=True):
        import glob
        import matplotlib.pyplot as plt
        
        self.average_months=[]
        self.average_years=[]
        sea_ice_monthly_paths=glob.glob(sea_ice_path+"*_*")
        
        m_int=0
        
        for monthly_path in sea_ice_monthly_paths:
            sea_ice_files=glob.glob(monthly_path+"/"+"*.nc")
            f_int=0
        
            for file in sea_ice_files:
                print("Day Files:",f_int)
                ds=xr.open_dataset(file)
                ice_edge=ds["ice_edge"]
                if f_int==0:
                    ice_ds=ice_edge
                    
                else:
                    ice_ds=xr.concat([ice_ds,ice_edge],dim="time")
                f_int+=1    
            ice_ds=ice_ds.fillna(3)
            monthly_mean_ice=ice_ds.resample(time="M").sum()/len(sea_ice_files)
            monthly_mean_ice=monthly_mean_ice.fillna(3)    
                
            if test_plot:
                fig=plt.figure(figsize=(9,9))
                plt.imshow(monthly_mean_ice[0,:,:])
                print("Save Sea Ice Quicklook")
                fig.savefig("Quicklook_Sea_Ice_"+monthly_path[-6:-3]+"-"+\
                            monthly_path[-2:]+".png",dpi=200)
            if m_int==0:
                long_term_sea_ice_ds=monthly_mean_ice
            else:
                long_term_sea_ice_ds=xr.concat([long_term_sea_ice_ds,
                                                monthly_mean_ice],dim="time")
            m_int+=1
        
        merged_nc_fname="monthly_longterm_sea_ice_edge.nc"
        if save_merged_nc:
            long_term_sea_ice=long_term_sea_ice_ds.to_dataset()
            nc_compression=dict(zlib=True,complevel=9,dtype=np.float32)
            nc_encoding={var:nc_compression for var in long_term_sea_ice.variables}
            long_term_sea_ice.to_netcdf(\
                        path=sea_ice_path+merged_nc_fname,
                        mode="w",engine="netcdf4",format="NETCDF4",
                        encoding=nc_encoding)                
            
        return long_term_sea_ice_ds
    
    def process_era5_sea_ice(self,sea_ice_path,
                             test_plot=True,
                             save_merged_nc=True):
        import glob
        import matplotlib.pyplot as plt
        
        self.average_months=[]
        self.average_years=[]
        sea_ice_fname="merged_dry_run_integral.nc"
        
        m_int=0
        ds=xr.open_dataset(sea_ice_path+sea_ice_fname)
        ice_ds=ds["siconc"]
        print("Resample to monthly mean")
        monthly_mean_ice=ice_ds.resample(time="M").mean()
        
        if test_plot:
                fig=plt.figure(figsize=(9,9))
                plt.imshow(monthly_mean_ice[0,:,:])
                print("Save Sea Ice Quicklook")
                fig.savefig("Quicklook_Sea_Ice_1991_03.png",dpi=200)
        
        merged_nc_fname="monthly_longterm_sea_ice_edge.nc"
        if save_merged_nc:
            long_term_sea_ice=monthly_mean_ice.to_dataset()
            nc_compression=dict(zlib=True,complevel=9,dtype=np.float32)
            nc_encoding={var:nc_compression for var in long_term_sea_ice.variables}
            long_term_sea_ice.to_netcdf(\
                        path=sea_ice_path+merged_nc_fname,
                        mode="w",engine="netcdf4",format="NETCDF4",
                        encoding=nc_encoding)                
            
        return long_term_sea_ice

    def check_if_ERA_data_exists(self,data_type="hourly_pressure_levels"):
        """
        Checks if given files for days of a specific data type (total columns,
        hourly pressure level data and so on) exist and returns boolean for 
        given files that are comprised in a dictionary

        Parameters
        ----------
        data_type : str, optional
            str specifying the data type to be checked for existence.
            The default is "hourly_pressure_levels".

        
        Returns
        -------
        files_exist : dict
            Dictionary containing the boolean information if file exists.

        """
        file_end=".nc"
        files_exist={}
        
        for date in self.dates:
            year=date[0:4]
            month=date[4:6]
            if not data_type.startswith("monthly"):
                day=date[6:8]
                era_fname=data_type+"_"+year+"_"+month+"_"+day+file_end
                if data_type=="dry_run_integrals":
                    era_fname='dry_run_integrals_'+year+month+file_end
                if data_type=="dry_run_temp_850hPa":
                    era_fname='dry_run_temp850hPa_'+year+month+file_end
            else:
                if data_type.endswith("ice"):
                    file_end=".zip"
                era_fname=data_type+"_"+year+"_"+month+file_end
            # check if defined file exists    
            if not os.path.exists(self.data_path+era_fname):
                files_exist[era_fname]=False
                print("File not found: The ERA data",
                      era_fname," does not exist.")
            else:
                print("File ",self.data_path+era_fname,
                      " exists. Nothing to be done.")
                files_exist[era_fname]=True
        return files_exist
                       
    def check_if_ERA_data_exists_and_download(
                                self,
                                data_type="hydrometeor_pressure_levels"):
        
        """
        Checks

        Returns
        -------
        file_is_downloaded.

        """
        self.data_type=data_type
        self.download_funcs={
            "hydrometeor_pressure_levels":self.get_hourly_era5_hydrometeors_pressure_lvls,
            "total_columns":self.get_hourly_era5_total_columns,
            "monthly_average_total_columns":self.get_monthly_averages_total_columns,
            "monthly_sea_ice":self.download_satellite_sea_ice,
            "single_levels_daily_avg":self.get_daily_avg_single_levels,
            "temp_850hPa":self.get_hourly_era5_temp_850hPa}
        
        if data_type not in ["hydrometeor_pressure_levels",
                            "total_columns",
                            "monthly_average_total_columns",
                            "monthly_sea_ice","single_levels_daily_avg",
                            "temp_850hPa"]:
            raise NameError(data_type," is not included in ERA downloading types")
        else:
            files_exist=self.check_if_ERA_data_exists(data_type=self.data_type)
            i=0
            for file in files_exist.keys():
                if not files_exist[file]:
                    if not data_type.startswith("monthly"):
                        self.missing_dates.append(self.dates[i])
                    else:
                        self.missing_months.append(self.dates[i][0:6])
                i+=1
            if not data_type.startswith("monthly"):
                if not len(self.missing_dates)==0:
                    print("Following dates are missing and will be downloaded:",
                          self.missing_dates)
                    #Download data
                    self.download_funcs[data_type]()
            
                else:    
                    print(data_type, "are already downloaded for ",self.dates)
            else:
                if not len(self.missing_months)==0:
                    print("following months are missing and will be downloaded:",
                          self.missing_months)
                    #Download data
                    self.download_funcs[data_type]()
            
        return 

    def download_handler(self,do_levels=False,do_total_columns=True,
                         do_monthly_averages=False,do_temp_850hPa=False,
                         do_daily_average_single_levels=False):
        
        print("Download routine is started and will get the data desired")
        # HMP Total Columns       
        if do_total_columns:
            self.check_if_ERA_data_exists_and_download(
                                    data_type="total_columns")
        # HMC Total Columns       
        if do_levels:
            self.check_if_ERA_data_exists_and_download(
                                    data_type="hydrometeor_pressure_levels")
        # Monthly Averages
        # --> published on the sixth day of the following month
        if do_monthly_averages:
            self.check_if_ERA_data_exists_and_download(
                                    data_type="monthly_average_total_columns")
        if do_daily_average_single_levels:
            self.check_if_ERA_data_exists_and_download(
                                         data_type="single_levels_daily_avg")
        if do_temp_850hPa:
            self.check_if_ERA_data_exists_and_download(
                                         data_type="temp_850hPa")

class CARRA():
    def __init__(self,for_flight_campaign=True,
                 campaign="NAWDEX",research_flights=None,
                 carra_path=os.getcwd()):
        self.for_flight_campaign=for_flight_campaign
        self.data_path=carra_path
        if for_flight_campaign:
            self.campaign_name=campaign
            self.flight=""
        self.hours=['00:00','03:00','06:00',
                    '09:00','12:00','15:00', 
                    '18:00','21:00']
        self.carra_path=carra_path
        if not os.path.exists(self.carra_path):
            os.makedirs(self.carra_path)
    
    def load_vertical_carra_data(self,date,initial_time="06:00"):
        self.carra_file="CARRA_vertical_levels_"+date+"run_"+\
            initial_time[0:2]+".nc"
        self.ds=xr.open_dataset(self.carra_path+self.carra_file)
        self.ds=self.ds.drop(["step","time"])
        self.ds["longitude"]=self.ds["longitude"].where(
                                            self.ds["longitude"]<180,
                                            self.ds["longitude"]-360)
        
    def merge_all_files_for_given_flight(self):
        import glob
        file_date_list=glob.glob(self.carra_path+"*"+self.date+"*.nc")
        ds_list=[]
        for file in file_date_list:
            ds_item=self.load_vertical_carra_data(self.date,
                                                  initial_time=file[-4:-2]+\
                                                      ":00")
            ds_list.append(ds_item)
        self.ds=xr.concat(ds_list,dim="valid_time")
        
    def calc_specific_humidity_from_relative_humidity(self):
        """
        
        
        Returns
        -------
        None.
            
        """
        print("Calculate q from RH")
        #Using metpy functions to calculate specific humidity from RH
        pressure=self.ds["isobaricInhPa"].data
        pressure=pressure[:,np.newaxis]
        pressure=np.repeat(pressure,self.ds["t"].shape[1],axis=1)
        if len(self.ds["t"].shape)>=3:
            pressure=pressure[:,:,np.newaxis]
            pressure=np.repeat(pressure,self.ds["t"].shape[2],axis=2)
            pressure=pressure * units.hPa
    
        rh=self.ds["r"].data/100
        temperature=self.ds["t"].data * units.K
        mixing_ratio=mpcalc.mixing_ratio_from_relative_humidity(
            rh,temperature,pressure)
        specific_humidity=xr.DataArray(np.array(
                                    mpcalc.specific_humidity_from_mixing_ratio(
                                        mixing_ratio)),
                                   dims=["isobaricInhPa","y","x"])
    
        self.ds=self.ds.assign({"specific_humidity":specific_humidity})
 

    def calc_IVT_from_q(self):
        print("Calculate IVT from CARRA")
        g= 9.81
        list_timestamps=[pd.Timestamp(self.ds.valid_time.values)] 
    
        if len(list_timestamps)==1:
            self.carra_ivt=xr.Dataset(coords={"longitude":self.ds.longitude,
                                     "latitude":self.ds.latitude})
            nan_array=np.empty((self.ds["r"].shape[1],
                        self.ds["r"].shape[2]))
    
        else:
            self.carra_ivt=xr.Dataset(coords={"valid_time":self.ds.valid_time,
                                 "longitude":self.ds.longitude,
                                 "latitude":self.ds.latitude})
            nan_array=np.empty((self.ds["r"].shape[0],self.ds["r"].shape[2],
                        self.ds["r"].shape[3]))
    
        nan_array[:]=np.nan
    
        self.carra_ivt["IVT"]=xr.DataArray(data=nan_array.T,
                                           coords=self.carra_ivt.coords)
        self.carra_ivt["IVT_u"]=xr.DataArray(data=nan_array.T,
                                             coords=self.carra_ivt.coords)
        self.carra_ivt["IVT_v"]=xr.DataArray(data=nan_array.T,
                                             coords=self.carra_ivt.coords)
        self.carra_ivt["IWV_clc"]=xr.DataArray(data=nan_array.T,
                                               coords=self.carra_ivt.coords)
        for timestep in list_timestamps:
            print("Timestep:", timestep)
            if len(list_timestamps)==1:
                q_loc=np.array(self.ds["specific_humidity"].values)
                u_loc=np.array(self.ds["u"].values)
                v_loc=np.array(self.ds["v"].values)
        
            else:
                q_loc=self.ds["specific_humidity"][timestep,:,:].dropna()
                u_loc=self.ds["u"][timestep,:,:].dropna()
                v_loc=self.ds["v"][timestep,:,:].dropna()
            qu=q_loc*u_loc
            qv=q_loc*v_loc
            pres_index=pd.Series(self.ds["isobaricInhPa"].values*100)
        
            iwv_temporary=-1/g*np.trapz(q_loc,axis=0,x=pres_index)
            ivt_u_temporary=-1/g*np.trapz(qu,axis=0,x=pres_index)
            ivt_v_temporary=-1/g*np.trapz(qv,axis=0,x=pres_index)
            ivt_temporary=np.sqrt(ivt_u_temporary**2+ivt_v_temporary**2)
            
            self.carra_ivt["IVT"][timestep,:,:].values=ivt_temporary.T
            self.carra_ivt["IVT_u"][timestep,:,:].values=ivt_u_temporary.T
            self.carra_ivt["IVT_v"][timestep,:,:].values=ivt_v_temporary.T
            self.carra_ivt["IWV_clc"][timestep,:,:].values=iwv_temporary.T
        
    def calc_ivt_from_origin_carra_ds(self):
        self.calc_specific_humidity_from_relative_humidity()
        self.calc_IVT_from_q()
    
    
class CARRA_Downloader(CARRA):
    def __init__(self,dates,domain,for_flight_campaign=True,
                 campaign="NAWDEX",research_flights=None,
                 carra_path=os.getcwd(),initial_time="00:00"):
        
        super().__init__(for_flight_campaign,campaign,research_flights,carra_path)
        self.campaign=campaign
        self.dates=dates
        self.missing_dates=[]
        self.missing_months=[] # needed for monthly functions
        self.domain=domain
        self.initial_time=initial_time
        pass
    
    def download_total_column_data(self,single_date=True,variables=[
                                    'evaporation', 
                                    'skin_temperature',
                                    'total_column_integrated_water_vapour',
                                    'total_precipitation']):
        
        for date in self.dates:
            year=date[0:4]
            month=date[4:6]
            if not single_date:
                if month=="03":
                    days=[str(dd) for dd in np.arange(8,32)]
                elif month=="04":
                    days=[str(dd) for dd in np.arange(1,18)]
            else:
                days=[date[6:8]]
                
            lead_time_hours=["1","2","3"]
            print("Download single column values for moisture budget.")
            import cdsapi
        
            c = cdsapi.Client()
            fname='CARRA_vertical_columns_'
            c.retrieve(
                'reanalysis-carra-single-levels',
                {
                    'format': 'netcdf',
                    'domain': self.domain,
                    'level_type': 'surface_or_atmosphere',
                    'variable': variables,
                    'product_type': 'forecast',
                    'time': self.initial_time,
                    'year': year,
                    'month': month,
                    'day': days,
                    'leadtime_hour': lead_time_hours,
                    },self.data_path+fname+year+month+days[0]+'run_'+\
                        self.initial_time[0:2]+'.nc')
        
    def download_forecast_hydrometeor_profiles(self,single_date=True,
                            variables=['specific_cloud_ice_water_content',
                                       'specific_cloud_liquid_water_content',
                                       'specific_cloud_rain_water_content',
                                       'specific_cloud_snow_water_content']):
        #if self.initial_time=="00:00":
        #    lead_time_hours=[0,1,2,3,4,5,6,9,12,15,18,21,24]
        #elif self.intial_time=="06:00":
        lead_time_hours=["1","2","3"]
        import cdsapi

        c = cdsapi.Client()
        fname='CARRA_vertical_hydrometeors_'
        
        for date in self.dates:
            year=date[0:4]
            month=date[4:6]
            if not single_date:
                if month=="03":
                    days=[str(dd) for dd in np.arange(8,32)]
                elif month=="04":
                    days=[str(dd) for dd in np.arange(1,18)]
            else:
                days=[date[6:8]]
            c.retrieve(
                'reanalysis-carra-pressure-levels',
                {'format': 'netcdf',
                 'domain': self.domain,
                 'variable': variables,
                 'pressure_level': ['10', '20', '30',
                                    '50', '70', '100',
                                    '150', '200', '250',
                                    '300', '400', '500',
                                    '600', '700', '750',
                                    '800', '825', '850',
                                    '875', '900', '925',
                                    '950', '1000'],
                 'product_type': 'forecast',
                 'time': self.initial_time,
                 'leadtime_hour':lead_time_hours,
                 'year': year,
                 'month': month,
                 'day': days[0]},
                self.data_path+fname+year+month+days[0]+'run_'+\
                    self.initial_time[0:2]+'.nc')
                        
    def download_forecast_vertical_profiles(self,single_date=True,
                                   variables=['temperature',
                                              'relative_humidity',
                                              'u_component_of_wind',
                                              'v_component_of_wind',
                                              'geopotential']):
        #if self.initial_time=="00:00":
        #    lead_time_hours=[0,1,2,3,4,5,6,9,12,15,18,21,24]
        #elif self.intial_time=="06:00":
        lead_time_hours=["1","2","3"]
        import cdsapi

        c = cdsapi.Client()
        fname='CARRA_vertical_levels_'
        
        for date in self.dates:
            year=date[0:4]
            month=date[4:6]
            if not single_date:
                if month=="03":
                    days=[str(dd) for dd in np.arange(8,32)]
                elif month=="04":
                    days=[str(dd) for dd in np.arange(1,18)]
            else:
                days=[date[6:8]]
            c.retrieve(
                'reanalysis-carra-pressure-levels',
                {'format': 'netcdf',
                 'domain': self.domain,
                 'variable': variables,
                 'pressure_level': ['10', '20', '30',
                                    '50', '70', '100',
                                    '150', '200', '250',
                                    '300', '400', '500',
                                    '600', '700', '750',
                                    '800', '825', '850',
                                    '875', '900', '925',
                                    '950', '1000'],
                 'product_type': 'forecast',
                 'time': self.initial_time,
                 'leadtime_hour':lead_time_hours,
                 'year': year,
                 'month': month,
                 'day': days[0]},
                self.data_path+fname+year+month+days[0]+'run_'+\
                    self.initial_time[0:2]+'.nc') 
    
    def download_handler(self,do_levels=True,do_hydrometeors=True,
                         do_total_columns=False,
                         do_specific_levels=True):
        
        print("Download routine is started and will get the data desired")
        # HMP Total Columns       
        if do_total_columns:
           self.download_total_column_data()        # HMC Total Columns       
        
        if do_levels:
            self.download_forecast_vertical_profiles()
        
        if do_hydrometeors:
            self.download_forecast_hydrometeor_profiles()
        
        #if do_specific_levels:
        #    self.download_specific_levels(level_type="temp_850hPa")
        
        else:
            pass
    
        return None
###############################################################################
###############################################################################
"""
==============================================================================
                            MAIN SCRIPT BEGINNING
==============================================================================
"""
def main(campaign="NAWDEX"):
    central_path=os.getcwd()+"/../../../Work/GIT_Repository/"
    era_path=central_path+"/"+campaign+"/data/ERA-5/"
    carra_path=central_path+"/"+campaign+"/data/CARRA/"
    
    era_is_desired=True
    carra_is_desired=False
    carra_initial="15:00"
    domain="east_domain"
        
    # Check campaign name to specify flights and, i.e. dates to download
    if campaign.upper()=="NAWDEX":
        # Specifications:
        synthetic_campaign=False
        # is this analysis for campaign data?    
        analysing_campaign=True
    
        #flights to analyse
        flights=["RF10"]
        #flights={#"RF01":"17","RF02":"21","RF03":"23","RF04":"26",
                 #            "RF05":"27","RF06":"01","RF07":"06","RF08":"09",
                 #            "RF09":"10",
        #         "RF10":"13"}#,"RF11":"14","RF12":"15",
                 #            "RF13":"18"}
        months=["10"]
        dates=[#"20160917","20160921","20160923","20160926","20160927",
               #"20160927","20161001","20161006","20161009","20161010",
               "20161013"]#,"20161014","20161015","20161018"]
        
    elif campaign=="HALO_AC3_Dry_Run":
        # Specifications:
        synthetic_campaign=True
        # is this analysis for campaign data?    
        analysing_campaign=True
    
        # which flights to analyse?
        flights=["RF01","RF02","RF03","RF04"]
        month="03"
        dates=["20210323","20210324","20210325","20210326"]
            
    elif campaign.upper()=="NA_FEBRUARY_RUN":
        synthetic_campaign=True
        analysing_campaign=True
        flights=["SRF02"]#,"RF02","RF03","RF04","RF05"]
        # Siberian 
        # 20160423
        # 20180406? --> Yes but does not reach northward enough, block at coastline (Sca)
        # 20190329? --> No!!
        # 20160328? --> Somehow yes
        # NA Pathway 20160302, 20160312
        dates=["20180224"]#["20200419"]#["20160328"]20180427,20160408#,"20180226",
        #North-Atlantic: "20190319","20190329""20190416","20190420"]
    elif campaign=="Second_Synthetic_Study":
        synthetic_campaign=True
        analysing_campaign=True
        flights=["SRF02"]
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
        dates=["20110317"]       
        #dates=["20110423"]
    elif campaign=="Weather_Course":
        synthetic_campaign=False
        analysing_campaign=False
        flights=[]
        dates=["20210727"]
    elif campaign=="HALO_AC3":
        synthetic_campaign=False
        analysing_campaign=True
        flights=["RF08"]#["RF06"]#,"RF06"]
        dates=["20220321"]#["20220316"]#,"20220316"]
    else:
        raise Exception("The given campaign is not specified in the Downloader")
    
    hours_time=['00:00','01:00','02:00',
                '03:00','04:00','05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00']
    
    
    # load the config-file for further analysis
    config_file=data_config.load_config_file(os.getcwd(),"data_config_file")
    config_file["Data_Paths"]["campaign"]=campaign
    
    
    if not os.path.exists(central_path+"/"+config_file["Data_Paths"]["campaign"]):
        os.mkdir(central_path+"/"+config_file["Data_Paths"]["campaign"])
        os.mkdir(central_path+"/"+config_file["Data_Paths"]["campaign"]+"/data/")
    if not os.path.exists(era_path):
        os.mkdir(era_path)
        print(era_path," has created.")
    
    # print campaign for which to download data and check if it is the right one. 
    # define research area which depends on Campaign to analyse
    # research_area= [N, -E, E, S]
    
    if analysing_campaign:
        if config_file["Data_Paths"]["campaign"]=="NARVAL-II":
            research_area=[] # use tropical typical coordinates, 
                             #   for that show in upcoming paper
            pass
        elif config_file["Data_Paths"]["campaign"]=="EUREC4A":
            research_area=[] # use tropical typical coordinates, 
                             # for that show in upcoming paper
            pass        
        elif (config_file["Data_Paths"]["campaign"]=="NAWDEX") \
        or (config_file["Data_Paths"]["campaign"]=="HALO_AC3_Dry_Run") \
            or (config_file["Data_Paths"]["campaign"]=="NA_February_Run")\
                or (config_file["Data_Paths"]["campaign"]=="Second_Synthetic_Study"):
            research_area=[90, -75, 40,  80]
            if flights[0]=="RF01":
                research_area=[90, -75, 20,  60]
            if (flights[0]=="SRF06") or (flights[0]=="SRF07"):
                research_area=[90, -40, 20,  100]
            if config_file["Data_Paths"]["campaign"]=="HALO_AC3_Dry_Run":
                research_area=[90, -75, 50,  80]
        elif (config_file["Data_Paths"]["campaign"]=="HALO_AC3"):
            if flights[0]=="RF01" or flights[0]=="RF02" or \
                flights[0]=="RF03" or flights[0]=="RF04" or \
                    flights[0]=="RF05" or \
                        flights[0]=="RF07" or\
                            flights[0]=="RF08":
                research_area=[90,-75,50,60]
            elif flights[0]=="RF06" or flights[0]=="RF16":
                research_area=[90,-90,60,100]
        era5_downloader=ERA5_Downloader(
                        dates, research_area,
                        for_flight_campaign=True,
                        campaign=config_file["Data_Paths"]["campaign"],
                        research_flights=None,
                        era_path=era_path)
        carra_downloader=CARRA_Downloader(dates, domain,
                        for_flight_campaign=True,
                        campaign=config_file["Data_Paths"]["campaign"],
                        research_flights=None,
                        carra_path=carra_path,initial_time=carra_initial)
    
    else:
        if config_file["Data_Paths"]["campaign"]=="Weather_Course":
            research_area=[56,5.5,47,15.5]
            era5_downloader=ERA5_Downloader(
                        dates, research_area,
                        for_flight_campaign=False,
                        campaign=config_file["Data_Paths"]["campaign"],
                        research_flights=None,
                        era_path=era_path)
    
    
    #%% Download_Handler
    # HMP Profile data
    if era_is_desired:
        era5_downloader.download_handler(do_total_columns=True,do_levels=True,
                                         do_daily_average_single_levels=False)
    if carra_is_desired:
        carra_downloader.download_handler(do_total_columns=True,do_levels=True,
                                          do_hydrometeors=False)
if __name__=="__main__":
    #main(campaign="NA_February_Run")
    #main(campaign="NA_February_Run")
    main(campaign="HALO_AC3")#"Second_Synthetic_Study")