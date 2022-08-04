# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:05:50 2022

@author: u300737
"""

#Basics
import data_config
import os
# Calc Tools
import numpy as np
import pandas as pd

# Relevant created classes and modules
#import Flight_Campaign
#from Flight_Mapping import FlightMaps
from AR import Atmospheric_Rivers

#Grid Data
# Run routines
#import run_grid_data_on_halo # to run single days
import Campaign_AR_plotter # to run analysis for sequence of single days and create combined plots

# IVT variability
from IVT_Variability_handler import IVT_Variability_Plotter
#------------------------------------------------------------------------------#
# Plot scripts
import matplotlib.pyplot as plt

try:
    from typhon.plots import styles
except:
    print("Typhon module cannot be imported")

import Interp_Data_Plotting
import Moisture_Budget as Budgets
#sys.exit()
#-----------------------------------------------------------------------------#

import warnings
warnings.filterwarnings("ignore")

# Config File
config_file=data_config.load_config_file(os.getcwd(),"data_config_file")



#%% Get data from all flights
#
flight_dates={"North_Atlantic_Run":
              {"SRF02":"20180224",
               "SRF04":"20190319",#},
               "SRF07":"20200416",#},
               "SRF08":"20200419"
              },
              "Second_Synthetic_Study":
              {"SRF02":"20110317",
               "SRF03":"20110423",
               "SRF08":"20150314",
               "SRF09":"20160311",
               "SRF12":"20180225"
               }}

# Major configurations
campaign="Second_Synthetic_Study"
#"North_Atlantic_Run"#"Second_Synthetic_Study"#"North_Atlantic_Run"### 
do_plotting=False
instantan=True

# 
calc_hmp=True
calc_hmc=False
# What to use
use_era=True
use_carra=True
use_icon=False
grid_name="ERA5"

if use_icon:
    grid_name="ICON_2km"
elif use_carra:
    grid_name="CARRA"

flights=[*flight_dates[campaign].keys()]

Hydrometeors,HALO_Dict,cmpgn_cls=Campaign_AR_plotter.main(campaign=campaign,flights=flights,
                                          era_is_desired=use_era, 
                                          icon_is_desired=use_icon,
                                          carra_is_desired=use_carra,
                                          do_daily_plots=do_plotting,
                                          calc_hmp=calc_hmp,calc_hmc=calc_hmc,
                                          do_instantaneous=instantan)

HMCs,HALO_Dict,cmpgn_cls=Campaign_AR_plotter.main(campaign=campaign,flights=flights,
                                          era_is_desired=use_era, 
                                          icon_is_desired=use_icon,
                                          carra_is_desired=use_carra,
                                          do_daily_plots=do_plotting,
                                          calc_hmp=False,calc_hmc=True,
                                          do_instantaneous=instantan)


budget_data_path=cmpgn_cls.campaign_data_path+"budget/"
if not os.path.exists(budget_data_path):
    os.makedirs(budget_data_path)
            
budget_plot_path=cmpgn_cls.plot_path+"budget/"
if not os.path.exists(budget_plot_path):
    os.makedirs(budget_plot_path)

            
for flight in flights:
    if instantan:
        flight=flight+"_instantan"
        analysed_flight=flight.split("_")[0]
    else:
        analysed_flight=flight        
    Moisture_CONV=\
    Budgets.Moisture_Convergence(cmpgn_cls,flight,config_file,
                 grid_name=grid_name,do_instantan=instantan)
    Budget_plots=Budgets.Moisture_Budget_Plots(cmpgn_cls,flight,config_file,
                 grid_name=grid_name,do_instantan=instantan)

    for sector in ["cold_sector","core","warm_sector"]:
        if flight.startswith("SRF12"):
            if sector=="cold_sector":
                continue
        for number_of_sondes in [2,100]:    
            print(flight)
            #flight_dates=["2016"]
            ar_of_day="SAR_internal"
            working_path=os.getcwd()
            grid_name=Hydrometeors[analysed_flight]["AR_internal"].name
            
            AR_inflow,AR_outflow=Atmospheric_Rivers.locate_AR_cross_section_sectors(
                                    HALO_Dict,Hydrometeors,analysed_flight)
            TIVT_inflow,TIVT_outflow=Atmospheric_Rivers.calc_TIVT_of_sectors(
                                    AR_inflow,AR_outflow,grid_name)
            
            #%%
            #sector="core" # warm_sector, # cold_sector
            # Sonde number
            #number_of_sondes=2
            sondes_selection={}
            sondes_selection["inflow_"+sector]=np.linspace(0,AR_inflow["AR_inflow_"+sector].shape[0]-1,
                                                        num=number_of_sondes).astype(int)
            sondes_selection["outflow_"+sector]=np.linspace(0,AR_outflow["AR_outflow_"+sector].shape[0]-1,
                                                         num=number_of_sondes).astype(int)
            #%% Loc and locate sondes for regression method
            inflow_sondes_times=AR_inflow["AR_inflow_"+sector].index[\
                                    sondes_selection["inflow_"+sector]]
            outflow_sondes_times=AR_outflow["AR_outflow_"+sector].index[\
                                    sondes_selection["outflow_"+sector]]
            
            sondes_pos_inflow=AR_inflow["AR_inflow_"+sector][\
                                ["Halo_Lat","Halo_Lon"]].loc[inflow_sondes_times]
            sondes_pos_outflow=AR_outflow["AR_outflow_"+sector][\
                                ["Halo_Lat","Halo_Lon"]].loc[outflow_sondes_times]
            sondes_pos_all=pd.concat([sondes_pos_inflow,sondes_pos_outflow])
            #%%
            if not "q" in HMCs[analysed_flight]["AR_internal"].keys():
                HMCs[analysed_flight]["AR_internal"]["q"]=\
                    HMCs[analysed_flight]["AR_internal"]["specific_humidity"].copy()
            q_inflow_sondes=HMCs[analysed_flight]["AR_internal"]["q"].loc[\
                                                    inflow_sondes_times]
            q_outflow_sondes=HMCs[analysed_flight]["AR_internal"]["q"].loc[\
                                                    outflow_sondes_times]
            
            u_inflow_sondes=HMCs[analysed_flight]["AR_internal"]["u"].loc[\
                                                    inflow_sondes_times]
            u_outflow_sondes=HMCs[analysed_flight]["AR_internal"]["u"].loc[\
                                                    outflow_sondes_times]
            
            v_inflow_sondes=HMCs[analysed_flight]["AR_internal"]["v"].loc[\
                                                    inflow_sondes_times]
            v_outflow_sondes=HMCs[analysed_flight]["AR_internal"]["v"].loc[\
                                                    outflow_sondes_times]
            
            wind_inflow_sondes=np.sqrt(u_inflow_sondes**2+\
                                       v_inflow_sondes**2)
            
            moist_transport_inflow=q_inflow_sondes*wind_inflow_sondes
            
            wind_outflow_sondes=np.sqrt(u_outflow_sondes**2+\
                                        v_outflow_sondes**2)
            
            moist_transport_outflow=q_outflow_sondes*wind_outflow_sondes
            
            ###################################################################
            #%%
            # Old rough stuff
            ar_inflow=AR_inflow["AR_inflow"]
            ar_outflow=AR_outflow["AR_outflow"]

            Budget_plots.plot_AR_TIVT_cumsum_quicklook(
                ar_inflow,ar_outflow)
            
            IVT_Variability_Plotter.plot_inflow_outflow_IVT_sectors(cmpgn_cls,
                                                        AR_inflow,AR_outflow,
                                                        TIVT_inflow,TIVT_outflow,
                                                        grid_name,flight)
            HMCs[analysed_flight]["AR_internal"]["wind"]=np.sqrt(
                HMCs[analysed_flight]["AR_internal"]["u"]**2+\
                    HMCs[analysed_flight]["AR_internal"]["v"]**2)

            q_field=HMCs[analysed_flight]["AR_internal"]["q"].copy()
            wind_field=HMCs[analysed_flight]["AR_internal"]["wind"].copy()

            moisture_transport=q_field*wind_field

            q_sector_inflow=q_field.loc[AR_inflow["AR_inflow_"+sector].index]
            q_sector_outflow=q_field.loc[AR_outflow["AR_outflow_"+sector].index]

            wind_sector_inflow=wind_field.loc[AR_inflow["AR_inflow_"+sector].index]
            wind_sector_outflow=wind_field.loc[AR_outflow["AR_outflow_"+sector].index]

            moist_transport_sector_inflow=moisture_transport.loc[\
                                                AR_inflow["AR_inflow_"+sector].index]
            moist_transport_sector_outflow=moisture_transport.loc[\
                                                AR_outflow["AR_outflow_"+sector].index]

            pressure=q_field.columns.astype(float)
            Moisture_CONV.run_rough_budget_closure(wind_field,q_field,moisture_transport,
                                 wind_sector_inflow,wind_sector_outflow,
                                 q_sector_inflow,q_sector_outflow,
                                 moist_transport_sector_inflow,
                                 moist_transport_sector_outflow,pressure,
                                 AR_inflow,AR_outflow,sector=sector)
            ###################################################################
            #%%
            ### Prepare the pattern for regression method
            
            sondes_pos_all=Moisture_CONV.get_xy_coords_for_domain(sondes_pos_all)
            domain_values={}
            moist_transport_inflow=moist_transport_inflow.groupby(level=0).last()
            moist_transport_outflow=moist_transport_outflow.groupby(level=0).last()
            u_inflow_sondes=u_inflow_sondes.groupby(level=0).last()
            u_outflow_sondes=u_outflow_sondes.groupby(level=0).last()
            v_inflow_sondes=v_inflow_sondes.groupby(level=0).last()
            v_outflow_sondes=v_outflow_sondes.groupby(level=0).last()
            q_inflow_sondes=q_inflow_sondes.groupby(level=0).last()
            q_outflow_sondes=q_outflow_sondes.groupby(level=0).last()
            wind_inflow_sondes=wind_inflow_sondes.groupby(level=0).last()
            wind_outflow_sondes=wind_outflow_sondes.groupby(level=0).last()
            domain_values["transport"]=pd.concat([moist_transport_inflow,
                                                  moist_transport_outflow])
            domain_values["u"]=pd.concat([u_inflow_sondes,
                                                  u_outflow_sondes])
            
            domain_values["v"]=pd.concat([v_inflow_sondes,
                                                  v_outflow_sondes])
            domain_values["q"]=pd.concat([q_inflow_sondes,
                                                  q_outflow_sondes])
            domain_values["wind"]=pd.concat([wind_inflow_sondes,
                                                  wind_outflow_sondes])
            
            mean_u,dx_u,dy_u=Moisture_CONV.run_regression(sondes_pos_all,domain_values,
                                               "u")
            mean_v,dx_v,dy_v=Moisture_CONV.run_regression(sondes_pos_all,domain_values,
                                               "v")
            
            mean_qv,dx_qv,dy_qv=Moisture_CONV.run_regression(sondes_pos_all,domain_values,
                                               "transport")
            
            mean_q,dx_q_calc,dy_q_calc=Moisture_CONV.run_regression(sondes_pos_all,domain_values,
                                               "q")
            
            mean_scalar_wind,dx_scalar_wind,dy_scalar_wind=Moisture_CONV.run_regression(
                                    sondes_pos_all,domain_values,"wind")
            
            div_qv=(dx_qv+dy_qv)*1000
            #div_wind=(dx_u+dy_v)
            div_scalar_wind=(dx_scalar_wind+dy_scalar_wind)
            #div_mass=div_wind*domain_values["q"].mean(axis=0).values*1000
            div_scalar_mass=div_scalar_wind*\
                domain_values["q"].mean(axis=0).values*1000
            adv_q_calc=(dx_q_calc+dy_q_calc)*\
                domain_values["wind"].mean(axis=0).values*1000
            #adv_q=div_qv-div_mass
            adv_q_scalar=div_qv-div_scalar_mass
            
            # Plot the divergence results
            Budget_plots.plot_single_flight_and_sector_regression_divergence(
                                                    sector,number_of_sondes,
                                                    div_qv,div_scalar_mass,
                                                    adv_q_calc,adv_q_scalar)
            
            # Save the data
            budget_regression_profile_df=pd.DataFrame(data=np.nan,index=div_qv.index,
                                                      columns=["CONV","ADV","TRANSP"])
            budget_regression_profile_df["CONV"]=div_scalar_mass.values
            budget_regression_profile_df["ADV"]=adv_q_calc.values
            budget_regression_profile_df["TRANSP"]=div_qv.values
            
                
            budget_file_name=flight+"_AR_"+sector+"_"+grid_name+"_regr_sonde_no_"+\
                str(number_of_sondes)+".csv"
            budget_regression_profile_df.to_csv(path_or_buf=budget_data_path+budget_file_name)    