# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:05:35 2022

@author: u300737
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

#def merge_carra_and_era():
    
current_path=os.getcwd()

ivt_overall_data_path=current_path+"/../../../Work/GIT_Repository/"

flight_dates={#"2016-10-13":["NAWDEX","RF10"],
              "2011-03-17":["Second_Synthetic_Study","SRF02"],
              "2011-04-23":["Second_Synthetic_Study","SRF03"],
              "2015-03-14":["Second_Synthetic_Study","SRF08"],
              "2016-03-11":["Second_Synthetic_Study","SRF09"],
              "2018-02-24":["NA_February_Run","SRF02"],
              "2018-02-25":["Second_Synthetic_Study","SRF12"],
              "2019-03-19":["NA_February_Run","SRF04"],
              "2020-04-16":["NA_February_Run","SRF07"],
              "2020-04-19":["NA_February_Run","SRF08"]}


AR_IVT_dict={}
CARRA_ERA_stats=pd.DataFrame(data=np.nan,
                             index=[*flight_dates.keys()],
                             columns=["Rel_mean","Rel_max","Rel_std"])
for rf in [*flight_dates.keys()]:
    print(rf)
    flight=flight_dates[rf][1]
    data_path=ivt_overall_data_path+flight_dates[rf][0]+"/data/"
    era_path=data_path+"/ERA-5/"
    carra_path=data_path+"/CARRA/"
    era_file=glob.glob(era_path+"*"+flight+"_SAR*HMP*")[0]
    carra_file=glob.glob(carra_path+"*"+flight+"_SAR*HMP*")[0]
    print(era_file)
    print(carra_file)
    era_ivt=pd.read_csv(era_file)
    carra_ivt=pd.read_csv(carra_file)
    era_ivt["CARRA_IVT"]=carra_ivt["Interp_IVT"]
    AR_IVT_dict[rf]=era_ivt
    CARRA_ERA_stats["Rel_mean"].loc[rf]=era_ivt["CARRA_IVT"].mean()/\
                era_ivt["Interp_IVT"].mean()
    CARRA_ERA_stats["Rel_max"].loc[rf]=era_ivt["CARRA_IVT"].max()/\
                era_ivt["Interp_IVT"].max()
    CARRA_ERA_stats["Rel_std"].loc[rf]=era_ivt["CARRA_IVT"].std()/\
                era_ivt["Interp_IVT"].std()
                
mean_CARRA_ERA_comparison=CARRA_ERA_stats.mean(axis=0)
    
    #print(data_path)