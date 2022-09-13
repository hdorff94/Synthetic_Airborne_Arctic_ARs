# Synthetic_Airborne_Arctic_ARs

This is the GIT repository used to investigate the divergence of moisture transport in nine exemplaric Atmospheric River (AR) cases using ERA5 and CARRA. By that, we want to assess the feasibility of deriving moisture transport divergence from sporadic airborne soundings. At the current state, this is done using synthetic sondes that are created within the reanalysis grid datasets for synthetically created zig-zag flight pattern.
 
### List what needed to run and import for recreating all the manuscript data and figures:
#### Create Major AR cross-section data
```python 
import os 
major_script_path=os.getcwd()+"/major_scripts/"
sys.path.insert(major_script_path,1)

import campaignAR_plotter

campaignAR_plotter.main()
# inside main you can specify the flights that are commented out and the subcampaign for variable campaign_name
```

#### Figures
most of them require the plotting modules
```python 
import os 
plotting_script_path=os.getcwd()+"/plotting/"
sys.path.insert(plotting_script_path,1)
```

##### Create Figure 1
```python 
import ar_cases_overview_map
# this creates the AR overview maps showing IVT and flight track with sea-ice edge and isobars for all nine AR events
```
##### Create Figure 2
```python 
import ar_ivt_climatology_plotting
# this creates the KDE-plot of mean AR-IVT as a function of AR centered latitude (long-time series 1979-2020).
# The nine ARs during observation periods are highlighted therein for climalogical framing.
```
##### Create Figure 3
This is a sketch of HALO observing AR boxes for the representative moisture budget components. This was created independently with image processing software.
##### Create Figure 4

```python 
import ar_ivt_climatology_plotting
# this creates the KDE-plot of mean AR-IVT as a function of AR centered latitude (long-time series 1979-2020).
# The nine ARs during observation periods are highlighted therein for climalogical framing.
```

##### Create Figure 4
This is the aircraft flight time bar plot showing the duration for flying all AR cross-sections with given constant flight speed of 250 m/s.
Go to subpath "/notebooks/"
and open **Aircraft_Flight_Time.ipnyb**
##### Create Figure 5
This is the illustration of the AR sectors along the cross-section (warm prefrontal, core, cold postfrontal)
showing the positioning of seven synthetic sondes along the in- and outflow corridors used for divergence calculations.  
##### Create Figure 6
This is AR IVT shape multiplot indicating all nine inflow AR-IVT shapes. 
So far this plot is created manually by repeating the distance-based AR-IVT inflow cross-section plot. **Open issue to create a plot routine**
##### Create Figure 7
This is the distance based AR-IVT together with synthetic soundings and gaussian fit for CARRA-IVT compared to ERA5-IVT 
```python 
import campaignAR_plotter

campaignAR_plotter.main(campaign="Second_Synthetic_Study",flights=["SRF02"],calc_hmp=True,
                        use_era=True,use_carra_True,do_plotting=True)
# inside main you can specify the flights that are commented out and the subcampaign for variable campaign_name
```
##### Create Figure 8
This is the plot of AR-IVT along cross-section (so far with time on y-axis **need to be updated**) for 2011-04-23.
CARRA-IVT is shown as continuous representation together with sporadic sounding representation declining from 10 to 4 sondes and indicating TIVT.

##### Create Figure 9
This is the TIVT dependency on sounding frequency. For that, the following is required to create the data:
```python 
import os 
script_path=os.getcwd()+"/scripts/"
sys.path.insert(script_path,1)

import run_sonde_freq_ivt_var_analysis

```

##### Create Figure 10

##### Create Figure 11

##### Create Figure 12

##### Create Figure 4


