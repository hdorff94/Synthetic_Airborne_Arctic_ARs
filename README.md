# Synthetic_Airborne_Arctic_ARs 🌊 ❄️ 💦
This is the GIT repository used to investigate the divergence of moisture transport in nine exemplaric Atmospheric River (AR) cases using ERA5 and CARRA. By that, we want to assess the feasibility of deriving moisture transport divergence from sporadic airborne soundings. At the current state, this is done using synthetic sondes that are created within the reanalysis grid datasets for synthetically created zig-zag flight pattern.
 
# List what needed to run and import for recreating all the manuscript data and figures:
## Create Major AR cross-section data 💻
```python 
import os 
major_script_path=os.getcwd()+"/major_scripts/"
sys.path.insert(major_script_path,1)

import campaignAR_plotter

campaignAR_plotter.main()
# inside main you can specify the flights that are commented out and the subcampaign for variable campaign_name
```

## Figures in Chapters 🌐
most of them require the plotting modules
```python 
import os 
plotting_script_path=os.getcwd()+"/plotting/"
sys.path.insert(plotting_script_path,1)
```
### Introduction
no figures or data are explicitly presented in Sect. 1.
### Data
#### Figure 1 (AR Cases) 
```python 
import ar_cases_overview_map
# this creates the AR overview maps showing IVT and flight track with sea-ice edge and isobars for all nine AR events
```
#### Figure 2 (Climatological Reference)
```python 
import ar_ivt_climatology_plotting
# this creates the KDE-plot of mean AR-IVT as a function of AR centered latitude (long-time series 1979-2020).
# The nine ARs during observation periods are highlighted therein for climalogical framing.
```
### Methods
#### Figure 3 (Airborne Moisture Budget Observation)
This is a sketch of HALO observing AR boxes for the representative moisture budget components. This was created independently with image processing software.
#### Figure 4 (Flight Duration)
This is the aircraft flight time bar plot showing the duration for flying all AR cross-sections with given constant flight speed of 250 m/s.
Go to subpath "/notebooks/"
and open **Aircraft_Flight_Time.ipnyb**
#### Figure 5 (AR sectors approach)
This is the illustration of the AR sectors along the cross-section (warm prefrontal, core, cold postfrontal)
showing the positioning of seven synthetic sondes along the in- and outflow corridors used for divergence calculations. 
Go to subpath "/notebooks/" and open **AR_Sector_Illustration.ipnyb**
### IVT Variability
#### Figure 6 (AR-IVT shapes)
This is AR IVT shape multiplot indicating all nine inflow AR-IVT shapes. 
So far this plot is created manually by repeating the distance-based AR-IVT inflow cross-section plot. **Open issue to create a plot routine**
#### Figure 7 (Distance-based sonde representation of IVT)
This is the distance based AR-IVT together with synthetic soundings and gaussian fit for CARRA-IVT compared to ERA5-IVT 
```python 
import campaignAR_plotter

campaignAR_plotter.main(campaign="Second_Synthetic_Study",flights=["SRF02"],calc_hmp=True,
                        use_era=True,use_carra_True,do_plotting=True)
# inside main you can specify the flights that are commented out and the subcampaign for variable campaign_name
```
#### Figure 8 (Gaussian Fit)
This is the plot of AR-IVT along cross-section (so far with time on y-axis **need to be updated**) for 2011-04-23.
CARRA-IVT is shown as continuous representation together with sporadic sounding representation declining from 10 to 4 sondes and indicating TIVT.

#### Figure 9 (TIVT Sounding Frequency)
This is the TIVT dependency on sounding frequency. For that, the following is required to create the data:
```python 
import os 
script_path=os.getcwd()+"/scripts/"
sys.path.insert(script_path,1)

import run_sonde_freq_ivt_var_analysis

```
The figure as well as some correlations are created by the notebook **TIVT_IVT_Variability_analysis.ipynb**
#### Figure 10 (Vertical profiles of transport components) 
This is the vertical statistics of q,v, and moisture transport.
```python 
import wind_moisture_dominance_analysis

wind_moisture_dominance_analysis.main(figures_to_create="fig10")
```

#### Figure 11 (Variability Dominating Quantity) 
This shows the variability dominanting quantity for all the cases in a multiplot of vertical profiles.
```python 
import wind_moisture_dominance_analysis

wind_moisture_dominance_analysis.main(figures_to_create="fig11")
```
### IVT Divergence
#### Figure 12 (Frontal-specific IVT in- and outflow)
This shows the frontal sector based AR cross-sections for in- and outflow corridor in multiplot. 
It is created by running the notebook **AR_sector_multiplot_in-outflow.ipynb**  
This notebook basically runs multiplot_inflow_outflow_IVT sectors from ```python class IVT_Variability_Plotter```.
#### Figure 13 (Case Sector-Based Vertical Profiles of Divergence)
To create the divergence results just run:
```python 
import run_moisture_budget_closure_regression_method
```
the divergence values are then stored in the airborne data folder under budgets. Some additional figures are created and stored under "plots/budget/supplementary/"


