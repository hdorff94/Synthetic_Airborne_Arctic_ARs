# Synthetic_Airborne_Arctic_ARs 🌊 ❄️ 💦
This is the git repository used to investigate the divergence of moisture transport in nine exemplaric Atmospheric River (AR) cases using ERA5 and CARRA. By that, we want to assess the feasibility of deriving moisture transport divergence from sporadic airborne soundings. At the current state, this is done using synthetic sondes that are created within the reanalysis grid datasets for synthetically created zig-zag flight pattern.
 
# List what needed to run and import for recreating all the manuscript data and figures:
## Create Major AR cross-section data 💻
```python 
import os 
major_script_path=os.getcwd()+"/major_scripts/"
sys.path.insert(major_script_path,1)

import campaignAR_plotter

campaignAR_plotter.main()
# inside main you can specify the flights that are commented out and the subcampaign for variable campaign_name

# Current dates used:
# "20110317","20110423","20150314","20160311","20180224","20180225","20190319","20200416","20200419"
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
import plot_ar_cases_overview()
plot_ar_cases_overview(figure_to_create="fig01")
# this creates the AR overview maps showing IVT and flight track with sea-ice edge and isobars for all nine AR events.
# Note it is important to choose "fig01" as figure_to_create
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

#### Figure 4 (AR sectors approach)
This is the illustration of the AR sectors along the cross-section (warm prefrontal, core, cold postfrontal)
showing the positioning of seven synthetic sondes along the in- and outflow corridors used for divergence calculations. 
Go to subpath "/notebooks/" and open **AR_Sector_Illustration.ipnyb**

### IVT variability
#### Figure 5 (Distance-based sonde representation of IVT)
This is the distance based AR-IVT together with synthetic soundings and gaussian fit for CARRA-IVT compared to ERA5-IVT for the AR1.
```python 
import campaignAR_plotter

campaignAR_plotter.main(campaign="Second_Synthetic_Study",flights=["SRF02"],calc_hmp=True,
                        use_era=True,use_carra_True,do_plotting=True)
# inside main you can specify the flights that are commented out and the subcampaign for variable campaign_name
```
#### Figure 6 (TIVT Sounding Frequency)
This is the TIVT dependency on sounding frequency. For that, the following is required to create the data:
```python 
import os 
script_path=os.getcwd()+"/scripts/"
sys.path.insert(script_path,1)

import run_sonde_freq_ivt_var_analysis
```
The figure as well as some correlations are created by the notebook **TIVT_IVT_Variability_analysis.ipynb**

#### Figure 7 (Vertical profiles of transport components) 
This is the vertical statistics of q,v, and moisture transport.
```python 
import wind_moisture_dominance_analysis

wind_moisture_dominance_analysis.main(figures_to_create="fig07")
```

#### Figure 08 (Variability Dominating Quantity) 
This depicts the vertical distribution of the variability components (cov_norm, s_v & s_q) for all inflow cross-sections with the intercase average. 
Since this was part of the ancient fig11, it has be called "updated_fig11" in the main plotting routine
```python 
import wind_moisture_dominance_analysis

wind_moisture_dominance_analysis.main(figures_to_create="updated_fig11")
```

#### Figure 09 (Vertical cross-section AR event comparison)
This contour multiplot illustrates the vertical cross-section curtains of all ARs in terms of moisture transport distribution and its components.
```python 
import plot_ar_cases_overview

plot_ar_cases_overview.main(figure_to_create="fig09")
```

### IVT Divergence
To create the divergence results just run:
```python 
import run_moisture_budget_closure_regression_method
```
This script itself calls ```Moisture_Convergence.calc_moisture_convergence_from_regression_method```
the divergence values are then stored in the airborne data folder under budgets. Some additional figures are created and stored under "plots/budget/supplementary/"
#### Figure 10 (Frontal-specific IVT in- and outflow)
This shows the frontal sector based AR cross-sections for in- and outflow corridor in multiplot. 
It is created by running the notebook **AR_sector_multiplot_in-outflow.ipynb**  
This notebook basically runs multiplot_inflow_outflow_IVT sectors from ```python class IVT_Variability_Plotter```.
#### Figure 11 (Case Sector-Based Vertical Profiles of Divergence)
To create the single case values just run:
```python 
import plot_moisture_budget_results
figure_to_create="Fig11_single_case_sector_profiles"
plot_moisture_budget_results.main(figure_to_create=figure_to_create)
```
Inside the plot routine defines ```save_for_manuscript=True```
#### Figure 12 (AR events averaged Sector-Based Divergence, Campaign Overview)
This creates a boxplot showing the sector-based divergence statistics for the nine events intercomparing continuous and sonde-based divegence calculations.
```python 
import plot_moisture_budget_results
figure_to_create="fig12_campaign_divergence_overviews"
plot_moisture_budget_results.main(figure_to_create=figure_to_create)
```
### Instationarity
This section includes three figures and deals with the instationarity and how this changes our understanding of ARs from airborne perspective.  
In order to create the instantan moisture transport divergence components you have to run (similar to **IVT divergence**):
```python 
import run_moisture_budget_closure_regression_method
# set instantan as True
run.moisture_budget_closure_regression_method.main(instantan=True)
```
This script itself calls ```Moisture_Convergence.calc_moisture_convergence_from_regression_method```

#### Figure 13 (Instantan In- and Outflow)
To create the multiplot of in- and outflow comparison (IVT) you have two choices:
```python 
import instantan
figure_to_create="fig13_in_outflow_instantan"
instantan.main(figure_to_create=figure_to_create)
```
or alternatively, you run the notebook ```AR_Stationarity.ipynb``` 
#### Figure 14 (Daily Contribution Comparison: Instantan, Time Propagating Error Divergence) 
%#### Figure 14 (Daily Contribution to Moisture Budget Divergence Non Instantan - Instantan)
%To create the multiplot with comparing non-instantan minus instantan as a vertical profile just run:
This figure compares the daily contribution of IVT divergence to daily moisture budget for flight duration with instantan values.
The comparison refers to the cross-frontal section and is similar built as Fig. 12. Using the same data for the flight
```python 
import plot_moisture_budgets
figure_to_create="fig14_divergence_instantan_errorbars""
plot_moisture_budgets.main(figure_to_create=figure_to_create)
```
This routine internally runs Budgets.Moisture_Budget_Plots.moisture_convergence_time_instantan_comparison() comparing both ideal-based (continuous) 
frontal-specific divergence components. 

For detailed flight specific information, check for **Figure S4,S5**
#### Figure 15 (IVT Convergence Error Sounding Error and Instantaneous)
So far this the partioned comparison for sounding frequency and instantaneous error arising comparing both magnitudes.
It is called as:
```python 
import plot_moisture_budgets
figure_to_create="fig15_campaign_divergence_overview_instantan_comparison"
plot_moisture_budgets.main(figure_to_create=figure_to_create)
```

#### Supplementary plots
##### Figure S1 (Flight Duration)
This is the aircraft flight time bar plot showing the duration for flying all AR cross-sections with given constant flight speed of 250 m/s.
Go to subpath "/notebooks/"
and open **Aircraft_Flight_Time.ipnyb**
##### Figure S2 (AR-IVT shapes)
This is AR IVT shape multiplot indicating all nine inflow AR-IVT shapes. 
So far this plot is created manually by repeating the distance-based AR-IVT inflow cross-section plot. **Open issue to create a plot routine**
##### Figure S3 (Fig 12 but for instantan perspectives)
Runned by:
```python 
import plot_moisture_budgets_results
figure_to_create="fig_supplements_sonde_pos_comparison"
plot_moisture_budgets_results.main(figure_to_create=figure_to_create)
```
From the Budgets module, this runs sonde_divergence_error_bar() which is also needed for Figure 12.

##### Figure S4 (Sonde positioning)
Runned by:
```python 
import plot_moisture_budgets_results
figure_to_create="fig_supplements_sonde_pos_comparison"
plot_moisture_budgets_results.main(figure_to_create=figure_to_create)
```
this runs two fcts with the first being Inst_Budget_plots.compare_inst_sonde_pos()
and illustrates the sonde positioning for the instantan and continuous representation. 
If the sounding locations change is specified in the Instantan & Budget class (default on_flight_tracks=True). 
This argument has to be given for Budgets.get_overall_budgets()
    
#### Figure S5 (Daily comparison of Moisture Budget contribution for Continuous and Instantan)
Runned by:
```python 
import plot_moisture_budgets_results
figure_to_create="fig_supplements_sonde_pos_comparison"
plot_moisture_budgets_results.main(figure_to_create=figure_to_create)
```
analogue to above, the function creates S4 and S5. The relevant for S5 is: Inst_Budget_plots.mean_errors_per_flight()
This shows the sector-based contributions of the divergence components for each flight. On top of that, it compares
the continuous flight propagating and instantaneous values. Again, the sonde positioning has to be defined (default on_flight_tracks=True).

%```python 
%import instantan
%figure_to_create="fig16_conv_error"
%instantan.main(figure_to_create=figure_to_create)
%# this itself calls
%instantan.cls.plot_div_tern_instantan_comparison("CONV",save_as_manuscript=False) # if last is set True figure is stored for manuscript
%```

%#### Figure 17 (ADV Non Instantan - Instantan
%Very similar to Fig 16, same plot style with slight adaptations:
%```python 
%import instantan
%figure_to_create="fig17_adv_error"
%instantan.main(figure_to_create=figure_to_create)
%# this itself calls
%instantan.cls.plot_div_tern_instantan_comparison("ADV",save_as_manuscript=False) # if last is set True figure is stored for manuscript
%```
