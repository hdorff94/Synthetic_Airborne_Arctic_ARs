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
plotting_script_path=os.getcwd()+"/major_scripts/"
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

##### Create Figure 4

##### Create Figure 4

##### Create Figure 4

##### Create Figure 4

##### Create Figure 4

##### Create Figure 4

##### Create Figure 4

##### Create Figure 4

##### Create Figure 4

##### Create Figure 4


