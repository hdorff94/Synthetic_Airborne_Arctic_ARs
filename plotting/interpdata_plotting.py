# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:17:50 2021

@author: u300737
"""
import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from reanalysis import ERA5 as ERA5
from reanalysis import CARRA as CARRA
from ICON import ICON_NWP as ICON

from matplotlib.legend_handler import HandlerBase

#@staticmethod
def ar_cross_sections_overview_flights_vertical_profile(
        flight_dates,use_era,use_carra,
        use_icon,na_flights,snd_flights,do_meshing=True,
        use_cmasher=True):
    from matplotlib.ticker import NullFormatter
    
    ar_of_day=["AR_internal"]
    import campaignAR_plotter
    NA_Hydrometeors,NA_HALO_Dict,NA_cmpgn_cls=campaignAR_plotter.main(
                                        campaign="North_Atlantic_Run",
                                        flights=na_flights,
                                        ar_of_days=ar_of_day,
                                        era_is_desired=use_era, 
                                        icon_is_desired=use_icon,
                                        carra_is_desired=use_carra,
                                        do_daily_plots=False,
                                        calc_hmp=False,calc_hmc=True,
                                        do_instantaneous=False)
        
    SND_Hydrometeors,SND_HALO_Dict,SND_cmpgn_cls=campaignAR_plotter.main(
                                        campaign="Second_Synthetic_Study",
                                        flights=snd_flights,
                                        ar_of_days=ar_of_day,
                                        era_is_desired=use_era, 
                                        icon_is_desired=use_icon,
                                        carra_is_desired=use_carra,
                                        do_daily_plots=False,
                                        calc_hmp=False,calc_hmc=True,
                                        do_instantaneous=False)
            
    key_list=[*NA_Hydrometeors.keys()]
    for key in key_list:
        new_dict_entry=int(flight_dates["North_Atlantic_Run"][key])
        print(new_dict_entry)
        NA_Hydrometeors[new_dict_entry]=NA_Hydrometeors[key]
        NA_HALO_Dict[new_dict_entry]=NA_HALO_Dict[key]
        del NA_Hydrometeors[key], NA_HALO_Dict[key]

    key_list=[*SND_Hydrometeors.keys()]
    for key in key_list:
        new_dict_entry=int(flight_dates["Second_Synthetic_Study"][key])
        SND_Hydrometeors[new_dict_entry]=SND_Hydrometeors[key]
        SND_HALO_Dict[new_dict_entry]=SND_HALO_Dict[key]
        del SND_Hydrometeors[key], SND_HALO_Dict[key]

    campaign_Hydrometeors= dict(list(NA_Hydrometeors.items()) +\
                                list(SND_Hydrometeors.items()))
    campaign_Hydrometeors=dict(sorted(campaign_Hydrometeors.items()))

    campaign_HALO = dict(list(NA_HALO_Dict.items()) +\
                             list(SND_HALO_Dict.items())) 

    campaign_HALO=dict(sorted(campaign_HALO.items()))
    import matplotlib
    import matplotlib.pyplot as plt
    humidity_colormap   = "terrain_r"
    if use_cmasher:
        import cmasher as cmr
        humidity_colormap = "cmr.rainforest_r" 
    import seaborn as sns
    font_size=14
    matplotlib.rcParams.update({'font.size': font_size})
    cross_section_fig,ax=plt.subplots(figsize=(18,12),nrows=3,ncols=3)
    axes=ax.flatten()
    p=0
    plot_labels=["(a)","(b)","(c)",
                 "(d)","(e)","(f)",
                 "(g)","(h)","(i)"]
            
    for date in campaign_Hydrometeors.keys():
        print("Plotting Moisture Transport components for date ",date)
        center_int_idx=int(campaign_HALO[date]["inflow"].shape[0]/2)
        center_distance=campaign_HALO[date]["inflow"]["distance"].iloc[\
                                                        center_int_idx]
        distance=campaign_HALO[date]["inflow"]["distance"]-\
                        center_distance
            
        #Then tick and format with matplotlib:
        fig=plt.figure(figsize=(16,12))
        
        inflow_section_index=campaign_HALO[date]["inflow"].index
        try:
            moisture=campaign_Hydrometeors[date]["AR_internal"]["q"].loc[\
                                                        inflow_section_index]
        except:
            moisture=campaign_Hydrometeors[date]["AR_internal"][\
                                                "specific_humidity"].loc[\
                                                        inflow_section_index]
        wind=np.sqrt(campaign_Hydrometeors[date]["AR_internal"]["u"].loc[\
                                                        inflow_section_index]**2+\
        campaign_Hydrometeors[date]["AR_internal"]["v"].loc[\
                                                        inflow_section_index]**2)
        pressure=campaign_Hydrometeors[date]["AR_internal"]["u"].\
                                    columns.astype(float)
        
        corr_levels=pd.Series(data=np.nan,index=pressure)
        for height in corr_levels.index:
            corr_levels.loc[height]=wind[str(height)].corr(
                                        moisture[str(height)])
        
        # calc the correlation    
        pres_start=0#8
        # Specific humidity
        q_min=0
        q_max=5
        
        # Add wind 
        x_temp=distance
        y_temp=pressure[pres_start::]
        y,x=np.meshgrid(y_temp,x_temp)
        
        #Create new Hydrometeor content as a sum of all water contents
        moisture=moisture.replace(to_replace=np.nan,value=0.0)
        if do_meshing:
            Ci=axes[p].pcolormesh(x/1000,y,
                moisture.iloc[:,pres_start::].values*1000,
                vmin=q_min,vmax=q_max,
                cmap=humidity_colormap)    
        else:
            q_levels=np.linspace(0,5,100)
            Ci=axes[p].contourf(x/1000,y,
                moisture.iloc[:,pres_start::].values*1000,
                levels=q_levels,cmap=cm.get_cmap(
                    humidity_colormap,len(q_levels)-1),extend="max")
        
        upper_axes=axes[p].twiny()
        upper_axes.plot(corr_levels,pressure,ls="-",lw=3,color="k",ms=5)
        upper_axes.plot(corr_levels,pressure,ls="-.",lw=2,color="white",ms=5)
        
        upper_axes.invert_yaxis()
        upper_axes.set_xlim([-1.5,1.5])
        for corner in ["bottom","left","right","top"]:
            upper_axes.spines[corner].set_linewidth(0)
            upper_axes.set_xticks([])
        if p<3:
            upper_axes.spines["top"].set_linewidth(2)
            upper_axes.tick_params(length=4,width=2)
            upper_axes.set_xticks([-1.0,-0.5,0,0.5,1.0])
        moisture_levels=[5,10,20,30,40]
        wind_levels=[15,25,35]
        axes[p].set_xlim([-500,500])
        axes[p].set_xticks([-500,-250,0,250,500])
        if p<6:
            axes[p].set_xticklabels("")
        if p==1:
            upper_axes.set_xlabel("Correlation coefficient $r_{\mathrm{corr}}$",
                                  fontsize=font_size+4)
        if p==3:
            axes[p].set_ylabel("Pressure (hPa)",fontsize=font_size+4)
        if p==7:
            axes[p].set_xlabel("Lateral Distance (km)",fontsize=font_size+4)
        axes[p].text(0.015,0.8,"AR"+str(p+1),color="k",
                     transform=axes[p].transAxes,
                     bbox=dict(facecolor='lightgrey', edgecolor='black', 
                               boxstyle='round,pad=0.2'))
        axes[p].text(0.015,0.9,plot_labels[p],color="k",
                     transform=axes[p].transAxes)
        #axes[p].text(0.015,0.1,"$\overline{r_{corr}}="+\
        #             str(np.round(corr_levels.mean()),2),
        #             transform=axes[p].transAxes,
        #             bbox=dict(facecolor="lightgrey",edgecolor="k",
        #                       boxstyle="round,pad=0.2"))
        CS2=axes[p].contour(x/1000,y,wind.iloc[:,pres_start::],
                           levels=wind_levels,colors=["grey","plum","magenta"],
                           linestyles="--",linewidths=3.0)
        
        axes[p].invert_yaxis()
        axes[p].set_yscale("log")
        axes[p].yaxis.set_minor_formatter(NullFormatter())
        
        if p%3==0:
            axes[p].set_yticks([1000,850,700,500])
            axes[p].set_yticklabels(["1000","850","700","500"])
            
        else:
            axes[p].set_yticks([1000,850,700,500])
            axes[p].set_yticklabels("")
        
        for axis in ["left","bottom"]:
            axes[p].spines[axis].set_linewidth(2)
            axes[p].tick_params(length=4,width=2)
        
        #ax[1].set_xlabel("Relative standard deviation")
        
        axes[p].axvline(0,ls="--",color="k")
        axes[p].clabel(CS2,fontsize=16,fmt='%1d $\mathrm{ms}^{-1}$',inline=1)
        wv_flux=moisture*wind #halo_era5_hmc["wind"]
        moisture_flux=1/9.82*wv_flux*1000
        CS=axes[p].contour(x/1000,y,moisture_flux.iloc[:,pres_start::],
                       levels=moisture_levels,colors="k",
                       linestyles=["--","-"],linewidths=1.0)
        axes[p].clabel(CS,fontsize=16,fmt='%1.1f',inline=1)
        axes[p].set_ylim([1000,400])
        sns.despine(ax=axes[p],offset=10)
        p+=1
        
    cbar_ax = cross_section_fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb=cross_section_fig.colorbar(Ci, cax=cbar_ax)
    cb.set_label("Specific humidity (g/kg)",fontsize=font_size+4)
    cb.set_ticks([0,1,2,3,4,5])
    plt.subplots_adjust(hspace=0.1)
    if use_era:
        grid_name="ERA5_"
    if use_carra:
        grid_name="CARRA_"
    # add a overall colorbar for specific humidity    
    supplement_path=SND_cmpgn_cls.plot_path+\
                "/../../../Synthetic_AR_Paper/Manuscript/Paper_Plots/"
    fig_name="fig10_"+grid_name+"AR_inflow_cross_sections_overview.png"
    cross_section_fig.savefig(supplement_path+fig_name,
                              dpi=300,bbox_inches="tight")
    print("Figure saved as: ",supplement_path+fig_name)
        
        
class HandlerBoxPlot(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,trans):
        a_list = []
        a_list.append(matplotlib.lines.Line2D(
                        np.array([0, 0, 1, 1, 0])*width-xdescent, 
                        np.array([0.25, 0.75, 0.75, 0.25, 0.25])*\
                            height-ydescent)) # box

        a_list.append(matplotlib.lines.Line2D(
                        np.array([0.5,0.5])*width-xdescent,
                        np.array([0.75,1])*height-ydescent)) # top vert line

        a_list.append(matplotlib.lines.Line2D(
                        np.array([0.5,0.5])*width-xdescent,
                        np.array([0.25,0])*height-ydescent)) # bottom vert line

        a_list.append(matplotlib.lines.Line2D(
                        np.array([0.25,0.75])*width-xdescent,
                        np.array([1,1])*height-ydescent)) # top whisker

        a_list.append(matplotlib.lines.Line2D(
                        np.array([0.25,0.75])*width-xdescent,
                        np.array([0,0])*height-ydescent)) # bottom whisker

        a_list.append(matplotlib.lines.Line2D(
                        np.array([0,1])*width-xdescent,
                        np.array([0.5,0.5])*height-ydescent, lw=2)) # median
        for a in a_list:
            a.set_color(orig_handle.get_color())
        return a_list

class ERA_HALO_Plotting(ERA5):
    def __init__(self,flight,ar_of_day=None,plot_path=os.getcwd(),
                 synthetic_campaign=False):
        self.flight=flight[0]
        self.ar_of_day=ar_of_day
        self.plot_path=plot_path
        self.synthetic_campaign=synthetic_campaign
    def specify_plotting(self):
    
        #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!############
        pd.plotting.register_matplotlib_converters()
        #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!############
        
        import cartopy.crs as ccrs
        set_font=22
        matplotlib.rcParams.update({'font.size':set_font})
        return set_font

    def plot_radar_era5_time_series(self,radar,halo_era5,dropsondes,
                                    flight,plot_path,save_figure=False):
        
        if not self.synthetic_campaign:
            fig,axs=plt.subplots(2,1,figsize=(16,9),sharex=True)
        else:
            axs=[]
            fig,single_ax=plt.subplots(figsize=(16,9))
            axs.append(single_ax)
        axs[0].plot(halo_era5["Interp_LWP"].index.time,halo_era5["Interp_LWP"],
                    color="red",label="LWP")
        axs[0].plot(halo_era5["Interp_IWP"].index.time,halo_era5["Interp_IWP"],
                    color="blue",label="IWP")
        axstwin=axs[0].twinx()
        
        axstwin.plot(halo_era5["Interp_IWV"].index.time,halo_era5["Interp_IWV"],
                     color="orange",label="IWV")
        if not dropsondes=={}:
            if not flight[0]=="RF08":
                dropsondes["IWV"]=dropsondes["IWV"].loc[halo_era5.index[0]:
                                                halo_era5.index[-1]]
                # RF08 has only one or no (?) dropsonde which makes the plotting
                # more complicated
                axstwin.plot(dropsondes["IWV"].index.time,
                         np.array(dropsondes["IWV"]),
                         linestyle='',markersize=10,marker='v',color="orange",
                         markeredgecolor="black")
                axstwin.set_ylabel("ERA5, Sondes:\n IWV (kg$\mathrm{m}^{-2}$)")
        else:
            axstwin.set_ylabel("ERA5 IWV (kg$\mathrm{m}^{-2}$)")
        axs[0].set_ylabel("Hydrometeor Path (g$\mathrm{m}^{-2}$)")
        axstwin.legend(loc="upper right")
        if not self.synthetic_campaign:
            hamp={}
            hamp["Reflectivity"]=radar["Reflectivity"]\
                                    .loc[halo_era5["Interp_IWV"].index[0]:\
                                         halo_era5["Interp_IWV"].index[-1]]
            y=radar["height"][:]#/1000
            print("Start plotting HAMP Cloud Radar")
            axs[1].set_yticks([0,2000,4000,6000,8000,10000])
            levels=np.arange(-32,32.0,1.0)# 
            try:
                C1=axs[1].contourf(hamp["Reflectivity"].index.time,
                        y,hamp["Reflectivity"].T,levels,
                        cmap=cm.get_cmap("temperature",len(levels)-1),
                        extend="both")
                print("Typhon color schemes")
            
            except:
                print("Standard color schemes")
                C1=axs[1].contourf(hamp["Reflectivity"].index.time,
                               y,hamp["Reflectivity"].T,levels,
                               cmap=cm.get_cmap('viridis',len(levels)-1),
                               extend="both")
            
            for label in axs[1].xaxis.get_ticklabels()[::2]:
                    label.set_visible(False)
            cb = fig.colorbar(C1,ax=axs[1],
                              orientation="horizontal",shrink=0.5)
            cb.set_label('Reflectivity (dBZ)')
            labels = levels[::8]
            cb.set_ticks(labels)
            axs[1].set_xlabel('')
            axs[1].set_ylabel('Altitude (m)')
            axs[1].set_yticklabels(["0","2000","4000","6000","8000","10000"])
            axs[1].set_ylim([0,10000])
            for axis in ["left","bottom"]:
                axs[1].spines[axis].set_linewidth(3)
                axs[1].tick_params(length=10,width=3)
        
            
        axs[0].set_xlabel('Time (UTC)')
        ymax=500
        if halo_era5["Interp_LWP"].max()>500 or \
           halo_era5["Interp_IWP"].max()>500:
            ymax=800
        axs[0].set_ylim([0,ymax])
        for axis in ["left","bottom"]:
                axs[0].spines[axis].set_linewidth(3)
                axs[0].tick_params(length=10,width=3)
        
        axs[0].legend(loc="upper left")
        sns.despine(offset=10)

        plt.subplots_adjust(hspace=0.2)
        figname="HALO_ERA5_"+self.flight+".png"    
        if self.ar_of_day!=None:
            figname=self.ar_of_day+"_"+figname
        if save_figure:
            fig.savefig(plot_path+figname,dpi=300,bbox_inches="tight")
            print("Figure saved as: ",plot_path+figname)
        else:
            print(figname, "not saved as file")
            
        return None
    
    
    def plot_hamp_reflectivity_quicklook(self,HAMP,start,end,plot_path,savefig=False):
        fig=plt.figure(figsize=(16,7))
        matplotlib.rcParams.update({'font.size':22})
        hamp=HAMP.copy()
        hamp["Reflectivity"]=hamp["Reflectivity"].loc[start:end]
        ax1=fig.add_subplot(111)
        y=HAMP["Heights"][:]#/1000
        print("Start plotting HAMP Cloud Radar")
        levels=np.arange(-30,+20.0,2.0)# 
        C1=ax1.contourf(hamp["Reflectivity"].index.time,y,hamp["Reflectivity"].T,
                        levels,cmap=cm.get_cmap('viridis',len(levels)-1),extend="both")
        for label in ax1.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        cb = plt.colorbar(C1)
        cb.set_label('Reflectivity in dBZ')
        labels = levels[::2]
        cb.set_ticks(labels)   
        ax1.set_xlabel('Time in UTC')
        ax1.set_ylabel('Altitude in m ')
        ax1.grid()
        ax1.set_ylim(0,10000)
        return None    
        
    def plot_radar_era5_hwc(self,radar,halo_era5_hmc,flight,path,
                            colormap,save_figure=True):    
        fig=plt.figure(figsize=(16,30))
        ax1=fig.add_subplot(411)
        x_temp=halo_era5_hmc["IWC"].index.time
        y_temp=np.log10(np.array(halo_era5_hmc["IWC"].columns.astype(float)))#/1000
        y,x=np.meshgrid(y_temp,x_temp)
        levels=np.linspace(-4,1,50)
        C1=ax1.pcolormesh(x,y,(halo_era5_hmc["IWC"]*1000),
                          norm=colors.LogNorm(vmin=1e-4,vmax=1e1),cmap=colormap)
        for label in ax1.xaxis.get_ticklabels()[::5]:
            label.set_visible(False)
            #ax1.set_xticks([HAMP["Reflectivity"].index.time[::4]])
        cb = plt.colorbar(C1,extend="both")
        cb.set_label('IWC in g/kg')
        #ax1.set_yscale("log")
        ax1.set_ylim([np.log10(200),3])
        ax1.set_yticks([np.log10(200),np.log10(300),np.log10(500),
                        np.log10(700),np.log10(850),np.log10(1000)])
        ax1.set_yticklabels([200,300,500,700,850,1000])
        ax1.set_xlabel('')
        ax1.invert_yaxis()
        
        ax2=fig.add_subplot(412)
        #y=np.array(halo_era5_hmc["LWC"].columns.astype(float))#/1000
        levels=np.linspace(-4,1,50)
        C2=ax2.pcolormesh(x,y,(halo_era5_hmc["LWC"]*1000),
                          norm=colors.LogNorm(vmin=1e-4,vmax=1e1),cmap=colormap)
        for label in ax2.xaxis.get_ticklabels()[::5]:
            label.set_visible(False)
            #ax1.set_xticks([HAMP["Reflectivity"].index.time[::4]])
        cb = plt.colorbar(C2,extend="both")
        cb.set_label('LWC in  g/kg')
        #ax1.set_yscale("log")
        ax2.set_ylim([np.log10(200),3])
        ax2.set_yticks([np.log10(200),np.log10(300),np.log10(500),
                        np.log10(700),np.log10(850),np.log10(1000)])
        ax2.set_yticklabels([200,300,500,700,850,1000])
        ax2.set_xlabel('')
        ax2.invert_yaxis()
        
        ax3=fig.add_subplot(413)
        #y=np.array(halo_era5_hmc["PWC"].columns.astype(float))#/1000
        levels=np.linspace(-4,1,50)
        C3=ax3.pcolormesh(x,y,(halo_era5_hmc["PWC"]*1000), 
                          norm=colors.LogNorm(vmin=1e-4,vmax=1e1),cmap=colormap)
    
        for label in ax3.xaxis.get_ticklabels()[::5]:
            label.set_visible(False)
            #ax1.set_xticks([HAMP["Reflectivity"].index.time[::4]])
        cb = plt.colorbar(C3,extend="both")
        cb.set_label('PWC in g/kg')
        #ax1.set_yscale("log")
        ax3.set_ylim([np.log10(200),3])
        ax3.set_yticks([np.log10(200),np.log10(300),np.log10(500),np.log10(700),
                        np.log10(850),np.log10(1000)])
        ax3.set_yticklabels([200,300,500,700,850,1000])
        ax3.set_xlabel('')
        ax3.invert_yaxis()
        ax4=fig.add_subplot(414)
        y=np.array(radar["height"][:])#/1000
        print("Start plotting HAMP Cloud Radar")
        levels=np.arange(-30,+20.0,2.0)# 
        C4=ax4.contourf(radar["Reflectivity"].index.time,y,radar["Reflectivity"].T,
                            levels,cmap=cm.get_cmap(colormap,len(levels)-1),extend="both")
        for label in ax1.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            #ax1.set_xticks([HAMP["Reflectivity"].index.time[::4]])
        cb = plt.colorbar(C4)
        cb.set_label('Reflectivity in dBZ')
        labels = levels[::2]
        cb.set_ticks(labels)   
            #ax1.plot(HAMP["Reflectivity"].index.time,HAMP["Halo"]['Altitude'][:],linewidth=2.0,color='black')
            #ax1.set_title('HAMP: Reflectivity factor') 
        ax4.set_xlabel('Time in UTC')
        ax4.set_ylabel('Altitude in m ')
        #ax4.grid()
        ax4.set_ylim(0,12000)
        #    labels = levels[::2]
        #    cb.set_ticks(labels)   
            #ax1.plot(HAMP["Reflectivity"].index.time,HAMP["Halo"]['Altitude'][:],linewidth=2.0,color='black')
            #ax1.set_title('HAMP: Reflectivity factor') #
        #    ax1.set_xlabel('Time in UTC')
        #    ax1.set_ylabel('Altitude in m ')
        #    ax1.grid()
        fig_name="HMC_HALO_ERA5_"+flight+".png"
        fig.savefig(path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", fig_name)
        return None
    
    def plot_radar_era5_combined_hwc(self,radar,halo_era5_hmc,flight_date,
                                     colormap,save_figure=True):    
        
        fig=plt.figure(figsize=(16,14))
        if not self.synthetic_campaign:
            ax1=fig.add_subplot(211)
        else:
            ax1=fig.add_subplot(111)
        x_temp=halo_era5_hmc["IWC"].index.time
        y_temp=np.log10(np.array(halo_era5_hmc["IWC"].columns.astype(float)))
        y,x=np.meshgrid(y_temp,x_temp)
        levels=np.linspace(-4,3,80)
        
        #Create new Hydrometeor content as a sum of all water contents
        halo_era5_hmc["TWC"]=halo_era5_hmc["IWC"][:]+\
                                halo_era5_hmc["LWC"][:]+halo_era5_hmc["PWC"][:]
        C1=ax1.pcolormesh(x,y,(halo_era5_hmc["TWC"]*1000),
                          norm=colors.LogNorm(vmin=1e-4,vmax=1e3),
                          cmap=colormap)
        
        cb = plt.colorbar(C1,extend="both")
        cb.set_label('ERA-5: \n Water Content in g/kg')
        
        ax1.set_ylim([np.log10(200),3])
        ax1.set_ylabel("Pressure in hPa")
        ax1.set_yticks([np.log10(200),np.log10(300),
                        np.log10(500),np.log10(700),
                        np.log10(850),np.log10(1000)])
        
        ax1.set_yticklabels([200,300,500,700,850,1000])
        ax1.set_xlabel('')
        ax1.invert_yaxis()
        
        labels = levels[::2]
        if not self.synthetic_campaign:
            ax2=fig.add_subplot(212)
            y=np.array(radar["height"][:])#/1000
            print("Start plotting HAMP Cloud Radar")
            levels=np.arange(-30,+20.0,2.0)# 
            C2=ax2.contourf(radar["Reflectivity"].index.time,
                        y,radar["Reflectivity"].T,levels,
                        cmap=cm.get_cmap(colormap,len(levels)-1),extend="both")
            cb = plt.colorbar(C2)
            cb.set_label('HALO Radar: \n Reflectivity in dBZ')
            labels= levels[::2]
            cb.set_ticks(labels)   
            ax2.set_xlabel('Time in UTC')
        
            ax2.set_ylabel('Altitude in m ')
            ax2.set_ylim(0,12000)
            ax2.set_yticks([0,2000,4000,6000,8000,10000,12000])
            ax2.set_yticklabels([0,2000,4000,6000,8000,10000,12000])
            for label in ax2.yaxis.get_ticklabels()[::2]:
                label.set_visible(False)
        
        fig_name="Combined_HMC_HALO_ERA5_"+self.flight+"_"+flight_date+".png"
        if self.ar_of_day is not None:
                fig_name=self.ar_of_day+"_"+fig_name
            
            
        fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", self.plot_path+fig_name)
        return None
    
    def two_H_plot_radar_era5_combined_hwc(self,era5_on_halo,radar,halo_era5_hmc,
                                           date,colormap,start,end,
                                           do_masking=False,save_figure=True,
                                           low_level=False):    
        import matplotlib.dates as mdates
        hours = mdates.MinuteLocator(byminute=[0,20,40,60],interval = 1)
        h_fmt = mdates.DateFormatter('%H:%M')
    
        #Then tick and format with matplotlib:
        fig=plt.figure(figsize=(32,10))
        
        #Liquid water
        if not self.synthetic_campaign:
            #Then tick and format with matplotlib:
            fig=plt.figure(figsize=(32,10))
        
            fig_constellation=221
        else:
            #Then tick and format with matplotlib:
            fig=plt.figure(figsize=(32,20))
            fig_constellation=311
        
        ax1=fig.add_subplot(fig_constellation)
        x_temp=halo_era5_hmc["LWC"].loc[start:end].index#.time
        y_temp=np.log10(np.array(halo_era5_hmc["LWC"].columns.astype(float)))
        y,x=np.meshgrid(y_temp,x_temp)
        levels=np.linspace(-4,2,50)
        y=np.array(halo_era5_hmc["Geopot_Z"].loc[start:end])
        # if orography is present or other things occur, 
        # nan values can occur at height levels
        nan_height=np.argwhere(np.isnan(y.mean(axis=0)))
        if not nan_height.shape==(0,1):
            nan_height_max=nan_height.max()
        else:
            nan_height_max=-1
        y=y[:,nan_height_max+1::]
        x=x[:,nan_height_max+1::]
        
        #Create new Hydrometeor content as a sum of all water contents
        cutted_hmc=halo_era5_hmc["LWC"].iloc[:,nan_height_max+1::]
        
        if do_masking:
            cutted_hmc_masked,masked_df=era5_on_halo.\
                                            apply_orographic_mask_to_era(
                                                cutted_hmc,variable="LWC")
        
        C1=ax1.pcolormesh(x,y,(cutted_hmc*1000),
                          norm=colors.LogNorm(vmin=1e-4,vmax=1e3),
                          cmap=colormap)
        cb = plt.colorbar(C1,extend="both")
        cb.set_label('ERA-5: \n Liquid Water Content in g/kg')
        
        if low_level:
            ax1.set_ylim(0,3000)
            ax1.set_yticks([0,500, 1000,1500,  2000,
                            2500, 3000, 3500, 4000, 
                            4500, 5000])
            ax1.set_yticklabels([0,500, 1000, 1500, 2000,2500, 3000,
                                 3500, 4000, 4500, 5000])
        else:
            ax1.set_ylim(0,12000)
            ax1.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax1.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
        for label in ax1.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        ax1.set_ylabel('Altitude in m ')
        
        ax1.set_xlabel('')
        ax1.xaxis.set_major_locator(hours)
        ax1.xaxis.set_major_formatter(h_fmt)
        
        #Ice water
        ax2=fig.add_subplot(fig_constellation+2)
        levels=np.linspace(-4,2,50)
        
        #Create new Hydrometeor content as a sum of all water contents
        cutted_hmc=halo_era5_hmc["IWC"].loc[start:end]
        if do_masking:
            cutted_hmc=cutted_hmc*era5_on_halo.mask_df
            print("Orographic Masking is applied")
        else:
            print("No Orographic Masking is applied/needed")
        
        cutted_hmc=halo_era5_hmc["IWC"].iloc[:,nan_height_max+1::]
        
        C2=ax2.pcolormesh(x,y,(cutted_hmc*1000),
                          norm=colors.LogNorm(vmin=1e-4,vmax=1e3),
                          cmap=colormap)
        cb = plt.colorbar(C2,extend="both")
        cb.set_label('ERA-5: \n Ice Water Content in g/kg')
        
        if low_level:
            ax2.set_ylim(0,3000)
            ax2.set_yticks([0,500, 1000,1500,  2000,2500, 
                            3000, 3500, 4000, 4500, 5000])
            ax2.set_yticklabels([0,500, 1000, 1500, 2000,2500, 
                                 3000, 3500, 4000, 4500, 5000])
        else:
            ax2.set_ylim(0,12000)
            ax2.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax2.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
        for label in ax2.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        ax2.set_ylabel('Altitude in m ')
        
        ax2.set_xlabel('')
        ax2.xaxis.set_major_locator(hours)
        ax2.xaxis.set_major_formatter(h_fmt)
        
        
        ax3=fig.add_subplot(fig_constellation+1)
        levels=np.linspace(-4,2,50)
        
        #Create new Hydrometeor content as a sum of all water contents
        halo_era5_hmc["NWC"]=halo_era5_hmc["IWC"]+\
                                halo_era5_hmc["LWC"]+\
                                halo_era5_hmc["PWC"]
        cutted_hmc=halo_era5_hmc["NWC"].loc[start:end]\
            .iloc[:,nan_height_max+1::]
        if do_masking:
            cutted_hmc=cutted_hmc*era5_on_halo.mask_df
        #ERA5().apply_orographic_mask_to_era(cutted_hmc,
        #      variable="PWC")
        C3=ax3.pcolormesh(x,y,(cutted_hmc*1000),
                          norm=colors.LogNorm(vmin=1e-4,vmax=1e3),
                          cmap=colormap)
        cb = plt.colorbar(C3,extend="both")
        cb.set_label('ERA-5: \n Total Water Content in g/kg')
        
        if low_level:
            ax3.set_ylim(0,3000)
            ax3.set_yticks([0,500, 1000,1500,  2000,2500,
                            3000, 3500, 4000, 4500, 5000])
            ax3.set_yticklabels([0,500, 1000, 1500, 2000,2500, 
                                 3000, 3500, 4000, 4500, 5000])
        else:
            ax3.set_ylim(0,12000)
            ax3.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax3.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
        for label in ax3.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        ax3.set_ylabel('Altitude in m ')
        
        ax3.set_xlabel('')
        ax3.xaxis.set_major_locator(hours)
        ax3.xaxis.set_major_formatter(h_fmt)
        if not self.synthetic_campaign:
            ax4=fig.add_subplot(fig_constellation+3)
            y=np.array(radar["height"][:])#/1000
            print("Start plotting HAMP Cloud Radar")
            levels=np.arange(-30,+20.0,2.0)# 
            C4=ax4.contourf(radar["Reflectivity"].loc[start:end].index,y,
                        radar["Reflectivity"].loc[start:end].T,levels,
                        cmap=cm.get_cmap(colormap,len(levels)-1),extend="both")
        
            cb = plt.colorbar(C4)
            cb.set_label('HALO Radar: \n Reflectivity in dBZ')
            labels = levels[::2]
            cb.set_ticks(labels)   
            ax4.set_xlabel('Time in UTC')
            ax4.xaxis.set_major_locator(hours)
            ax4.xaxis.set_major_formatter(h_fmt)
        
        
            if low_level:
                ax4.set_ylim(0,3000)
                ax4.set_yticks([0,500, 1000,1500,  2000,2500, 
                                3000, 3500, 4000, 4500, 5000])
                ax4.set_yticklabels([0,500, 1000, 1500, 2000,2500,
                                     3000, 3500, 4000, 4500, 5000])
        
            else:
                ax4.set_ylim(0,12000)
                ax4.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
                ax4.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
            
            for label in ax4.yaxis.get_ticklabels()[::2]:
                label.set_visible(False)
        
        if low_level:
            fig_name="Combined_HMC_HALO_ERA5_low_level_"+\
            self.flight+"_"+date+"_"+\
            str(start)[-8:-6]+str(start)[-5:-3]+\
            "_"+str(end)[-8:-6]+str(end)[-5:-3]+".png"

        else:
            fig_name="Combined_HMC_HALO_ERA5_"+\
                        self.flight+"_"+date+"_"+\
                        str(start)[-8:-6]+str(start)[-5:-3]+\
                        "_"+str(end)[-8:-6]+str(end)[-5:-3]+".png"
            
        if self.ar_of_day is not None:
                fig_name=self.ar_of_day+"_"+fig_name
        
        fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", self.plot_path+fig_name)
        return None
    
    def plot_HALO_AR_ERA_thermodynamics(self,era5_on_halo,radar,halo_era5_hmc,
                                        date,start,end,do_masking=False,
                                        save_figure=True, low_level=False):    
        import matplotlib.dates as mdates
        #from matplotlib.colors import Boundarynorm
        hours = mdates.MinuteLocator(byminute=[0,30],
                                     interval = 1)
        h_fmt = mdates.DateFormatter('%H:%M')
    
        theta_colormap      = "RdYlBu_r"
        humidity_colormap   = "terrain_r"
        #Then tick and format with matplotlib:
        fig=plt.figure(figsize=(16,12))
        
        
        # Get high reflectivities
        if not self.synthetic_campaign:
            high_dbZ_index=radar["Reflectivity"][\
                                    radar["Reflectivity"]>15].any(axis=1)
            high_dbZ=radar["Reflectivity"].loc[high_dbZ_index]
                #Liquid water
        
        ax1=fig.add_subplot(211)
        x_temp=halo_era5_hmc["theta_e"].loc[start:end].index
        y_temp=np.log10(np.array(halo_era5_hmc["theta_e"]\
                                 .columns.astype(float)))
        y,x=np.meshgrid(y_temp,x_temp)
        
        levels=np.linspace(285,350,66)
        if self.flight=="RF10":
            levels=np.linspace(275,330,56)
        if low_level:
            levels=np.linspace(285,320,36)
            if self.flight=="RF10":
                levels=np.linspace(275,310,26)
        y=np.array(halo_era5_hmc["Geopot_Z"].loc[start:end])
        nan_height=np.argwhere(np.isnan(y.mean(axis=0)))
        if not nan_height.shape==(0,1):
            nan_height_max=nan_height.max()
        else:
            nan_height_max=-1
        y=y[:,nan_height_max+1::]
        x=x[:,nan_height_max+1::]
        
        
        cutted_theta=halo_era5_hmc["theta_e"].loc[start:end].iloc[\
                                                        :,nan_height_max+1::]
        if do_masking:
            cutted_theta=cutted_theta*era5_on_halo.mask_df
        #cutted_theta=cutted_theta*#ERA5().apply_orographic_mask_to_era(cutted_theta,
                         #                                    threshold=0.3,
                         #                                    variable="Theta_e")
        C1=ax1.pcolormesh(x,y,cutted_theta,cmap=theta_colormap,
                          vmin=levels[0],vmax=levels[-1])
        cb = plt.colorbar(C1,extend="both")
        cb.set_label('ERA-5: $\Theta_{e}$ in K')
        
        if low_level:
            ax1.set_ylim(0,6000)
            ax1.set_yticks([0,500, 1000,1500,2000,2500, 3000, 
                            3500, 4000, 4500, 5000,5500,6000])
            ax1.set_yticklabels([0,500, 1000, 1500, 2000,2500,3000, 3500,
                                 4000, 4500, 5000,5500,6000])
            marker_factor=5800
        else:
            ax1.set_ylim(0,12000)
            ax1.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax1.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
            marker_factor=11000
        if not self.synthetic_campaign:
            marker_pos=marker_factor*np.ones(len(high_dbZ))
            high_dbZ_scatter=ax1.scatter(high_dbZ.index,marker_pos,
                                    s=35,color="white",marker="D",
                                    linewidths=0.2,edgecolor="k")
        
        
        for label in ax1.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        ax1.set_ylabel('Altitude in m ')
        ax1.set_xlabel('')
        ax1.xaxis.set_major_locator(hours)
        ax1.xaxis.set_major_formatter(h_fmt)
        
        #Specific humidity
        q_min=0
        q_max=4
        
        if low_level:
            q_min=0
            q_max=5
        
        # Add wind 
        halo_era5_hmc["wind"]=np.sqrt(halo_era5_hmc["u"]**2+\
                                      halo_era5_hmc["v"]**2)
        
        ax2=fig.add_subplot(212)
        levels=np.linspace(q_min,q_max,50)
        
        #Create new Hydrometeor content as a sum of all water contents
        cutted_hmc=halo_era5_hmc["q"].loc[start:end]
        cutted_hmc=cutted_hmc.iloc[:,nan_height_max+1::]
        if do_masking:
            cutted_hmc=cutted_hmc*era5_on_halo.mask_df
        
        C2=ax2.pcolormesh(x,y,cutted_hmc*1000,
                          vmin=q_min,vmax=q_max,
                          cmap=humidity_colormap)
        mixing_ratio=cutted_hmc
        
        moisture_levels=[5,10,20,30,40]
        wind_levels=[10,20,40,60]
        if low_level:
            moisture_levels=[5,10,20,30,35]
            wind_levels=[10,20,30]
        wind=halo_era5_hmc["wind"].loc[start:end].iloc[:,nan_height_max+1::]
        if do_masking:
            wind=wind*era5_on_halo.mask_df
        
        wv_flux=mixing_ratio*wind 
        moisture_flux=1/9.82*wv_flux*1000
        #moisture_flux=moisture_flux
        
        CS=ax2.contour(x,y,moisture_flux,
                       levels=moisture_levels,colors="k",
                       linestyles="-",linewidths=1.0)
        
        CS2=ax2.contour(x,y,wind,
                       levels=wind_levels,colors="magenta",
                       linestyles="--",linewidths=1.0)
        
        # Contour lines and Colorbar specifications
        ax2.clabel(CS,fontsize=16,fmt='%1.1f',inline=1)
        ax2.clabel(CS2,fontsize=16,fmt='%1.1f',inline=1)
        
        cb = plt.colorbar(C2,extend="max")
        cb.set_label('ERA-5: \n Specific Humidity in g/kg')
        
        if low_level:
            ax2.set_ylim(0,6000)
            ax2.set_yticks([0,500, 1000,1500,2000,2500, 3000, 
                            3500, 4000, 4500, 5000,5500,6000])
            ax2.set_yticklabels([0,500, 1000, 1500, 2000,2500,3000, 3500,
                                 4000, 4500, 5000,5500,6000])
        
            fig_name="Thermodynamics_HALO_ERA5_low_level_"+\
                        self.flight+"_"+date+"_"+str(start)[-8:-6]+\
                        str(start)[-5:-3]+"_"+str(end)[-8:-6]+\
                        str(end)[-5:-3]+".png"
            
        else:
            ax2.set_ylim(0,12000)
            ax2.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax2.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
            fig_name="Thermodynamics_HMC_HALO_ERA5_"+self.flight+"_"+date+"_"+\
                        str(start)[-8:-6]+str(start)[-5:-3]+"_"+\
                        str(end)[-8:-6]+str(end)[-5:-3]+".png"
            
        for label in ax2.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        
        ax2.set_ylabel('Altitude in m ')
        ax2.set_xlabel('')
        ax2.xaxis.set_major_locator(hours)
        ax2.xaxis.set_major_formatter(h_fmt)
        if self.ar_of_day is not None:
                fig_name=self.ar_of_day+"_"+fig_name
        
        fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", self.plot_path+fig_name)
        return None
    
    def internal_leg_representativeness(self,campaign_cls,era5_on_halo,
                                        flight,halo_df,halo_era5_hmc):
        import seaborn as sns
        import matplotlib.dates as mdates
        hours = mdates.MinuteLocator(byminute=[0,20,40,60],interval = 1)
        h_fmt = mdates.DateFormatter('%H:%M')
        # cloud particles water
        # and precipitation
        #fig=plt.figure(figsize=(22,10))
        #fig_constellation=111
        #ax1=fig.add_subplot(fig_constellation)
        
        
        # Create new Hydrometeor content as a sum of all water contents
        # Droplets
        cutted_hmc=halo_era5_hmc.copy()
        # Preprocess
        x_temp=cutted_hmc["LWC"].index
        y_temp=np.log10(np.array(cutted_hmc["LWC"].columns.astype(float)))
        y,x=np.meshgrid(y_temp,x_temp)
        y=np.array(cutted_hmc["Geopot_Z"])
        
        # if orography is present or other things occur, 
        # nan values can occur at height levels
        nan_height=np.argwhere(np.isnan(y.mean(axis=0)))
        if not nan_height.shape==(0,1):
            nan_height_max=nan_height.max()
        else:
            nan_height_max=-1
        y=y[:,nan_height_max+1::]
        x=x[:,nan_height_max+1::]
        
        cutted_hmc["DWC"]=cutted_hmc["IWC"]+\
                                cutted_hmc["LWC"]
        cutted_hmc["DWC"][cutted_hmc["DWC"]<1e-5]=np.nan
        cutted_hmc["DWC"]=cutted_hmc["DWC"].resample("1min").mean()
        cutted_hmc["Geopot_Z"]=cutted_hmc["Geopot_Z"].resample("1min").mean()
        
        
        
        #Droplets
        cutted_hmc["DWC"]=np.log10(cutted_hmc["DWC"].\
                                    iloc[:,nan_height_max+1::].values.reshape(-1))
        cutted_hmc["Geopot_Z"]=cutted_hmc["Geopot_Z"].\
                                            iloc[:,nan_height_max+1::].\
                                                values.reshape(-1)
        cutted_hmc["Geopot_Z"][cutted_hmc["Geopot_Z"]>10000]=np.nan
        new_cutted_hmc=pd.DataFrame(columns=["DWC","Geopot_Z"],index=range(cutted_hmc["Geopot_Z"].shape[0]))
        new_cutted_hmc["DWC"]=cutted_hmc["DWC"]
        new_cutted_hmc["Geopot_Z"]=cutted_hmc["Geopot_Z"]
        new_cutted_hmc.dropna(subset=["DWC","Geopot_Z"],inplace=True)
        
        new_cutted_hmc["species"]="internal"
        new_cutted_hmc["weights"]=np.ones_like(new_cutted_hmc["DWC"])/new_cutted_hmc["DWC"].shape[0]
        
        # Precipitation is already existent for ERA5
        # Create new Hydrometeor content as a sum of all water contents
        #cutted_hmc["PWC"]=halo_era5_hmc["PWC"].iloc[:,nan_height_max+1::].\
        #                        values.reshape(-1)
        file_name="hydrometeors_pressure_levels_"+campaign_cls.years[flight]+\
                                    campaign_cls.flight_month[flight]+\
                                    campaign_cls.flight_day[flight]+".nc"    
        import xarray as xr
        hmc_ds=xr.open_dataset(campaign_cls.campaign_data_path+"ERA-5/"+file_name)
        
        time_hour_index=halo_df.index[int(halo_df.shape[0]/2)].hour
        dwc_ds=hmc_ds[["ciwc","clwc","z"]]
        dwc_ds=dwc_ds.sel(latitude=slice(halo_df.latitude.max(),
                                          halo_df.latitude.min()),
                          longitude=slice(halo_df.longitude.min(),
                                           halo_df.longitude.max()))
        dwc_ds["cdwc"]=dwc_ds["ciwc"][time_hour_index,:,:,:]+dwc_ds["clwc"][time_hour_index,:,:,:]
        dwc_df=pd.DataFrame()
        dwc_df["DWC"]=np.log10(dwc_ds["cdwc"].to_dataframe()["cdwc"].values.reshape(-1)*1000)
        dwc_df["Geopot_Z"]=dwc_ds["z"][time_hour_index,:,:,:].to_dataframe()["z"].values.reshape(-1)
        dwc_df["DWC"][dwc_df["DWC"]<-5]=np.nan
        dwc_df["Geopot_Z"][dwc_df["Geopot_Z"]>10000]=np.nan
        dwc_df.dropna(subset=["DWC","Geopot_Z"],inplace=True)
        dwc_df["species"]="all"
        dwc_df["weights"]=np.ones_like(dwc_df["DWC"])/dwc_df["DWC"].shape[0]
        
        entire_dwc=pd.concat([dwc_df,new_cutted_hmc])
        
        # Plot
        axs = sns.JointGrid(data= new_cutted_hmc,x='DWC', y='Geopot_Z', space=1)
        axs.x = dwc_df.DWC
        axs.y = dwc_df.Geopot_Z
        #axs.plot_joint(sns.distplot,kind="hist",s=5,c='k', marker='x')
        axs.plot_joint(sns.kdeplot,kind="hist",shade=True, 
                       color='salmon',zorder=0,levels=3,alpha=0.6)
        
        axs.ax_joint.scatter('DWC', 'Geopot_Z', data=new_cutted_hmc, c='grey',
                             marker='x',s=3,alpha=0.7)
        # drawing pdf instead of histograms on the marginal axes
        axs.ax_marg_x.cla()
        axs.ax_marg_y.cla()
        sns.kdeplot(ax=axs.ax_joint,data=new_cutted_hmc,x="DWC",y="Geopot_Z",
                    shade=False,levels=3,
                    color="black")
        sns.distplot(new_cutted_hmc.DWC, ax=axs.ax_marg_x,color="grey")
        sns.distplot(new_cutted_hmc.Geopot_Z, ax=axs.ax_marg_y, color="grey",
                     vertical=True)
        sns.distplot(dwc_df.DWC, ax=axs.ax_marg_x,color="salmon")
        sns.distplot(dwc_df.Geopot_Z, ax=axs.ax_marg_y, color="salmon",
                     vertical=True)
        axs.ax_marg_x.set_yticklabels("")
        axs.ax_marg_y.set_xticklabels("")
        axs.ax_marg_x.set_ylabel("")
        axs.ax_marg_y.set_xlabel("")
        axs.ax_marg_x.set_xticklabels("")
        axs.ax_marg_y.set_yticklabels("")
        axs.ax_marg_x.set_xlabel("")
        axs.ax_marg_y.set_ylabel("")
        
        axs.ax_joint.set_ylim([0,10000])
        axs.ax_joint.set_yticks([0, 2000, 4000,6000,8000,10000])
        axs.ax_joint.set_yticklabels([0, 2, 4,6,8,10])
        axs.ax_joint.set_xlabel("Droplet Water Content in $g/kg$")
        axs.ax_joint.set_ylabel('Altitude in km')
        axs.ax_joint.set_xticks([-5,-4, -3, -2,-1,0])
        axs.ax_joint.set_xticklabels(["$10^{-5}$","$10^{-4}$","$10^{-3}$","$10^{-2}$",
                                      "$10^{-1}$","$10^{-0}$"])
        axs.ax_joint.set_xlim([-5,0])
        legend_handles=[mpatches.Patch(color="darkblue",label="AR sector"),
                        ]
        axs.ax_joint.legend("")
        # drawing pdf instead of histograms on the marginal axes
        sns.despine(offset=5)
        fig_name=self.plot_path+self.flight+\
                    "_ERA5_Internal_Leg_Hydrometeor_Representativeness.png"
        plt.savefig(fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",fig_name)
class CARRA_HALO_Plotting(CARRA):
    def __init__(self,flight,plot_path,ar_of_day=None,synthetic_campaign=False):
        self.flight=flight[0]
        self.ar_of_day=ar_of_day
        self.plot_path=plot_path
        self.synthetic_campaign=synthetic_campaign  
    
    def plot_specific_humidity_profile(self,halo_carra,halo_df,dropsondes,
                           radar,date,start,end,
                           with_ivt=False,do_masking=False,
                           save_figure=True,low_level=False,AR_sector="all"):
        
        humidity_colormap   = "terrain_r"
        
        fig=plt.figure(figsize=(14,7))
        ax2=fig.add_subplot(111)
        if not self.synthetic_campaign:
            # Get high reflectivities
            high_dbZ_index=radar["Reflectivity"]\
                                [radar["Reflectivity"]>15].any(axis=1)
            high_dbZ=radar["Reflectivity"].loc[high_dbZ_index]
        # Specific humidity
        q_min=0
        q_max=6
        
        if low_level:
            q_min=0
            q_max=6
        
        # Drop nan rows
        halo_carra["q"]=halo_carra["specific_humidity"].dropna(how="all").reset_index()\
                                .drop_duplicates(subset="index",keep="first")\
                                    .set_index("index")
        halo_carra["u"]=halo_carra["u"].dropna(how="all").reset_index()\
                                .drop_duplicates(subset="index",keep="first")\
                                    .set_index("index")
        halo_carra["v"]=halo_carra["v"].dropna(how="all").reset_index()\
                                .drop_duplicates(subset="index",keep="first")\
                                    .set_index("index")
        #halo_carra["p"]=halo_carra["p"].dropna(how="all").reset_index()\
        #                        .drop_duplicates(subset="index",keep="first")\
        #                            .set_index("index")
        halo_carra["Z_Height"]=halo_carra["z"].loc[halo_carra["q"].index]
        halo_carra["Z_Height"]=halo_carra["Z_Height"].reset_index()\
                                .drop_duplicates(subset="index",keep="first")\
                                    .set_index("index")
        halo_lat=halo_df["latitude"].loc[pd.DatetimeIndex(halo_carra["q"].index)]\
                    .reset_index().drop_duplicates(subset="index",keep="first")\
                        .set_index("index")
        # Add wind 
        halo_carra["wind"]=np.sqrt(halo_carra["u"]**2+halo_carra["v"]**2)
        x_temp=halo_df.groupby(level=0).first().index#halo_lat.values
        y_temp=halo_carra["q"].columns
        
        #levels=np.linspace(q_min,q_max,50)
        y=np.array(halo_carra["Z_Height"].loc[halo_carra["q"].index])
        #y.fillna(method='ffill',axis=0,inplace=True)
        #y.fillna(method='bfill',axis=0,inplace=True)
        #y=np.array(y)
        y_temp,x=np.meshgrid(y_temp,x_temp)
        
        #Create new Hydrometeor content as a sum of all water contents
        cutted_hmc=halo_carra["q"]
        #if do_masking:
        #    cutted_hmc=cutted_hmc*era5_on_halo.mask_df
        cutted_hmc=cutted_hmc.replace(to_replace=np.nan,value=0.0)
        C2=ax2.pcolormesh(x,y,cutted_hmc*1000,
                          vmin=q_min,vmax=q_max,
                          cmap=humidity_colormap)
        mixing_ratio=cutted_hmc
        
        
        moisture_levels=[5,10,20,30,40]
        wind_levels=[10,20,40,60]
        if low_level:
            moisture_levels=[5,10,20,30,35]
            wind_levels=[10,20,30]
        wind=halo_carra["wind"]
        #if do_masking:
        #    wind=wind*era5_on_halo.mask_df
        
        wv_flux=mixing_ratio*wind #halo_era5_hmc["wind"]
        moisture_flux=1/9.82*wv_flux*1000
        
        CS=ax2.contour(x,y,moisture_flux.loc[start:end],
                       levels=moisture_levels,colors="k",
                       linestyles="-",linewidths=1.0)
        
        CS2=ax2.contour(x,y,halo_carra["wind"].loc[start:end],
                       levels=wind_levels,colors="magenta",
                       linestyles="--",linewidths=1.0)
        # Has to be changed for this function, as dropsondes are considered
        # by their latitude position --> which latitude to choose?!
        #if not self.synthetic_campaign:
        #    ax2.scatter(dropsondes["Latitude"],
        #             np.ones(dropsondes["IVT"].shape[0])*5500,
        #             s=50,marker='v',color="lightgreen",
        #             edgecolor="black",label="Dropsondes")
        # 
        #     for sonde_index in dropsondes["IVT"].index:
        #         ax2.axvline(x=sonde_index,ymin=0,ymax=5450,
        #                 color="black",ls="--",alpha=0.6)
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
            
        
            fig_name="_Specific_Humidity_HALO_CARRA_low_level_"+self.flight+\
                "_"+date+"_"+str(start)[-8:-6]+str(start)[-5:-3]+\
                "_"+str(end)[-8:-6]+\
                str(end)[-5:-3]+".png"
            
        else:
            ax2.set_ylim(0,12000)
            ax2.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax2.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
            fig_name="_Specific_Humidity_HALO_CARRA_"+self.flight+"_"+date+"_"+\
                str(start)[-8:-6]+str(start)[-5:-3]+"_"+str(end)[-8:-6]+\
                str(end)[-5:-3]+".png"
        
        for label in ax2.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        
        ax2.set_xlabel("Time (UTC)")
        if not self.synthetic_campaign:
            marker_pos=np.ones(len(high_dbZ))*5900
            high_dbZ_scatter=ax2.scatter(high_dbZ.index,marker_pos,
                                     s=55,color="white",marker="D",
                                     linewidths=0.4,edgecolor="k")
       
        ax2.set_ylabel('Altitude in m ')
        ax2.set_xlabel('')
        ax2.tick_params("both",length=5,width=1.5,which="major")
        if self.ar_of_day is not None:
                fig_name=self.ar_of_day+"_"+AR_sector+fig_name
        
        fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", self.plot_path+fig_name)
        return None

class ICON_HALO_Plotting(ICON):
    def __init__(self,cmpgn_cls,
                 flight,plot_path,ar_of_day=None,synthetic_campaign=False):
        self.cmpgn=cmpgn_cls
        self.flight=flight[0]
        self.ar_of_day=ar_of_day
        self.plot_path=plot_path
        self.synthetic_campaign=synthetic_campaign
        
    ###########################################################################
    ######          TEMPORARY ICON FUNCTIONS     ##############################
    ###########################################################################
    def plot_IVT_icon_era5_sondes(self,halo_era5,dropsondes,last_index,date,
                                  with_ICON=True,with_CARRA=True,
                                  with_dropsondes=True,
                                  save_figure=True,
                                  synthetic_icon_lat=None):
        import matplotlib
        import seaborn as sns
        matplotlib.rcParams.update({"font.size":24})
        
        fig=plt.figure(figsize=(16,7))
        # ERA-5 IVT
        ax1=fig.add_subplot(111)
        ax1.set_ylabel("IVT (kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)")
    
        ax1.plot(halo_era5["Interp_IVT"].index.time,
                halo_era5["Interp_IVT"],ls='--',lw=3,
                color="green",label="ERA-5")
        
        lower_lim=halo_era5["Interp_IVT"].min()//50*50
        upper_lim=halo_era5["Interp_IVT"].max()//50*50+100
        
        if with_ICON:
            # ICON IVT
            icon_data_path=self.plot_path+\
                "/../../../../data/"+"ICON_LEM_2KM/"+self.flight+"/"    
            # ICON IVT
            #if synthetic_icon_lat == 0:
            #    icon_data_path=icon_major_path
            #else:
            #    icon_data_path=icon_major_path+"Latitude_"+\
            #                    str(synthetic_icon_lat)+"/"
            ivt_icon_file=self.flight+"_"+self.ar_of_day+\
                "_ICON_Interpolated_IVT.csv"
            if self.synthetic_campaign:
                ivt_icon_file="Synthetic_"+ivt_icon_file
            if os.path.isfile(icon_data_path+ivt_icon_file):
                print("ICON IVT exists and will be included")    
                icon_ivt=pd.read_csv(icon_data_path+ivt_icon_file,index_col=0)
                icon_ivt.index=pd.DatetimeIndex(icon_ivt.index)
                icon_ivt=icon_ivt.iloc[0:last_index]
                ax1.plot(icon_ivt["IVT"].index.time,
                         icon_ivt["IVT"],ls='-',lw=3,
                         color="darkgreen",label="ICON-2km")
        
                lower_lim=icon_ivt["IVT"].min()//50*50
                upper_lim=icon_ivt["IVT"].max()//50*50+100
            else:
                print("ICON IVT does not exists or cannot be handled")
            
        if with_CARRA:
            carra_data_path=self.cmpgn.campaign_data_path+"CARRA/"
            ivt_carra_file=self.flight+"_"+self.ar_of_day+\
                "_HMP_CARRA_HALO_"+date+".csv"
            if self.synthetic_campaign:
                ivt_carra_file="Synthetic_"+ivt_carra_file
            if os.path.isfile(carra_data_path+ivt_carra_file):
                print("CARRA IVT exists and will be included")    
                carra_ivt=pd.read_csv(carra_data_path+ivt_carra_file,
                                     index_col=0)
                carra_ivt.index=pd.DatetimeIndex(carra_ivt.index)
                carra_ivt=carra_ivt.iloc[0:last_index]
                ax1.plot(carra_ivt["Interp_IVT"].index.time,
                         carra_ivt["Interp_IVT"],ls='-',lw=3,
                         color="black",label="CARRA")
        
                lower_lim=carra_ivt["Interp_IVT"].min()//50*50
                upper_lim=carra_ivt["Interp_IVT"].max()//50*50+100
            
        # Dropsondes IVT
        if with_dropsondes:
            if not self.synthetic_campaign:
                if not self.flight=="RF08":
                    try:
                        dropsondes["IVT"]=dropsondes["IVT"].loc[\
                                    halo_era5.index[0]:halo_era5.index[-1]]
                        # RF08 has only one or no (?) dropsonde 
                        # which makes the plotting more complicated
                        ax1.plot(dropsondes["IVT"].index.time,
                                 np.array(dropsondes["IVT"]),
                                 linestyle='',markersize=15,marker='v',
                                 color="orange",markeredgecolor="black",
                                 label="Dropsondes")
                    except:
                        pass
        ax1.set_ylim([lower_lim,upper_lim])
        ax1.legend(loc="upper center",ncol=3)
        ax1.set_xlabel('')
        ax1.set_xlabel('Time (UTC)')
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(3)
        ax1.tick_params(length=10,width=3)
        sns.despine(offset=10)
        if with_dropsondes:
            figname=self.flight+"_IVT_HALO_ERA5_Sonde"
        else:
            figname=self.flight+"_IVT_HALO_ERA5"
        if self.ar_of_day!=None:
            figname=self.ar_of_day+"_"+figname
        if with_CARRA:
            figname=figname+"_CARRA"
        if with_ICON:
            figname=figname+"_ICON"
        figname=figname+".png"    
        if save_figure:
            fig.savefig(self.plot_path+figname,dpi=600,bbox_inches="tight")
            print("Figure saved as: ",self.plot_path+figname)
        else:
            print(figname, "not saved as file")
        return None        

    def plot_hmp_icon_era5_sondes(self,radar,icon_hmp,halo_era5,dropsondes,
                                  last_index,plot_ivt=True,save_figure=True,
                                  synthetic_icon_lat=None):
        import seaborn as sns
        if plot_ivt:
            fig,axs=plt.subplots(4,1,figsize=(20,30),sharex=True)
            
            # ERA-5 IVT
            axs[0].set_ylabel("ERA-5,ICON,Sondes:\n IVT"+\
                              "(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)")
    
            axs[0].plot(halo_era5["Interp_IVT"].index.time,
                    halo_era5["Interp_IVT"],ls='--',lw=2,
                    color="green",label="ERA5")
            # ICON IVT
            icon_major_path="C:/Users/u300737/Desktop/PhD_UHH_WIMI/Work/"+\
                                "GIT_Repository/NAWDEX/data/ICON_LEM_2KM/"
            
            if synthetic_icon_lat == 0:
                icon_data_path=icon_major_path
            else:
                icon_data_path=icon_major_path+"Latitude_"+\
                    str(synthetic_icon_lat)+"/"
            
            if not self.synthetic_campaign:
                # Dropsondes IVT
                if not self.flight=="RF08":
                    dropsondes["IVT"]=dropsondes["IVT"].loc[\
                                        halo_era5.index[0]:halo_era5.index[-1]]
                    # RF08 has only one or no (?) dropsonde 
                    # which makes the plotting more complicated
                    axs[0].plot(dropsondes["IVT"].index.time,
                         np.array(dropsondes["IVT"]),
                         linestyle='',markersize=15,marker='v',
                         color="lightgreen",markeredgecolor="black",
                         label="Dropsondes")
        
            axs[0].set_ylim([10,500])
            axs[0].legend(loc="upper center",ncol=3,fontsize=12)
            axs[0].set_xlabel('')
            plot_no=1
        else:
            fig,axs=plt.subplots(3,1,figsize=(16,22),sharex=True)
            plot_no=0
        halo_era5=halo_era5.loc[icon_hmp.index]
        if plot_ivt:
            fig,axs=plt.subplots(4,1,figsize=(16,26),sharex=True)
            # ERA-5 IVT
            axs[0].set_ylabel("ERA-5,ICON,Sondes:\n IVT "+\
                              "(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)")
        
            axs[0].plot(halo_era5["Interp_IVT"].index.time,
                    halo_era5["Interp_IVT"],ls='--',lw=2,
                    color="green",label="ERA5")
            # ICON IVT
            ivt_icon_file=self.flight+"_"+self.ar_of_day+\
                            "_ICON_Interpolated_IVT.csv"
            if os.path.isfile(icon_data_path+ivt_icon_file):
                print("ICON IVT exists and will be included")    
                icon_ivt=pd.read_csv(icon_data_path+ivt_icon_file,index_col=0)
                icon_ivt.index=pd.DatetimeIndex(icon_ivt.index)
                icon_ivt=icon_ivt.iloc[0:last_index]
                axs[0].plot(icon_ivt["IVT"].index.time,
                            icon_ivt["IVT"],ls='-',lw=3,
                            color="darkgreen",label="ICON-IVT")
            
            else:
                print("ICON IVT does not exists or cannot be handled")
            
            if not self.synthetic_campaign:
                # Dropsondes IVT
                if not self.flight=="RF08":
                    dropsondes["IVT"]=dropsondes["IVT"].loc[\
                                        halo_era5.index[0]:halo_era5.index[-1]]
                    # RF08 has only one or no (?) dropsonde 
                    # which makes the plotting more complicated
                    axs[0].plot(dropsondes["IVT"].index.time,
                         np.array(dropsondes["IVT"]),
                         linestyle='',markersize=15,marker='v',
                         color="lightgreen",markeredgecolor="black",
                         label="Dropsondes")
        
            axs[0].set_ylim([50,halo_era5["Interp_IVT"].max()//100*100+100])
            axs[0].legend(loc="upper center",ncol=3)
            axs[0].set_xlabel('')
            plot_no=1
        else:
            fig,axs=plt.subplots(3,1,figsize=(16,22),sharex=True)
            plot_no=0
        if not self.synthetic_campaign:
            #IWV Plot
            if not self.flight=="RF08":
                dropsondes["IWV"]=dropsondes["IWV"].loc[\
                                        halo_era5.index[0]:halo_era5.index[-1]]
                # RF08 has only one or no (?) dropsonde 
                # which makes the plotting more complicated
                axs[plot_no].plot(dropsondes["IWV"].index.time,
                         np.array(dropsondes["IWV"]),
                         linestyle='',markersize=15,marker='v',color="orange",
                         markeredgecolor="black",label="Dropsondes")
                #axs[0].set_legend(loc="upper right")
        axs[plot_no].set_xlabel('')
        axs[plot_no].legend(loc="upper left",ncol=3,fontsize=12)
        
        if not self.synthetic_campaign:
            axs[plot_no].set_ylabel("ERA-5,ICON,Sondes:\n IWV "+\
                                "(kg$\mathrm{m}^{-2}$)")
        else:
            axs[plot_no].set_ylabel("ERA-5,ICON:\n IWV "+\
                                "(kg$\mathrm{m}^{-2}$)")
        axs[plot_no].plot(halo_era5["Interp_IWV"].index.time,
                    halo_era5["Interp_IWV"],ls='--',
                    color="brown",label="ERA5-IWV")
        
        axs[plot_no].plot(icon_hmp["Interp_IWV"].index.time,
                    icon_hmp["Interp_IWV"],ls='-',lw=2,
                    color="orange",label="ICON-IWV")
        if not self.synthetic_campaign:
        
            if not self.flight=="RF08":
                dropsondes["IWV"]=dropsondes["IWV"].loc[\
                                        halo_era5.index[0]:halo_era5.index[-1]]
            
                # RF08 has only one or no (?) dropsonde which makes the plotting
                # more complicated
                axs[plot_no].plot(dropsondes["IWV"].index.time,
                         np.array(dropsondes["IWV"]),
                         linestyle='',markersize=15,marker='v',color="orange",
                         markeredgecolor="black")
        
        axs[plot_no].set_xlabel('')
        axs[plot_no].legend(loc="upper left")
        new_plot_no=plot_no+1
        
        axs[new_plot_no].plot(halo_era5["Interp_LWP"].index.time,
                              halo_era5["Interp_LWP"],
                              ls='--',color="salmon",label="ERA5-LWP")
        
        axs[new_plot_no].plot(icon_hmp["Interp_LWP"].index.time,
                              icon_hmp["Interp_LWP"],
                              ls='-',lw=2,color="darkred",
                              label="ICON-LWP")
        
        axs[new_plot_no].plot(halo_era5["Interp_IWP"].index.time,
                              halo_era5["Interp_IWP"],ls='--',
                              color="lightblue",label="ERA5-IWP")
        
        axs[new_plot_no].plot(icon_hmp["Interp_IWP"].index.time,
                              icon_hmp["Interp_IWP"],
                              ls='-',lw=2,color="darkblue",
                              label="ICON-IWP")
        
        axs[new_plot_no].set_ylabel("Hydrometeor Path (g$\mathrm{m}^{-2}$)")
        
        if not self.synthetic_campaign:
            # high radar reflectivities
            high_dbZ_index=radar["Reflectivity"]\
                        [radar["Reflectivity"]>15].any(axis=1)
        
            high_dbZ=radar["Reflectivity"].loc[high_dbZ_index]
            marker_pos=np.ones(high_dbZ.shape[0])*750
                
            axs[new_plot_no].scatter(high_dbZ.index.time,marker_pos,
                                        s=50,color="grey",marker="D",
                                        linewidths=0.2,edgecolor="k",
                                        label="radar dBZ > 15")
        
        axs[new_plot_no].legend(loc="upper center",ncol=3,fontsize=12)
        
        plt.subplots_adjust(hspace=0.2)
        axs[new_plot_no].set_xlabel('')
        axs[new_plot_no].set_ylim([0,500])
        
        if not self.synthetic_campaign:
            hamp={}
            new_plot_no=new_plot_no+1
            hamp["Reflectivity"]=radar["Reflectivity"].loc\
                                [halo_era5["Interp_IWV"].index[0]:\
                                 halo_era5["Interp_IWV"].index[-1]]
        
            y=radar["height"][:]#/1000
            print("Start plotting HAMP Cloud Radar")
            axs[new_plot_no].set_yticks([0,2000,4000,6000,8000,10000])
            axs[new_plot_no].set_ylim([0,12000])
            axs[new_plot_no].set_yticklabels(["0","2","4","6","8","10"])
            axs[new_plot_no].set_ylabel("Altitude (km)")
            levels=np.arange(-30,30.0,1.0)# 
            try:
                C1=axs[new_plot_no].contourf(hamp["Reflectivity"].index.time,y,
                                         hamp["Reflectivity"].T,levels,
                                         cmap=cm.get_cmap(\
                                                "temperature",len(levels)-1),
                                             extend="both")
            except:
                C1=axs[new_plot_no].contourf(hamp["Reflectivity"].index.time,y,
                                         hamp["Reflectivity"].T,levels,
                                         cmap=cm.get_cmap('viridis',
                                                          len(levels)-1),
                                         extend="both")
            
            for label in axs[new_plot_no].xaxis.get_ticklabels()[::8]:
                label.set_visible(False)
            cb = fig.colorbar(C1,ax=axs[new_plot_no],
                          orientation="horizontal",shrink=0.5)
            cb.set_label('Reflectivity (dBZ)')
            labels = levels[::8]
            cb.set_ticks(labels)   
        
        axs[new_plot_no].set_xlabel('Time (UTC)')
        sns.despine(offset=1)
        figname=self.ar_of_day+"_"+self.flight+"_HALO_ICON_ERA5"+".png"    
        if save_figure:
            fig.savefig(self.plot_path+figname,dpi=300,bbox_inches="tight")
            print("Figure saved as: ",self.plot_path+figname)
        else:
            print(figname, "not saved as file")
        return None        

    def plot_HALO_AR_ICON_thermodynamics(self,halo_icon,halo_era5,dropsondes,
                                         radar,date,path,
                                         start,end,icon_data_path,
                                         with_ivt=False,do_masking=False,
                                         save_figure=True,low_level=False):    
        
        import matplotlib
        import matplotlib.dates as mdates
        from matplotlib import gridspec
        ########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!############
        pd.plotting.register_matplotlib_converters()
        #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!############
        
        set_font=18
        matplotlib.rcParams.update({'font.size':set_font})
        plt.rc('axes',linewidth=1.5)
        hours = mdates.MinuteLocator(byminute=[0,30],
                                     interval = 1)
        h_fmt = mdates.DateFormatter('%H:%M')
    
        #theta_colormap      = "jet"
        humidity_colormap   = "terrain_r"
        #Then tick and format with matplotlib:
        if with_ivt:
            fig=plt.figure(figsize=(16,10))
            gs= gridspec.GridSpec(2,2,height_ratios=[1,2],width_ratios=[1,0.14])
            ax1=plt.subplot(gs[0,0])
            # ERA-5 IVT
            ax1.set_ylabel("ERA-5,ICON,Sondes:\n IVT"+\
                           "(kg$\mathrm{m}^{-1}\mathrm{s}^{-1}$)")
    
            ax1.plot(halo_era5["Interp_IVT"].index.time,
               halo_era5["Interp_IVT"],ls='--',lw=2,
                color="green",label="ERA5")
            
            # ICON IVT
            ivt_icon_file=self.flight+"_"+self.ar_of_day+\
                            "_ICON_Interpolated_IVT.csv"
            if self.synthetic_campaign:
                ivt_icon_file="Synthetic_"+ivt_icon_file
            if os.path.isfile(icon_data_path+ivt_icon_file):
                print("ICON IVT exists and will be included")    
                icon_ivt=pd.read_csv(icon_data_path+ivt_icon_file,index_col=0)
                icon_ivt.index=pd.DatetimeIndex(icon_ivt.index)
                ax1.plot(icon_ivt["IVT"].index.time,
                icon_ivt["IVT"],ls='-',lw=3,color="darkgreen",label="ICON-IVT")
        
            #else:
            #    print("ICON IVT does not exists or cannot be handled")
            if not self.synthetic_campaign:
                # Dropsondes IVT
                if not self.flight[0]=="RF08":
                    dropsondes["IVT"]=dropsondes["IVT"].loc[\
                                    halo_era5.index[0]:\
                                    halo_era5.index[-1]]
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
            ax1.set_xlim([halo_era5.index.time[0],halo_era5.index.time[-1]])
            ax1.tick_params("both",length=5,width=1.5,which="major")
            
            ax1.xaxis.set_major_locator(hours)
            ax1.xaxis.set_major_formatter(h_fmt)
        
            ax2=plt.subplot(gs[1,:])
        else:
            fig=plt.figure(figsize=(16,7))
            ax2=fig.add_subplot(111)
        if not self.synthetic_campaign:
            # Get high reflectivities
            high_dbZ_index=radar["Reflectivity"]\
                                [radar["Reflectivity"]>15].any(axis=1)
            high_dbZ=radar["Reflectivity"].loc[high_dbZ_index]
        # Specific humidity
        q_min=0
        q_max=5
        
        if low_level:
            q_min=0
            q_max=5
        
        # Add wind 
        
        halo_icon["wind"]=np.sqrt(halo_icon["u"]**2+halo_icon["v"]**2)
        halo_icon["wind"]=halo_icon["wind"].iloc[:,10:]
        halo_icon["q"]=halo_icon["q"].iloc[:,10:]
        x_temp=halo_icon["q"].loc[start:end].index
        y_temp=halo_icon["q"].loc[start:end].columns
        y,x=np.meshgrid(y_temp,x_temp)
        
        #levels=np.linspace(q_min,q_max,50)
        y=halo_icon["Z_Height"].iloc[:,10:].loc[start:end]
        y.fillna(method='ffill',axis=0,inplace=True)
        y.fillna(method='bfill',axis=0,inplace=True)
        y=np.array(y)
        #Create new Hydrometeor content as a sum of all water contents
        cutted_hmc=halo_icon["q"].loc[start:end]
        #if do_masking:
        #    cutted_hmc=cutted_hmc*era5_on_halo.mask_df
        cutted_hmc=cutted_hmc.replace(to_replace=np.nan,value=0.0)
        C2=ax2.pcolormesh(x,y,cutted_hmc*1000,
                          vmin=q_min,vmax=q_max,
                          cmap=humidity_colormap)
        mixing_ratio=cutted_hmc
        #mpcalc.mixing_ratio_from_specific_humidity(halo_era5_hmc["q"])
        
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
        if not self.synthetic_campaign:
            ax2.scatter(dropsondes["IVT"].index,
                     np.ones(dropsondes["IVT"].shape[0])*5500,
                     s=50,marker='v',color="lightgreen",
                     edgecolor="black",label="Dropsondes")
        
            for sonde_index in dropsondes["IVT"].index:
                ax2.axvline(x=sonde_index,ymin=0,ymax=5450,
                        color="black",ls="--",alpha=0.6)
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
            
        
            fig_name="Specific_Humidity_HALO_ICON_low_level_"+self.flight+\
                "_"+date+"_"+str(start)[-8:-6]+str(start)[-5:-3]+\
                "_"+str(end)[-8:-6]+\
                str(end)[-5:-3]+".png"
            
        else:
            ax2.set_ylim(0,12000)
            ax2.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax2.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
            fig_name="Specific_Humidity_HALO_ICON_"+self.flight+"_"+date+"_"+\
                str(start)[-8:-6]+str(start)[-5:-3]+"_"+str(end)[-8:-6]+\
                str(end)[-5:-3]+".png"
        
        for label in ax2.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        
        ax2.set_xlabel("Time (UTC)")
        if not self.synthetic_campaign:
            marker_pos=np.ones(len(high_dbZ))*5900
            high_dbZ_scatter=ax2.scatter(high_dbZ.index,marker_pos,
                                     s=55,color="white",marker="D",
                                     linewidths=0.4,edgecolor="k")
       
        ax2.set_ylabel('Altitude in m ')
        ax2.set_xlabel('')
        ax2.xaxis.set_major_locator(hours)
        ax2.xaxis.set_major_formatter(h_fmt)
        ax2.tick_params("both",length=5,width=1.5,which="major")
        if self.ar_of_day is not None:
                fig_name=self.ar_of_day+"_"+fig_name
        
        fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", self.plot_path+fig_name)
        return None
    
    def plot_radar_icon_hwc(self,radar,halo_icon,colormap,
                            save_figure=True):    
        fig=plt.figure(figsize=(16,24))
        ax1=fig.add_subplot(311)
        
        # Add wind 
        x_temp=halo_icon["qi"].index.time
        y_temp=halo_icon["qi"].columns
        y,x=np.meshgrid(y_temp,x_temp)
        
        #levels=np.linspace(q_min,q_max,50)
        y=halo_icon["Z_Height"]
        y.fillna(method='ffill',axis=0,inplace=True)
        y.fillna(method='bfill',axis=0,inplace=True)
        y=np.array(y)
        #Create new Hydrometeor content as a sum of all water contents
        cutted_hmc=halo_icon["qi"]
        cutted_hmc[cutted_hmc<1e-8]=np.nan
        #if do_masking:
        #    cutted_hmc=cutted_hmc*era5_on_halo.mask_df
        cutted_hmc=cutted_hmc.replace(to_replace=np.nan,value=0.0)
        C1=ax1.pcolormesh(x,y,cutted_hmc*1000,norm=colors.LogNorm(vmin=1e-4,
                                                                  vmax=1e1),
                          cmap=colormap)
        levels=np.linspace(-4,1,50)
        for label in ax1.xaxis.get_ticklabels()[::5]:
            label.set_visible(False)
        ax1.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
        ax1.set_yticklabels([0, 2, 4,6,8,10,12])
        ax1.set_ylabel('Altitude (km) ')
            
        cb = plt.colorbar(C1,extend="both")
        cb.set_label('IWC in g/kg')
        #ax1.set_yscale("log")
        #ax1.set_ylim([np.log10(200),3])
        #ax1.set_yticks([np.log10(200),np.log10(300),np.log10(500),np.log10(700),np.log10(850),np.log10(1000)])
        #ax1.set_yticklabels([200,300,500,700,850,1000])
        ax1.set_xlabel('')
        #ax1.invert_yaxis()
        
        ax2=fig.add_subplot(312)
        #y=np.array(halo_era5_hmc["LWC"].columns.astype(float))#/1000
        levels=np.linspace(-4,1,50)
        cutted_hmc=halo_icon["qs"]
        cutted_hmc[cutted_hmc<1e-8]=np.nan
        C2=ax2.pcolormesh(x,y,halo_icon["qs"]*1000,
                          norm=colors.LogNorm(vmin=1e-4,vmax=1e1),
                          cmap=colormap)
        for label in ax2.xaxis.get_ticklabels()[::5]:
            label.set_visible(False)
            #ax1.set_xticks([HAMP["Reflectivity"].index.time[::4]])
        cb = plt.colorbar(C2,extend="both")
        cb.set_label('SWC in  g/kg')
        #ax1.set_yscale("log")
        ax2.set_xlabel('')
        ax2.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
        ax2.set_yticklabels([0,2,4,6,8,10,12])
        ax2.set_ylabel('Altitude (km) ')
        
        ax4=fig.add_subplot(313)
        y=np.array(radar["height"][:])#/1000
        print("Start plotting HAMP Cloud Radar")
        levels=np.arange(-30,+20.0,2.0)# 
        C4=ax4.contourf(radar["Reflectivity"].index.time,y,
                        radar["Reflectivity"].T,levels,
                        cmap=cm.get_cmap(colormap,len(levels)-1),extend="both")
        for label in ax1.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
            #ax1.set_xticks([HAMP["Reflectivity"].index.time[::4]])
        cb = plt.colorbar(C4)
        cb.set_label('Reflectivity in dBZ')
        labels = levels[::2]
        cb.set_ticks(labels)   
        ax4.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
        ax4.set_yticklabels([0, 2, 4,6,8,10,12])
        
        ax4.set_xlabel('Time in UTC')
        ax4.set_ylabel('Altitude (km) ')
        ax4.set_ylim(0,12000)
        fig_name="HMC_HALO_ICON_"+self.flight+".png"
        fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", fig_name)
        return None
    
    def plot_AR_q_lat(self,halo_icon,halo_df,dropsondes,
                           radar,date,path,start,
                           end,icon_data_path,
                           with_ivt=False,do_masking=False,
                           save_figure=True,low_level=False):
        
        humidity_colormap   = "terrain_r"
        
        fig=plt.figure(figsize=(14,7))
        ax2=fig.add_subplot(111)
        if not self.synthetic_campaign:
            # Get high reflectivities
            high_dbZ_index=radar["Reflectivity"]\
                                [radar["Reflectivity"]>15].any(axis=1)
            high_dbZ=radar["Reflectivity"].loc[high_dbZ_index]
        # Specific humidity
        q_min=0
        q_max=6
        
        if low_level:
            q_min=0
            q_max=6
        
        # Drop nan rows
        halo_icon["q"]=halo_icon["q"].dropna(how="all").reset_index()\
                                .drop_duplicates(subset="index",keep="first")\
                                    .set_index("index")
        halo_icon["u"]=halo_icon["u"].dropna(how="all").reset_index()\
                                .drop_duplicates(subset="index",keep="first")\
                                    .set_index("index")
        halo_icon["v"]=halo_icon["v"].dropna(how="all").reset_index()\
                                .drop_duplicates(subset="index",keep="first")\
                                    .set_index("index")
        halo_icon["p"]=halo_icon["p"].dropna(how="all").reset_index()\
                                .drop_duplicates(subset="index",keep="first")\
                                    .set_index("index")
        halo_icon["Z_Height"]=halo_icon["Z_Height"].loc[halo_icon["q"].index]
        halo_icon["Z_Height"]=halo_icon["Z_Height"].reset_index()\
                                .drop_duplicates(subset="index",keep="first")\
                                    .set_index("index")
        halo_lat=halo_df["latitude"].loc[halo_icon["q"].index].reset_index()\
                                .drop_duplicates(subset="index",keep="first")\
                                    .set_index("index")
        # Add wind 
        halo_icon["wind"]=np.sqrt(halo_icon["u"]**2+halo_icon["v"]**2)
        x_temp=halo_lat.values
        y_temp=halo_icon["q"].columns
        
        #levels=np.linspace(q_min,q_max,50)
        y=np.array(halo_icon["Z_Height"].loc[halo_icon["q"].index])
        #y.fillna(method='ffill',axis=0,inplace=True)
        #y.fillna(method='bfill',axis=0,inplace=True)
        #y=np.array(y)
        y_temp,x=np.meshgrid(y_temp,x_temp)
        
        #Create new Hydrometeor content as a sum of all water contents
        cutted_hmc=halo_icon["q"]
        #if do_masking:
        #    cutted_hmc=cutted_hmc*era5_on_halo.mask_df
        cutted_hmc=cutted_hmc.replace(to_replace=np.nan,value=0.0)
        C2=ax2.pcolormesh(x,y,cutted_hmc*1000,
                          vmin=q_min,vmax=q_max,
                          cmap=humidity_colormap)
        mixing_ratio=cutted_hmc
        
        
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
        # Has to be changed for this function, as dropsondes are considered
        # by their latitude position --> which latitude to choose?!
        #if not self.synthetic_campaign:
        #    ax2.scatter(dropsondes["Latitude"],
        #             np.ones(dropsondes["IVT"].shape[0])*5500,
        #             s=50,marker='v',color="lightgreen",
        #             edgecolor="black",label="Dropsondes")
        # 
        #     for sonde_index in dropsondes["IVT"].index:
        #         ax2.axvline(x=sonde_index,ymin=0,ymax=5450,
        #                 color="black",ls="--",alpha=0.6)
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
            
        
            fig_name="LAT_Specific_Humidity_HALO_ICON_low_level_"+self.flight+\
                "_"+date+"_"+str(start)[-8:-6]+str(start)[-5:-3]+\
                "_"+str(end)[-8:-6]+\
                str(end)[-5:-3]+".png"
            
        else:
            ax2.set_ylim(0,12000)
            ax2.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
            ax2.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
            fig_name="LAT_Specific_Humidity_HALO_ICON_"+self.flight+"_"+date+"_"+\
                str(start)[-8:-6]+str(start)[-5:-3]+"_"+str(end)[-8:-6]+\
                str(end)[-5:-3]+".png"
        
        for label in ax2.yaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        
        ax2.set_xlabel("Latitude ($^{\circ}$ N)")
        if not self.synthetic_campaign:
            marker_pos=np.ones(len(high_dbZ))*5900
            high_dbZ_scatter=ax2.scatter(high_dbZ.index,marker_pos,
                                     s=55,color="white",marker="D",
                                     linewidths=0.4,edgecolor="k")
       
        ax2.set_ylabel('Altitude in m ')
        ax2.set_xlabel('')
        ax2.tick_params("both",length=5,width=1.5,which="major")
        if self.ar_of_day is not None:
                fig_name=self.ar_of_day+"_"+fig_name
        
        fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as: ", self.plot_path+fig_name)
        return None
    
    def internal_leg_representativeness(self,campaign_cls,icon_cls,icon_on_halo,
                                        flight,halo_df,halo_icon_hmc):
        import seaborn as sns
        
        # Preprocess
        x_temp=halo_icon_hmc["qi"].index
        y_temp=np.log10(np.array(halo_icon_hmc["qi"].columns.astype(float)))
        y,x=np.meshgrid(y_temp,x_temp)
        y=np.array(halo_icon_hmc["Z_Height"])
        # if orography is present or other things occur, 
        # nan values can occur at height levels
        nan_height=np.argwhere(np.isnan(y.mean(axis=0)))
        #if not nan_height.shape==(0,1):
        #    nan_height_max=nan_height.max()
        #else:
        #    nan_height_max=-1
        nan_height_max=9
        y=y[:,:]
        x=x[:,nan_height_max+1::]
        
        # Create new Hydrometeor content as a sum of all water contents
        # Droplets
        halo_icon_hmc["DWC"]=(halo_icon_hmc["qi"]+\
                              halo_icon_hmc["qs"]+\
                              halo_icon_hmc["qr"]+\
                              halo_icon_hmc["qc"])*1000
        halo_icon_hmc["DWC"][halo_icon_hmc["DWC"]<1e-5]=np.nan
        halo_icon_hmc["DWC"]=halo_icon_hmc["DWC"].resample("10s").mean()
        halo_icon_hmc["Z_Height"]=halo_icon_hmc["Z_Height"].resample("10s").mean()
        
        cutted_hmc=pd.DataFrame()
        #Droplets
        cutted_hmc["DWC"]=np.log10(halo_icon_hmc["DWC"].\
                                    iloc[:,nan_height_max+1::].values.reshape(-1))
        cutted_hmc["Z_Height"]=halo_icon_hmc["Z_Height"].values.reshape(-1)
        cutted_hmc["Z_Height"][cutted_hmc["Z_Height"]>10000]=np.nan
        cutted_hmc.dropna(subset=["DWC","Z_Height"],inplace=True)
        
        cutted_hmc["species"]="internal"
        cutted_hmc["weights"]=np.ones_like(cutted_hmc["DWC"])/\
                                cutted_hmc["DWC"].shape[0]
        
        file_name_ice="Ice_Content_ICON_"+flight+"_14UTC"+".nc"    
        file_name_snow="Snow_Content_ICON_"+flight+"_14UTC"+".nc"
        file_name_liquid="Liquid_Content_ICON_"+flight+"_14UTC"+".nc"
        file_name_rain="Rain_Content_ICON_"+flight+"_14UTC"+".nc"
        
        file_name_height="Z_Height_ICON_"+flight+"_14UTC"+".nc"
        
        import xarray as xr
        ice_ds=xr.open_dataset(campaign_cls.data_path+"ICON_LEM_2KM/"+\
                               file_name_ice)
        snow_ds=xr.open_dataset(campaign_cls.data_path+"ICON_LEM_2KM/"+\
                                file_name_snow)
        liquid_ds=xr.open_dataset(campaign_cls.data_path+"ICON_LEM_2KM/"+\
                                  file_name_liquid)
        rain_ds=xr.open_dataset(campaign_cls.data_path+"ICON_LEM_2KM/"+\
                                file_name_rain)
        
        height_ds=xr.open_dataset(campaign_cls.data_path+"ICON_LEM_2KM/"+\
                                  file_name_height)
        dwc_ds=xr.Dataset()
        dwc_ds["qi"]=ice_ds["qi"]
        dwc_ds["qs"]=snow_ds["qs"]
        dwc_ds["qr"]=rain_ds["qr"]
        dwc_ds["qc"]=liquid_ds["qc"]
        dwc_ds["Z_Height"]=height_ds["z_mc"]
        # Cut ICON data to region of interest
        lat_range=[halo_df.latitude.min(),halo_df.latitude.max()]
        lon_range=[halo_df.longitude.min(),halo_df.longitude.max()]
        domain_idx=icon_cls.get_indexes_for_given_area(self,dwc_ds,lat_range,
                                                       lon_range)
        dwc_ds=dwc_ds.isel(ncells=domain_idx)
        # Change time information
        dwc_ds=ICON.adapt_icon_time_index(self,dwc_ds,"2016-10-13",flight[0])
        #dwc_ds=dwc_ds.resample(time="1h").mean()
        #dwc_ds["cdwc"]=xr.DataArray(data=np.array(dwc_ds["qi"][0,:,:])+\
        #                                 np.array(dwc_ds["qs"][0,:,:]),
        #                            coords=dwc_ds.coords)
        
        #Resample to the hour value
        resample_dwc_df=pd.DataFrame(data=np.array(dwc_ds["qi"][0,:,:])+\
                                          np.array(dwc_ds["qs"][0,:,:])+\
                                          np.array(dwc_ds["qr"][0,:,:])+\
                                          np.array(dwc_ds["qc"][0,:,:]))
        
        resample_z_df=pd.DataFrame(data=np.array(dwc_ds["Z_Height"][:,:])) 
        
        #resample_dwc_df=resample_dwc_df.resample("1h").mean()
        #resample_z_df=resample_z_df.resample("1h").mean()
        
        dwc_df=pd.DataFrame()
        dwc_df["DWC"]=np.log10(resample_dwc_df.values.reshape(-1)*1000)
        dwc_df["Z_Height"]=resample_z_df.values.reshape(-1)
        dwc_df["DWC"][dwc_df["DWC"]<-5]=np.nan
        dwc_df["Z_Height"][dwc_df["Z_Height"]>10000]=np.nan
        dwc_df.dropna(subset=["DWC","Z_Height"],inplace=True)
        dwc_df["species"]="all"
        dwc_df["weights"]=np.ones_like(dwc_df["DWC"])/dwc_df["DWC"].shape[0]
        
        print("SELECT Sample. This takes long")
        entire_dwc=pd.concat([dwc_df,cutted_hmc])
        # Plot
        random_int_logs=np.random.randint(dwc_df.shape[0],size=100000)
        dwc_df=dwc_df.iloc[random_int_logs,:]
        axs = sns.JointGrid(data= dwc_df,x='DWC', y='Z_Height', space=1,
                            height=10)
        #axs.x = dwc_df.DWC
        #axs.y = dwc_df.Z_Height
        
        axs.plot_joint(sns.kdeplot,kind="hist",shade=True, 
                       color='blue',zorder=0,levels=3,alpha=0.5)
        
        axs.ax_joint.scatter('DWC', 'Z_Height', data=cutted_hmc, c='grey',
                             marker='x',s=1,alpha=0.3)
        
        # drawing pdf instead of histograms on the marginal axes
        axs.ax_marg_x.cla()
        axs.ax_marg_y.cla()
        sns.kdeplot(ax=axs.ax_joint,data=cutted_hmc,x="DWC",y="Z_Height",
                    shade=False,levels=3,
                    color="black")
        sns.distplot(cutted_hmc.DWC, ax=axs.ax_marg_x,color="k",
                    hist_kws={"weights":np.ones_like(cutted_hmc.DWC)/\
                              len(cutted_hmc.DWC)})
        sns.distplot(cutted_hmc.Z_Height, ax=axs.ax_marg_y, color="k",
                    hist_kws={"weights":np.ones_like(cutted_hmc.Z_Height)/\
                              len(cutted_hmc.Z_Height)},
                    vertical=True)
        sns.distplot(dwc_df.DWC, ax=axs.ax_marg_x,color="blue",
                     hist_kws={"weights":np.ones_like(dwc_df.DWC)/\
                              len(dwc_df.DWC)})
        sns.distplot(dwc_df.Z_Height, ax=axs.ax_marg_y, color="blue",
                     hist_kws={"weights":np.ones_like(dwc_df.Z_Height)/\
                              len(dwc_df.Z_Height)},vertical=True)
        axs.ax_marg_x.set_yticklabels("")
        axs.ax_marg_y.set_xticklabels("")
        axs.ax_marg_x.set_ylabel("")
        axs.ax_marg_y.set_xlabel("")
        axs.ax_marg_x.set_xticklabels("")
        axs.ax_marg_y.set_yticklabels("")
        axs.ax_marg_x.set_xlabel("")
        axs.ax_marg_y.set_ylabel("")
        
        axs.ax_joint.set_ylim([0,10000])
        axs.ax_joint.set_yticks([0, 2000, 4000,6000,8000,10000])
        axs.ax_joint.set_yticklabels([0, 2, 4,6,8,10])
        axs.ax_joint.set_xlabel("Specific Water Content in $g/kg$")
        axs.ax_joint.set_ylabel('Altitude in km')
        axs.ax_joint.set_xticks([-5, -3, -1,1])
        axs.ax_joint.set_xticklabels(["$10^{-5}$","$10^{-3}$",
                                      "$10^{-1}$","$10^{1}$"])
        axs.ax_joint.set_xlim([-5,1])
        # drawing pdf instead of histograms on the marginal axes
        sns.despine(offset=5)
        #create legend objects
            #
            #axs.ax_joint.legend()
        fig_name=self.plot_path+self.flight+\
                    "_ICON_Internal_Leg_Hydrometeor_Representativeness.png"
        plt.savefig(fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",fig_name)
    
    def mean_internal_leg_representativeness(self,campaign_cls,icon_cls,
                                             icon_on_halo,flight,
                                             halo_df,halo_icon_hmc,
                                             simple_stats=True):
        
        import seaborn as sns
        import matplotlib
        matplotlib.rcParams.update({"font.size":16})
        # Create new Hydrometeor content as a sum of frozen and unfrozen
        # water contents
        halo_icon_hmc["DWC"]=(halo_icon_hmc["qr"]+\
                              halo_icon_hmc["qc"])*1000
        halo_icon_hmc["FWC"]=(halo_icon_hmc["qi"]+\
                              halo_icon_hmc["qs"])*1000
        
        halo_dwc_df=pd.DataFrame(data=np.array(halo_icon_hmc["DWC"].iloc[:,10::]),
                                 index=pd.DatetimeIndex(\
                                        np.array(halo_icon_hmc["DWC"].index)),
                                 columns=np.array(halo_icon_hmc["Z_Height"][:])\
                                             .mean(axis=0))
        halo_dwc_df[halo_dwc_df<1e-5]=np.nan
        
        halo_fwc_df=pd.DataFrame(data=np.array(halo_icon_hmc["FWC"].iloc[:,10::]),
                                 index=pd.DatetimeIndex(\
                                        np.array(halo_icon_hmc["FWC"].index)),
                                 columns=np.array(halo_icon_hmc["Z_Height"][:])\
                                             .mean(axis=0))
        halo_fwc_df[halo_fwc_df<1e-5]=np.nan
        stats_halo_dwc_df=pd.DataFrame()
        stats_halo_fwc_df=pd.DataFrame()
        stats_icon_dwc_df=pd.DataFrame()
        stats_icon_fwc_df=pd.DataFrame()
        
        stats_halo_dwc_df=halo_dwc_df.quantile([0.05,0.25,0.5,
                                                0.75,0.95],axis=0)
        stats_halo_fwc_df=halo_fwc_df.quantile([0.05,0.25,0.5,
                                                0.75,0.95],axis=0)
            
        #---------------------------------------------------------------------#
        # get squared ICON domain
        file_name_ice="Ice_Content_ICON_"+flight+"_14UTC"+".nc"    
        file_name_snow="Snow_Content_ICON_"+flight+"_14UTC"+".nc"
        file_name_liquid="Liquid_Content_ICON_"+flight+"_14UTC"+".nc"
        file_name_rain="Rain_Content_ICON_"+flight+"_14UTC"+".nc"
        
        file_name_height="Z_Height_ICON_"+flight+"_14UTC"+".nc"
        
        
        import xarray as xr
        ice_ds=xr.open_dataset(campaign_cls.data_path+\
                               "ICON_LEM_2KM/"+file_name_ice)
        snow_ds=xr.open_dataset(campaign_cls.data_path+\
                                "ICON_LEM_2KM/"+file_name_snow)
        liquid_ds=xr.open_dataset(campaign_cls.data_path+\
                                  "ICON_LEM_2KM/"+file_name_liquid)
        rain_ds=xr.open_dataset(campaign_cls.data_path+\
                                "ICON_LEM_2KM/"+file_name_rain)
        
        height_ds=xr.open_dataset(campaign_cls.data_path+\
                                  "ICON_LEM_2KM/"+file_name_height)
        # Assign new dataset
        hmc_ds=xr.Dataset()
        hmc_ds["qi"]=ice_ds["qi"]
        hmc_ds["qs"]=snow_ds["qs"]
        hmc_ds["qr"]=rain_ds["qr"]
        hmc_ds["qc"]=liquid_ds["qc"]
        hmc_ds["Z_Height"]=height_ds["z_mc"]
        
        # Cut ICON data to region of interest
        lat_range=[halo_df.latitude.min(),halo_df.latitude.max()]
        lon_range=[halo_df.longitude.min(),halo_df.longitude.max()]
        domain_idx=icon_cls.get_indexes_for_given_area(self,hmc_ds,
                                                       lat_range,lon_range)
        hmc_ds=hmc_ds.isel(ncells=domain_idx)
        # Change time information
        hmc_ds=ICON.adapt_icon_time_index(self,hmc_ds,"2016-10-13",flight[0])
        
        #Resample to the hour value
        icon_dwc_df=pd.DataFrame(data=(np.array(hmc_ds["qr"][0,:,:])+\
                                          np.array(hmc_ds["qc"][0,:,:]))*1000)
        icon_fwc_df=pd.DataFrame(data=(np.array(hmc_ds["qi"][0,:,:])+\
                                          np.array(hmc_ds["qs"][0,:,:]))*1000)
       
        stats_icon_dwc_df=icon_dwc_df.quantile([0.05,0.25,0.5,
                                                0.75,0.95],axis=1)
        stats_icon_fwc_df=icon_fwc_df.quantile([0.05,0.25,0.5,
                                                0.75,0.95],axis=1)
        
        #if simple_stats:
        #    stats_icon_dwc_df["mean"]=icon_dwc_df.mean(axis=1)
        #    stats_icon_fwc_df["mean"]=icon_fwc_df.mean(axis=1)
        #    stats_icon_dwc_df["std"]=icon_dwc_df.std(axis=1)
        #    stats_icon_fwc_df["std"]=icon_fwc_df.std(axis=1)
        
        icon_z_df=pd.DataFrame(data=np.array(hmc_ds["Z_Height"][:,:])) 
        stats_icon_dwc_df.columns=icon_z_df.mean(axis=1)
        stats_icon_fwc_df.columns=icon_z_df.mean(axis=1)
        
        # Plotting
        
        vertical_var_fig=plt.figure(figsize=(10,10))
        
        # melted hydrometeors 
        ax1=vertical_var_fig.add_subplot(121)
        boxplot_dwc=halo_dwc_df.loc[:,::5]
        boxplot_fwc=halo_fwc_df.loc[:,::5]
        
        boxplot_positions=boxplot_dwc.isnull().sum(axis=0)
        boxplot_positions=boxplot_positions!=boxplot_dwc.shape[0]
        
        boxplot_positions_fwc=boxplot_fwc.isnull().sum(axis=0)
        boxplot_positions_fwc=boxplot_positions!=boxplot_fwc.shape[0]
        
        boxplot_dwc=boxplot_dwc.loc[:,boxplot_positions]
        boxplot_fwc=boxplot_fwc.loc[:,boxplot_positions_fwc]
        boxplot_dwc_columns=np.array(boxplot_dwc.columns)
        boxplot_fwc_columns=np.array(boxplot_fwc.columns)
        boxplot_positions=boxplot_dwc.columns
        boxplot_positions_fwc=boxplot_fwc.columns
        boxplot_values=boxplot_dwc.values
        #filtered_boxplot_dwc:
        # Filter data using np.isnan
        mask = ~np.isnan(boxplot_dwc)
        
        # do it as for loop over all columns and add whiskerplot
        # --> to be done tomorrow
        for col in boxplot_dwc_columns:
            bp=ax1.boxplot(boxplot_dwc[col].dropna().values,
                        positions=[float(col)/1000],
                        notch=False, widths=150/1000,
                        vert=False,patch_artist=True,
                        boxprops={"facecolor":"lightgrey",
                                  "linewidth":3,"alpha":0.3},
                        whiskerprops={"linewidth":3},medianprops={"linewidth":3})
            ax1.set_yticklabels("")
        
        ax1.fill_betweenx(stats_icon_dwc_df.columns/1000,
                          stats_icon_dwc_df.loc[0.05,:],
                          x2=stats_icon_dwc_df.loc[0.95,:],
                          color="lightblue")
        ax1.fill_betweenx(stats_icon_dwc_df.columns/1000,
                          stats_icon_dwc_df.loc[0.25,:],
                          x2=stats_icon_dwc_df.loc[0.75,:],
                          color="lightskyblue")
        
        #ax1.plot(stats_icon_dwc_df.loc[0.5,:],
        #         stats_icon_dwc_df.columns/1000,
        #         ls="--",lw=3,color="white")
        ax1.set_yticks([0,2,4,6,8,10])
        ax1.set_yticklabels(["0","2","4","6","8","10"])
        ax1.set_ylim([0,10])
        
        # Frozen hydrometeors
        ax2=vertical_var_fig.add_subplot(122)
        
        ax2.fill_betweenx(stats_icon_fwc_df.columns/1000,
                          stats_icon_fwc_df.loc[0.05,:],
                          x2=stats_icon_fwc_df.loc[0.95,:],
                          color="lightgreen")
        
        ax2.fill_betweenx(stats_icon_fwc_df.columns/1000,
                          stats_icon_fwc_df.loc[0.25,:],
                          x2=stats_icon_fwc_df.loc[0.75,:],
                          color="mediumseagreen",label="entire AR sector")
        
        ax2.plot(stats_icon_fwc_df.loc[0.5,:],
                 stats_icon_fwc_df.columns/1000,
                 ls="--",lw=3,color="white")
        
        for col in boxplot_fwc_columns:
            bp2=ax2.boxplot(boxplot_fwc[col].dropna().values,
                        positions=[float(col)/1000],
                        notch=False, widths=150/1000,
                        vert=False,patch_artist=True,
                        boxprops={"facecolor":"lightgrey",
                                  "linewidth":3,"alpha":0.3},
                        whiskerprops={"linewidth":3},
                        medianprops={"linewidth":3})
            ax2.set_yticklabels("")
       
        ax2.set_ylim([0,10])
        ax2.set_yticks([0,2,4,6,8,10])
        ax2.set_ylim([0,10])
        ax1_xlim_max=1.0
        ax2_xlim_max=0.5
        ax1.set_ylabel("Altitude (km)")
        ax1.set_xlabel("Water Content (g/kg)")
        ax2.set_xlabel("Water Content (g/kg)")
        ax1.set_xlim([0,ax1_xlim_max])
        ax2.set_xlim([0,ax2_xlim_max])
        ax1.set_xticks([0,ax1_xlim_max/2,ax1_xlim_max])
        ax2.set_xticks([0,ax2_xlim_max/2,ax2_xlim_max])
        ax1.text(0.55*ax1_xlim_max,9,
                 "a) Melted \nHydrometeors",
                 color="blue",fontsize=20,fontweight="bold")
        
        ax2.text(0.55*ax2_xlim_max,9,
                 "b) Frozen \nHydrometeors",
                 color="darkgreen",fontsize=20,fontweight="bold")
        
        for axis in ["left","bottom"]:
            ax1.spines[axis].set_linewidth(3)
        ax1.tick_params(length=10,width=3)
        
        for axis in ["left","bottom"]:
            ax2.spines[axis].set_linewidth(3)
        ax2.tick_params(length=10,width=3)
        #ax2.legend(loc="lower right")
        sns.despine(offset=10)
        
        legend_lnobject=matplotlib.lines.Line2D([0,1],[0,1],
                                                color="black",lw=3)
        patch_object=matplotlib.patches.Patch(color="lightgreen",
                                              edgecolor="black",
                                              linewidth=4,
                                              label="entire AR sector")
        
        ax2.legend([patch_object,legend_lnobject],["entire AR sector","HALO curtain"],
                   handler_map={legend_lnobject:HandlerBoxPlot()},
                   handleheight=3, loc="lower right",fontsize=16)
        fig_name="Mean_Hydrometeor_Internal_Representation.png"
        vertical_var_fig.savefig(self.plot_path+\
                                 fig_name,
                                 dpi=300,bbox_inches="tight")        
        print("Figure saved as:", self.plot_path+fig_name)
        return None
    
        #ax2.fill_betweenx(stats_icon_fwc_df.index[::2]/1000,
        #                  stats_icon_fwc_df["mean"].iloc[::2]-\
        #                      stats_icon_fwc_df["std"].iloc[::2],
        #                  x2=stats_icon_fwc_df["mean"][::2]+\
        #                      stats_icon_fwc_df["std"].iloc[::2],
        #                      color="lightgreen")

        #ax2.errorbar(y=stats_halo_fwc_df.index[::2]/1000,
        #             x=stats_halo_fwc_df["mean"].iloc[::2],
        #             xerr=stats_halo_fwc_df["std"].iloc[::2],
        #             color="darkgreen",label="HALO")
                
        #ax2.fill_betweeny()
        
        #        dwc_df=pd.DataFrame()
        #        dwc_df["DWC"]=np.log10(resample_dwc_df.values.reshape(-1)*1000)
        #        dwc_df["Z_Height"]=resample_z_df.values.reshape(-1)
        #        dwc_df["DWC"][dwc_df["DWC"]<-5]=np.nan
        #        dwc_df["Z_Height"][dwc_df["Z_Height"]>10000]=np.nan
        #        dwc_df.dropna(subset=["DWC","Z_Height"],inplace=True)
        #        dwc_df["species"]="all"
        #        dwc_df["weights"]=np.ones_like(dwc_df["DWC"])/dwc_df["DWC"].shape[0]
            
#        print("SELECT Sample. This takes long")
#        entire_dwc=pd.concat([dwc_df,cutted_hmc])
#        # Plot
        # random_int_logs=np.random.randint(dwc_df.shape[0],size=100000)
        # dwc_df=dwc_df.iloc[random_int_logs,:]
        # axs = sns.JointGrid(data= dwc_df,x='DWC', y='Z_Height', space=1)
        # #axs.x = dwc_df.DWC
        # #axs.y = dwc_df.Z_Height
        
        # axs.plot_joint(sns.kdeplot,kind="hist",shade=True, 
        #                color='blue',zorder=0,levels=3,alpha=0.5)
        
        # axs.ax_joint.scatter('DWC', 'Z_Height', data=cutted_hmc, c='grey',
        #                      marker='x',s=1,alpha=0.3)
        
        # # drawing pdf instead of histograms on the marginal axes
        # axs.ax_marg_x.cla()
        # axs.ax_marg_y.cla()
        # sns.kdeplot(ax=axs.ax_joint,data=cutted_hmc,x="DWC",y="Z_Height",
        #             shade=False,levels=3,
        #             color="black")
        # sns.distplot(cutted_hmc.DWC, ax=axs.ax_marg_x,color="k",
        #             hist_kws={"weights":np.ones_like(cutted_hmc.DWC)/len(cutted_hmc.DWC)})
        # sns.distplot(cutted_hmc.Z_Height, ax=axs.ax_marg_y, color="k",
        #             hist_kws={"weights":np.ones_like(cutted_hmc.Z_Height)/\
        #                       len(cutted_hmc.Z_Height)},
        #             vertical=True)
        # sns.distplot(dwc_df.DWC, ax=axs.ax_marg_x,color="blue",
        #              hist_kws={"weights":np.ones_like(dwc_df.DWC)/\
        #                       len(dwc_df.DWC)})
        # sns.distplot(dwc_df.Z_Height, ax=axs.ax_marg_y, color="blue",
        #              hist_kws={"weights":np.ones_like(dwc_df.Z_Height)/\
        #                       len(dwc_df.Z_Height)},vertical=True)
        # axs.ax_marg_x.set_yticklabels("")
        # axs.ax_marg_y.set_xticklabels("")
        # axs.ax_marg_x.set_ylabel("")
        # axs.ax_marg_y.set_xlabel("")
        # axs.ax_marg_x.set_xticklabels("")
        # axs.ax_marg_y.set_yticklabels("")
        # axs.ax_marg_x.set_xlabel("")
        # axs.ax_marg_y.set_ylabel("")
        
        # axs.ax_joint.set_ylim([0,10000])
        # axs.ax_joint.set_yticks([0, 2000, 4000,6000,8000,10000])
        # axs.ax_joint.set_yticklabels([0, 2, 4,6,8,10])
        # axs.ax_joint.set_xlabel("Specific Water Content in $g/kg$")
        # axs.ax_joint.set_ylabel('Altitude in km')
        # axs.ax_joint.set_xticks([-5, -3, -1,1])
        # axs.ax_joint.set_xticklabels(["$10^{-5}$","$10^{-3}$",
        #                               "$10^{-1}$","$10^{1}$"])
        # axs.ax_joint.set_xlim([-5,1])
        # # drawing pdf instead of histograms on the marginal axes
        # sns.despine(offset=5)
        # #create legend objects
        #     #
        #     #axs.ax_joint.legend()
        # fig_name=self.plot_path+self.flight+\
        #             "_ICON_Internal_Leg_Hydrometeor_Representativeness.png"
        # plt.savefig(fig_name,dpi=300,bbox_inches="tight")
        # print("Figure saved as:",fig_name)
    
    def old_temporary(self):
           print("this function contains nothing and just stores old commented",
                 "routines")
           """
           #Liquid water
           #  ax1=fig.add_subplot(211)
           #  x_temp=halo_era5_hmc["theta_e"].loc[start:end].index#.time
           #  y_temp=np.log10(np.array(halo_era5_hmc["theta_e"].columns.astype(float)))#/1000
           #  y,x=np.meshgrid(y_temp,x_temp)
            
           #  levels=np.linspace(285,350,66)
           #  if flight=="RF10":
           #      levels=np.linspace(275,330,56)
           #  if low_level:
           #      levels=np.linspace(285,320,36)
           #      if flight=="RF10":
           #          levels=np.linspace(275,310,26)
           #  y=np.array(halo_era5_hmc["Z_Geopot"].loc[start:end])
            
           #  #Create new Hydrometeor content as a sum of all water contents
           #  cutted_theta=halo_era5_hmc["theta_e"].loc[start:end]
           #  if do_masking:
           #      cutted_theta=cutted_theta*era5_on_halo.mask_df
           #  #cutted_theta=cutted_theta*#ERA5().apply_orographic_mask_to_era(cutted_theta,
           #                   #                                    threshold=0.3,
           #                   #                                    variable="Theta_e")
           # # norm=BoundaryNorm(levels,ncolors=plt.get_cmap(theta_colormap),clip=True)
           #  C1=ax1.pcolormesh(x,y,cutted_theta,cmap=theta_colormap,
           #                    vmin=levels[0],vmax=levels[-1])
           #  cb = plt.colorbar(C1,extend="both")
           #  cb.set_label('ERA-5: $\Theta_{e}$ in K')
            
           #  if low_level:
           #      ax1.set_ylim(0,6000)
           #      ax1.set_yticks([0,500, 1000,1500,2000,2500, 3000, 
           #                      3500, 4000, 4500, 5000,5500,6000])#, 4000)#,6000,8000,10000,12000])
           #      ax1.set_yticklabels([0,500, 1000, 1500, 2000,2500,3000, 3500,
           #                           4000, 4500, 5000,5500,6000])#, 4000)#,6000,8000,10000,12000])
           #      marker_pos=np.ones(len(high_dbZ))*5800
           #  else:
           #      ax1.set_ylim(0,12000)
           #      ax1.set_yticks([0, 2000, 4000,6000,8000,10000,12000])
           #      ax1.set_yticklabels([0, 2000, 4000,6000,8000,10000,12000])
           #      marker_pos=np.ones(len(high_dbZ))*11000
            
           #  high_dbZ_scatter=ax1.scatter(high_dbZ.index,marker_pos,
           #                              s=35,color="white",marker="D",
           #                              linewidths=0.2,edgecolor="k")
            
            
           #  for label in ax1.yaxis.get_ticklabels()[::2]:
           #      label.set_visible(False)
           #  ax1.set_ylabel('Altitude in m ')
           #  ax1.set_xlabel('')
           #  ax1.xaxis.set_major_locator(hours)
           #  ax1.xaxis.set_major_formatter(h_fmt)
           """