#!/usr/bin/env python

import sys

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
from datetime import date

def piecewise_exponential_cumulative(ages, k1, k2, m=94):
    ### this function calculates a piecewise exponential function, whose integral equals one
    # k1 is the decay rate of first exponential
    # k2 is decay rate of second exponetial
    # m is the boundary between k1 and k2
    # this function represents the steady-state cumulative age distribution for patch ages given harvest rates for 
    # young and mature forests (k1 and k2) where m is the age that distinguishes the two age types
    n_age_bins = len(ages)
    
    # calculate the frequency of youngest age bin
    a = 1./((1./k1)+np.exp(-k1*m)*((-1/k1)+(1/k2)))
    
    # calculate the first piecewise exponential set
    vals = a * np.exp(-k1*ages)
    ##print(len(vals[m:]))
    ##print(len(np.arange(n_age_bins-m)))
    ##print(len(ages.isel(slice(m-n_age_bins, None))))
    
    # calculate the second piecewise exponential set, beginning with the last element ofthe first set
    vals[m:] = vals[m]*np.exp(-k2*np.arange(n_age_bins-m))
    
    # rescale last age bin to account for all of the truncated ages beyond
    #vals[-1] = vals[-1]/k2

    vals_cumulative = vals.cumsum()
    return(vals_cumulative)

# this is the year that LUH2 transient data starts
year_luh2_start = 850

if len(sys.argv) < 4:
    print("Three command-line arguments needed: ")
    print("first is the path to the LUH data file as processed for FATES. ")
    print("second is the year for which to calculate steady-state.")
    print("third is a grid name (used only for output filename, e.g. ne30 or 4x5)")
    sys.exit(4)
    
#fin_luh2 = xr.open_dataset('LUH2_states_transitions_management.timeseries_4x5_hist_simyr1650-2015_c240216.nc')
fin_luh2 = xr.open_dataset(sys.argv[1])

nages = 300
ntime_total = len(fin_luh2.time)

# calculate the year that you want to calculate a steady-state logging rate for
year_to_calc_steadystate = sys.argv[2]
if (year_to_calc_steadystate < year_luh2_start) or (year_to_calc_steadystate > year_luh2_start + ntime_total):
    print("year_to_calc_steadystate must be within the time span of data file")
    print(year_to_calc_steadystate)
    sys.exit(4)

gridname = sys.argv[3]

    
print(fin_luh2.data_vars)
if "lon" in fin_luh2.coords or "lsmlon" in fin_luh2.coords:
    reg_grid = True
    IM = len(fin_luh2.lon)
    JM = len(fin_luh2.lat)
    if "lsmlon" in fin_luh2.dims:
        latname = "lsmlat"
        lonname = "lsmlon"
        lon_coord = fin_luh2.lsmlon
        lat_coord = fin_luh2.lsmlat
    else:
        latname = "lat"
        lonname = "lon"
        lon_coord = fin_luh2.lon
        lat_coord = fin_luh2.lat
else:
    lsmlon = False
    reg_grid = False
    IM = len(fin_luh2.lndgrid)

ages = xr.DataArray(np.arange(nages), dims=['age'], coords=[np.arange(nages)])

age_mature=94

#print(fin_luh2.dims)

area_prim = fin_luh2.primf + fin_luh2.primn

rate_newsec_creation = fin_luh2.primf_harv + fin_luh2.primn_harv + fin_luh2.primf_to_secdn + fin_luh2.primn_to_secdf  + fin_luh2.urban_to_secdf  + fin_luh2.urban_to_secdn  + fin_luh2.c3ann_to_secdf  + fin_luh2.c3ann_to_secdn  + fin_luh2.c4ann_to_secdf  + fin_luh2.c4ann_to_secdn  + fin_luh2.c3per_to_secdf  + fin_luh2.c3per_to_secdn  + fin_luh2.c4per_to_secdf  + fin_luh2.c4per_to_secdn  + fin_luh2.c3nfx_to_secdf  + fin_luh2.c3nfx_to_secdn  + fin_luh2.pastr_to_secdf  + fin_luh2.pastr_to_secdn  + fin_luh2.range_to_secdf  + fin_luh2.range_to_secdn 

rate_sec_loss_transitions = fin_luh2.secdf_to_urban + fin_luh2.secdf_to_c3ann + fin_luh2.secdf_to_c4ann + fin_luh2.secdf_to_c3per + fin_luh2.secdf_to_c4per + fin_luh2.secdf_to_c3nfx + fin_luh2.secdf_to_pastr + fin_luh2.secdf_to_range + fin_luh2.secdn_to_urban + fin_luh2.secdn_to_c3ann + fin_luh2.secdn_to_c4ann + fin_luh2.secdn_to_c3per + fin_luh2.secdn_to_c4per + fin_luh2.secdn_to_c3nfx + fin_luh2.secdn_to_pastr + fin_luh2.secdn_to_range

secmf_harv = fin_luh2.secmf_harv
secyf_harv = fin_luh2.secyf_harv
secnf_harv = fin_luh2.secnf_harv

sectot = fin_luh2.secdf + fin_luh2.secdn
#print(sectot)
#sys.exit(4)

if reg_grid:
    ### initialize the secondary forest age as having all zero age at start of dataset
    age_dist = xr.DataArray(np.zeros((ntime_total,nages,JM,IM)), dims=['time','age',latname,lonname], coords=[fin_luh2.time, ages, lat_coord, lon_coord])
    age_dist[0,0,:,:] = sectot.isel(time=0)

    generic_zeros_array = xr.DataArray(np.zeros((ntime_total,JM,IM)), dims=['time',latname,lonname], coords=[fin_luh2.time, lat_coord, lon_coord])
else:
    ### initialize the secondary forest age as having all zero age at start of dataset
    age_dist = xr.DataArray(np.zeros((ntime_total,nages, IM)), dims=['time','age','lndgrid'], coords=[fin_luh2.time, ages, fin_luh2.lndgrid])
    age_dist[0,0,:] = sectot.isel(time=0)
    generic_zeros_array = xr.DataArray(np.zeros((ntime_total, IM)), dims=['time','lndgrid'], coords=[fin_luh2.time, fin_luh2.lndgrid])

secy_dist = generic_zeros_array.copy()#xr.DataArray(np.zeros((ntime_total,JM,IM)), dims=['time',latname,lonname], coords=[fin_luh2.time, fin_luh2.lsmlat, fin_luh2.lsmlon])
secm_dist = generic_zeros_array.copy()#xr.DataArray(np.zeros((ntime_total,JM,IM)), dims=['time',latname,lonname], coords=[fin_luh2.time, fin_luh2.lsmlat, fin_luh2.lsmlon])

k_secloss_dist = generic_zeros_array.copy()#xr.DataArray(np.zeros((ntime_total,JM,IM)), dims=['time',latname,lonname], coords=[fin_luh2.time, fin_luh2.lsmlat, fin_luh2.lsmlon])
k_secyhar_dist = generic_zeros_array.copy()#xr.DataArray(np.zeros((ntime_total,JM,IM)), dims=['time',latname,lonname], coords=[fin_luh2.time, fin_luh2.lsmlat, fin_luh2.lsmlon])
k_secthar_dist = generic_zeros_array.copy()#xr.DataArray(np.zeros((ntime_total,JM,IM)), dims=['time',latname,lonname], coords=[fin_luh2.time, fin_luh2.lsmlat, fin_luh2.lsmlon])
k_secmhar_dist = generic_zeros_array.copy()#xr.DataArray(np.zeros((ntime_total,JM,IM)), dims=['time',latname,lonname], coords=[fin_luh2.time, fin_luh2.lsmlat, fin_luh2.lsmlon])


for t in range(ntime_total-1):
    if t%50 == 0:
        print('starting year ',t)
    secy = age_dist.isel(time=t).isel(age=slice(0,age_mature)).sum(dim='age')
    secm = age_dist.isel(time=t).isel(age=slice(age_mature,None)).sum(dim='age')
    if reg_grid:
        age_dist[t+1,0,:,:] = rate_newsec_creation.isel(time=t)
    else:
        age_dist[t+1,0,:] = rate_newsec_creation.isel(time=t)
    k_secloss = np.minimum(1.,np.nan_to_num(rate_sec_loss_transitions.isel(time=t)/sectot.isel(time=t)))
    k_secyhar = np.minimum(1.,np.nan_to_num(secyf_harv.isel(time=t)/secy))
    k_secthar = np.minimum(1.,np.nan_to_num(secnf_harv.isel(time=t)/sectot.isel(time=t)))
    k_secmhar = np.minimum(1.,np.nan_to_num(secmf_harv.isel(time=t)/secm))

    for age in range(age_mature):
        age_dist[t+1,age+1] = age_dist[t,age] * (1. - np.minimum(1.,(k_secloss + k_secyhar + k_secthar)))
        age_dist[t+1,0] = age_dist[t+1,0] + age_dist[t,age] * np.minimum(1.,(k_secyhar + k_secthar))
    for age in range(age_mature,nages-2):
        age_dist[t+1,age+1] = age_dist[t,age] * (1.- np.minimum(1.,(k_secloss + k_secmhar + k_secthar)))
        age_dist[t+1,0] = age_dist[t+1,0] + age_dist[t,age] * np.minimum(1.,(k_secmhar + k_secthar))
    age_dist[t+1,nages-1] = (age_dist[t,nages-1] + age_dist[t,nages-2]) * (1.- np.minimum(1.,(k_secloss + k_secmhar + k_secthar)))
    age_dist[t+1,0] = age_dist[t+1,0] + (age_dist[t,nages-1] + age_dist[t,nages-2]) * np.minimum(1.,(k_secmhar + k_secthar))

    if reg_grid:
        secy_dist[t,:,:] = secy
        secm_dist[t,:,:] = secm
        k_secloss_dist[t,:,:] = k_secloss
        k_secyhar_dist[t,:,:] = k_secyhar
        k_secthar_dist[t,:,:] = k_secthar
        k_secmhar_dist[t,:,:] = k_secmhar
    else:
        secy_dist[t,:] = secy
        secm_dist[t,:] = secm
        k_secloss_dist[t,:] = k_secloss
        k_secyhar_dist[t,:] = k_secyhar
        k_secthar_dist[t,:] = k_secthar
        k_secmhar_dist[t,:] = k_secmhar        


age_dist_cum = age_dist.cumsum(dim='age')


mean_age = (age_dist * ages).sum(dim='age') / age_dist.sum(dim='age')


age_dist_cum_norm = age_dist_cum / sectot


median_secondary_age = np.abs(age_dist_cum_norm.fillna(0.) - 0.5).argmin(dim='age', skipna=True)
median_secondary_age = (median_secondary_age * sectot / sectot)


if reg_grid:
    generic_nan_array = xr.DataArray(np.nan * np.ones([JM,IM]), dims=[latname,lonname], coords=[fin_luh2.lat, fin_luh2.lon])
else:
    generic_nan_array = xr.DataArray(np.nan * np.ones([IM]), dims=['lndgrid'], coords=[fin_luh2.lndgrid])

sec_young_harvestrate_rel = generic_nan_array.copy()#xr.DataArray(np.nan * np.ones([JM,IM]), dims=[latname,lonname], coords=[fin_luh2.lat, fin_luh2.lon])
sec_mature_harvestrate_rel = generic_nan_array.copy()#xr.DataArray(np.nan * np.ones([JM,IM]), dims=[latname,lonname], coords=[fin_luh2.lat, fin_luh2.lon])


time = year_to_calc_steadystate - year_luh2_start
n_age_max = 150

x = ages.isel(age=slice(0,n_age_max)).data
fails = generic_nan_array.copy()#xr.DataArray(np.nan * np.ones([JM,IM]), dims=[latname,lonname], coords=[fin_luh2.lat, fin_luh2.lon])
if reg_grid:
    for i in range(IM):
        for j in range(JM):
            if latname == "lsmlat":
                if sectot.sel(lsmlat=j,lsmlon=i,time=time) > 0.:
                    y = age_dist_cum_norm.isel(lsmlat=j,lsmlon=i,time=time,age=slice(0,n_age_max)).data
                else:
                    continue
            else:
                if sectot.isel(lat=j,lon=i,time=time) > 0.:
                    y = age_dist_cum_norm.isel(lat=j,lon=i,time=time,age=slice(0,n_age_max)).data
                else:
                    continue               
            try:
                params = curve_fit(piecewise_exponential_cumulative, x, y, p0=[0.01,0.01], bounds=[0,1])
                #print(params)
                sec_young_harvestrate_rel[j,i] = params[0][0]
                sec_mature_harvestrate_rel[j,i] = params[0][1]
            except:
                fails[j,i] = 1.
                sec_young_harvestrate_rel[j,i] = 0.05  ### set to some nominal value
                sec_mature_harvestrate_rel[j,i] = 0.05  ### set to some nominal value
else:
    for i in range(IM):
        if sectot.sel(lndgrid=i,time=time) > 0.:
            y = age_dist_cum_norm.isel(lndgrid=i,time=time,age=slice(0,n_age_max)).data
            try:
                params = curve_fit(piecewise_exponential_cumulative, x, y, p0=[0.01,0.01], bounds=[0,1])
                #print(params)
                sec_young_harvestrate_rel[i] = params[0][0]
                sec_mature_harvestrate_rel[i] = params[0][1]
            except:
                fails[i] = 1.
                sec_young_harvestrate_rel[i] = 0.05  ### set to some nominal value
                sec_mature_harvestrate_rel[i] = 0.05  ### set to some nominal value    
        


sec_young_harvestrate_rel.plot()


sec_mature_harvestrate_rel.plot()


fails.plot()


# replace nans with zeros in harvest rates
sec_young_harvestrate_rel = sec_young_harvestrate_rel.fillna(0.)
sec_mature_harvestrate_rel = sec_mature_harvestrate_rel.fillna(0.)


n_ts_out = 500
fout = fin_luh2.sel(time=[time]*n_ts_out)


# first zero all transitions
varlist =  list(fout.variables)
matchers = ['_to_','_harv','_bioh']
matching_vars = [s for s in varlist if any(xs in s for xs in matchers)]
#matching_vars

# Loop over variables governing transition rates and zero them
for var in matching_vars:
    # Access the variable data
    fout[var] = fout[var] * 0.


# now rewrite the secondary harvest rates based on the fits above
if reg_grid:
    for i_ts in range(n_ts_out):
        fout.secyf_harv[i_ts,:,:] = sec_young_harvestrate_rel * (fout.secdf.isel(time=0) + fout.secdn.isel(time=0)).data
        fout.secmf_harv[i_ts,:,:] = sec_mature_harvestrate_rel * (fout.secdf.isel(time=0) + fout.secdn.isel(time=0)).data
else:
    for i_ts in range(n_ts_out):
        fout.secyf_harv[i_ts,:] = sec_young_harvestrate_rel * (fout.secdf.isel(time=0) + fout.secdn.isel(time=0)).data
        fout.secmf_harv[i_ts,:] = sec_mature_harvestrate_rel * (fout.secdf.isel(time=0) + fout.secdn.isel(time=0)).data

fout['time'] = np.arange(n_ts_out)
fout['YEAR'][:] = np.arange(n_ts_out)

fout.to_netcdf('LUH2_states_transitions_management.timeseries_'+gridname+'_hist_steadystate_'+str(year_to_calc_steadystate)+'_'+str(date.today())+'.nc')


