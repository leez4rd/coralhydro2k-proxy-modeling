#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:28:13 2021

@author: alyssaatwood
"""

# unmodified original code, with the only exception being the directories to get the data 

import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power 
from plotnine import *
import h5py                  # for reading in .mat files
from netCDF4 import Dataset  # for reading netcdf file
from datetime import date, datetime, timedelta   # to convert "days since XXX" to date
import xarray as xr
import datetime as dt2         # to create a datetime object
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from pandas.tseries.offsets import MonthBegin
from fit_bivariate import bivariate_fit

def decyrs_to_datetime(decyrs):      # converts date in decimal years to a datetime object (from https://stackoverflow.com/questions/20911015/decimal-years-to-datetime-in-python)
    output = np.empty([len(decyrs)],dtype='datetime64[s]')
    for l in range(len(decyrs)):
        start = decyrs[l]                   # this is the input (time in decimal years)
        year = int(start)
        rem = start - year
        base = datetime(year, 1, 1)
        result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
        result2 = result.strftime("%Y-%m-%d")   # truncate to just keep YYYY-MM-DD (get rid of hours, secs, and minutes)
        #print(l, result2)
        output[l] = np.datetime64(result2)
    return output

# Loosely based off: https://community.alteryx.com/t5/Alteryx-Designer-Discussions/Round-a-date-to-the-nearest-month/td-p/624262
def round_nearest_month(values):      # rounds a numpy datetime64 array to the nearest month (1st of the nearest month)
    dat = {'date': values}
    df = pd.DataFrame(data=dat)
    d = df['date'].dt.day
    for i in range(len(d)):
        if d[i] > 15:
            df.date[i] = pd.to_datetime(df.at[i, 'date']) + MonthBegin(1)   # ceiling month
        else:
            df.date[i] = df.date[i] - pd.Timedelta('1 day') * (df.date.dt.day[i] - 1)  # floor month
            #df.date = df.index.floor('M')
    df_array = np.array(df)        # converts pandas dataframe back to numpy array
    output = df_array[:,0]       # remove singleton dimension
    return output

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (idx, array[idx])   # return index and its value as a tuple

def oma(X):
   μ = X.mean(axis=0)
   λ, V = LA.eig(np.cov(X.T))
   imin = λ.argmin()
   a = V[:,[imin]] # the trailing eigenvector 
   b = -a[:2]/a[2]
   b0 = μ.dot(np.vstack([-b, [1.0]]))
   return np.vstack([b0, b])

def tls(Z):
   μ = Z.mean(axis=0)
   U, s, Vt = LA.svd(Z-μ)
   a = (Vt.T)[:,[2]] # The trailing right singular vector 
   b = -a[:2]/a[2]
   b0 = μ.dot(np.vstack([-b, [1.0]]))
   return np.vstack([b0, b])

def quad(d, W):
   return np.diag(np.dot(W.T, np.dot(np.diag(d), W)))

def bls(X, varX, Y, varY, tol=1e-5, verbose=False, PC=False): 
   n, p = X.shape
   W = fractional_matrix_power(np.cov(X.T), -0.5) 
   Ra = np.column_stack((np.ones(n), X))
   varR = varX
   if PC:
      μx = X.mean(axis=0)
      R = (X-μx) @ W
      Ra = np.column_stack((np.ones(n), R))
      # apply quadratic form to each row of varX:
      varR = np.apply_along_axis(lambda d: quad(d, W), 1, varX)
      
   Z = np.hstack((Ra[:, 1:], Y)) 
   bnew = tls(Z)
   # print(bnew)
   b = np.zeros((bnew.shape[0], 1)) 
   Rt = Ra.T
   varRt = varR.T
   
   for k in range(100): 
       b = bnew
       ei = Y - (Ra @ b)
       Sei2 = varY + varR @ np.square(b[1:])
       wi = 1.0 / Sei2
       ei2 = np.square(ei*wi)
       R = Rt @ (Ra*wi)
       g = Rt @ (Y*wi) + np.vstack(([0.0], varRt @ ei2)) * b 
       bnew = LA.lstsq(R, g, rcond=None)[0]
       if all(np.abs(b-bnew)<tol):
          break 
   if PC:
       bnew = np.vstack([bnew[0], W @ bnew[1:]]) 
       
   return bnew

def lsq(X, varX, Y, varY):
   n = Y.shape[0]
   μx = X.mean(axis=0)
   μy = Y.mean()
   Xa = np.column_stack((np.ones(n), X))
   Bols = LA.lstsq(Xa.T @ Xa, Xa.T @ Y, rcond=None)[0]
   Bbls = bls(X, varX, Y, varY)
   Bpcbls = bls(X, varX, Y, varY, PC=True)
   B = np.hstack((Bols, Bbls, Bpcbls)).T
   # intercepts for the marginal functions (for plotting)
   intm = μy - μx * B[:,1:]
   method = pd.Categorical(["OLS", "BLS", "PCBLS"],["OLS", "BLS", "PCBLS"], ordered=True)
   label = [f"δ18Oc = {b1:.3f}SST + {b2:.3f}SSS + {b0:.2f}" for b1, b2, b0 in zip(B[:,1], B[:,2], B[:,0])]
   Xcor = np.corrcoef(X, rowvar=False)[1,0]
   return pd.DataFrame({"method": method, "b0": B[:,0], "b1": B[:,1], "b2": B[:,2], "int1": intm[:,0], "int2": intm[:,1], "label": label, "Xcor": round(Xcor,2)})

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

#============================================================================
# Read in data
#============================================================================

# Read in data from csv file
#d = pd.read_csv("./Cocos-Keeling_annual.csv", sep="[±,]", engine="python") 
#d.head(3)
#Y = d[['d18Oc']].values
#varY = np.square(d[['d18Oc']].values)
#Y[:3]
# Read in SST and SSS data
#X = d[['ersstv5', 'HadEN4']].values
#varX = np.square(d[['ersstv5e', 'HadEN4e']].values)

#============================================================================
# Read in coral d18O data
#============================================================================

# Indian Ocean corals
coralid1 = 'HE18COC01'    # Hennekam (2018) - Cocos (Keeling) Islands
# coralid1 = 'HE18COC02'
# coralid1 = 'ZI08MAY01'
#coralid1 = 'ST13MAL01'    # Storz (2013) - Rasdhoo Atoll, Maldives
#coralid1 = 'ZI08MAY01'    # Zinke (2008) - Mayotte
#coralid1 = 'PF19LAR01'    # Pfeiffer (2004) - St. Gilles Reef, La Réunion

# Central Pacific corals
#coralid1 = 'NU11PAL01'    # Nurhati (2011) - Palmyra (Tier 1)
#coralid1 = 'NU09FAN01'    # Nurhati (2009) - Fanning Island (Tier 2)
#coralid1 = 'NU09CHR01'    # Nurhati (2009) - Christmas Island (Tier 2)

# Others
# coralid1 = 'WU14CLI01'    # Wu (2014) - Clipperton
# coralid1 = 'GO08BER01'    # Goodkin (2008) - Bermuda
#coralid = 'ST13MAL01'    # Storz (2013) - Maldives
#coralid1 = 'CA14TIM01'    # Cayharini (2014) - Timor, Indonesia
#coralid1 = 'CO03CHBA01A' # more salinity variation (from slack)
# coralid1 = 'CA13TUR01' # nan error 

# Define analysis interval
time_step = 'year'
#time_step = 'bimonthly'

#dir = '/Users/alyssaatwood/Dropbox/Florida_State/Research/Research_projects/CoralHydro2k/Coral_database/'
#dir = '/Users/alyssa_atwood/Desktop/Dropbox/Florida_State/Research/Research_projects/CoralHydro2k/Coral_database/'
f = h5py.File('./hydro2kv0_5_2.mat','r')
data1 = f.get('ch2k/'+coralid1+'/d18O')

dataset = f.get('ch2k/')
coralnames = list(dataset.keys())

timec01 = np.array(data1[0,:])-1/24     # time in fractional year CE (subtract 1/24 so that time corresponds to start of month (as with sst datasets), rather than mid-month)

d18Oc1 = np.array(data1[1,:])           # Convert to NumPy array
timec1 = decyrs_to_datetime(timec01)    # convert time from decimal year YYYY.FF to datetime object (YYYY-MM-DD)
nt = len(timec1)

# Analytic error in d18Oc (assumes the values are in per mille!)
data_analerr = f.get('ch2k/'+coralid1+'/d18O_errAnalytic')
d18Oc_analerr = data_analerr[0,:]      # analytical error is a single value (in per mille)

print(timec1.shape)
print(timec1)
# Round time steps to nearest month (just a couple days off in some months)
timec1mo = round_nearest_month(timec1)
#timec1mo.dtypes    # print data type for pandas dataframe

if 'coralid2' in locals():                  # if a second coral id (coralid2) exixts, merge the two data sets
    data2 = f.get('ch2k/'+coralid2+'/d18O')
    timec02 = np.array(data2[0,:])          # time in fractional year CE
    d18Oc2 = np.array(data2[1,:])           # Convert to NumPy array
    timec2 = decyrs_to_datetime(timec02)    # convert time from decimal year YYYY.FF to datetime object (YYYY-MM-DD)
    timec2mo = round_nearest_month(timec2)

    # Merge records
    timec = np.concatenate((timec1mo,timec2mo),axis=0)
    d18Oc = np.concatenate((d18Oc1,d18Oc2),axis=0)
else:
    timec = timec1mo
    d18Oc = d18Oc1
    
#data_errunits = f.get('ch2k/'+coralid+'/units_d18O_errAnalytic')
#d18Ocerr_units = np.array2string(data_errunits)           #  Read in string from file usually this is a single value (in per mille)
#d18Ocerr_units = bytes(f.get('ch2k/'+coralid+'/units_d18O_errAnalytic')[:]).decode('utf-8')

# Create DataArray object by specifying coordinates and dimension names (xarray) from numpy array
d18Oc = xr.DataArray(d18Oc, coords=[timec], dims=["time"])
d18Oc = d18Oc.dropna(dim='time', how='any', thresh=None)  # drop all nans in array (across 0th dim)

d18Oc = d18Oc.sortby(d18Oc.time,ascending=True)                   # sort arrays by time
timec = d18Oc.time


latc = f.get('ch2k/'+coralid1+'/lat')
lonc = f.get('ch2k/'+coralid1+'/lon')  # lon = -180:180
latc = np.array(latc)                 # Convert to NumPy array
lonc = np.array(lonc)                 # Convert to NumPy array
print(latc, lonc)

#============================================================================
# Read in obs SST and uncertainty data
#============================================================================

#f2 = Dataset(dir + 'ersstev5_uncertainty.nc')          # total uncertainty estimate time x lat x lon (based on 1000-member ensemble reconstruction; Matt Fischer sent to me) 
#time_sster_all = f2.variables['time'][:]               # time (1/1854-12/2017; units = "days since 1900-01-01 00:00:00"): NOTE DIFFERENT UNITS AND ENDPOINT TO ERSSTV5 (SAME START TIME THOUGH)
#sster_all  = f2.variables['ut'][:,:,:]                 # time x lat x lon 
#lat_sster_all = f2.variables['latitude'][:]            # lat(89)
#lon_sster_all = f2.variables['longitude'][:]           # lon (180)

#============================================================================
# Read in ERSSTv5
#============================================================================
#dir = '/Users/alyssaatwood/Dropbox/Obs_datasets/'
#dir = '/Users/alyssa_atwood/Desktop/Dropbox/Obs_datasets/'
ds = xr.open_dataset('./sst_mnmean_noaa_ersstv5.nc',decode_times=True)         # ERSST v5
print(ds)

# Change longitude array from 0:360 to -180:180
lon_name = 'lon'  # whatever name is in the data
# Adjust lon values to make sure they are within (-180, 180)
ds['_longitude_adjusted'] = xr.where(ds[lon_name] > 180, ds[lon_name] - 360, ds[lon_name])
# reassign the new coords to as the main lon coords and sort DataArray using new coordinate values
ds = (
    ds
    .swap_dims({lon_name: '_longitude_adjusted'})
    .sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)})
    .drop(lon_name))
ds = ds.rename({'_longitude_adjusted': lon_name})
print(ds)

# Read in variables
sst_all = ds['sst']
time_sst_all = ds['time']             # ERSSTv5: time (1/1854-12/2020; units = "days since 1800-1-1 00:00:00")
lat_sst_all = sst_all['lat'].values   # get coordinates from dataset
lon_sst_all = sst_all['lon'].values   # lon = 0:360
#years = sst_all['time'].values
#nt , ny, nx = sst_all.shape
#print(years)
print(time_sst_all)
print(type(time_sst_all))

# Get indices and sst data at closest grid point to coral
(indlat_sst, latval_sst) = find_nearest(lat_sst_all, latc)
(indlon_sst, lonval_sst) = find_nearest(lon_sst_all, lonc)
#(indtime_sst, timeval_sst) = find_nearest(time_sst_all , timec[0])   # coral data steps on midpoints, sst data steps on first day of month... find closest time in sst data set to start of coral data
sst_f = sst_all[:,indlat_sst,indlon_sst]
print(latval_sst)
print(timec[0])

# Match ages of SST and SSS data to coral data
#sst_final = sst.sel(time = timec, method='nearest')
print(sst_f)

#============================================================================
# Read in ERSSTv5 uncertainties 
#============================================================================
ds = xr.open_dataset('./ersstev5_uncertainty.nc',decode_times=True)         # ERSST v5
# Change longitude array from 0:360 to -180:180
lon_name = 'longitude'  # whatever name is in the data
# Adjust lon values to make sure they are within (-180, 180)
ds['_longitude_adjusted'] = xr.where(ds[lon_name] > 180, ds[lon_name] - 360, ds[lon_name])
# reassign the new coords to as the main lon coords and sort DataArray using new coordinate values
ds = (
    ds
    .swap_dims({lon_name: '_longitude_adjusted'})
    .sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)})
    .drop(lon_name))
ds = ds.rename({'_longitude_adjusted': lon_name})
print(ds)

sster_all = ds['ut']
time_sster_all = ds['time']             # ERSSTv5: time (1/1854-12/2020; units = "days since 1800-1-1 00:00:00")
lat_sster_all = sster_all['latitude'].values   # get coordinates from dataset
lon_sster_all = sster_all['longitude'].values

(indlat_sster, latval_sster) = find_nearest(lat_sster_all, latc)
(indlon_sster, lonval_sster) = find_nearest(lon_sster_all, lonc)
sster_f = sster_all[:,indlat_sster,indlon_sster]
print(time_sster_all)

# Match ages of SST and SSS data to coral data
#sster_final = sster.sel(time = timec, method='nearest')
print(sster_f)

#============================================================================
# Read in obs SSS and uncertainty data
#============================================================================
#dir = '/Users/alyssaatwood/Dropbox/Obs_datasets/Salinity/HadEN4/'
#dir = '/Users/alyssa_atwood/Desktop/Dropbox/Obs_datasets/Salinity/HadEN4/'
ds = xr.open_dataset('./sss_HadleyEN4.2.1g10_190001-201012.nc',decode_times=True)         # ERSST v5
print(ds)
sss_all = ds['sss']
time_sss_all = ds['time']             # HadEN4: time x lat x lon (time = 1900:2010)
lat_sss_all = sss_all['lat'].values   # get coordinates from dataset
lon_sss_all = sss_all['lon'].values
print(time_sss_all)

# Get indices and sst data at closest grid point to coral
(indlat_sss, latval_sss) = find_nearest(lat_sss_all, latc)
(indlon_sss, lonval_sss) = find_nearest(lon_sss_all, lonc)
sss_f = sss_all[:,indlat_sss,indlon_sss]

# Match ages of SST and SSS data to coral data
#sss_final = sss.sel(time = timec, method='nearest')
print(sss_f)

#============================================================================
# Read in HadEN4 uncertainties 
#============================================================================
ds = xr.open_dataset('./sss_HadleyEN4.2.1g10_190001-202012_salinityerrorSD.nc',decode_times=True)         # ERSST v5
ssser_all = ds['SALT_ERR_STD']
time_ssser_all = ds['TIME']               # HadEN4 error: time x 1 x lat x lon (time = 1900:2020)
lat_ssser_all = ssser_all['LAT'].values   # get coordinates from dataset
lon_ssser_all = ssser_all['LONN180_180'].values

# Get rid of singleton depth dimension - Create new DataArray object by specifying coordinates and dimension names (xarray) from numpy array
ssser_all = xr.DataArray(ssser_all[:,0,:,:], coords=[time_ssser_all,lat_ssser_all,lon_ssser_all], dims=["time","lat","lon"])

(indlat_ssser, latval_ssser) = find_nearest(lat_ssser_all, latc)
(indlon_ssser, lonval_ssser) = find_nearest(lon_ssser_all, lonc)
ssser_f = ssser_all[:,indlat_ssser,indlon_ssser]
print(time_ssser_all)

# Match ages of SST and SSS data to coral data
#ssser_final = ssser.sel(time = timec, method='nearest')
print(ssser_f)


# Interpolate sst and coral data to monthly climatology
#f_cont = interp1d(lat,pr_tavg_cont[:,ilon],kind='linear')   # creates a fn y(x)
         
         
#f1 = Dataset(dir + 'sst_mnmean_noaa_ersstv5.nc')       # ERSST v5
#time_sst_all = f1.variables['time'][:]           # ERSSTv5: time (1/1854-12/2020; units = "days since 1800-1-1 00:00:00")
#sst_all  = f1.variables['sst'][:,:,:]            # time x lat x lon
#lat_sst_all = f1.variables['lat'][:]             # lat
#lon_sst_all = f1.variables['lon'][:]             # lon 
#(indlat_sst, latval_sst) = find_nearest(lat_sst_all, latc)
#(indlon_sst, lonval_sst) = find_nearest(lon_sst_all, lonc)
#sst = sst_all[:,indlat_sst,indlon_sst]
#print(time_sst_all)

# Convert date from "days since XX" to decimal year
#days = time_sst_all[0]         # This may work for floats in general, but using integers is more precise (e.g. days = int(9465.0))
#start = date(1800,1,1)      # This is the "days since" part
#delta = timedelta(days)     # Create a time delta object from the number of days
#offset = start + delta      # Add the specified number of days to start date
#print(offset)               # >>>  this is the date in format YYYY-MM-DD (eg 2015-12-01)
#print(type(offset))         # >>>  <class 'datetime.date'>
#mo = offset.month         # To Get month from date object
#yr = offset.year          # To Get year from date object
#decyr = yr + (mo-1)/12 + 1/12/2  # steps on midpoints of months (jan = X.04, feb = @.0.12, etc)
#print(offset[0])
#print(decyr[0])

#date_time = datetime.fromtimestamp(offset)
#d = offset.strftime("%z")
#print(d)

#============================================================================
# Average sst and coral data into years (can also calculate seasonal averages using 'time.season')

#d18Oc_yr = d18Oc.groupby('time.year').mean('time')   # annual mean    
#d18Oc_yr = d18Oc.groupby('time.month').mean('time')   # monthly climatology    
#d18Oc_yr = d18Oc.groupby('time.month').mean('time')[0:2]   # monthly climatology of Dec-Mar

#print(d18Oc.time[0:24])     # print months of first 11 data points
#print(d18Oc["time.month"][0:24])     # print months of first 11 data points
#print(d18Oc[0:24])     # print months of first 11 data points

#nyr = np.int(len(d18Oc)/12)   # nearest integer, rounded down

# Initialize DataArrays
#d18Oc_tropyr = xr.DataArray('NA', coords=[np.arange(nyr)], dims=["time"])  # initialize empty DataArray dims = nyr
#sst_tropyr = xr.DataArray('NA', coords=[np.arange(nyr)], dims=["time"])  # initialize empty DataArray dims = nyr
#sster_tropyr = xr.DataArray('NA', coords=[np.arange(nyr)], dims=["time"])  # initialize empty DataArray dims = nyr
#sss_tropyr = xr.DataArray('NA', coords=[np.arange(nyr)], dims=["time"])  # initialize empty DataArray dims = nyr
#ssser_tropyr = xr.DataArray('NA', coords=[np.arange(nyr)], dims=["time"])  # initialize empty DataArray dims = nyr

# Avg data over the tropical year (Apr 1 - Mar 31) (Verified data for HE18COC01)
coral_years = np.array(d18Oc.time.dt.year)   # array of years in coral data

#=============================================================================
# Select common time period
#=============================================================================

#startyr = coral_years[0]        # start on first year of coral data (this may be old!)
#startyr = 1975           # start on a specified year
#startyr = 1979           # start on a specified year
#startyr = 1960           # start on a specified year
startyr = coral_years[0]           # start on a specified year
endyr = coral_years[-1]
nyr = endyr-startyr      # set the last year for the tropical averages as the final year of coral data
print(nyr)
print(d18Oc.time.dt.year)
print(d18Oc)

# Truncate data to min/max of overlapping ages of all data sets
#t1 = max(d18Oc.time[0],sst_final.time[0],sss_final.time[0],sster_final.time[0],ssser_final.time[0])       # find latest start date of all data sets
#t2 = min(d18Oc.time[-1],sst_final.time[-1],sss_final.time[-1],sster_final.time[-1],ssser_final.time[-1])  # find earliest end date of all data sets
#d18Oc = d18Oc.sel(time=slice(t1, t2))
#sst_final = sst_final.sel(time=slice(t1, t2))
#sss_final = sss_final.sel(time=slice(t1, t2))
#sster_final = sster_final.sel(time=slice(t1, t2))
#ssser_final = ssser_final.sel(time=slice(t1, t2))

# Select time period
t1 = dt2.datetime(startyr, 1, 1)     
t2 = dt2.datetime(endyr, 12, 31) 
#t1 = d18Oc.time[0]   
#t2 = d18Oc.time[-1] 
d18Oc = d18Oc.sel(time=slice(t1, t2))
sst_f = sst_f.sel(time=slice(t1, t2))
sss_f = sss_f.sel(time=slice(t1, t2))
sster_f = sster_f.sel(time=slice(t1, t2))
ssser_f = ssser_f.sel(time=slice(t1, t2))

#=============================================================================

# Interpolate all data to SST dates (monthly data)
#sst_interp = sst_final         
#d18Oc_interp = d18Oc.interp(time=sst_final.time)
#sss_interp = sss_final.interp(time=sst_final.time)
#sster_interp = sster_final.interp(time=sst_final.time)
#ssser_interp = ssser_final.interp(time=sst_final.time)

# Interpolate all data to coral dates
#d18Oc_interp = d18Oc 
#sst_interp = sst_final.interp(time=d18Oc.time)
#sss_interp = sss_final.interp(time=d18Oc.time)
#sster_interp = sster_final.interp(time=d18Oc.time)
#ssser_interp = ssser_final.interp(time=d18Oc.time)

#d18Oc_bin = d18Oc.groupby("time.month").mean()  # calculates monthly climatology

#=============================================================================
# Bin all data from the same site into 2-month bins
#=============================================================================
# Detrend the data
#d18Oc_dt1 = signal.detrend(d18Oc)
#sst_dt1 = signal.detrend(sst_final)
#sss_dt1 = signal.detrend(sss_final)

# Detrend the SST, SSS, and d18Oc data, but retain the intercept (so just remove the trend but the values aren't centered around 0)
time_d18Oc = d18Oc.time.dt.year+d18Oc.time.dt.month/12-1/24 - d18Oc.time.dt.year[0]       # subtract the first year so the intercept is defined at start year 
time_sst = sst_f.time.dt.year+sst_f.time.dt.month/12-1/24 - sst_f.time.dt.year[0]
time_sss = sss_f.time.dt.year+sss_f.time.dt.month/12-1/24 - sss_f.time.dt.year[0]
print(time_sst)


# Detrend  the data but retain the intercept
d18O_res = stats.linregress(time_d18Oc, d18Oc)
sst_res = stats.linregress(time_sst, sst_f)
sss_res = stats.linregress(time_sss, sss_f)

d18Oc_dt1 = d18Oc-d18O_res.slope*time_d18Oc
sst_dt1 = sst_f-sst_res.slope*time_sst
sss_dt1 = sss_f-sss_res.slope*time_sss

sster_dt1 = sster_f                    # Don't detrend the errors (confirmed with Matt F.)
ssser_dt1 = ssser_f                    # Don't detrend the errors 

# Convert back to DataArrays (detrend converts to numpy array)
#d18Oc_dt = xr.full_like(d18Oc, d18Oc_dt1)
d18Oc_dt = xr.DataArray(d18Oc_dt1, coords=[d18Oc.time], dims=["time"])
sst_dt = xr.DataArray(sst_dt1, coords=[sst_f.time], dims=["time"])
sss_dt = xr.DataArray(sss_dt1, coords=[sss_f.time], dims=["time"])
sster_dt = xr.DataArray(sster_dt1, coords=[sster_f.time], dims=["time"])
ssser_dt = xr.DataArray(ssser_dt1, coords=[ssser_f.time], dims=["time"])

# Downsample to 2-month bins (nan if no data in a given month), using time bins from d18Oc data
#   preferred over monthly interpolation, which can introduce errors
#d18Oc_bin = d18Oc_dt.resample(time="2M",label="left",loffset="MS").mean()       # bin into 2-month windows on the start of the next month
##delta = np.array(sst_dt.time[0]) - np.array(d18Oc_bin.time[0])
##sst_bin = sst_dt.resample(time="2M",label = "left",loffset=delta).mean()   
#sst_bin = sst_dt.resample(time="2M",label="left",loffset="MS").mean()   
#sss_bin = sss_dt.resample(time="2M",label="left",loffset="MS").mean()   
#sster_bin = sster_dt.resample(time="2M",label="left",loffset="MS").mean()   
#ssser_bin = ssser_dt.resample(time="2M",label="left",loffset="MS").mean()  

#sst_2mo = sst_dt.resample(time="2M",label = "left",loffset="MS").mean()     # this steps on the 1st day of every other month
#d18Oc_2mo = d18Oc_dt.resample(time="2M",label="left",loffset="MS").mean()   # bin into 2-month windows on the start of the next month


#groupby_bins: 3bins (int or array-like) – If bins is an int, it defines the number of equal-width bins 
#  in the range of x. However, in this case, the range of x is extended by .1% on each side to include the min or max values of x. If bins is a sequence it defines the bin edges allowing for non-uniform bin width. No extension of the range of x is done in this case.
  #nbins = int(len(d18Oc_dt))
  #lb = np.array(d18Oc.time[0:-1])    # time_bins labeled with START of bin (must be in the form of numpy array)
  #d18Oc_bin = d18Oc_dt.groupby_bins('time',bins=d18Oc.time,labels=lb).mean()   # group the data into the same time bins
# Define bins from sst data (get error when define from coral data if have multiple records b/c of duplicated bins)
lb = np.array(sst_dt.time[::2])    # every other sst time step = every other month. time_bins labeled with START of bin (must be in the form of numpy array)
lb = lb[0:-1]    # Bin labels must be one fewer than the number of bin edges
d18Oc_bin = d18Oc_dt.groupby_bins('time',bins=sst_dt.time[::2],labels=lb).mean()   # group the data into the same time bins
sst_bin = sst_dt.groupby_bins('time',bins=sst_dt.time[::2],labels=lb).mean()   
sss_bin = sss_dt.groupby_bins('time',bins=sst_dt.time[::2],labels=lb).mean()
sster_bin = sster_dt.groupby_bins('time',bins=sst_dt.time[::2],labels=lb).mean()
ssser_bin = ssser_dt.groupby_bins('time',bins=sst_dt.time[::2],labels=lb).mean()


#=============================================================================
# Average data and propogate errors over the tropical year (Apr 1-Mar 31)
#=============================================================================
if time_step == 'year':
    # Initialize arrays
    d18Oc_final = np.empty([nyr], dtype=float)
    d18Ocerr_final = np.empty([nyr], dtype=float)
    sst_final = np.empty([nyr], dtype=float)
    sster_final = np.empty([nyr], dtype=float)
    sss_final = np.empty([nyr], dtype=float)
    ssser_final = np.empty([nyr], dtype=float)
    yr_d18Ocfinal = np.empty([nyr], dtype=float)
    yr_sstfinal = np.empty([nyr], dtype=float)
    yr_sssfinal = np.empty([nyr], dtype=float)
    
    count = 0
    for i in range(nyr):
       t1 = dt2.datetime(startyr+count, 4, 1)     # take time slice of dates in the tropical year (Apr 1-Mar 31)
       t2 = dt2.datetime(startyr+1+count, 3, 31)  # t1 = Apr 1, Year 1; t2 = Mar 31, Year 2
       #yr[i] = startyr+count
    
       # d18O and error
       sub = d18Oc_bin.sel(time_bins=slice(t1, t2))       # select time slice of the tropical year
       d18Oc_final[i] = sub.mean(axis=0,skipna='True')
       # Calc total error in d18Oc
       d18Oc_se = np.array(sub.std(axis=0,skipna='True')/(len(sub)**0.5))   # calculate standard error of the data for that year and convert to numpy array
       #d18Ocerr_final[i] = ((d18Oc_analerr**2) + (d18Oc_se**2))**0.5  # this is the combined error in d18Oc from: (1) standard deviation of the mean over the year, and (2) the analyical error, combined in quadrature. See Eqn 4.26 in Taylor "Error Analysis" Textbook)
       # From Matt Fischer:
       #sigma_b = d18Oc_se
       #b = d18Oc_final[i]
       #sigma_slope = np.abs(b)*((sigma_b/b)**2)**0.5
       sigma_slope = d18Oc_se
       d18Ocerr_final[i] = ((sigma_slope**2) + (d18Oc_analerr**2))**0.5  # this is the combined error in d18Oc from: (1) standard deviation of the mean over the year, and (2) the analyical error, combined in quadrature. 
       #d18Ocerr_final[i] = d18Oc_analerr/(len(sub)**0.5)  # this is the combined error in d18Oc from: (1) standard deviation of the mean over the year, and (2) the analyical error, combined in quadrature. 
       yr_d18Ocfinal[i] = pd.to_datetime(np.array(sub.time_bins[0])).year    # year of start of bins (convert to np array, then pandas datetime object to extract year)
     
       # SST and error
       sub = sst_bin.sel(time_bins=slice(t1, t2))       # select time slice of the tropical year
       sst_final[i] = sub.mean(axis=0,skipna='True')
       yr_sstfinal[i] = pd.to_datetime(np.array(sub.time_bins[0])).year    # year of start of bins (convert to np array, then pandas datetime object to extract year)
       
       sub2 = sster_bin.sel(time_bins=slice(t1, t2))       # select time slice of the tropical year
       #sster_final[i] = sub.mean(axis=0)
       # Combine two sources of error:
       #sigma_avger = sub.mean(axis=0)/(len(sub)**0.5)                                    # to convert a monthly standard error into an annual one, the general rule is to divide by sqrt(N)
       #sub2 = sst_interp.sel(time=slice(t1, t2))
       #sigma_se = np.array(sub2.std(axis=0)/(len(sub2)**0.5))   # calculate standard error of the data for that year and convert to numpy array
       #sigma_slope = np.abs(b)*((sigma_b/b)**2)**0.5
       sigma_slope = np.array(sub.std(axis=0,skipna='True')/(len(sub)**0.5))   # calculate standard error of the data for that year and convert to numpy array
       #sster_final[i] = sigma_avger  # this is the combined error in d18Oc from: (1) standard deviation of the mean over the year, and (2) the analyical error, combined in quadrature.
       sst_analerr = ((np.sum(sub2**2))**0.5)/len(sub2)
       sster_final[i] = ((sigma_slope**2) + (sst_analerr**2))**0.5  # this is the combined error in d18Oc from: (1) standard deviation of the mean over the year, and (2) the analyical error, combined in quadrature. 
    
       # SSS and error
       sub = sss_bin.sel(time_bins=slice(t1, t2))       # select time slice of the tropical year
       sss_final[i] = sub.mean(axis=0,skipna='True')
       yr_sssfinal[i] = pd.to_datetime(np.array(sub.time_bins[0])).year    # year of start of bins (convert to np array, then pandas datetime object to extract year)

       sub2 = ssser_bin.sel(time_bins=slice(t1, t2))       # select time slice of the tropical year
       #ssser_final[i] = sub.mean(axis=0)
       # Combine two sources of error:
       #sigma_avger = sub.mean(axis=0)/(len(sub)**0.5)                                    # to convert a monthly standard error into an annual one, the general rule is to divide by sqrt(N)
       #sub2 = sss_interp.sel(time=slice(t1, t2))
       #sigma_se = np.array(sub2.std(axis=0)/(len(sub2)**0.5))   # calculate standard error of the data for that year and convert to numpy array
       #ssser_final[i] = ((sigma_se**2) + (sigma_avger**2))**0.5  # this is the combined error in d18Oc from: (1) standard deviation of the mean over the year, and (2) the analyical error, combined in quadrature.
       sigma_slope = np.array(sub.std(axis=0,skipna='True')/(len(sub)**0.5))   # calculate standard error of the data for that year and convert to numpy array
       sss_analerr = ((np.sum(sub2**2))**0.5)/len(sub2)
       #ssser_final[i] = sigma_avger  # this is the combined error in d18Oc from: (1) standard deviation of the mean over the year, and (2) the analyical error, combined in quadrature.
       ssser_final[i] = ((sigma_slope**2) + (sss_analerr**2))**0.5  # this is the combined error in d18Oc from: (1) standard deviation of the mean over the year, and (2) the analyical error, combined in quadrature. 
       count = count + 1

else:
    t1 = dt2.datetime(startyr, 4, 1)     # take time slice of dates in the tropical year (Apr 1-Mar 31)
    t2 = dt2.datetime(endyr, 3, 31)  # t1 = Apr 1, Year 1; t2 = Mar 31, Year 2
    d18Oc_final = d18Oc_bin.sel(time_bins=slice(t1, t2))
    d18Ocerr_final = np.ones([len(d18Oc_final)], dtype=float)*d18Oc_analerr
    sst_final = sst_bin.sel(time_bins=slice(t1, t2))   
    sss_final = sss_bin.sel(time_bins=slice(t1, t2)) 
    sster_final = sster_bin.sel(time_bins=slice(t1, t2))  
    ssser_final = ssser_bin.sel(time_bins=slice(t1, t2)) 
    yr_d18Ocfinal = d18Oc_final.time_bins
    yr_sstfinal = sst_final.time_bins
    yr_sssfinal = sss_final.time_bins
  
#sst_yr = sst.groupby('time.year').mean('time')        # 1854-2020
#sster_yr = sster.groupby('time.year').mean('time')    # 1854-2016
#sss_yr = sss.groupby('time.year').mean('time')        # 1854-2020
#ssser_yr = ssser.groupby('time.year').mean('time')    # 1854-2016

# Match ages of coral data to SST and SSS data
#sst_yr_final = sst_yr.sel(year = d18Oc_yr.year, method='nearest')
#sster_yr_final = sster_yr.sel(year = d18Oc_yr.year, method='nearest')
#sss_yr_final = sss_yr.sel(year = d18Oc_yr.year, method='nearest')
#ssser_yr_final = ssser_yr.sel(year = d18Oc_yr.year, method='nearest')

# Create new DataArray object by specifying coordinates and dimension names (xarray) from numpy array = 
#d18Ocerr_tropyrf = xr.DataArray(d18Ocerr_tropyr, coords=[yr], dims=["start_year"])
#sst_tropyrf = xr.DataArray(sst_tropyr, coords=[yr], dims=["start_year"])
#sster_tropyrf = xr.DataArray(sster_tropyr, coords=[yr], dims=["start_year"])
#sss_tropyrf = xr.DataArray(sss_tropyr, coords=[yr], dims=["start_year"])
#ssser_tropyrf = xr.DataArray(ssser_tropyr, coords=[yr], dims=["start_year"])

#Y = xr.DataArray(d18Oc_tropyr, coords=[yr], dims=["start_year"])                    # d18Oc yearly values 
#varY = xr.DataArray(np.square(d18Ocerr_tropyr), coords=[yr], dims=["start_year"])         # (d18Oc errors)^2
#X = xr.DataArray([sst_tropyr, sss_tropyr], coords=[range(2),yr], dims=["vars","start_year"])
#varX = xr.DataArray(np.square([sster_tropyr,ssser_tropyr]), coords=[range(2),yr], dims=["vars","start_year"])


# Format data for Matt's code:
if time_step == 'year':
    nt = nyr
else:
    nt = len(d18Oc_final)
    
Y = np.zeros((nt,1))
Y[:,0] = d18Oc_final    # d18Oc yearly values 

varY = np.zeros((nt,1))         
varY[:,0] = np.square(d18Ocerr_final)         # (d18Oc errors)^2

X = np.zeros((nt,2))
X[:,0] = sst_final
X[:,1] = sss_final

varX = np.zeros((nt,2))
varX[:,0] = np.square(sster_final)
varX[:,1] = np.square(ssser_final)

# Matt's original code
#Y = d[['dOa']].values                   # d18Oc yearly values
#varY = np.square(d[['dOae']].values)    # (d18Oc errors)^2
#X = d[['ersstv5', 'HadEN4']].values     # [SST values, SSS values]
#varX = np.square(d[['ersstv5e', 'HadEN4e']].values) # [SST errors^2, SSS errors^2]


#============================================================================
# Perform WLS/York (2004) regression
#
# bivariate_fit(xi, yi, dxi, dyi, ri=0.0, b0=1.0, maxIter=1e6):
#    """Make a linear bivariate fit to xi, yi data using York et al. (2004).
#
#    This is an implementation of the line fitting algorithm presented in:
#    York, D et al., Unified equations for the slope, intercept, and standard
#    errors of the best straight line, American Journal of Physics, 2004, 72,
#    3, 367-375, doi = 10.1119/1.1632486
#
#    See especially Section III and Table I. The enumerated steps below are
#    citations to Section III
#
#    Parameters:
#      xi, yi      x and y data points
#      dxi, dyi    errors for the data points xi, yi
#      ri          correlation coefficient for the weights
#      b0          initial guess b
#      maxIter     float, maximum allowed number of iterations
#
#    Returns:
#      a           y-intercept, y = a + bx
#      b           slope
#      S           goodness-of-fit estimate
#      sigma_a     standard error of a
#      sigma_b     standard error of b
#
#    Usage:
#    [a, b] = bivariate_fit( xi, yi, dxi, dyi, ri, b0, maxIter)
#============================================================================

# Regress d18Oc onto SSS
[a1, b1, S1, cov_matrix1] = bivariate_fit(sss_final, d18Oc_final, ssser_final, d18Ocerr_final, ri=0.0, b0=1.0, maxIter=1e6)   # here X =[SSS], Y = [d18Oc]

# Regress d18Oc onto SST
[a2, b2, S2, cov_matrix2] = bivariate_fit(sst_final, d18Oc_final, sster_final, d18Ocerr_final, ri=0.0, b0=1.0, maxIter=1e6)   # here X =[SST], Y = [d18Oc]

# Regress SST onto SSS
[a3, b3, S3, cov_matrix3] = bivariate_fit(sss_final, sst_final, ssser_final, sster_final, ri=0.0, b0=1.0, maxIter=1e6)   # here X =[SSS], Y = [SST]

# Regress d18Oc onto (-0.21*SST + SSS)
[a4, b4, S4, cov_matrix4] = bivariate_fit(0.21*sst_final+sss_final, d18Oc_final, ssser_final, d18Ocerr_final, ri=0.0, b0=1.0, maxIter=1e6)   # here X =[SSS], Y = [d18Oc]

#============================================================================
# Regression Plots
#============================================================================

plt.plot(yr_sssfinal, sss_final, 'o', label = 'time series')
plt.show()


# Initialise the subplot function using number of rows and columns
# figure, axis = plt.subplots(1, 2)
# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(1, 2)

# d18Oc vs SSS
axis[0].plot(sss_final, d18Oc_final, 'o', label = 'original data', color='k')
axis[0].plot(sss_final, a1 + b1*sss_final, 'r', label="δ18Oc = {0:.3f}*SSS + {1:.2f}".format(b1, a1))
axis[0].set_xlabel('SSS (%)')
axis[0].set_ylabel('Coral $\delta^{18}$O ($\perthousand$)')
axis[0].legend(fontsize=8)

# d18Oc vs SST -- MODIFIED, now plots something else 
axis[1].plot(sst_final, d18Oc_final, 'o', label = 'original data', color='k')
axis[1].plot(sst_final, a2 + sst_final*b2, 'orange', label="δ18Oc = {0:.3f}*SST + {1:.2f}".format(b2, a2))
axis[1].set_xlabel('SSS (%)')
axis[1].set_ylabel('Coral $\delta^{18}$O ($\perthousand$) plus SST term')
axis[1].legend(fontsize=8)

#plt.savefig(recordID+'_wls.pdf', bbox_inches='tight')
plt.show()

# SST vs SSS
plt.plot(sss_final, sst_final, 'o', label = 'original data', color='k')
plt.plot(sss_final, a3 + b3*sss_final, 'b', label="SST = {0:.3f}*SSS + {1:.2f}".format(b3, a3))
plt.xlabel('SSS (%)')
plt.ylabel('SST (°C)')
plt.legend(fontsize=8)
# d18Oc + 0.21SST vs SSS
plt.plot(sss_final, d18Oc_final, 'o', label = 'original data', color='k')
plt.plot(sss_final, d18Oc_final, 'r', label="δ18Oc = {0:.3f}*SSS + {1:.2f}".format(b4, a4))
plt.xlabel('SSS (%)')
plt.ylabel('Coral $\delta^{18}$O ($\perthousand$)')
plt.legend(fontsize=8)
plt
plt.plot(sst_final, d18Oc_final, 'o', label = 'original data', color='k')
plt.plot(sst_final, d18Oc_final, 'r', label="δ18Oc = {0:.3f}*SSS + {1:.2f}".format(b4, a4))
plt.xlabel('SSS (%)')
plt.ylabel('Coral $\delta^{18}$O ($\perthousand$)')
plt.legend(fontsize=8)
plt.show()
plt.plot(sss_final, d18Oc_final + 0.21*sst_final, 'o', label = 'original data', color='k')
plt.plot(sss_final, a4 + b4*sss_final, 'r', label="δ18Oc = {0:.3f}*SSS + {1:.2f}".format(b4, a4))
plt.xlabel('SSS (%)')
plt.ylabel('Coral $\delta^{18}$O ($\perthousand$)')
plt.legend(fontsize=8)

plt.plot(sss_final, d18Oc_final + 0.21*sst_final, 'o', label = 'original data', color='k')
plt.plot(sss_final, a4 + b4*sss_final, 'r', label="δ18Oc = {0:.3f}*SSS + {1:.2f}".format(b4, a4))
plt.xlabel('SSS (%)')
plt.ylabel('Coral $\delta^{18}$O ($\perthousand$)')
plt.legend(fontsize=8)


'''
# d18Oc vs SSS
axis[0].plot(sss_final, d18Oc_final, 'o', label = 'original data', color='k')
axis[0].plot(sss_final, a1 + b1*sss_final, 'r', label="δ18Oc = {0:.3f}*SSS + {1:.2f}".format(b1, a1))
axis[0].set_xlabel('SSS (%)')
axis[0].set_ylabel('Coral $\delta^{18}$O ($\perthousand$)')
axis[0].legend(fontsize=8)

# d18Oc vs SST
axis[1].plot(sst_final, d18Oc_final, 'o', label = 'original data', color='k')
axis[1].plot(sst_final, a2 + b2*sst_final, 'orange', label="δ18Oc = {0:.3f}*SST + {1:.2f}".format(b2, a2))
axis[1].set_xlabel('SST (°C)')
axis[1].set_ylabel('Coral $\delta^{18}$O ($\perthousand$)')
axis[1].legend(fontsize=8)
'''
# plt.savefig(coralid1+'_wls.pdf', bbox_inches='tight')
plt.show()

# SST vs SSS
plt.plot(sss_final, sst_final, 'o', label = 'original data', color='k')
plt.plot(sss_final, a3 + b3*sss_final, 'b', label="SST = {0:.3f}*SSS + {1:.2f}".format(b3, a3))
plt.xlabel('SSS (%)')
plt.ylabel('SST (°C)')
plt.legend(fontsize=8)
plt.savefig(coralid1+'_SSTvsSSS.pdf', bbox_inches='tight')
plt.show()

#============================================================================
# Time Series Plots
#============================================================================

#============================================================================
# Plot 1: Raw SST and d18Oc
#============================================================================
time = range(startyr,endyr)
fig, ax = plt.subplots(figsize=(10,4))
plt.gca().invert_yaxis()
#fig, ax = plt.subplots(figsize=(15,5))
#ax.plot(d18Oc_interp.time,d18Oc_interp,color='black',label='d18Oc')
ax.plot(d18Oc.time,d18Oc,color='black',label='d18Oc')

ax2=ax.twinx()
#ax2.plot(sst_final.time,sst_final,color='blue',label='sst')
ax2.plot(sst_f.time,sst_f,color='blue',label='sst')
#ax1 = plt.gca()
#ax1.set_xticks([1980,1990,2000,2010,2020])
lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#plt.ylim(-5.,-4.)
#plt.xlim(startyr,endyr)
ax.set_xlabel('Year (CE)')
ax.set_title('SST and d18Oc (raw)')
plt.savefig(coralid1+'_sst_ts_raw.pdf', bbox_inches='tight')
plt.show()

#============================================================================
# Plot 2: Raw SSS and d18Oc
#============================================================================
time = range(startyr,endyr)
fig, ax = plt.subplots(figsize=(10,4))
plt.gca().invert_yaxis()
#fig, ax = plt.subplots(figsize=(15,5))
#ax.plot(d18Oc_interp.time,d18Oc_interp,color='black',label='d18Oc')
ax.plot(d18Oc.time,d18Oc,color='black',label='d18Oc')

ax2=ax.twinx()
plt.gca().invert_yaxis()
#ax2.plot(sst_final.time,sst_final,color='blue',label='sst')
ax2.plot(sss_f.time,sss_f,color='red',label='sst')
#ax1 = plt.gca()
#ax1.set_xticks([1980,1990,2000,2010,2020])
lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#plt.ylim(-5.,-4.)
#plt.xlim(startyr,endyr)
ax.set_xlabel('Year (CE)')
ax.set_title('SSS and d18Oc (raw)')
plt.savefig(coralid1+'_sss_ts_raw.pdf', bbox_inches='tight')
plt.show()

#============================================================================
# Plot 3: Bimonthly SST and d18Oc
#============================================================================
time = range(startyr,endyr)
fig, ax = plt.subplots(figsize=(10,4))
plt.gca().invert_yaxis()
#fig, ax = plt.subplots(figsize=(15,5))
#ax.plot(d18Oc_interp.time,d18Oc_interp,color='black',label='d18Oc')
ax.plot(d18Oc_bin.time_bins,d18Oc_bin,color='black',label='d18Oc')

ax2=ax.twinx()
#ax2.plot(sst_final.time,sst_final,color='blue',label='sst')
ax2.plot(sst_bin.time_bins,sst_bin,color='blue',label='sst')
#ax1 = plt.gca()
#ax1.set_xticks([1980,1990,2000,2010,2020])
lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#plt.ylim(-5.,-4.)
#plt.xlim(startyr,endyr)
ax.set_xlabel('Year (CE)')
ax.set_title('SST and d18Oc (bimonthly binned)')
plt.savefig(coralid1+'_sst_ts.pdf', bbox_inches='tight')
plt.show()

#============================================================================
# Plot 4: Detrended SST and d18Oc
#============================================================================

time = range(startyr,endyr)
fig, ax = plt.subplots(figsize=(10,4))
plt.gca().invert_yaxis()
#fig, ax = plt.subplots(figsize=(15,5))
#ax.plot(d18Oc_interp.time,d18Oc_interp,color='black',label='d18Oc')
ax.plot(d18Oc_dt.time,d18Oc_dt,color='black',label='d18Oc')

ax2=ax.twinx()
#ax2.plot(sst_final.time,sst_final,color='blue',label='sst')
ax2.plot(sst_dt.time,sst_dt,color='blue',label='sst')
#ax1 = plt.gca()
#ax1.set_xticks([1980,1990,2000,2010,2020])
lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#plt.ylim(-5.,-4.)
#plt.xlim(startyr,endyr)
ax.set_xlabel('Year (CE)')
ax.set_title('SST and d18Oc (detrended)')
plt.savefig(coralid1+'_sst_detrend_ts.pdf', bbox_inches='tight')
plt.show()


#============================================================================
# Plot 5: Bimonthly SSS and d18Oc
#============================================================================

time = range(startyr,endyr)
fig, ax = plt.subplots(figsize=(10,4))
plt.gca().invert_yaxis()
#fig, ax = plt.subplots(figsize=(15,5))
#ax.plot(d18Oc_interp.time,d18Oc_interp,color='black',label='d18Oc')
ax.plot(d18Oc_bin.time_bins,d18Oc_bin,color='black',label='d18Oc')

ax2=ax.twinx()
plt.gca().invert_yaxis()
#ax2.plot(sst_final.time,sst_final,color='blue',label='sst')
ax2.plot(sss_bin.time_bins,sss_bin,color='red',label='sss')
#ax1 = plt.gca()
#ax1.set_xticks([1980,1990,2000,2010,2020])
lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#plt.ylim(-5.,-4.)
#plt.xlim(startyr,endyr)
ax.set_xlabel('Year (CE)')
ax.set_title('SSS and d18Oc (bimonthly binned)')
plt.savefig(coralid1+'_sss_ts.pdf', bbox_inches='tight')
plt.show()

#============================================================================
# Plot 6: Yearly avg'd SST
#============================================================================

time = range(startyr,endyr)
fig, ax = plt.subplots(figsize=(10,4))
plt.gca().invert_yaxis()
#fig, ax = plt.subplots(figsize=(15,5))
#ax.plot(d18Oc_interp.time,d18Oc_interp,color='black',label='d18Oc')
ax.plot(yr_d18Ocfinal,d18Oc_final,color='black',label='d18Oc')

ax2=ax.twinx()
#ax2.plot(sst_final.time,sst_final,color='blue',label='sst')
ax2.plot(yr_sstfinal,sst_final,color='blue',label='sst')
#ax1 = plt.gca()
#ax1.set_xticks([1980,1990,2000,2010,2020])
lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#ax2.set_ylim(26,28)
#plt.xlim(startyr,endyr)
ax.set_xlabel('Year (CE)')
ax.set_title('SST and d18Oc (trop year avg)')
plt.savefig(coralid1+'_sstyr_ts.pdf', bbox_inches='tight')
plt.show()

#============================================================================
# Plot 7: Yearly avg'd SSS
#============================================================================

time = range(startyr,endyr)
fig, ax = plt.subplots(figsize=(10,4))
plt.gca().invert_yaxis()
#fig, ax = plt.subplots(figsize=(15,5))
#ax.plot(d18Oc_interp.time,d18Oc_interp,color='black',label='d18Oc')
ax.plot(yr_d18Ocfinal,d18Oc_final,color='black',label='d18Oc')

ax2=ax.twinx()
plt.gca().invert_yaxis()
#ax2.plot(sst_final.time,sst_final,color='blue',label='sst')
ax2.plot(yr_sssfinal,sss_final,color='red',label='sss')
#ax1 = plt.gca()
#ax1.set_xticks([1980,1990,2000,2010,2020])
lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#ax2.set_ylim(26,28)
#plt.xlim(startyr,endyr)
ax.set_xlabel('Year (CE)')
ax.set_title('SSS and d18Oc (trop year avg)')
plt.savefig(coralid1+'_sssyr_ts.pdf', bbox_inches='tight')
plt.show()

#============================================================================
# Plot 8: Trend in SST
#============================================================================

plt.plot(time_sst, sst_f, 'o', label='original data')
plt.plot(time_sst, sst_res.intercept + sst_res.slope*time_sst, 'r', label='fitted line')
plt.xlabel('year from start yr')
plt.ylabel('SST')
plt.title('Trend in SST')
plt.legend()
plt.savefig(coralid1+'_trend_sst.pdf', bbox_inches='tight')
plt.show()

#============================================================================
# Plot 9: Trend in SSS
#============================================================================

plt.plot(time_sss, sss_f, 'o', label='original data')
plt.plot(time_sss, sss_res.intercept + sss_res.slope*time_sss, 'r', label='fitted line')
plt.xlabel('year from start yr')
plt.ylabel('SSS')
plt.title('Trend in SSS')
plt.legend()
plt.savefig(coralid1+'_trend_sss.pdf', bbox_inches='tight')
plt.show()
