#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:28:13 2021

@author: alyssaatwood
"""


import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power 
# from plotnine import *
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
import sys
import os
from datetime import datetime
from cftime import DatetimeProlepticGregorian

# sys.stderr = open(os.devnull, "w") # display error output 

# show arrays or not
PRINT_STUFF = False 

def average_xarrays(*arrays):
    avg = 0
    for array in arrays:
        # print(array)
        avg += (array['time'].data)


def decyrs_to_datetime(decyrs):      # converts date in decimal years to a datetime object (from https://stackoverflow.com/questions/20911015/decimal-years-to-datetime-in-python)
    
    output = np.empty([len(decyrs)],dtype='O')
   
    for l in range(len(decyrs)):
        start = decyrs[l]                   # this is the input (time in decimal years)
        year = int(start)
        rem = start - year
        base = datetime(year, 1, 1)
        result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
        # result2 = result.strftime("%Y-%m-%d")   # truncate to just keep YYYY-MM-DD (get rid of hours, secs, and minutes)
      
        output[l] = result
    return output

'''
alternate version using datetime library 
def decyrs_to_datetime(decyrs):
    results = []
    for i in range(len(decyrs)):
        year = int(decyrs[i])
        rem = decyrs[i] - year

        base = datetime(year, 1, 1)
        temp = (base.replace(year=base.year + 1) - base).total_seconds()
        result =  base + timedelta(seconds = temp * rem)
        results += [result]
    return results
'''


# Loosely based off: https://community.alteryx.com/t5/Alteryx-Designer-Discussions/Round-a-date-to-the-nearest-month/td-p/624262
# may need to rewrite this function 
'''
def round_nearest_month(values):      # rounds a numpy datetime64 array to the nearest month (1st of the nearest month)
    # values = pd.to_datetime(values, unit='s')
    for i in range(len(values)):
        # why did this not do anything?
        values[i] = np.datetime64(values[i]).astype('datetime64[M]')
    
    dat = {'date': values}
    # why does this line generate an error? 
    # FIGURED IT OUT: https://stackoverflow.com/questions/31917964/python-numpy-cannot-convert-datetime64ns-to-datetime64d-to-use-with-numba
    # may just need to rewrite this function to avoid going through pandas 

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
'''

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



#more salinity variation 
#coralid1 = 'CO95TUNG01A'

'''
CO95TUNG01A
CO14OSPA02A
CO03CHBA01A

'''

# *** Reference coral ID's *** # 

# Indian Ocean coras
#coralid1 = 'HE18COC01'    # Hennekam (2018) - Cocos (Keeling) Islands
#coralid2 = 'HE18COC02'
#coralid1 = 'ST13MAL01'    # Storz (2013) - Rasdhoo Atoll, Maldives
#coralid1 = 'ZI08MAY01'    # Zinke (2008) - Mayotte
#coralid1 = 'PF19LAR01'    # Pfeiffer (2004) - St. Gilles Reef, La Réunion

# Central Pacific corals
#coralid1 = 'NU11PAL01'    # Nurhati (2011) - Palmyra (Tier 1)
#coralid1 = 'NU09FAN01'    # Nurhati (2009) - Fanning Island (Tier 2)
#coralid1 = 'NU09CHR01'    # Nurhati (2009) - Christmas Island (Tier 2)

# Others
#coralid1 = 'WU14CLI01'    # Wu (2014) - Clipperton
#coralid1 = 'GO08BER01'    # Goodkin (2008) - Bermuda
#coralid1 = 'ST13MAL01'    # Storz (2013) - Maldives
#coralid1 = 'CA14TIM01'    # Cayharini (2014) - Timor, Indonesia

<<<<<<< HEAD
=======
coralid1 = 'SM06LKF01'


>>>>>>> b7012cfc260c8abe47537585a00e6cca4e76fae9

# Define analysis interval
time_step = 'year'
# time_step = 'bimonthly'

#============================================================================
# Read in coral d18O data
#============================================================================

f = h5py.File('hydro2kv0_5_2(1).mat','r')

if len(sys.argv) > 1:

	# if running script from command line
    coralid1 = str(sys.argv[2]) 

else:

	# choose the coral record you want to analyze here:

    # coralid1 = 'GO08BER01'    # Goodkin (2008) - Bermuda -- cannot interpolate with duplicate values 
    # coralid1 = 'PF19LAR01'

    # the Maldives 
    
    # coralid1 = 'ST13MAL01' 

    # error sleuthing
    # coralid1 = 'AB08MEN01'
    # coralid1 = 'CA13PEL01' # undiagnosed nan error


    # this one generates a time range error

    # coralid1 = 'AB20MEN07'

    # this one generates a shape error 
    # coralid1 = 'CA13TUR01'

    # coralid1 = 'HE18COC01' # used to be fine, now generating error 
    # coralid1 = 'ZI08MAY01'
    # this one gives us "nonetype is not iterable" which means it isn't finding the ID in the data
    # coralid1 = 'CO95TUNG01A'

    # coralid1 = 'NU11PAL01'    # Nurhati (2011) - Palmyra (Tier 1)
    # coralid1 = 'NU09FAN01'    # Nurhati (2009) - Fanning Island (Tier 2) -- this one looks bad 
    # coralid1 = 'NU09CHR01'    # Nurhati (2009) - Christmas Island (Tier 2)
    # coralid1 = 'AS05GUA01'
    # coralid1 = 'MO20KOI01'
<<<<<<< HEAD
    # coralid1 = 'SM06LKF01'


    coralid1 = 'SM06LKF01'


=======
    coralid1 = 'SM06LKF01'
>>>>>>> b7012cfc260c8abe47537585a00e6cca4e76fae9

    # exit(0)

data1 = f.get('ch2k/'+coralid1+'/d18O')

# print(data1[0])

# clean nan's --  
'''
try:
    mask = ~pd.isna(data1[0]) & ~pd.isna(data1[1])

    dates = data1[0][mask]
    datavals = data1[1][mask]
except: 
    pass
'''

data_cleaned = data1 # np.asarray([dates, datavals])
dataset = f.get('ch2k/')
coralnames = list(dataset.keys())

timec01 = (data_cleaned[0,:])-1/24     # time in fractional year CE (subtract 1/24 so that time corresponds to start of month (as with sst datasets), rather than mid-month)
d18Oc1 = np.array(data_cleaned[1,:])           # Convert to NumPy array


timec1 = decyrs_to_datetime(timec01)    # convert time from decimal year YYYY.FF to datetime object (YYYY-MM-DD)

# Analytic error in d18Oc (assumes the values are in per mille!)
data_analerr = f.get('ch2k/'+coralid1+'/d18O_errAnalytic')
d18Oc_analerr = data_analerr[0,:]      # analytical error is a single value (in per mille)


# round time steps to nearest month (just a couple days off in some months)
timec1mo = round_nearest_month(timec1) 

timec1mo = timec1  
if False:                  # if a second coral id (coralid2) exixts, merge the two data sets
    data2 = f.get('ch2k/'+coralid2+'/d18O')
    timec02 = np.array(data2[0,:])          # time in fractional year CE
    d18Oc2 = np.array(data2[1,:])           # Convert to NumPy array
    timec2 = decyrs_to_datetime(timec02)    # convert time from decimal year YYYY.FF to datetime object (YYYY-MM-DD)
    timec2mo = round_nearest_month(timec2)

    # Merge records
    timec = np.concatenate((timec1mo,timec2mo),axis=0) # will result in division by zero if same yr
    d18Oc = np.concatenate((d18Oc1,d18Oc2),axis=0)
else:
    timec = timec1mo
    d18Oc = d18Oc1



# create DataArray object by specifying coordinates and dimension names (xarray) from numpy array


nan_mask = np.isnan(d18Oc)        # this is where d18O data ins nan
d18Oc = d18Oc[~nan_mask]
timec = timec[~nan_mask]

d18Oc = xr.DataArray(d18Oc, coords=[timec], dims=["time"])
# d18Oc = d18Oc.dropna(dim='time', how='any', thresh=None)  # drop all nans in array (across 0th dim)

d18Oc = d18Oc.sortby(d18Oc.time,ascending=True)                   # sort arrays by time
timec = d18Oc.time





latc = f.get('ch2k/'+coralid1+'/lat')
lonc = f.get('ch2k/'+coralid1+'/lon')  # lon = -180:180


latc = np.array(latc)                 # Convert to NumPy array
lonc = np.array(lonc)                 # Convert to NumPy array

<<<<<<< HEAD
for el in lonc:
    if el < 0:
        el += 360
=======
lonc = lonc + 180

>>>>>>> b7012cfc260c8abe47537585a00e6cca4e76fae9

##printt(latc, lonc)

#============================================================================
# Read in obs SST and uncertainty data
#============================================================================

dir = ''
ds = xr.open_dataset(dir+'sst_mnmean_noaa_ersstv5.nc',decode_times=True)         # ERSST v5

# Change longitude array from 0:360 to -180:180 -- MAKE THIS A FUNCTION 
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
# ds = ds.dropna(dim = "lat", how = "any")

# Read in variables
sst_all = ds['sst']
# print(sst_all.values)
time_sst_all = ds['time']             # ERSSTv5: time (1/1854-12/2020; units = "days since 1800-1-1 00:00:00")


lat_sst_all = sst_all['lat'].values   # get coordinates from dataset
lon_sst_all = sst_all['lon'].values   # lon = 0:360

# get indices and sst data at closest grid point to coral
(indlat_sst, latval_sst) = find_nearest(lat_sst_all, latc)
(indlon_sst, lonval_sst) = find_nearest(lon_sst_all, lonc)


# (indtime_sst, timeval_sst) = find_nearest(time_sst_all , timec[0])   # coral data steps on midpoints, sst data steps on first day of month... find closest time in sst data set to start of coral data
sst_f = sst_all[:,indlat_sst,indlon_sst]
# print(sst_f.values)
# match ages of SST and SSS data to coral data
# sst_final = sst.sel(time = timec, method='nearest')


#============================================================================
# Read in ERSSTv5 uncertainties 
#============================================================================
ds = xr.open_dataset(dir+'ersstev5_uncertainty.nc',decode_times=True)         # ERSST v5
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
# ds = ds.dropna(dim = "latitude", how = "any")

sster_all = ds['ut']
time_sster_all = ds['time']             # ERSSTv5: time (1/1854-12/2020; units = "days since 1800-1-1 00:00:00")
lat_sster_all = sster_all['latitude'].values   # get coordinates from dataset
lon_sster_all = sster_all['longitude'].values

(indlat_sster, latval_sster) = find_nearest(lat_sster_all, latc)
(indlon_sster, lonval_sster) = find_nearest(lon_sster_all, lonc)
sster_f = sster_all[:,indlat_sster,indlon_sster]

# Match ages of SST and SSS data to coral data
#sster_final = sster.sel(time = timec, method='nearest')


#============================================================================
# Read in obs SSS and uncertainty data
#============================================================================
dir = '' 
ds = xr.open_dataset(dir+'sss_HadleyEN4.2.1g10_190001-201012.nc',decode_times=True)         # ERSST v5

# ds = ds.interpolate_na(dim = "lat", method = "linear", fill_value = "extrapolate")

# UNCOMMENTED THIS LINE 
# ds = ds.dropna(dim = "lat", how = "any") # -- dropping nans caused major inconsistencies in data for some reason 
# if this causes problems elsewhere, we can do it conditionally or with try / except 

sss_all = ds['sss']


time_sss_all = ds['time']             # HadEN4: time x lat x lon (time = 1900:2010)
lat_sss_all = sss_all['lat'].values   # get coordinates from dataset
lon_sss_all = sss_all['lon'].values

# Get indices and sst data at closest grid point to coral
(indlat_sss, latval_sss) = find_nearest(lat_sss_all, latc)
(indlon_sss, lonval_sss) = find_nearest(lon_sss_all, lonc)



sss_f = sss_all[:,indlat_sss,indlon_sss]


# Match ages of SST and SSS data to coral data
#sss_final = sss.sel(time = timec, method='nearest')


#============================================================================
# Read in HadEN4 uncertainties in sea surface salinity 
#============================================================================
ds = xr.open_dataset(dir+'sss_HadleyEN4.2.1g10_190001-202012_salinityerrorSD.nc',decode_times=True)         # ERSST v5

# ds = ds.dropna(dim = "LAT", how = "any") # not exactly sure why we are dropping along this dimension but im not questioning it yet

ssser_all = ds['SALT_ERR_STD']
time_ssser_all = ds['TIME']               # HadEN4 error: time x 1 x lat x lon (time = 1900:2020)
lat_ssser_all = ssser_all['LAT'].values   # get coordinates from dataset
lon_ssser_all = ssser_all['LONN180_180'].values


# Get rid of singleton depth dimension - Create new DataArray object by specifying coordinates and dimension names (xarray) from numpy array

ssser_all = xr.DataArray(ssser_all[:,0,:,:], coords=[time_ssser_all,lat_ssser_all,lon_ssser_all], dims=["time","lat","lon"])

# ssser_all = xr.DataArray(ssser_all, coords=[time_ssser_all,lat_ssser_all,lon_ssser_all], dims=["time","lat","lon"])
(indlat_ssser, latval_ssser) = find_nearest(lat_ssser_all, latc)
(indlon_ssser, lonval_ssser) = find_nearest(lon_ssser_all, lonc)
ssser_f = ssser_all[:,indlat_ssser,indlon_ssser]

#============================================================================#







#=============================================================================
# Select common (i.e., overlapping for all datasets) time period
#=============================================================================

coral_years = np.array(d18Oc.time.dt.year)   # array of years in coral data
if PRINT_STUFF:
    print("Range of years in coral record: ")
    print(coral_years)

startyr = 1980 # coral_years[0]  # 1980       # start on a specified year
endyr = coral_years[-1]   # 2000
nyr = endyr-startyr      # set the last year for the tropical averages as the final year of coral data

# Truncate data to min/max of overlapping ages of all data sets
# t1 = max(d18Oc.time[0],sst_final.time[0],sss_final.time[0],sster_final.time[0],ssser_final.time[0])       # find latest start date of all data sets
# t2 = min(d18Oc.time[-1],sst_final.time[-1],sss_final.time[-1],sster_final.time[-1],ssser_final.time[-1])  # find earliest end date of all data sets
# d18Oc = d18Oc.sel(time=slice(t1, t2))
# sst_final = sst_final.sel(time=slice(t1, t2))
# sss_final = sss_final.sel(time=slice(t1, t2))
# sster_final = sster_final.sel(time=slice(t1, t2))
# ssser_final = ssser_final.sel(time=slice(t1, t2))

# Select time period

t1 = datetime(startyr, 1, 1)     
t2 = datetime(endyr, 12, 31) 
# t1 = d18Oc.time[0]   
# t2 = d18Oc.time[-1] 



# is the sel method specific to xarray? what could we do instead? 
# it seems like this is just slicing the array to include times in [t1, t2]
# a bitmask might be useful here too 

d18Oc = d18Oc.sel(time=slice(t1, t2))
sst_f = sst_f.sel(time=slice(t1, t2))
sss_f = sss_f.sel(time=slice(t1, t2))
sster_f = sster_f.sel(time=slice(t1, t2))
ssser_f = ssser_f.sel(time=slice(t1, t2))


# print(sss_f.values)
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

# Detrend the SST, SSS, and d18Oc data, but retain the intercept (so just remove the trend but the values aren't centered around 0)
# print(d18Oc)

# nan_mask = [~np.isnan(sss_final) & ~np.isnan(sst_final) & ~np.isnan(d18Oc_final)
#     & ~np.isnan(ssser_final) & ~np.isnan(sster_final) & ~np.isnan(d18Ocerr_final)]# 
# 

# # remove nans before computing composite variable 
# d18Oc_final = d18Oc_final[tuple(nan_mask)]
# sst_final = sst_final[tuple(nan_mask)]
# sster_final = sster_final[tuple(nan_mask)]
# d18Ocerr_final = d18Ocerr_final[tuple(nan_mask)]
# sss_final = sss_final[tuple(nan_mask)]
# ssser_final = ssser_final[tuple(nan_mask)]
# d18O_plus_SST =  d18O_plus_SST_err[tuple(nan_mask)]
# d18O_plus_SST_err =  d18O_plus_SST_err[tuple(nan_mask)]
# yr_sssfinal = yr_sssfinal[tuple(nan_mask)]


time_d18Oc = d18Oc.time.dt.year+d18Oc.time.dt.month/12-1/24 - d18Oc.time.dt.year[0]       # subtract the first year so the intercept is defined at start year 
time_sst = sst_f.time.dt.year+sst_f.time.dt.month/12-1/24 - sst_f.time.dt.year[0]
time_sss = sss_f.time.dt.year+sss_f.time.dt.month/12-1/24 - sss_f.time.dt.year[0]


# Detrend the data but retain the intercept
d18O_res = stats.linregress(time_d18Oc, d18Oc)
sst_res = stats.linregress(time_sst, sst_f)
sss_res = stats.linregress(time_sss, sss_f)



d18Oc_dt1 = d18Oc-d18O_res.slope*time_d18Oc
sst_dt1 = sst_f-sst_res.slope*time_sst
sss_dt1 = sss_f-sss_res.slope*time_sss

sster_dt1 = sster_f                    # Don't detrend the errors (confirmed with Matt F.)
ssser_dt1 = ssser_f                    # Don't detrend the errors 

# Convert back to DataArrays (detrend converts to numpy array)


d18Oc_dt = xr.DataArray(d18Oc_dt1, coords=[d18Oc.time], dims=["time"])
sst_dt = xr.DataArray(sst_dt1, coords=[sst_f.time], dims=["time"])
sss_dt = xr.DataArray(sss_dt1, coords=[sss_f.time], dims=["time"])
sster_dt = xr.DataArray(sster_dt1, coords=[sster_f.time], dims=["time"])
ssser_dt = xr.DataArray(ssser_dt1, coords=[ssser_f.time], dims=["time"])
# should the analytic error be in here? 

# this step may be generating nans - all the time_bins rows contain nans
lb = np.array(sst_dt.time[::2])    
lb = lb[0:-1]    


# need to remove duplicates for interpolation 
# d18Oc_dt = d18Oc_dt.dropna(dim = "time", how = "any") # maybe there was a nan that slipped through that's messing up the mean?
# d18Oc_dt = d18Oc_dt.interpolate_na(dim = "time", method = "linear")
# print("label array: ")
# print(lb)

lb = lb[~np.isnan(lb)]

# print("label array after nan removal: ")
# print(lb)
#  print(type(lb))

sst_dt = sst_dt[~np.isnan(sst_dt)]
if PRINT_STUFF: 
    print("BEFORE BINNING: ")
    print(d18Oc_dt)
        

# why does binnning generate nan's even when interpolating? 
# is it just that there is no data for certain two month periods? 


d18Oc_bin = d18Oc_dt.groupby_bins('time',bins=sst_dt.time[::2],labels=lb).mean()   
sst_bin = sst_dt.groupby_bins('time',bins=sst_dt.time[::2],labels=lb).mean()   
sss_bin = sss_dt.groupby_bins('time',bins=sst_dt.time[::2],labels=lb).mean()
sster_bin = sster_dt.groupby_bins('time',bins=sst_dt.time[::2],labels=lb).mean()
ssser_bin = ssser_dt.groupby_bins('time',bins=sst_dt.time[::2],labels=lb).mean()


# d18Oc_bin = d18Oc_bin.interpolate_na(dim = "time_bins", method = "linear")

if PRINT_STUFF:
    print("AFTER BINNING AND INTERPOLATING AGAIN: ")
    print(d18Oc_bin)

# ad hoc fix for straggler nans
# d18Oc_bin = d18Oc_bin.dropna(dim = "time_bins", how = "any") 
# ssser_bin = ssser_bin.dropna(dim = "time_bins", how = "any")
# sster_bin = sster_bin.dropna(dim = "time_bins", how = "any")
# sss_bin = sss_bin.dropna(dim = "time_bins", how = "any")
# sst_bin = sst_bin.dropna(dim = "time_bins", how = "any")


# print(d18Oc_bin)
# print(d18Oc_analerr)
# print(sster_bin)
#=============================================================================
# Average data and propagate errors over the tropical year (Apr 1-Mar 31)
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

    d18O_plus_SST = np.empty([nyr], dtype = float )
    d18O_plus_SST_err  = np.empty([nyr], dtype = float )

    for i in range(nyr):
        # sub array ends up empty bc there is nothing in the slice after removing nan's ? 

        t1 = dt2.datetime(startyr+i, 4, 1)     # take time slice of dates in the tropical year (Apr 1-Mar 31)
        t2 = dt2.datetime(startyr+1+i, 3, 31)  # t1 = Apr 1, Year 1; t2 = Mar 31, Year 2
        #yr[i] = startyr+i
        
        # d18O and error
        # d18Oc_bin = d18Oc_bin.dropna(dim = "time_bins", how = "any")
        

        sub = d18Oc_bin.sel(time_bins=slice(t1, t2))       # select time slice of the tropical year
        
        if PRINT_STUFF and i == 25: 
            print("BINNED AND CLEANED: ")
            print(d18Oc_bin)
            print("sub is: ")
            print(d18Oc_bin.sel(time_bins = slice(t1, t2)))
            print("T1 and T2 are: ", t1, t2)

        # in many instances, sub is empty; what do we do in these cases? 

        try:
            x = sub[0] # attempt to reference first element
            # will throw an error if empty
        except:
            # if there is nothing in this two month period, interpolate?
            continue



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
        try: 
            yr_sssfinal[i] = pd.to_datetime(np.array(sub.time_bins[0])).year    # year of start of bins (convert to np array, then pandas datetime object to extract year)
        except:
            # interpolate here instead 
            yr_sssfinal[i] = 0

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


    # mask to remove all nans through every array according to joint condition 
    nan_mask = [~np.isnan(sss_final) & ~np.isnan(sst_final) & ~np.isnan(d18Oc_final)
    & ~np.isnan(ssser_final) & ~np.isnan(sster_final) & ~np.isnan(d18Ocerr_final)]

    
    # remove nans before computing composite variable 
    d18Oc_final = d18Oc_final[tuple(nan_mask)]
    sst_final = sst_final[tuple(nan_mask)]
    sster_final = sster_final[tuple(nan_mask)]
    d18Ocerr_final = d18Ocerr_final[tuple(nan_mask)]
    sss_final = sss_final[tuple(nan_mask)]
    ssser_final = ssser_final[tuple(nan_mask)]
    d18O_plus_SST =  d18O_plus_SST_err[tuple(nan_mask)]
    d18O_plus_SST_err =  d18O_plus_SST_err[tuple(nan_mask)]
    yr_sssfinal = yr_sssfinal[tuple(nan_mask)]

    # print(sst_final)
    for i in range(len(sss_final)):
        d18O_plus_SST[i] = d18Oc_final[i] + 0.21*sst_final[i]
        d18O_plus_SST_err[i] = d18Ocerr_final[i] + 0.21*0.21*sster_final[i]+2*0.21*np.cov(d18Oc_final,sst_final)[0][1]

else:
    
    t1 = dt2.datetime(startyr, 4, 1)     # take time slice of dates in the tropical year (Apr 1-Mar 31)
    t2 = dt2.datetime(endyr, 3, 31)  # t1 = Apr 1, Year 1; t2 = Mar 31, Year 2
    
    # print(startyr)
    # print(endyr)
   #  print(d18Oc_bin)


    d18Oc_final = d18Oc_bin.sel(time_bins=slice(t1, t2))
    d18Ocerr_final = np.ones([len(d18Oc_final)], dtype=float)*d18Oc_analerr
    sst_final = sst_bin.sel(time_bins=slice(t1, t2))   
    sss_final = sss_bin.sel(time_bins=slice(t1, t2)) 
    sster_final = sster_bin.sel(time_bins=slice(t1, t2))  
    ssser_final = ssser_bin.sel(time_bins=slice(t1, t2)) 
    yr_d18Ocfinal = d18Oc_final.time_bins
    yr_sstfinal = sst_final.time_bins
    yr_sssfinal = sss_final.time_bins
    d18O_plus_SST = np.empty(len(d18Oc_final))
    d18O_plus_SST_err = np.empty(len(d18Ocerr_final))

    # mask to remove all nans through every array according to joint condition 
    nan_mask = [~np.isnan(sss_final) & ~np.isnan(sst_final) & ~np.isnan(d18Oc_final)
    & ~np.isnan(ssser_final) & ~np.isnan(sster_final) & ~np.isnan(d18Ocerr_final)]

    # NOTE: for bimonthly, we need to use xarray built in function to avoid type error
    # remove nans before computing composite variable 
    
    d18Oc_final = d18Oc_final[tuple(nan_mask)]
    sst_final = sst_final[tuple(nan_mask)]
    sster_final = sster_final[tuple(nan_mask)]
    d18Ocerr_final = d18Ocerr_final[tuple(nan_mask)]
    sss_final = sss_final[tuple(nan_mask)]
    ssser_final = ssser_final[tuple(nan_mask)]
    d18O_plus_SST =  d18O_plus_SST_err[tuple(nan_mask)]
    d18O_plus_SST_err =  d18O_plus_SST_err[tuple(nan_mask)]
    
    '''
    d18Oc_final = d18Oc_final.dropna(dim='time', how='any', thresh=None)  # drop all nans in array (across 0th dim)
    sst_final = sst_final.dropna(dim='time', how='any', thresh=None)  # drop all nans in array (across 0th dim)
    sster_final = sster_final.dropna(dim='time', how='any', thresh=None)  # drop all nans in array (across 0th dim)
    d18Ocerr_final = d18Ocerr_final.dropna(dim='time', how='any', thresh=None)  # drop all nans in array (across 0th dim)
    sss_final = sss_final.dropna(dim='time', how='any', thresh=None)  # drop all nans in array (across 0th dim)
    ssser_final = ssser_final.dropna(dim='time', how='any', thresh=None)  # drop all nans in array (across 0th dim)
    d18O_plus_SST =  d18O_plus_SST_err.dropna(dim='time', how='any', thresh=None)  # drop all nans in array (across 0th dim)
    d18O_plus_SST_err =  d18O_plus_SST_err.dropna(dim='time', how='any', thresh=None)  # drop all nans in array (across 0th dim)
    '''

    # yr_sssfinal = yr_sssfinal[nan_mask]
    
    for i in range(len(sss_final)):
        d18O_plus_SST[i] = d18Oc_final[i] + 0.21*sst_final[i]
        d18O_plus_SST_err[i] = d18Ocerr_final[i] + 0.21*0.21*sster_final[i]+2*0.21*np.cov(d18Oc_final,sst_final)[0][1]
        

# last pass nan removal; enforcing shape consistency

nan_mask = [~np.isnan(sss_final) & ~np.isnan(sst_final) & ~np.isnan(d18Oc_final)
& ~np.isnan(ssser_final) & ~np.isnan(sster_final) & ~np.isnan(d18Ocerr_final)]

'''
d18O_plus_SST = d18O_plus_SST[nan_mask]
d18O_plus_SST_err = d18O_plus_SST_err[nan_mask]
sster_final = sster_final[nan_mask]
yr_sssfinal = yr_sssfinal[nan_mask]

sss_final = sss_final[nan_mask]
ssser_final = ssser_final[nan_mask]


'''








PRINT_STUFF = False

if PRINT_STUFF: 
    print("coral dataset: ")
    print(data1)
    print("isotope ratios after first cleaning: ")
    # print(datavals)
    print("dates after first cleaning: ")
    # print(dates)

    print("Times from xarray operation: ")
    print(time_d18Oc)
    print(time_sst)
    print(time_sss)

    print("SST pipeline: ")
    print(sst_f)
    print(sst_dt)
    print(sst_bin)

    print("SSS pipeline: ")
    print(sss_all)
    print(sss_f)
    print(sss_dt)
    print(sss_bin)

    print("SST error pipeline: ")
    print(sster_f)
    print(sster_dt)
    print(sster_bin)

    print("SSS error pipeline: ")
    print(ssser_f)
    print(ssser_dt)
    print(ssser_bin)

    print("Isotope ratio error: ")
    print(d18Ocerr_final)

    print("Composite variable is: ")
    print(d18O_plus_SST)

    print("Error in composite variable is: ")
    print(d18O_plus_SST_err)

    print("SSS final is: ")
    print(sss_final)

    print("Error in SSS final is: ")
    print(ssser_final)

    print("sub is: ")
    # print(sub)

    print("isotope ratios final: ")
    print(d18Oc_final)
  
  
[a1, b1, S1, cov_matrix1] = bivariate_fit(sss_final, d18Oc_final, ssser_final, d18Ocerr_final, ri=0.0, b0=1.0, maxIter=1e6)   # here X =[SSS], Y = [d18Oc]

# Regress d18Oc + 0.21*SST onto SSS
# print("what is this returning?")
# print(bivariate_fit(sss_final, d18O_plus_SST, ssser_final, d18O_plus_SST_err, ri=0.0, b0=1.0, maxIter=1e6)[0])   # here X =[SSS], Y = [SST])
if time_step == 'bimonthly':
    [a4, b4, S4, cov_matrix4] = bivariate_fit(np.unique(sss_final.values), np.unique(d18O_plus_SST), np.unique(ssser_final.values), np.unique(d18O_plus_SST_err), ri=0.0, b0=1.0, maxIter=1e6) # here X =[SSS], Y = [SST]
else:
     [a4, b4, S4, cov_matrix4] = bivariate_fit(sss_final, d18O_plus_SST, ssser_final, d18O_plus_SST_err, ri=0.0, b0=1.0, maxIter=1e6) # here X =[SSS], Y = [SST]

# print(sss_final)
# print(d18O_plus_SST)
# print(ssser_final)
# print(d18O_plus_SST_err)


(r4m,p4m) = stats.pearsonr(sss_final, d18O_plus_SST) # (Pearson's correlation coefficient: scipy.stats.pearsonr(x,y)) 
print(b4)
print(r4m)
print(p4m)

sss_sst_covariance = np.cov(sss_final,sst_final)[0][1]
print(sss_sst_covariance)

if time_step == 'bimonthly':
    mean_sss = np.mean(sss_final.values)
    print(mean_sss)
    std_sss = np.std(sss_final.values)
    print(std_sss)
    std_ratio = np.std(sss_final.values) / np.std(sst_final.values)
    print(std_ratio)
else:
    mean_sss = np.mean(sss_final)
    print(mean_sss)
    std_sss = np.std(sss_final)
    print(std_sss)
    std_ratio = np.std(sss_final) / np.std(sst_final)
    print(std_ratio)


print(S4) # built-in goodness of fit metric 


# only plot the a2 regression, along with a single salinity time series and a single coral time series (commented out for now)
A2_PLOTS = False
if A2_PLOTS:
    plt.plot(sss_final, d18O_plus_SST, 'o', label = 'd18Oc plus SST - data', color='k')
    plt.title(coralid1)
    plt.plot(sss_final, a4 + b4*sss_final, 'r', label="δ18Oc + 0.21*SST = {0:.3f}*SSS + {1:.2f}".format(b4, a4))
    plt.xlabel('SSS (%)')
    plt.ylabel('Coral $\delta^{18}$O ($\perthousand$) + 0.21*SST')
    plt.legend(fontsize=8)
    plt.text(50, 10, 'R^2 = {0:.2f}'.format(r4m*r4m), horizontalalignment='center',verticalalignment='center') #transform=ax.transAxes: indicates that the coordinates are given relative to the axes bounding box, with (0, 0) being the lower left of the axes and (1, 1) the upper right. 
    plt.show()

    # plt.savefig(coralid1 + '_a2_regression.jpg')

    # time = range(startyr,endyr)
#     fig, ax = plt.subplots(figsize=(10,4))
#     plt.gca().invert_yaxis()
#     fig, ax = plt.subplots(figsize=(15,5))
#     ax.plot(d18Oc_interp.time,d18Oc_interp,color='black',label='d18Oc')
#     ax.plot(yr_d18Ocfinal,d18Oc_final,color='black',label='d18Oc')# 

#     ax2=ax.twinx()
#     plt.gca().invert_yaxis()
#     ax2.plot(sst_final.time,sst_final,color='blue',label='sst')
#     ax2.plot(yr_sssfinal,sss_final,color='red',label='sss')
#     ax1 = plt.gca()
#     ax1.set_xticks([1980,1990,2000,2010,2020])
#     lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
#     ax2.set_ylim(26,28)
#     plt.xlim(startyr,endyr)
#     # ax.set_xlabel('Year (CE)')
#     # ax.set_title('SSS and d18Oc (trop year avg)')
#     # plt.savefig(coralid1+'_sssyr_ts.jpg', bbox_inches='tight')
#     plt.show()


    time = range(startyr,endyr)
    fig, ax = plt.subplots(figsize=(10,4))
    plt.gca().invert_yaxis()
    #fig, ax = plt.subplots(figsize=(15,5))
    #ax.plot(d18Oc_interp.time,d18Oc_interp,color='black',label='d18Oc')
    ax.plot(d18Oc.time,d18Oc,color='black',label='d18Oc')

    # ax2=ax.twinx()
    plt.gca().invert_yaxis()
    ax.plot(sst_f.time,sst_f,color='red',label='sst')
    ax.plot(sss_f.time,sss_f,color='blue',label='sss')
    #ax1 = plt.gca()
    #ax1.set_xticks([1980,1990,2000,2010,2020])
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    #plt.ylim(-5.,-4.)
    #plt.xlim(startyr,endyr)
    ax.set_xlabel('Year (CE)')
    ax.set_title('SSS and d18Oc (raw)')
    # plt.savefig(coralid1+'_sss_ts_raw.pdf', bbox_inches='tight')
    plt.show()

# rest of the plots-- time series, etc 

MAKE_PLOTS = False


if MAKE_PLOTS: 
    plt.plot(sss_final, d18O_plus_SST, 'o', label = 'd18Oc plus SST - data', color='k')

    plt.plot(sss_final, a4 + b4*sss_final, 'r', label="δ18Oc + 0.21*SST = {0:.3f}*SSS + {1:.2f}".format(b4, a4))
    plt.xlabel('SSS (%)')
    plt.ylabel('Coral $\delta^{18}$O ($\perthousand$) + 0.21*SST')
    plt.legend(fontsize=8)
    (r4m,p4m) = stats.pearsonr(sss_final, d18O_plus_SST) # (Pearson's correlation coefficient: scipy.stats.pearsonr(x,y)) 
    plt.text(0.5, 0.8, 'R^2 = {0:.2f}'.format(r4m*r4m), horizontalalignment='center',verticalalignment='center') #transform=ax.transAxes: indicates that the coordinates are given relative to the axes bounding box, with (0, 0) being the lower left of the axes and (1, 1) the upper right. 
    plt.show()

    plt.plot(yr_sssfinal, sss_final, 'o', label = 'Time series: SSS')
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
    # fig, ax = plt.subplots(figsize=(15,5))
    # ax.plot(d18Oc_interp.time,d18Oc_interp,color='black',label='d18Oc')
    ax.plot(d18Oc.time,d18Oc,color='black',label='d18Oc')

    ax2=ax.twinx()
    ax2.plot(sst_f.time,sst_f,color='blue',label='sst')
    # ax2.plot(sst_f.time,sst_f,color='blue',label='sst')
    # ax1 = plt.gca()
    # ax1.set_xticks([1980,1990,2000,2010,2020])
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    # plt.ylim(-5.,-4.)
    # plt.xlim(startyr,endyr)
    ax.set_xlabel('Year (CE)')
    ax.set_title('SST and d18Oc (raw)')
    # plt.savefig(coralid1+'_sst_ts_raw.pdf', bbox_inches='tight')
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

"""
Current issues: 

HE18COC01
When performing xarray groupby operation for time binning, previous removal of nan's results in empty bins 
(a) interpolate instead of removing nan's before binning
(b) remove nan's after binning
(c) remove nan's before binning, but interpolate if a time bin is empty (may be the same as a)

Even after using interpolation and dropna to clean d18Oc_bin,
the 'sub' array is still empty for certain t1 and t2
What should we do about this?  


AB08MEN01
Everything looks good in the arrays, but the end plot of the regression is just ... very off ?

General:
The a2 values we are getting seem way off, esp when compared to the values of delta_SW vs SSS in the literature 
Need to fix round nearest month function 
Why does using startyr = coral_years[0] produce weird results?




"""

'''
# Drop data from all arrays at times that are nans in d18O
#d18Oc_bin_nan = d18Oc_bin.dropna('time_bins')
nan_mask = np.isnan(d18Oc_bin) # this is where d18O data ins nan
d18Oc_bin_nan = d18Oc_bin[~nan_mask]
sst_bin_nan = sst_bin[~nan_mask]
sss_bin_nan = sss_bin[~nan_mask]
sster_bin_nan = sster_bin[~nan_mask]
ssser_bin_nan = ssser_bin[~nan_mask]
d18Ocerr_bin_nan = d18Ocerr_bin[~nan_mask] 
# Regress d18Oc onto (-0.21*SST + SSS)
[a4m, b4m, S4m, cov_matrix4m] = bivariate_fit(sss_bin_nan.values, d18Oc_SSS_bin_nan.values, ssser_bin_nan.values, d18Oc_SSSerr_bin_nan, ri=0.0, b0=1.0, maxIter=1e6) # DataArray.values converts xarray to numpy array (d18Oc_SSSerr_bin is a numpy array, all others are DataArrays)
sigma_b4m = cov_matrix4m[0,0] # standard error of the slope
sigma_a4m = cov_matrix4m[1,1] # standard error of the intercept
(r4m,p4m) = stats.pearsonr(sss_bin_nan.values, d18Oc_SSS_bin_nan.values) # (Pearson's correlation coefficient: scipy.stats.pearsonr(x,y)) 

axis[0].text(0.5, 0.8, 'R^2 = {0:.2f}'.format(r1m*r1m), horizontalalignment='center',verticalalignment='center', transform=axis[0].transAxes) #transform=ax.transAxes: indicates that the coordinates are given relative to the axes bounding box, with (0, 0) being the lower left of the axes and (1, 1) the upper right. 




'''
