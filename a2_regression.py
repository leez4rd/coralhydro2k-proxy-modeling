#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:28:13 2021

@authors: alyssaatwood and lee
"""


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
import sys
import os
from datetime import datetime
from cftime import DatetimeProlepticGregorian

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


# more generalized way to do this? 
def read_in_sss(lat, lon):
    #============================================================================
    # Read in obs SSS and uncertainty data
    #============================================================================

    dir = '' # make this generalizable using path library 
    ds = xr.open_dataset('./sss_HadleyEN4.2.1g10_190001-201012.nc',decode_times=True)         # ERSST v5

    sss_all = ds['sss']


    time_sss_all = ds['time']             # HadEN4: time x lat x lon (time = 1900:2010)
    lat_sss_all = sss_all['lat'].values   # get coordinates from dataset
    lon_sss_all = sss_all['lon'].values

    # Get indices and sst data at closest grid point to coral
    (indlat_sss, latval_sss) = find_nearest(lat_sss_all, lat)
    (indlon_sss, lonval_sss) = find_nearest(lon_sss_all, lon)

    sss_f = sss_all[:,indlat_sss,indlon_sss]

    # Match ages of SST and SSS data to coral data
    # sss_final = sss.sel(time = timec, method='nearest')

    return sss_f

def read_in_sst(lat, lon):
    #============================================================================
    # Read in obs SST and uncertainty data
    #============================================================================

    dir = ''
    ds = xr.open_dataset('./sst_mnmean_noaa_ersstv5.nc',decode_times=True)         # ERSST v5

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
    (indlat_sst, latval_sst) = find_nearest(lat_sst_all, lat)
    (indlon_sst, lonval_sst) = find_nearest(lon_sst_all, lon)
    # (indtime_sst, timeval_sst) = find_nearest(time_sst_all , timec[0])   # coral data steps on midpoints, sst data steps on first day of month... find closest time in sst data set to start of coral data
    
    sst_f = sst_all[:,indlat_sst,indlon_sst]
    
    # match ages of SST and SSS data to coral data
    # sst_final = sst.sel(time = timec, method='nearest')
    return sst_f


def read_in_sst_err(lat, lon):
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

    sster_all = ds['ut']
    time_sster_all = ds['time']             # ERSSTv5: time (1/1854-12/2020; units = "days since 1800-1-1 00:00:00")
    lat_sster_all = sster_all['latitude'].values   # get coordinates from dataset
    lon_sster_all = sster_all['longitude'].values

    (indlat_sster, latval_sster) = find_nearest(lat_sster_all, lat)
    (indlon_sster, lonval_sster) = find_nearest(lon_sster_all, lon)
    sster_f = sster_all[:,indlat_sster,indlon_sster]

    # Match ages of SST and SSS data to coral data
    #sster_final = sster.sel(time = timec, method='nearest')


    return sster_f

def read_in_sss_err(lat, lon):

    #============================================================================
    # Read in HadEN4 uncertainties in sea surface salinity 
    #============================================================================
    ds = xr.open_dataset('./sss_HadleyEN4.2.1g10_190001-202012_salinityerrorSD.nc',decode_times=True)         # ERSST v5

    ssser_all = ds['SALT_ERR_STD']
    time_ssser_all = ds['TIME']               # HadEN4 error: time x 1 x lat x lon (time = 1900:2020)
    lat_ssser_all = ssser_all['LAT'].values   # get coordinates from dataset
    lon_ssser_all = ssser_all['LONN180_180'].values


    # Get rid of singleton depth dimension - Create new DataArray object by specifying coordinates and dimension names (xarray) from numpy array

    ssser_all = xr.DataArray(ssser_all[:,0,:,:], coords=[time_ssser_all,lat_ssser_all,lon_ssser_all], dims=["time","lat","lon"])

    # ssser_all = xr.DataArray(ssser_all, coords=[time_ssser_all,lat_ssser_all,lon_ssser_all], dims=["time","lat","lon"])
    (indlat_ssser, latval_ssser) = find_nearest(lat_ssser_all, lat)
    (indlon_ssser, lonval_ssser) = find_nearest(lon_ssser_all, lon)
    ssser_f = ssser_all[:,indlat_ssser,indlon_ssser]

    return ssser_f

    #============================================================================#

def read_in_coral_record(coralid1):

    #============================================================================
    # Read in coral d18O data
    #============================================================================
    # dataset = f.get('ch2k/')
    # coralnames = list(dataset.keys())
    
    f = h5py.File('hydro2kv0_5_2(1).mat','r')

    data1 = f.get('ch2k/'+coralid1+'/d18O')

    

    timec01 = (data1[0,:])-1/24     # time in fractional year CE (subtract 1/24 so that time corresponds to start of month (as with sst datasets), rather than mid-month)
    d18Oc1 = np.array(data1[1,:])           # Convert to NumPy array


    timec1 = decyrs_to_datetime(timec01)    # convert time from decimal year YYYY.FF to datetime object (YYYY-MM-DD)

    # Analytic error in d18Oc (assumes the values are in per mille!)
    data_analerr = f.get('ch2k/'+coralid1+'/d18O_errAnalytic')
    d18Oc_analerr = data_analerr[0,:]      # analytical error is a single value (in per mille)


    # round time steps to nearest month (just a couple days off in some months)
    timec1mo = round_nearest_month(timec1) 

    timec1mo = timec1  
    
    '''
    need to fix this!
    if False:                  # if a second coral id (coralid2) exixts, merge the two data sets
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
    '''




    # create DataArray object by specifying coordinates and dimension names (xarray) from numpy array

    d18Oc = xr.DataArray(d18Oc1, coords=[timec1mo], dims=["time"])
    # d18Oc = d18Oc.dropna(dim='time', how='any', thresh=None)  # drop all nans in array (across 0th dim)

    d18Oc = d18Oc.sortby(d18Oc.time,ascending=True)                   # sort arrays by time
    timec = d18Oc.time


    lat = f.get('ch2k/'+coralid1+'/lat')
    lon = f.get('ch2k/'+coralid1+'/lon')  # lon = -180:180
    lat = np.array(lat)                 # Convert to NumPy array
    lon = np.array(lon)                 # Convert to NumPy array
    
    return [lat, lon, d18Oc]


# make sure this is actually mutating the objects
def detrend_data(sst_f, sss_f, sster_f, ssser_f, d18Oc):
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

    # d18Oc_dt = xr.DataArray(d18Oc_dt1, coords=[d18Oc.time], dims=["time"])
#     sst_dt = xr.DataArray(sst_dt1, coords=[sst_f.time], dims=["time"])
#     sss_dt = xr.DataArray(sss_dt1, coords=[sss_f.time], dims=["time"])
#     sster_dt = xr.DataArray(sster_dt1, coords=[sster_f.time], dims=["time"])
#     ssser_dt = xr.DataArray(ssser_dt1, coords=[ssser_f.time], dims=["time"])

    d18Oc = xr.DataArray(d18Oc_dt1, coords=[d18Oc.time], dims=["time"])
    sst_f = xr.DataArray(sst_dt1, coords=[sst_f.time], dims=["time"])
    sss_f = xr.DataArray(sss_dt1, coords=[sss_f.time], dims=["time"])
    sster_f = xr.DataArray(sster_dt1, coords=[sster_f.time], dims=["time"])
    ssser_f = xr.DataArray(ssser_dt1, coords=[ssser_f.time], dims=["time"])

    # this step may be generating nans - all the time_bins rows contain nans
    lb = np.array(sst_f.time[::2])    
    lb = lb[0:-1]    


    lb = lb[~np.isnan(lb)]
    sst_f = sst_f[~np.isnan(sst_f)]
    return lb





# bimonthly binning 
def bimonthly_binning(dataset, lb):
    '''
    d18Oc_bin = d18Oc_f.groupby_bins('time',bins=sst_f.time[::2],labels=lb).mean()   
    sst_bin = sst_f.groupby_bins('time',bins=sst_f.time[::2],labels=lb).mean()   
    sss_bin = sss_f.groupby_bins('time',bins=sst_f.time[::2],labels=lb).mean()
    sster_bin = sster_f.groupby_bins('time',bins=sst_f.time[::2],labels=lb).mean()
    ssser_bin = ssser_f.groupby_bins('time',bins=sst_f.time[::2],labels=lb).mean()
    '''
    print(dataset)
    print(len(lb))
    print(len(dataset.time[::2]))
    print(dataset.time[::2])


    lb = [x for x in range(len(dataset.time[::2]) - 1)] # temp fix
    binned = dataset.groupby_bins('time', bins = dataset.time[::2], labels = lb).mean()
    return binned

# again, make sure this is mutating the objects and not creating copies 
def select_common_time_period(d18Oc, sst_f, sss_f, sster_f, ssser_f):

    #=============================================================================
    # Select common (i.e., overlapping for all datasets) time period
    #=============================================================================

    coral_years = np.array(d18Oc.time.dt.year)   # array of years in coral data
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

    return [startyr, endyr]



def find_a2(d18Oc_bin, sst_bin, sss_bin, sster_bin, ssser_bin, endyr, coralid = 'OS14UCP01', resolution='bimonthly', startyr = 1980):
    if resolution == 'year':
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
        
        # yr_sssfinal = yr_sssfinal[nan_mask]
        
        for i in range(len(sss_final)):
            d18O_plus_SST[i] = d18Oc_final[i] + 0.21*sst_final[i]
            d18O_plus_SST_err[i] = d18Ocerr_final[i] + 0.21*0.21*sster_final[i]+2*0.21*np.cov(d18Oc_final,sst_final)[0][1]
            

    # Regress d18Oc + 0.21*SST onto SSS

    if resolution == 'bimonthly':
        [a4, b4, S4, cov_matrix4] = bivariate_fit(sss_final.values, d18O_plus_SST, ssser_final.values, d18O_plus_SST_err, ri=0.0, b0=1.0, maxIter=1e6) # here X =[SSS], Y = [SST]
    else:
         [a4, b4, S4, cov_matrix4] = bivariate_fit(sss_final, d18O_plus_SST, ssser_final, d18O_plus_SST_err, ri=0.0, b0=1.0, maxIter=1e6) # here X =[SSS], Y = [SST]

    (r4m,p4m) = stats.pearsonr(sss_final, d18O_plus_SST) # (Pearson's correlation coefficient: scipy.stats.pearsonr(x,y)) 
    print(b4)
    print(r4m)
    print(p4m)

    return [b4, r4m, p4m]







def main():

    lat, lon, d18Oc = read_in_coral_record('OS14UCP01')


    SSS_data = read_in_sss(lat, lon)
    SST_data = read_in_sst(lat, lon)
    SSS_error_data = read_in_sss_err(lat, lon)
    SST_error_data = read_in_sst_err(lat, lon)
    
    startyr, endyr = select_common_time_period(d18Oc, SST_data, SSS_data, SST_error_data, SSS_error_data)
    
    lb = detrend_data(SST_data, SSS_data, SST_error_data, SSS_error_data, d18Oc)


    d18Oc_bin = bimonthly_binning(d18Oc, lb)  
    sst_bin = bimonthly_binning(SST_data, lb) 
    sss_bin = bimonthly_binning(SSS_data, lb)
    sster_bin = bimonthly_binning(SSS_error_data, lb)
    ssser_bin =  bimonthly_binning(SST_error_data, lb)

    

    result = find_a2(d18Oc_bin, sst_bin, sss_bin, sster_bin, ssser_bin, endyr, 'OS14UCP01', 'bimonthly')

    print(result)


if __name__ == '__main__':
    main()