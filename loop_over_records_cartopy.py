
# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs 
import matplotlib.pyplot as plt
from matplotlib import colors 
import numpy as np
import h5py
import os 
import subprocess 
import seaborn as sns 
import xarray as xr 
import pandas as pd 
from datetime import datetime
import xesmf as xe
import scipy
import scipy.sparse
import math
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# make the map global rather than have it zoom in to
# the extents of any plotted data
ax.set_global()

ax.stock_img()
ax.coastlines()


# filter by coral species and site_type
dataframe = pd.read_csv('Database_Metadata_CH2k - Sheet1.csv')
# subset = dataframe['Core ID' == recordID]
# subset = np.where(subset['Species'].str.contains(pat = '[pP]orites'), regex = True)
# if recordID in list(subset['Core ID'])... (only loop if Porites)

# analogous code for site type?


# m.fillcontinents(color = 'white',lake_color='#46bcec')


# read in database
dir = ''
dir = ''
f = h5py.File(dir+'hydro2kv0_5_2(1).mat','r')
dataset = f.get('ch2k/')
coralnames = list(dataset.keys()) #last key here is 'version'? 

# loop through record ID's 
lats = []
lons = []
a2s = []
dotsizes = []
mean_sss_all = []
std_sss_all = []
all_covariances = []
std_ratios = []
GOFs = []
data_dict = {}
error_count = 0
problem_corals = []


stop_early_count = 0 
for recordID in coralnames:
#for recordID in coralnames[0:5]:        # to test code
	if stop_early_count == -10:
		break
	stop_early_count += 1

	# eventually need to make the calculation of a2 its owsn function with recordID as a parameter
	# right now it just interfaces with command line to call the script and keeps chugging if there are errors

	print("Current ID:")
	print('\n', recordID)

	current_record = f.get('ch2k/'+recordID+'/d18O')
	error_encountered = False
	try:
		# result = subprocess.check_output(['python3', 'regression:testing_version.py', '-i', recordID])
		# result = subprocess.check_output(['python3', 'pyWLS_d18Oc_AA(1).py', '-i', recordID])
		# result = subprocess.check_output(['python3', 'cleaned_up_proxy_modeling.py', '-i', recordID])
		#result = subprocess.check_output(['python3', 'ch2k_regression.py', '-i', recordID])
		result = subprocess.check_output(['python3', 'ch2k_regression_AArev.py', '-i', recordID]).decode('ascii')
	except:
		error_encountered = True
		result = 0
		a2 = 0
		print("Error encountered in script, ignoring...")
		problem_corals += [recordID]
		error_count += 1

	if not error_encountered:
		#split_results = str(result).split('\\n')
		split_results = str(result).split('\n')
		print(len(split_results))
		print(split_results)
		# a2 = float(split_results[0][2:len(split_results[0])]) 
		a2_string = str(split_results[0])
		r4m = float(split_results[1])
		p4m = float(split_results[2])
		covariance = float(split_results[3])
		mean_sss = float(split_results[4])
		sss_std = float(split_results[5])
		std_ratio = float(split_results[6])
		goodness_of_fit = float(split_results[7])
		a2 = float(a2_string[0:len(a2_string)])
        


		

	# a2 = float(split_results[0][2:len(split_results[0])]) 
	
	# print(split_results[len(split_results)-2])

	print("a2 is...")

	print('\n', a2)
	try:
		latc = np.array(f.get('ch2k/'+recordID+'/lat'))[0][0]
		lonc = np.array(f.get('ch2k/'+recordID+'/lon'))[0][0]
		 
		latc = float(latc)
		lonc = float(lonc) 
	except:
		print("no latitude longitude pair for this key")

	if not np.isnan(a2):
		if np.abs(a2) < 2.5 and a2 != 0:
			data_dict[recordID] = (latc, lonc, a2)
			# lat, lon = m(latc, lonc)
			lats += [latc]
			lons += [lonc]
			a2s += [a2] # normalized to 5 temporarily, should be max a2 that we trust 
			dotsizes += [r4m*r4m*100] # multiply R^2 to exaggerate size differences 
			print("latitudes: ", lats)
			print("longitudes ", lons)
			print("a2's: ", a2s)
			print("R^2 values: ", dotsizes)

			# do we need to do this for every record? 
			mean_sss_all += [mean_sss]
			std_sss_all += [sss_std]
			all_covariances += [covariance]
			std_ratios += [std_ratio]
			GOFs += [goodness_of_fit]
			print(mean_sss_all)
			print(all_covariances)
			print("COVARIANCE IS ")
			print(covariance)

		# random thoughts
		# if this returns an array, do we average them? 
		# otherwise could feed this to scatterplot directly?

		# maybe use a dictionary comprehension with recordID as key and lat, lon tuple as value
		# populate this dictionary as we loop


		# maybe better to just output a three tuple of (lat, lon, a2)
	#except: 
	#	print("Error encountered, ignoring...")

# someone on stack overflow suggested switching lat and lon here... confusing but we will see if it works 
# a2s = colors.Normalize(a2s)
# copya2 = a2s
# a2s = [x for x in a2s if x != 0] # ignore all cases where a2 was zero (ie script fails)
# lons = [x for x,y in lons, copya2 if y != 0] # ignore all cases where a2 was zero (ie script fails)
# lats = [x for x,y in lats, copya2 if y != 0] # ignore all cases where a2 was zero (ie script fails)
# dotsizes  =[x for x in dotsizes if x != 0]

print("latitudes")
print(lats)
print("longitudes")
print(lons)
print("a2 values: ")
print(a2s)
print("R^2 values: ")
print(dotsizes)
print("Mean SSS values: ")
print(mean_sss_all)
print("SD of SSS: ")
print(std_sss_all)
print("Covariances between SSS and SST: ")
print(all_covariances)
print("Ratios between the SD's")
print(std_ratios)

print("mapping data: ")
print(data_dict)



print("Total number of problem corals: ", error_count)
print("List of culprits: ", problem_corals)


a2_plot = ax.scatter(lons, lats, c=a2s, s=dotsizes, vmin = -0.5, vmax = 2, cmap = "gnuplot", transform=ccrs.PlateCarree()) # latlon = True, zorder = 2)
plt.colorbar(a2_plot)


plt.show()

a2s = np.array(a2s)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# some scatterplots
# plt.figure()
sns.scatterplot(mean_sss_all, a2s, size = dotsizes, legend = "brief", ax = ax1)
ax1.set_xlabel("Mean SSS")
ax1.set_ylabel("a2")
# ax1.savefig('a2s_and_mean_SSS.jpg')
# plt.figure()
sns.scatterplot(std_sss_all, a2s, size = dotsizes, legend = "brief", ax = ax2)
ax2.set_xlabel("Standard Deviation of SSS")
ax2.set_ylabel("a2")
# ax2.savefig('a2s_and_SSS_SD.jpg')
# plt.figure()
sns.scatterplot(all_covariances, a2s, size = dotsizes, legend = "brief", ax = ax3)
ax3.set_xlabel("Covariance of SSS and SST")
ax3.set_ylabel("a2")
# ax3.savefig('a2s_and_covariance.jpg')
# plt.figure()
sns.scatterplot(std_ratios, a2s, size = dotsizes, legend = "brief", ax = ax4)
ax4.set_xlabel("ratio of SD's of SSS, SST")
ax4.set_ylabel("a2")
# ax4.savefig('a2s_and_SD_ratio.jpg')
plt.show()

plt.figure()

# creating map to plot data on 
'''
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')


m = Basemap(llcrnrlon=-180,
            llcrnrlat=-75,
            urcrnrlon=180,
            urcrnrlat=75,
            lat_0=0,
            lon_0=180,
            projection='merc',
            resolution = 'h',
            area_thresh=10000.,
            )

m.drawcoastlines()


try:
	parallels = np.arange(0.,81,10.)
	# labels = [left,right,top,bottom]
	m.drawparallels(parallels,labels=[False,True,True,False])
	meridians = np.arange(10.,351.,20.)
	m.drawmeridians(meridians,labels=[True,False,False,True])
except:
	print("error drawing parallels and meridians")
	pass

'''



# for color mesh, color variable needs to be 2 dimensionally indexed
# ie cov(SSS, SST) at latitude longitude needs to be stored as value in a matrix 
# lats, lons, and all_covariances are our input lists, so we need to build a color matrix
# such that color matrix [lats[i], lons[i]] = all_covariances[i]
lons = np.asarray(lons)
lats = np.asarray(lats)

# lats, lons = m(lons, lats)



'''
# this only plots covariances where our coral records are located 
# looks kinda bad 
color_matrix = np.empty([len(lons), len(lats)])
for i in range(len(lons)):
	for j in range(len(lats)):
		if i == j:
			color_matrix[i, i] = all_covariances[i]
		else:
			color_matrix[i, j] = 0



longitudes, latitudes = np.meshgrid(lons, lats)
print(latitudes)
print("test")
print(color_matrix)
print(lons)
m.pcolormesh(lons, lats, color_matrix,
             latlon=True, cmap='coolwarm', zorder = 1, vmin = -0.05, vmax = 0.05)

'''


plt.show()
#plt.savefig('a2'+time_step+'.pdf',bbox_inches='tight')
plt.savefig('a2.pdf',bbox_inches='tight')


#============================================================================
# Read in  SST data and format for pcolormesh 
#============================================================================

dir = ''
ds = xr.open_dataset(dir+'sst_mnmean_noaa_ersstv5.nc',decode_times=True)         # ERSST v5

# Change longitude array from 0:360 to -180:180 (if not already in this format)-- MAKE THIS A FUNCTION 
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

# ds = ds.sel(time=slice("2000-01-01", "2000-01-02"))
print(ds)

# Regrid data onto common grid
ds_regrid = xr.Dataset({"lat": (["lat"], np.arange(-90, 92, 2.0)),"lon": (["lon"], np.arange(-180, 182, 2.0)),})  # set up regridder
regridder = xe.Regridder(ds, ds_regrid, "bilinear")   # build regridder
ds_regrid = regridder(ds)                             # apply regridder to data

sst = ds_regrid['sst'].values
sst_all = ds_regrid['sst']
time_sst_all = ds_regrid['time']             # HadEN4: time x lat x lon (time = 1900:2010)
lat_sst_all = sst_all['lat']
lon_sst_all = sst_all['lon']

lat = ds_regrid['lat'][:].values
lon = ds_regrid['lon'][:].values
lon2, lat2 = np.meshgrid(lon, lat)

#============================================================================
# Read in  SSS data and format for pcolormesh 
#============================================================================dir = '/Users/alyssa_atwood/Desktop/Dropbox/Obs_datasets/Salinity/HadEN4/'

dir = '/Users/alyssa_atwood/Desktop/Dropbox/Obs_datasets/Salinity/HadEN4/'
dir = ''
ds = xr.open_dataset(dir+'sss_HadleyEN4.2.1g10_190001-201012.nc',decode_times=True)  
# ds = ds.interpolate_na(dim = "lat", method = "linear", fill_value = "extrapolate")

# Change longitude array from 0:360 to -180:180 (if not already in this format)-- MAKE THIS A FUNCTION 
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

# Regrid data onto common grid
regridder = xe.Regridder(ds, ds_regrid, "bilinear")   # build regridder
ds_regrid = regridder(ds)                             # apply regridder to data

sss_all = ds_regrid['sss']
time_sss_all = ds_regrid['time']             # HadEN4: time x lat x lon (time = 1900:2010)
lat_sss_all = sss_all['lat']
lon_sss_all = sss_all['lon']

#============================================================================
# Read in  SSS error data and format for pcolormesh 
#============================================================================
ds = xr.open_dataset(dir+'sss_HadleyEN4.2.1g10_190001-202012_salinityerrorSD.nc',decode_times=True)         

# Change longitude array from 0:360 to -180:180 (if not already in this format)-- MAKE THIS A FUNCTION 
lon_name = 'LONN180_180'  # whatever name is in the data
# Adjust lon values to make sure they are within (-180, 180)
ds['_longitude_adjusted'] = xr.where(ds[lon_name] > 180, ds[lon_name] - 360, ds[lon_name])
# reassign the new coords to as the main lon coords and sort DataArray using new coordinate values
ds = (
    ds
    .swap_dims({lon_name: '_longitude_adjusted'})
    .sel(**{'_longitude_adjusted': sorted(ds._longitude_adjusted)})
    .drop(lon_name))
ds = ds.rename({'_longitude_adjusted': lon_name})

# Get rid of singleton depth dimension 
#ds = ds.drop_dims('LEV1_1', errors='raise')  # seemed to delete the actual variables

# Rename dimensions so that regridder can identify them
ds = ds.rename({'LAT': 'lat','LONN180_180': 'lon','TIME': 'time'})

# Regrid data onto common grid
regridder = xe.Regridder(ds, ds_regrid, "bilinear")   # build regridder
ds_regrid = regridder(ds)                             # apply regridder to data

ssser_all = ds_regrid['SALT_ERR_STD']
time_ssser_all = ds_regrid['time']               # HadEN4 error: time x 1 x lat x lon (time = 1900:2020)
lat_ssser_all = ssser_all['lat']
lon_ssser_all = ssser_all['lon']

#============================================================================
# Read in  SST error data and format for pcolormesh 
#============================================================================

dir = ''
ds = xr.open_dataset(dir+'ersstev5_uncertainty.nc',decode_times=True)    

# Change longitude array from 0:360 to -180:180 (if not already in this format)-- MAKE THIS A FUNCTION 
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

# Regrid data onto common grid
regridder = xe.Regridder(ds, ds_regrid, "bilinear")   # build regridder
ds_regrid = regridder(ds)                             # apply regridder to data

sster_all = ds_regrid['ut']
time_sster_all = ds_regrid['time']             # ERSSTv5: time (1/1854-12/2020; units = "days since 1800-1-1 00:00:00")
lat_sster_all = sster_all['lat']
lon_sster_all = sster_all['lon']

#============================================================================
# Calculate covariance 
#============================================================================

startyr_cov = 1980        # start and end on a specified year (HadEN34 only goes to Dec. 2010)
endyr_cov = 2010
# Select time period
t1_cov = datetime(startyr_cov, 1, 1)     
t2_cov = datetime(endyr_cov, 12, 31) 

sst_all = sst_all.sel(time=slice(t1_cov, t2_cov))
sss_all = sss_all.sel(time=slice(t1_cov, t2_cov))

longitudes = []
latitudes = []
sst_fs = []
cov_fs = []
sdratio_fs = []
cov_fs2= np.empty([len(lat),len(lon)], dtype=float)

for ilat in range(len(lat)):
    for ilon in range(len(lon)):
        latitude = lat[ilat]
        longitude = lon[ilon]        
        longitudes.append(longitude)
        latitudes.append(latitude)
        sst_f = sst_all[:,ilat,ilon]   		# get salinity and temperature at this latitude and longitude 
        sss_f = sss_all[:,ilat,ilon]
 		# sst_fs.append(sst_f)

        # get errors at this latitude and longitude 
        sster_f = sst_all[:, ilat, ilon]
        ssser_f = sss_all[:, ilat, ilon]

        # note: need to slice to same time period, this is just a proof of concept to see if we can map successfully
        # calculate covariance, correlation 
        #this_covariance = np.cov(sst_f.values[0:1000], sss_f.values[0:1000])[0][1]
        this_covariance = np.cov(sst_f.values, sss_f.values)[0][1]
        cov_fs += [(this_covariance)]
        cov_fs = np.asarray(cov_fs)
        cov_fs2[ilat,ilon] = this_covariance
        #if latitude == 0 and longitude == 172:
        #    y=x
        
 		# corr = np.corrcoef(sst_f, sss_f)
 		# coverr = np.cov(sster_f, ssser_f)
 		# correrr = np.corrcoef(sster_f, ssser_f)
 		# calculate the ratio of the SD's of the errors 
 		# sd_sst = np.std(sst_f)
 		# sd_sss = np.std(sss_f)

 		# additional statistical features...




# Doesn't work with nans
#cov_fs2 = xr.cov(da_a, da_b, dim="time", ddof=1) #ddof (int, optional) – If ddof=1, covariance is normalized by N-1, giving an unbiased estimate, else normalization is by N.

# sst_fs = np.asarray(sst_fs)

# longitudes, latitudes = np.meshgrid(longitudes, latitudes)

#ax.pcolormesh(np.asarray(longitudes), np.asarray(latitudes), cov_fs,
#             latlon=True, cmap='coolwarm')
ax.pcolormesh(lon, lat, cov_fs2, cmap='coolwarm')

plt.show()


# Plot map of covariance
levels = np.arange(-0.7,0.75,0.05)
cmap = np.array(['seismic'])
title = np.array(['cov(SST,SSS)'])
cbar_title = np.array(['Covariance'])

fig1, ax = plt.subplots(nrows=1, figsize=(8,30), subplot_kw={'projection':ccrs.PlateCarree(central_longitude=180)})
ax.set_extent([90,300,-35, 35], ccrs.PlateCarree()) # 90E - 140W and ~25S to ~25N
ax.set_yticks([-30,-20,-10,0,10,20,30])
ax.set_xticks([-90,-45,0,45,90])
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
im = ax.contourf(lon, lat, cov_fs2, transform=ccrs.PlateCarree(), levels=levels, cmap=cmap[0], extend='both')
ax.coastlines()
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.03)
cbar.set_label(cbar_title[0])
plt.rcParams.update({'font.size': 16})     # changes font size in all plot components
print(fig1)
plt.savefig('cov.pdf',bbox_inches='tight')
