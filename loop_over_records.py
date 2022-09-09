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
f = h5py.File('hydro2kv0_5_2(1).mat','r')
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


stop_early_count = 0 
for recordID in coralnames:
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
		result = subprocess.check_output(['python3', 'ch2k_regression.py', '-i', recordID])
	except:
		error_encountered = True
		result = 0
		a2 = 0
		print("Error encountered in script, ignoring...")

	if not error_encountered:
		split_results = str(result).split('\\n')
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
		a2 = float(a2_string[2:len(a2_string)])


		

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
			print(lats)
			print(lons)
			print(a2s)
			print(dotsizes)

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


"""
read in SST data and format for pcolormesh 

"""

ds = xr.open_dataset('sst_mnmean_noaa_ersstv5.nc',decode_times=True)         # ERSST v5
# ds = ds.sel(time=slice("2000-01-01", "2000-01-02"))
print(ds)

# ds['sst'].plot(cmap='jet', vmax=0.02)
# plt.show()



lat = ds['lat'][:]
lon = ds['lon'][:]
lon, lat = np.meshgrid(lon, lat)

print(lon)
print(lat)

sst = ds['sst'].values
sst_all = ds['sst']

""" read in SSS data """ 
ds = xr.open_dataset('sss_HadleyEN4.2.1g10_190001-201012.nc',decode_times=True)         # ERSST v5

# ds = ds.interpolate_na(dim = "lat", method = "linear", fill_value = "extrapolate")

# UNCOMMENTED THIS LINE 
# ds = ds.dropna(dim = "lat", how = "any") # -- dropping nans caused major inconsistencies in data for some reason 
# if this causes problems elsewhere, we can do it conditionally or with try / except 

sss_all = ds['sss']


time_sss_all = ds['time']             # HadEN4: time x lat x lon (time = 1900:2010)
lat_sss_all = sss_all['lat'].values   # get coordinates from dataset
lon_sss_all = sss_all['lon'].values



""" read in error data """
ds = xr.open_dataset('sss_HadleyEN4.2.1g10_190001-202012_salinityerrorSD.nc',decode_times=True)         # ERSST v5

# ds = ds.dropna(dim = "LAT", how = "any") # not exactly sure why we are dropping along this dimension but im not questioning it yet

ssser_all = ds['SALT_ERR_STD']
time_ssser_all = ds['TIME']               # HadEN4 error: time x 1 x lat x lon (time = 1900:2020)
lat_ssser_all = ssser_all['LAT'].values   # get coordinates from dataset
lon_ssser_all = ssser_all['LONN180_180'].values


# Get rid of singleton depth dimension - Create new DataArray object by specifying coordinates and dimension names (xarray) from numpy array

ssser_all = xr.DataArray(ssser_all[:,0,:,:], coords=[time_ssser_all,lat_ssser_all,lon_ssser_all], dims=["time","lat","lon"])

# ssser_all = xr.DataArray(ssser_all, coords=[time_ssser_all,lat_ssser_all,lon_ssser_all], dims=["time","lat","lon"])

ds = xr.open_dataset('ersstev5_uncertainty.nc',decode_times=True)         # ERSST v5
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










longitudes = []
latitudes = []
sst_fs = []
cov_fs = []
sdratio_fs = []

for latitude in range(-80, 80):
	for longitude in range(-180, 180):
		longitudes.append(longitude)
		latitudes.append(latitude)
		# get salinity and temperature at this latitude and longitude 
		sst_f = sst_all[:,latitude,longitude]
		sss_f = sss_all[:,latitude,longitude]
		# sst_fs.append(sst_f)

		# get errors at this latitude and longitude 
		ssser_f = sss_all[:, latitude, longitude]
		sster_f = sst_all[:, latitude, longitude]


		# note: need to slice to same time period, this is just a proof of concept to see if we can map successfully
		# calculate covariance, correlation 
		this_covariance = np.cov(sst_f.values[0:1000], sss_f.values[0:1000])[0][1]
		cov_fs += [(this_covariance)]
		cov_fs = np.asarray(cov_fs)
		# corr = np.corrcoef(sst_f, sss_f)
		# coverr = np.cov(sster_f, ssser_f)
		# correrr = np.corrcoef(sster_f, ssser_f)
		# calculate the ratio of the SD's of the errors 
		# sd_sst = np.std(sst_f)
		# sd_sss = np.std(sss_f)


		# additional statistical features...


# sst_fs = np.asarray(sst_fs)

# longitudes, latitudes = np.meshgrid(longitudes, latitudes)

ax.pcolormesh(np.asarray(longitudes), np.asarray(latitudes), cov_fs,
             latlon=True, cmap='coolwarm')


plt.show()

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





# mapping part
# assuming we are fed a dictionary of the form recordID: (lat, lon, a2)
# maybe put this in above for loop instead? 
'''
lat_min = -40
lat_max = 40
lon_min = 90
lon_max = 300


x = np.empty(1000)
y = np.empty(1000)
for i in range(1000):
	x[i] = np.random.randint(-300, 300)
	y[i] = np.random.randint(-180, 180)

x, y = m(list(x), list(y)'''

# plt.scatter(x,y,z,marker='o',color='Red', zorder = 3)
# plt.show()
