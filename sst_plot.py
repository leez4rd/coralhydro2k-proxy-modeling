from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import colors 
import numpy as np
import h5py
import os 
import subprocess 
import seaborn as sns 
from scipy.ndimage.filters import gaussian_filter
from netCDF4 import Dataset, date2index
from datetime import datetime
import xarray as xr

data = Dataset('sst_mnmean_noaa_ersstv5.nc')
# timeindex = date2index(datetime(1980, 1, 15),
#                        data.variables['time'])
ds = xr.open_dataset('sst_mnmean_noaa_ersstv5.nc',decode_times=True)         # ERSST v5
sss_ds = xr.open_dataset('sss_HadleyEN4.2.1g10_190001-201012.nc',decode_times=True)         # ERSST v5
sss_ds = sss_ds.sel(time=slice("2000-01-01", "2000-01-02"))

ds = ds.sel(time=slice("1980-01-01", "1980-01-2"))
print(ds)

ds['sst'].plot(cmap='jet', vmax=0.02)
plt.show()



lat = ds['lat'][:]
lon = ds['lon'][:]
lon, lat = np.meshgrid(lon, lat)

print(lon)
print(lat)

sst = ds['sst'].values
print(np.shape(sst))
sst = sst[0][:][:]
print(np.shape(sst))

# lat_sst_all = sst['lat'].values   # get coordinates from dataset
# lon_sst_all = sst['lon'].values   # lon = 0:360


# sst_f = sst[:,lat,lon]


# print(lat_sst_all)
# print(lon_sst_all)

# sst = sst[~np.isnan(sst)]


fig = plt.figure(figsize=(10, 8))
'''
m = Basemap(projection='lcc', resolution='c',
            width=8E6, height=8E6, 
            lat_0=-80, lon_0=-100,)

            '''
# m.shadedrelief(scale=0.5)

m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

'''
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
'''

m.pcolormesh(lon, lat, sst,
             latlon=True, cmap='coolwarm')
plt.clim(-8, 8)
m.drawcoastlines(color='lightgray')

plt.title('Sea surface temperature ')
plt.colorbar(label='temperature  (°C)');

plt.show()



lat = sss_ds['lat'][:]
lon = sss_ds['lon'][:]
lon, lat = np.meshgrid(lon, lat)


sss = sss_ds['sss'].values
print(sss)
sss = sss[:][:]


m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

'''
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
'''

m.pcolormesh(lon, lat, sss,
             latlon=True, cmap='magma')
plt.clim(-8, 8)
m.drawcoastlines(color='lightgray')

plt.title('Sea surface salinity ')
plt.colorbar(label='salinity');

plt.show()










# data = gaussian_filter(np.random.randn(20, 40), sigma=2)


# filter by coral species and site_type

# creating map to plot data on 
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

m.fillcontinents(color = 'green',lake_color='#46bcec')


# fig, ax = plt.subplots(figsize=(15, 5))
# sns.heatmap(data=sst, cbar_kws={'pad': 0.02}, ax=m)
m.contour(np.arange(.5, ds.shape[1]), np.arange(.5, ds.shape[0]), ds, colors='yellow')


# a2_plot = m.scatter(lons, lats, c=a2s, s=dotsizes, vmin = -1.5, vmax = 1.5 , cmap = "coolwarm", latlon = True)
# plt.colorbar(a2_plot)
plt.show()