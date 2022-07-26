from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import colors 
import numpy as np
import h5py
import os 
import subprocess 
import seaborn as sns 

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

# m.drawcoastlines()
# m.fillcontinents(color = 'white',lake_color='#46bcec')


# read in database
f = h5py.File('hydro2kv0_5_2.mat','r')
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
data_dict = {}


stop_early_count = 0 
for recordID in coralnames:
	if stop_early_count == 10:
		break
	stop_early_count += 1

	# eventually need to make the calculation of a2 its own function with recordID as a parameter
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
			print(mean_sss_all)
			print(all_covariances)

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

a2s = np.array(a2s)

'''
sns.scatterplot(mean_sss_all, a2s, size = dotsizes, legend = "brief")
plt.xlabel("Mean SSS")
plt.ylabel("a2")
plt.savefig('a2s_and_mean_SSS.jpg')
sns.scatterplot(std_sss_all, a2s, size = dotsizes, legend = "brief")
plt.xlabel("Standard Deviation of SSS")
plt.ylabel("a2")
plt.savefig('a2s_and_SSS_SD.jpg')
sns.scatterplot(all_covariances, a2s, size = dotsizes, legend = "brief")
plt.xlabel("Covariance of SSS and SST")
plt.ylabel("a2")
plt.savefig('a2s_and_covariance.jpg')

sns.scatterplot(std_ratios, a2s, size = dotsizes, legend = "brief")
plt.xlabel("ratio of SD's of SSS, SST")
plt.ylabel("a2")
plt.savefig('a2s_and_SD_ratio.jpg')
'''

a2_plot = m.scatter(lons, lats, c=a2s, s=dotsizes, vmin = -3, vmax = 3 , cmap = "coolwarm", latlon = True)
plt.colorbar(a2_plot)


# for color mesh, color variable needs to be 2 dimensionally indexed
# ie cov(SSS, SST) at latitude longitude needs to be stored as value in a matrix 
# lats, lons, and all_covariances are our input lists, so we need to build a color matrix
# such that color matrix [lats[i], lons[i]] = all_covariances[i]
lons = np.asarray(lons)
lats = np.asarray(lats)

lats, lons = m(lons, lats)
print(lats)
print(lons)

print("st")
print(np.max(lons))
color_matrix = np.empty([int(np.max(lons)) + 1, int(np.max(lats)) + 1])
print(lons)
print(lats)
for i in range(len(lats)):
 	color_matrix[int(lons[i]), int(lats[i])] = all_covariances[i]

print(color_matrix)

longitudes, latitudes = np.meshgrid(lons, lats)

print(longitudes)
print(latitudes)

m.pcolormesh(lons, lats, color_matrix,
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
