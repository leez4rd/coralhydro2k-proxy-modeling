from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os 
import subprocess 

# DISCLAIMER: very hacky right now, just trying to get something going 

# creating map to plot data on 
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')


m = Basemap(llcrnrlon=40,
            llcrnrlat=-60,
            urcrnrlon=260,
            urcrnrlat=60,
            lat_0=0,
            lon_0=180,
            projection='merc',
            resolution = 'h',
            area_thresh=10000.,
            )
m.drawcoastlines()


# read in database
f = h5py.File('hydro2kv0_5_2.mat','r')
dataset = f.get('ch2k/')
coralnames = list(dataset.keys()) #last key here is 'version'? 

# loop through record ID's 

data_dict = {}
for recordID in coralnames:
	try:

		# eventually need to make the calculation of a2 its own function with recordID as a parameter
		# right now it just interfaces with command line to call the script and keeps chugging if there are errors

		print("Current ID:")
		print('\n', recordID)

		current_record = f.get('ch2k/'+recordID+'/d18O')
		result = subprocess.check_output(['python3', 'regression:testing_version.py', '-i', recordID])
		
		
		split_results = str(result).split('\\n')
		
		a2 = float(split_results[len(split_results)-2])

		print("a2 is...")
		print('\n', a2)
		latc = np.array(f.get('ch2k/'+recordID+'/lat'))
		lonc = np.array(f.get('ch2k/'+recordID+'/lon')) 

		data_dict[recordID] = (latc, lonc, a2)
		
		#a2_plot = map.scatter(latc, lonc, c=a2, s=30, cmap=plt.cm.jet)

		# random thoughts
		# if this returns an array, do we average them? 
		# otherwise could feed this to scatterplot directly?

		# maybe use a dictionary comprehension with recordID as key and lat, lon tuple as value
		# populate this dictionary as we loop
		# maybe better to just output a three tuple of (lat, lon, a2)
	except: 
		print("Data is Nonetype, ignoring...")


plt.show()
print(data_dict)



'''
# mapping part
# assuming we are fed a dictionary of the form recordID: (lat, lon, a2)
# maybe put this in above for loop instead? 

# **** currently just a scatterplot of random values over the Pacific ***** #

lat_min = -40
lat_max = 40
lon_min = 90
lon_max = 300


'''



'''
x = np.empty(1000)
y = np.empty(1000)
for i in range(1000):
	x[i] = np.random.randint(-300, 300)
	y[i] = np.random.randint(-180, 180)

x, y = m(list(x), list(y)'''

#plt.scatter(x,y,z,marker='o',color='Red', zorder = 3)
plt.show()



            
