#!/usr/bin/env python

#First import the netcdf4 library
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import numpy as np  # http://code.google.com/p/netcdf4-python/
import matplotlib
import math
import os
matplotlib.use("GTKAgg")
from pylab import *
#import matplotlib.pyplot as plt
import pdb
import netCDF4 as nc
import sys
import argparse
from m6toolbox  import section2quadmesh
from static_shelf_comparison import *

##########################################################  Main Program   #########################################################################
####################################################################################################################################################

def main():
	parser = argparse.ArgumentParser()
	args = parser.parse_args()


	#Plotting flats
	save_figure=True

	#General flags
	rotated=True

	#What to plot?
        plot_horizontal_field=False
        plot_cross_section=True

	#Which simulation to use
	#simulation='high_melt'
	simulation='fixed_01'

	#Defining path
	Geometry_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/Melt_on_high_melt_with_decay/'
	if simulation=='high_melt':
		Berg_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/Melt_on_high_melt_with_decay/'
		Berg_path_init='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/Melt_on_high_melt_with_decay_initialize/'
	elif simulation=='fixed_01':
		Berg_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/fixed_speed_Moving_berg_trimmed_shelf_from_zero_small_step_u01/'
		Berg_path_init='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/fixed_speed_Moving_berg_trimmed_shelf_from_zero_small_step_u01/'
	else:
		return

	#Geometry files
	ocean_geometry_filename=Geometry_path +'ocean_geometry.nc'
	ice_geometry_filename=Geometry_path+'/MOM_Shelf_IC.nc'
	ISOMIP_IC_filename=Geometry_path+'ISOMIP_IC.nc'
	
	#Berg files
	#Berg_ocean_file_init=Berg_path_init+'00010101.ocean_month.nc'
	if simulation=='high_melt':
		Berg_ocean_file1=Berg_path+'00010101.ocean_month.nc'
		Berg_ocean_file2=Berg_path+'00060101.ocean_month.nc'
		Berg_ocean_file2=Berg_path+'00060101.ocean_month.nc'
	if simulation=='fixed_01':
		extension='.ocean_month.nc'
		#extension='.prog.nc'
		#extension='.icebergs_month.nc'
		#extension='.ocean_month_z.nc'
		Berg_ocean_file1=Berg_path+'00010101' 
		Berg_ocean_file1=Berg_path+'00010107' 
		Berg_ocean_file2=Berg_path+'00010107' 
		Berg_ocean_file3=Berg_path+'00010206' 

	Berg_ocean_file_list=np.array([Berg_ocean_file1 +extension ,Berg_ocean_file2+ extension ,Berg_ocean_file3 + extension])
	Iceberg_file_list=np.array([Berg_ocean_file1 +'.icebergs_month.nc' ,Berg_ocean_file2+ '.icebergs_month.nc' ,Berg_ocean_file3 + '.icebergs_month.nc'])
	
	Berg_icebergs_file=Berg_path+'00010101.icebergs_month.nc'

	#Berg_ocean_file='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/Melt_on_with_decay/'+ '00010101.ocean_month.nc'

	#Load static fields
	(depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated=rotated)	
	
	#Defining figure characteristics
	#fig=plt.figure(figsize=(10,10),facecolor='grey')
	#fig = plt.figure(facecolor='black')
	#ax = fig.add_subplot(111,axisbg='gray')
	time_slice_num=np.array([9, 24, 24])  #When using prog 
	#time_slice_num=np.array([220,580,580])



	######################################################################################################################
	################################  Plotting melt comparison  ##########################################################
        ######################################################################################################################
        
        if plot_horizontal_field is True:
		fig=plt.figure(figsize=(15,10),facecolor='grey')
		ylim_min=550.
		ylim_max=750.
		for n in range(3):
			flipped=False
			field='spread_area'  ;vmin=0.0  ; vmax=3.0
			field='temp'  ;vmin=-2.0  ; vmax=-1.5
			field='sst'  ;vmin=-1.8  ; vmax=-1.5
			#field='spread_uvel' ;vmin=-0.01  ; vmax=0.01
			filename=Berg_ocean_file_list[n]

			(data1,time)=load_and_compress_data(filename,field=field,time_slice='',time_slice_num=time_slice_num[n],rotated=rotated,direction='xy',dir_slice=None, dir_slice_num=1, \
					return_time=True)
			time_str=str(int(np.round(time)))

			
			mask_using_bergs=True
			if mask_using_bergs is True:
				iceberg_filename=Iceberg_file_list[n]
				print iceberg_filename
				ice_data=load_and_compress_data(iceberg_filename,field='spread_area',time_slice='',time_slice_num=time_slice_num[n],\
						rotated=rotated,direction='xy',dir_slice=None, dir_slice_num=1)
				data1=mask_ice(data1,ice_data)
			#data1=mask_ocean(data1,shelf_area)
			plt.subplot(1,3,n+1)
			plot_data_field(data1,x,y,vmin,vmax,flipped,colorbar=True,cmap='jet',title='Time = '+ time_str + ' days',xlabel='x (km)',ylabel='y (km)',ylim_min=ylim_min, ylim_max=ylim_max)  


	####################################################################################################################################################
	####################################################  Cross Section     ############################################################################
	###################################################################################################################################################
	if plot_cross_section is True:
		fig=plt.figure(figsize=(10,10),facecolor='grey')
		ax = fig.add_subplot(111,axisbg='gray')
		if rotated is True:
			direction='yz'
			dist=yvec
		else:
			direction='xz'
			dist=xvec

		for n in range(3):
			plot_anomaly=False
			vertical_coordinate='layers'  #'z'
			#vertical_coordinate='z'
			time_slice=None
			#field='u'  ; vmin=-0.1  ; vmax=0.1    ; vanom=0.3 ; cmap='seismic'
			#field='v'  ; vmin=-0.1  ; vmax=0.1    ; vanom=0.3 ; cmap='seismic'
			field='temp'  ; vmin=-2.0  ; vmax=1.0 ; vanom=0.3 ; cmap='jet'
			#field='salt'  ; vmin=34  ; vmax=34.7  ;vdiff=0.05  ; vanom=0.05 ; cmap='jet'
			
			filename=Berg_ocean_file_list[n]
			if vertical_coordinate=='z':
				filename=filename.split('.nc')[0] + '_z.nc'

			print filename

			(data1,time)=load_and_compress_data(filename,field , time_slice, time_slice_num=time_slice_num[n], direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated,\
					return_time=True)
			time_str=str(int(np.round(time)))
			elevation1 = get_vertical_dimentions(filename,vertical_coordinate, time_slice, time_slice_num=time_slice_num[n],\
					direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
			(y1 ,z1 ,data1) =interpolated_onto_vertical_grid(data1, elevation1, dist, vertical_coordinate)

			if plot_anomaly is True:
				data0=load_and_compress_data(filename,field , time_slice=None, time_slice_num=0, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
				elevation0 = get_vertical_dimentions(filename,vertical_coordinate, time_slice=None, time_slice_num=-1, \
						direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
				(y0 ,z0 ,data0) =interpolated_onto_vertical_grid(data0, elevation0, dist, vertical_coordinate)
				data1=data1-data0
				vmin=-vanom  ; vmax=vanom
			

			plt.subplot(3,1,n+1)
			plot_data_field(data1, y1, z1, vmin, vmax, flipped=False, colorbar=True, cmap=cmap,title='Time = '+ time_str + ' days')
			xmin=450.  ;xmax=750.
			plt.xlim([xmin,xmax])
			
			surface=np.squeeze(elevation1[0,:])
			plot(dist, surface,'black')
			#data1_tmp=data1
			#tol=0.000000000001
			#data1[np.where(abs(data1)<tol)]=10000.
			#data1[np.where(abs(data1)>tol)]=-10000.
			#masked_array = np.ma.array (data1, mask=np.isnan(data1))
			#levels1=np.array([0.05,999.])
			#cNorm2=mpl.colors.Normalize(vmin=400, vmax=1000)
			#print y1.shape,z1.shape
			#CS = contourf(y1, z1[, data1_tmp,levels=levels1, hatches=[' '], fill=False,cmap='Greys',norm=cNorm2 )
			#print data1
		
		#For plotting purposes
		field=field+'_'+ vertical_coordinate
		if plot_anomaly is True:
			field=field+'_anomaly'





	plt.tight_layout()


	if save_figure==True:
		output_file='Figures/snapshots_'+simulation +'_'+ field + '.png'
		plt.savefig(output_file,dpi=300,bbox_inches='tight')
		print 'Saving ' ,output_file
		#print 'Saving file not working yet'

	#fig.set_size_inches(9,4.5)
	plt.show()
	print 'Script complete'



if __name__ == '__main__':
	main()
	#sys.exit(main())














