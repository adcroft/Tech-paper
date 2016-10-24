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
	
	#General Flags
	rotated=True

	#What to plot?
	plot_three_horizontal_field=True
	plot_bt_stream_comparison=False
	plot_temperature_cross_section=False

	#Defining path
	Shelf_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Shelf/Melt_on_without_decay_with_spreading_trimmed_shelf/'
	Berg_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/Melt_on_without_decay_with_spreading_trimmed_shelf/'

	#Geometry files
	ocean_geometry_filename=Shelf_path +'ocean_geometry.nc'
	ice_geometry_filename=Shelf_path+'/MOM_Shelf_IC.nc'
	ISOMIP_IC_filename=Shelf_path+'ISOMIP_IC.nc'
	
	#Berg files
	Berg_ocean_file=Berg_path+'00010101.ocean_month.nc'
	Berg_iceberg_file=Berg_path+'00010101.icebergs_month.nc'


	#Load static fields
	(depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated)	
	grounding_line=find_grounding_line(depth, shelf_area, ice_base, x,y, xvec, yvec)
	
	#Defining figure characteristics
	#fig=plt.figure(figsize=(15,10),facecolor='grey')
	#fig = plt.figure(facecolor='black')
	#ax = fig.add_subplot(111,axisbg='gray')

	######################################################################################################################
	################################  Plotting melt comparison  ##########################################################
	######################################################################################################################
	
	if plot_three_horizontal_field is True:
		#Plotting flags
		plot_vel_vel_sst=False
		plot_depth_spread_mass_mass=True

		plot_flag=''
		fig=plt.figure(figsize=(15,10))
		flipped=False
		time_slice='' 
		slice_num=1
		time_slice='mean'
		cmap='jet'
		N=3

		if plot_vel_vel_sst is True:
			field_list=np.array(['vo','uo','sst'])
			title_list=np.array(['Meridional surface velocity','Zonal surface velocity','SST'])
			filename_list=np.array([Berg_iceberg_file,Berg_iceberg_file,Berg_iceberg_file])
			vmin_list=np.array([-0.01, -0.01, -1.9])
			vmax_list=np.array([0.01, 0.01, -0.5])
			cmap_list=np.array(['bwr', 'bwr', 'jet'])
			scale_list=np.array([1,1,1])

		if plot_depth_spread_mass_mass is True:
			plotting_flag='depth_spread_mass_mass'
			field_list=np.array(['D','spread_mass','mass'])
			title_list=np.array(['Ocean Bottom','Ice Shelf Draft', 'Ice Shelf Draft (no spreading)'])
			mask_grounded=np.array([False, False, False])
			filename_list=np.array([ocean_geometry_filename,Berg_iceberg_file,Berg_iceberg_file])
			vmin_list=np.array([0, 0.0,0.0 ])
			vmax_list=np.array([1000, 1000, 1000])
			cmap_list=np.array(['jet', 'jet', 'jet'])
			rho_ice=918.0 ; rho_sw=1025.0;
			scale_data=True   ; 
			#scale_list=np.array([1,1/rho_ice,1/rho_ice])
			scale_list=np.array([1,1/rho_sw,1/rho_sw])

		for k in range(3):
			field=field_list[k]
			vmin=vmin_list[k]
			vmax=vmax_list[k]
			cmap=cmap_list[k]
			plot_flag=plot_flag+'_'+ field
			filename=filename_list[k]

			plt.subplot(1,N,k+1)
			data=load_and_compress_data(filename,field=field,time_slice=time_slice,time_slice_num=slice_num,rotated=rotated) 
			plt.plot(xvec,grounding_line, linewidth=3.0,color='black')
			#data1=mask_ocean(data1,shelf_area)
			if mask_grounded[k] is True:
				data=mask_grounded_ice(data,depth,ice_base)
			if scale_data is True:
				data=data*scale_list[k]
			plot_data_field(data,x,y,'',vmin,vmax,flipped,colorbar=True,cmap=cmap,title=title_list[k],xlabel='x (km)',ylabel='')	




	######################data###############################################################################################
	################################  Plotting Bt stream function ########################################################
	######################################################################################################################

	if plot_bt_stream_comparison is True:
		fig=plt.figure(figsize=(5,10))
		vmin=-1*(10**3)  ; vmax=1*(10**3)
		flipped=False
		field='barotropic_sf'
		plot_flag='field'

		data1=calculate_barotropic_streamfunction(Berg_ocean_file,depth,ice_base,time_slice='mean',time_slice_num=-1,rotated=rotated)
		plot_data_field(data1,x,y,'',vmin,vmax,flipped,colorbar=True,cmap='jet',title='Bergs',xlabel='x (km)',ylabel='y (km)')	


	######################data###############################################################################################
	################################  Plotting Cross Section        ########################################################
	######################################################################################################################


	if plot_temperature_cross_section is True:
		fig=plt.figure(figsize=(10,5))
		plot_anomaly=True
		time_slice=''
		#vertical_coordinate='z'
		vertical_coordinate='layers'  #'z'
		field='temp'  ; vmin=-2.0  ; vmax=1.0  ;vdiff=0.1   ; vanom=0.3
		#field='salt'  ; vmin=34  ; vmax=34.7  ;vdiff=0.05  ; vanom=0.05
		plot_flag='field'
		filename=Berg_ocean_file
		
		if vertical_coordinate=='z':
			filename=filename.split('.nc')[0] + '_z.nc'
		


		data=load_and_compress_data(filename,field , time_slice, time_slice_num=-1, direction='yz' ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		data[np.where(data==0.)]=np.nan
		elevation = get_vertical_dimentions(filename,vertical_coordinate, time_slice, time_slice_num=-1, direction='yz' ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		(y ,z ,data) =interpolated_onto_vertical_grid(data, elevation, yvec, vertical_coordinate)

		if plot_anomaly is True:
			data0=load_and_compress_data(filename,field , time_slice=None, time_slice_num=0, direction='yz' ,dir_slice=None, dir_slice_num=20,rotated=rotated)
			elevation0 = get_vertical_dimentions(filename,vertical_coordinate, time_slice=None, time_slice_num=-1, direction='yz' ,dir_slice=None, dir_slice_num=20,rotated=rotated)
			(y0 ,z0 ,data0) =interpolated_onto_vertical_grid(data0, elevation0, yvec, vertical_coordinate)
			data=data-data0
			vmin=-vanom  ; vmax=vanom

		plot_data_field(data, y, z, '', vmin, vmax, flipped=False, colorbar=True, cmap='jet')
		
		#For plotting purposes
		field=field+'_'+ vertical_coordinate




	plt.tight_layout()


	if save_figure==True:
		output_file='Figures/static_shelf_solo' + plot_flag + '.png'
		plt.savefig(output_file,dpi=300,bbox_inches='tight')
		print 'Saving ' ,output_file

	#fig.set_size_inches(9,4.5)
	plt.show()
	print 'Script complete'



if __name__ == '__main__':
	main()
	#sys.exit(main())














