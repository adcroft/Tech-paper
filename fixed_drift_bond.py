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


	#Plotting flags
	save_figure=True

	#General flags
	rotated=True

	#What to plot?
	plot_horizontal_field=True
	plot_temperature_cross_section=False


	if (plot_horizontal_field is False) and  (plot_temperature_cross_section is False):
		print 'You must select an option'
		return


	#Defining path

	Fixed_berg_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/completed/fixed_speed_Moving_berg_trimmed_shelf/'
	Drift_berg_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/completed/drifting_Moving_berg_trimmed_shelf/'
	Bond_berg_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/completed/Bond_drifting_Moving_berg_trimmed_shelf/'

	#Geometry files
	ocean_geometry_filename=Fixed_berg_path +'ocean_geometry.nc'
	ice_geometry_filename=Fixed_berg_path+'MOM_Shelf_IC.nc'
	ISOMIP_IC_filename=Fixed_berg_path+'ISOMIP_IC.nc'
	
	#Berg files
	Fixed_ocean_file=Fixed_berg_path+'00060101.ocean_month.nc'
	Drift_ocean_file=Drift_berg_path+'00060101.ocean_month.nc'
	Bond_ocean_file=Bond_berg_path+'00060101.ocean_month.nc'
	Fixed_prog_file=Fixed_berg_path+'00060101.prog.nc'
	Drift_prog_file=Drift_berg_path+'00060101.prog.nc'
	Bond_prog_file=Bond_berg_path+'00060101.prog.nc'
	Fixed_iceberg_file=Fixed_berg_path+'00060101.icebergs_month.nc'
	Drift_iceberg_file=Drift_berg_path+'00060101.icebergs_month.nc'
	Bond_iceberg_file=Bond_berg_path+'00060101.icebergs_month.nc'


	#Load static fields
	(depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated=rotated)	
	grounding_line=find_grounding_line(depth, shelf_area, ice_base, x,y, xvec, yvec)

	
	#Defining figure characteristics
	fig=plt.figure(figsize=(15,8))
	#fig=plt.figure(figsize=(15,10),facecolor='grey')
	#fig = plt.figure(facecolor='black')
	#ax = fig.add_subplot(111,axisbg='gray')

	######################################################################################################################
	################################  Plotting Horizontal field  ##########################################################
	######################################################################################################################
	
	if plot_horizontal_field is True:
		cmap='jet'
		time_slice=''
		flipped=False
		#field='uo' ; vmin=-0.03  ; vmax=0.03 ; cmap='bwr'; time_slice='mean'
		#field='vo' ; vmin=-0.03  ; vmax=0.03 ; cmap='bwr'; time_slice='mean'

		#field='sst' ;  vmin=0.0  ; vmax=3.0
		#field='spread_area' ; vmin=0.0  ; vmax=3.0
		field='spread_mass' ; vmin=0.0  ; vmax=500000.
		#field='spread_uvel' ; vmin=-2.0  ; vmax=1.0

		data1=load_and_compress_data(Fixed_iceberg_file,field=field,time_slice=time_slice,time_slice_num=-1,rotated=rotated)
		data2=load_and_compress_data(Drift_iceberg_file,field=field,time_slice=time_slice,time_slice_num=-1,rotated=rotated) 
		data3=load_and_compress_data(Bond_iceberg_file,field=field,time_slice=time_slice,time_slice_num=-1,rotated=rotated) 

		#data1=mask_ocean(data1,shelf_area)
		#data2=mask_ocean(data2,shelf_area)
		#data3=mask_ocean(data3,shelf_area)
		plt.subplot(1,3,1)
		plot_data_field(data1,x,y,vmin,vmax,flipped,colorbar=True,cmap='jet',title='Fixed',xlabel='x (km)',ylabel='y (km)', ylim_min=500.,ylim_max=750.)	
		plt.plot(xvec,grounding_line, linewidth=3.0,color='black')
		plt.subplot(1,3,2)
		plot_data_field(data2,x,y,vmin,vmax,flipped,colorbar=True,cmap='jet',title='Drift',xlabel='x (km)',ylabel='y (km)', ylim_min=500.,ylim_max=750.)	
		plt.plot(xvec,grounding_line, linewidth=3.0,color='black')
		plt.subplot(1,3,3)
		plot_data_field(data3,x,y,vmin,vmax,flipped,colorbar=True,cmap='jet',title='Bonds',xlabel='x (km)',ylabel='y (km)', ylim_min=500.,ylim_max=750.)	
		plt.plot(xvec,grounding_line, linewidth=3.0,color='black')


	######################data###############################################################################################
	################################   Plotting cross section        ########################################################
	#########################################################################################################################


	if plot_temperature_cross_section is True:

		plot_anomaly=False
		time_slice=''
		#vertical_coordinate='z'
		vertical_coordinate='layers'  #'z'
		#field='temp'  ; vmin=-2.0  ; vmax=1.0  ;vdiff=0.1   ; vanom=0.3
		field='salt'  ; vmin=34  ; vmax=34.7  ;vdiff=0.05  ; vanom=0.05
		cmap='jet'
		filename1=Fixed_ocean_file
		filename2=Drift_ocean_file
		filename3=Bond_ocean_file
		
		if rotated is True:
			direction='yz'
			dist=yvec
		else:
			direction='xz'
			dist=xvec

		
		if vertical_coordinate=='z':
			filename1=filename1.split('.nc')[0] + '_z.nc'
			filename2=filename2.split('.nc')[0] + '_z.nc'
		


		data1=load_and_compress_data(filename1,field , time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		elevation1 = get_vertical_dimentions(filename1,vertical_coordinate, time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		(y1 ,z1 ,data1) =interpolated_onto_vertical_grid(data1, elevation1, dist, vertical_coordinate)

		data2=load_and_compress_data(filename2,field , time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		elevation2 = get_vertical_dimentions(filename2,vertical_coordinate, time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		(y2 ,z2 ,data2) =interpolated_onto_vertical_grid(data2, elevation2, dist, vertical_coordinate)
		
		data3=load_and_compress_data(filename3,field , time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		elevation3 = get_vertical_dimentions(filename3,vertical_coordinate, time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		(y3 ,z3 ,data3) =interpolated_onto_vertical_grid(data3, elevation3, dist, vertical_coordinate)

		if plot_anomaly is True:
			data0=load_and_compress_data(filename1,field , time_slice=None, time_slice_num=0, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
			elevation0 = get_vertical_dimentions(filename1,vertical_coordinate, time_slice=None, time_slice_num=-1, \
					direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
			(y0 ,z0 ,data0) =interpolated_onto_vertical_grid(data0, elevation0, dist, vertical_coordinate)
			data1=data1-data0
			data2=data2-data0
			data3=data3-data0
			vmin=-vanom  ; vmax=vanom
			cmap='jet'


		plt.subplot(3,1,1)
		plot_data_field(data1, y1, z1, vmin, vmax, flipped=False, colorbar=True, cmap=cmap)
		plt.subplot(3,1,2)
		plot_data_field(data2, y2, z2, vmin, vmax, flipped=False, colorbar=True, cmap=cmap)
		plt.subplot(3,1,3)
		plot_data_field(data3, y3, z3, vmin, vmax, flipped=False, colorbar=True, cmap=cmap)
		
		#For plotting purposes
		field=field+'_'+ vertical_coordinate




	plt.tight_layout()


	if save_figure==True:
		output_file='Figures/fixed_drift_bond_' + field + '.png'
		plt.savefig(output_file,dpi=300,bbox_inches='tight')
		print 'Saving ' ,output_file

	#fig.set_size_inches(9,4.5)
	plt.show()
	print 'Script complete'



if __name__ == '__main__':
	main()
	#sys.exit(main())














