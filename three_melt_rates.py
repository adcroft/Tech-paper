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
	letter_labels=np.array(['(a)','(b)','(c)'])


	#Defining path
	path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/'
	extension='00060101.icebergs_month.nc'
	IceShelfMelt_file = path +  'ALE_z_After_melt_Collapse_diag_Strong_Wind/' +extension
	IcebergMelt_file = path + 'Berg_melt_ALE_z_After_melt_Collapse_diag_Strong_Wind/' + extension
	MixedMelt_file = path + 'Mixed_Melt_ALE_z_After_melt_Collapse_diag_Strong_Wind/' + extension

	#Geometry files
	Shelf_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Shelf/Melt_on_without_decay_with_spreading_trimmed_shelf/' 
	ocean_geometry_filename=Shelf_path +'ocean_geometry.nc'
	ice_geometry_filename=Shelf_path+'/MOM_Shelf_IC.nc'
	ISOMIP_IC_filename=Shelf_path+'ISOMIP_IC.nc'
	
	#Load static fields
	(depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated)	
      	grounding_line=find_grounding_line(depth, shelf_area, ice_base, x,y, xvec, yvec)
	#ice_front=find_ice_front(depth,shelf_area,x,y, xvec, yvec)


	
	#Defining figure characteristics
	fig, axes = plt.subplots(nrows=1,ncols=3)
	fig.set_size_inches(15.0,10.0, forward=True)

	######################################################################################################################
	################################  Plotting Horizontal field  ##########################################################
	######################################################################################################################
	
	if plot_horizontal_field is True:
		cmap='jet'
		time_slice=''
		time_slice_num=40
		flipped=False
		#field='uo' ; vmin=-0.03  ; vmax=0.03 ; cmap='bwr'; time_slice='mean'
		#field='vo' ; vmin=-0.03  ; vmax=0.03 ; cmap='bwr'; time_slice='mean'

		#field='sst' ;  vmin=0.0  ; vmax=3.0
		#field='spread_area' ; vmin=0.0  ; vmax=3.0
		field='spread_mass' ; vmin=0.0  ; vmax=500000.
		field='melt_m_per_year' ; vmin=0.0  ; vmax=15.
		#field='spread_uvel' ; vmin=-2.0  ; vmax=1.0


		#data1=mask_ocean(data1,shelf_area)
		#data2=mask_ocean(data2,shelf_area)
		#data3=mask_ocean(data3,shelf_area)
		filename_list=np.array([IceShelfMelt_file,IcebergMelt_file,MixedMelt_file])
		Title_list=np.array(['Ice Shelf Melt','Iceberg Melt','Mixed Melt'])
		#e_list=np.array([])
		ylo=80.
		yhi=450.
		for k in range(3):
			ax=plt.subplot(1,3,k+1)
			data=load_and_compress_data(filename_list[k],field=field,time_slice=time_slice,time_slice_num=time_slice_num,rotated=rotated)
			datamap=plot_data_field(data,x,y,vmin,vmax,flipped,colorbar=False ,cmap='jet',\
					title=Title_list[k],xlabel='x (km)',ylabel='', ylim_min=ylo,ylim_max=yhi, return_handle=True)	
			#plt.plot(xvec,grounding_line, linewidth=3.0,color='black')
			text(1,1,letter_labels[k], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)

			if k==0:
				plt.ylabel('y (km)',fontsize=20)
			if k>0:
				ax.set_yticks([])
		#Creating colorbar
		fig.subplots_adjust(right=0.85)
		cbar_ax = fig.add_axes([0.88,0.12 , 0.025, 0.75])
		cbar=fig.colorbar(datamap, cax=cbar_ax)
		cbar.set_label('(m/year)', rotation=90,fontsize=20)
		cbar.ax.tick_params(labelsize=20)


	#plt.tight_layout()
	if save_figure==True:
		output_file='Figures/three_melt_' + field + '.png'
		plt.savefig(output_file,dpi=300,bbox_inches='tight')
		print 'Saving ' ,output_file

	#fig.set_size_inches(9,4.5)
	plt.show()
	print 'Script complete'



if __name__ == '__main__':
	main()
	#sys.exit(main())














