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

def parseCommandLine():
        """
        Parse the command line positional and optional arguments.
        This is the highest level procedure invoked from the very end of the script.
        """

        parser = argparse.ArgumentParser(description=
        '''
        Plot snapshots of either surface or vertical sections, for Tech paper.
        ''',
        epilog='Written by Alon Stern, Dec. 2016.')
	
	#Adding an extra boolian type argument
        parser.register('type','bool',str2bool) # add type keyword to registries



	#Adding arguments:

	#Saving figure
	parser.add_argument('-save_figure', type='bool', default=False,
		                        help=''' When true, the figure produced by the script is saved''')
	
	#Plotting flags
	parser.add_argument('-ylim_min', type=float, default=240.0,
		                        help='''Minimum y used for plotting (only applies to horizontal sections)''')

	parser.add_argument('-ylim_max', type=float, default=440.0,
		                        help='''Minimum y used for plotting (only applies to horizontal sections)''')
	
	optCmdLineArgs = parser.parse_args()
	return optCmdLineArgs

def main(args):


	#Plotting flags
	save_figure=args.save_figure

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
	IcebergMelt_file = path + 'ALE_z_Berg_melt_After_melt_Collapse_diag_Strong_Wind/' + extension
	MixedMelt_file = path + 'ALE_z_Mixed_Melt_After_melt_Collapse_diag_Strong_Wind/' + extension

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
		ylo=args.ylim_min
		yhi=args.ylim_max
		for k in range(3):
			ax=plt.subplot(1,3,k+1)
			data=load_and_compress_data(filename_list[k],field=field,time_slice=time_slice,time_slice_num=time_slice_num,rotated=rotated)

			mask_open_ocean = True
			if mask_open_ocean == True:
				ice_data=load_and_compress_data(filename_list[k],field='spread_area' ,time_slice=time_slice,time_slice_num=time_slice_num,rotated=rotated)
				data=mask_ocean(data,ice_data)
				plot_data_field((ice_base*0),x,y,-5.0, 10.0,flipped,colorbar=False,cmap='Greys',title=title,xlabel='x (km)',ylabel='',ylim_min=ylo,\
						ylim_max=yhi,return_handle=False)


			datamap=plot_data_field(data,x,y,vmin,vmax,flipped,colorbar=False ,cmap='jet',\
					xlabel='x (km)',ylabel='', ylim_min=ylo,ylim_max=yhi, return_handle=True)	
					#title=Title_list[k],xlabel='x (km)',ylabel='', ylim_min=ylo,ylim_max=yhi, return_handle=True)	
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
	optCmdLineArgs= parseCommandLine()
	main(optCmdLineArgs)
	#sys.exit(main())














