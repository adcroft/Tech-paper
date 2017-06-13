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

	#General flags
	parser.add_argument('-rotated', type='bool', default=True,
		                        help=''' Rotates the figures so that latitude runs in the vertical (involves switching x and y. ''')
	parser.add_argument('-use_ALE', type='bool', default=True,
		                        help='''When true, it uses the results of the ALE simulations. When false, layed simulations are used.    ''')
	
	#What to plot?
	parser.add_argument('-fields_to_compare', type=str, default='plot_bt_stream_comparison',
		                        help=''' This flag determine whether to plot horizontal, vertical, or special case for barotropic stream function
					(should remove this later). Options are plot_bt_stream_comparison, plot_melt_comparison, plot_cross_section   ''')

	#Which file type to use
	parser.add_argument('-use_title_on_figure', type='bool', default=False,
		                        help=''' When true, figure the name of the field is written as title ''')

	parser.add_argument('-only_shelf_ALE', type='bool', default=False,
		                        help=''' When true, Figures compares Shelf ALE and Layer (must be used with use_ALE=True ''')


	parser.add_argument('-ylim_min', type=float, default=0.0,
		                        help='''Minimum y used for plotting (only applies to horizontal sections)''')

	parser.add_argument('-ylim_max', type=float, default=480.0,
		                        help='''Minimum y used for plotting (only applies to horizontal sections)''')


	optCmdLineArgs = parser.parse_args()
	return optCmdLineArgs


##########################################################  Main Program   #########################################################################
####################################################################################################################################################

def main(args):

	#Plotting flats
	save_figure=args.save_figure
	
	#General flags
	rotated=args.rotated
	use_ALE=args.use_ALE

	#What to plot?
	fields_to_compare=args.fields_to_compare


	#Defining path
	Path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/'
	Folder_name= 'Melt_on_without_decay_with_spreading_trimmed_shelf/' 
	Shelf_path=Path+'Shelf/' + Folder_name
	Berg_path=Path+'Bergs/' + Folder_name


	#Geometry files  (from non ALE version)
	ocean_geometry_filename=Shelf_path +'ocean_geometry.nc'
	ice_geometry_filename=Shelf_path+'/MOM_Shelf_IC.nc'
	ISOMIP_IC_filename=Shelf_path+'ISOMIP_IC.nc'
	
	#only used when comparing ALE and Layer
	Layer_Shelf_path=Path+'Shelf/' + Folder_name

	#Using ALE ice shelf
	use_ALE_flag=''
	if use_ALE is True:
		
		use_ALE_flag='ALE_z_'
		Folder_name='ALE_z_' +Folder_name
		Shelf_path=Path+'Shelf/' + Folder_name
		Berg_path=Path+'Bergs/' + Folder_name

		#Comparing Layer vs ALE
		if args.only_shelf_ALE is True:
			Berg_path=Layer_Shelf_path 
	
	print Shelf_path
	print Berg_path

	#Shelf files
	Shelf_ocean_file=Shelf_path+'00010101.ocean_month.nc'
	Shelf_ice_file=Shelf_path+'00010101.ice_month.nc'

	#Berg files
	Berg_ocean_file=Berg_path+'00010101.ocean_month.nc'
	Berg_ice_file=Berg_path+'00010101.ice_month.nc'
	Berg_iceberg_file=Berg_path+'00010101.icebergs_month.nc'

	#Other
	letter_labels=np.array(['(a)','(b)','(c)','(d)','(e)'])

	#Load static fields
	(depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated)	
      	grounding_line=find_grounding_line(depth, shelf_area, ice_base, x,y, xvec, yvec)
	ice_front=find_ice_front(depth,shelf_area,x,y, xvec, yvec)

	
	#Defining figure characteristics
	fig=plt.figure(figsize=(15,10),facecolor='white')
	#fig = plt.figure(facecolor='black')
	ax = fig.add_subplot(111,axisbg='white')

	#Scale
	ylim_min=args.ylim_min
	ylim_max=args.ylim_max

	######################################################################################################################
	################################  Plotting melt comparison  ##########################################################
	######################################################################################################################
	
	if fields_to_compare=='plot_melt_comparison':
		vmin=0.0  ; vmax=3.0  ; vdiff=3.0
		if use_ALE is True:
			vmin=0.0  ; vmax=7.0  ; vdiff=8.0
		flipped=False
		field='melt'
		time_slice='all'
		#field1='melt'
		#field2='melt_m_per_year'
		field1='UO'
		field2='UO'

		#data1=load_and_compress_data(Shelf_ocean_file,field=field1,time_slice=time_slice,time_slice_num=-1,rotated=rotated)
		#data2=load_and_compress_data(Berg_iceberg_file,field=field2,time_slice=time_slice,time_slice_num=-1,rotated=rotated) #field name is difference since it comes from a diff file.
		data1=load_and_compress_data(Shelf_ice_file,field=field1,time_slice=time_slice,time_slice_num=-1,rotated=rotated)
		data2=load_and_compress_data(Berg_ice_file,field=field2,time_slice=time_slice,time_slice_num=-1,rotated=rotated) #field name is difference since it comes from a diff file.

		#Masking out ocean and grounded ice
		mask_open_ocean=True
		if mask_open_ocean == True:
			mask_ocean_using_bergs=True
			if mask_ocean_using_bergs is True:
				ice_data=load_and_compress_data(Berg_iceberg_file,field='spread_area',time_slice=time_slice,time_slice_num=1,\
							rotated=rotated,direction='xy',dir_slice=None, dir_slice_num=1)
				data1=mask_ocean(data1,ice_data)
				data2=mask_ocean(data2,ice_data)
			else:
				data1=mask_ocean(data1,ice_base)
				data2=mask_ocean(data2,ice_base)
		mask_grounded=True
		if mask_grounded == True:
			data1=mask_grounded_ice(data1,depth,ice_base)
			data2=mask_grounded_ice(data2,depth,ice_base)
		print data1.shape
		T1=np.squeeze(data1[:,130,20])
		T2=np.squeeze(data2[:,130,20])
		#T1=np.mean(np.mean(data1,axis=2),axis=1)
		#T2=np.mean(np.mean(data2,axis=2),axis=1)

		#data1=mask_ocean(data1,shelf_area)
		#data2=mask_ocean(data2,shelf_area)
		#data3=mask_ocean(data1-data2,shelf_area)
		plt.plot(T1,color='r')
		plt.plot(T2, color='b')
		plt.show()

		


		#Plotting data




	plt.tight_layout()


	if save_figure==True:
		output_file='Figures/'+ use_ALE_flag +'static_shelf_comparison_' + field + '.png'
		plt.savefig(output_file,dpi=300,bbox_inches='tight')
		print 'Saving ' ,output_file

	#fig.set_size_inches(9,4.5)
	plt.show()
	print 'Script complete'



if __name__ == '__main__':
	optCmdLineArgs= parseCommandLine()
	main(optCmdLineArgs)

	#sys.exit(main())














