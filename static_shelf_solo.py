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

	#General flags
	parser.add_argument('-rotated', type='bool', default=True,
		                        help=''' Rotates the figures so that latitude runs in the vertical (involves switching x and y. ''')
	parser.add_argument('-use_ALE', type='bool', default=True,
		                        help='''When true, it uses the results of the ALE simulations. When false, layed simulations are used.    ''')
	
	#What to plot?
	parser.add_argument('-plot_horizontal_field', type='bool', default=True,
		                        help='''    ''')

	#Which file type to use
	parser.add_argument('-broken_shelf', type='bool', default=False,
		                        help=''' When true, figure plots data after ice shelf breakoff ''')
	
	parser.add_argument('-use_Mixed_Melt', type='bool', default=False,
		                        help=''' When true, figure plots using Mixed_melt_data ''')
	
	parser.add_argument('-three_fields_flag', type=str, default='plot_melt_sst_ustar_berg',
		                        help='''String determines which set three fields to plot. Options are plot_vel_vel_sst, plot_depth_spread_mass_mass
					plot_melt_sst_ustar_berg.  Other values will plot the 1 field given in the script''')
	
	parser.add_argument('-use_title_on_figure', type='bool', default=False,
		                        help=''' When true, figure the name of the field is written as title ''')

	parser.add_argument('-just_one_colorbar', type='bool', default=False,
		                        help=''' When true, a color bar is not plotted for each panel, and one colorbar is plotted for all three. ''')
	
	#Plotting parameters
	parser.add_argument('-cmap', type=str, default='jet',
		                        help='''Colormap to use when producing the figure ''')
	
	#parser.add_argument('-field', type=str, default='temp',
	#	                        help=''' Which field is being plotted  ''')
	
	#parser.add_argument('-vmin', type=float, default=0.0,
	#	                        help='''Minimum value used for plotting''')
	
	#parser.add_argument('-vmax', type=float, default=1.0,
	#	                        help='''Maximum values used for plotting''')
	
	#parser.add_argument('-vanom', type=float, default=0.3,
	#	                        help='''This is the color scale when plot_anomaly=True. Goes from [-vanom vanom]''')
	
	#parser.add_argument('-flipped', type='bool', default=False,
	#	                        help=''' The panel is flipped over so that it faces the other way.''')
	
	#parser.add_argument('-plot_anomaly', type='bool', default=False,
	#	                        help=''' If true, then figure plots the anomaly from the initial value, using color scale -vanom ''')

	#parser.add_argument('-vertical_coordinate', type=str, default='layers',
	#		help='''Describes which type of ocean_month file is being used. Options: layers, z''')

	parser.add_argument('-time_slice', type=str, default='',
			help='''Time slice tells the code whether to do a time mean or a snapshot. Options: mean, None (default is snapshot)''')
	
	parser.add_argument('-time_slice_num', type=int, default=59,
		                        help='''The index of the transect used (in the direction not plotted''')

	#parser.add_argument('-xmin', type=float, default=0.0,
	#	                        help='''Minimum x used for plotting (only applies to vertical sectins for now)''')

	#parser.add_argument('-xmax', type=float, default=960.0,
	#	                        help='''Minimum x used for plotting (only applies to vertical sectins for now)''')

	#parser.add_argument('-dir_slice_num', type=int, default=1,
	#	                        help='''The index of the transect used (in the direction not plotted''')

	parser.add_argument('-ylim_min', type=float, default=0.0,
		                        help='''Minimum y used for plotting (only applies to horizontal sections)''')

	parser.add_argument('-ylim_max', type=float, default=480.0,
		                        help='''Minimum y used for plotting (only applies to horizontal sections)''')



        optCmdLineArgs = parser.parse_args()
        return optCmdLineArgs

def main(args):
	#parser = argparse.ArgumentParser()
	#args = parser.parse_args()


	#Plotting flats
	save_figure=args.save_figure
	
	#General Flags
	rotated=args.rotated
	use_ALE=args.use_ALE
	broken_shelf=args.broken_shelf

	#What to plot?
	plot_horizontal_field=args.plot_horizontal_field
	#plot_bt_stream_comparison=False
	#plot_temperature_cross_section=False

	#Defining path
	#Defining path
	Path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/'
	Folder_name= 'Melt_on_without_decay_with_spreading_trimmed_shelf/' 
	Shelf_path=Path+'Shelf/' + Folder_name
	broken_path=''
	if broken_shelf is True:
		Folder_name= 'After_melt_Collapse_diag_Strong_Wind/' 
		Folder_name= 'After_melt_Collapse_diag_Strong_Wind/' 
		broken_path='_calved'
	Berg_path=Path+'Bergs/' + Folder_name

	#Geometry files
	ocean_geometry_filename=Shelf_path +'ocean_geometry.nc'
	ice_geometry_filename=Shelf_path+'/MOM_Shelf_IC.nc'
	ISOMIP_IC_filename=Shelf_path+'ISOMIP_IC.nc'
	

	#Using Mixed_melt_flag
	use_mixed_melt_flag=''
	if args.use_Mixed_Melt is True:
		use_mixed_melt_flag='Mixed_Melt_'
		Folder_name= 'Mixed_Melt_' +Folder_name

	#Using ALE ice shelf
	use_ALE_flag=''
	if use_ALE is True:
		use_ALE_flag='ALE_z_'
		Folder_name='ALE_z_' +Folder_name
		
		Shelf_path=Path+'Shelf/' + Folder_name
		Berg_path=Path+'Bergs/' + Folder_name
		
	#Berg files
	Berg_ocean_file_init=Berg_path+'00010101.ocean_month.nc'
	Berg_ocean_file=Berg_path+'00060101.ocean_month.nc'
	Berg_iceberg_file=Berg_path+'00060101.icebergs_month.nc'


	#Load static fields
	(depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated)	
	grounding_line=find_grounding_line(depth, shelf_area, ice_base, x,y, xvec, yvec)
	ice_front=find_ice_front(depth,shelf_area,x,y, xvec, yvec)
	
	#Defining figure characteristics
	#fig=plt.figure(figsize=(15,10),facecolor='grey')
	#fig = plt.figure(facecolor='black')
	#ax = fig.add_subplot(111,axisbg='gray')
	letter_labels=np.array(['(a)','(b)','(c)','(d)','(e)'])
	title=''
	
	#Deciding which time to plot (time_slice_num overridded sometimes)
	time_slice=args.time_slice
	time_slice_num=args.time_slice_num
	ylim_min=args.ylim_min
	ylim_max=args.ylim_max
		
	cmap=args.cmap

	######################################################################################################################
	################################  Plotting melt comparison  ##########################################################
	######################################################################################################################
	
	if plot_horizontal_field is True:
		#Plotting flags
		three_fields_flag=args.three_fields_flag
		plot_vel_vel_sst=False
		plot_depth_spread_mass_mass=False
		plot_melt_sst_ustar_berg=True

		plot_flag=''
		fig, axes = plt.subplots(nrows=1,ncols=3)
		fig.set_size_inches(15.0,10.0, forward=True)
		#fig.set_size_inches
		#fig=plt.figure(figsize=(15,10))

		flipped=False

		#if plot_vel_vel_sst is True:
		if three_fields_flag=='plot_vel_vel_sst':
			field_list=np.array(['vo','uo','sst'])
			title_list=np.array(['Meridional surface velocity','Zonal surface velocity','SST'])
			colorbar_unit_list=np.array(['Meridional velocity (m/s)','Zonal velocity (m/s)','Temp (C)'])
			filename_list=np.array([Berg_iceberg_file,Berg_iceberg_file,Berg_iceberg_file])
			vmin_list=np.array([-0.01, -0.01, -1.9])
			vmax_list=np.array([0.01, 0.01, -0.5])
			if use_ALE is True:
				vmin_list=np.array([-0.05, -0.05, -1.9])
				vmax_list=np.array([0.05, 0.05, -0.5])
			cmap_list=np.array(['bwr', 'bwr', 'jet'])
			mask_grounded=np.array([True,True, True])
			mask_open_ocean=np.array([False,False, False])
			scale_data=True   ; 
			scale_list=np.array([1,1,1])
		
		#if plot_melt_sst_ustar_berg is True:
		if three_fields_flag=='plot_melt_sst_ustar_berg':
			field_list=np.array(['melt_m_per_year','sst','ustar_iceberg'])
			title_list=np.array(['Melt rate','SST','u_star'])
			#colorbar_unit_list=np.array(['melt (m/year)','Temp (C)','u_star (m/s)'])
			colorbar_unit_list=np.array(['(m/yr)','(deg C)','(m/s)'])
			#title_list=np.array(['','',''])
			filename_list=np.array([Berg_iceberg_file,Berg_iceberg_file,Berg_iceberg_file])
			vmin_list=np.array([0.0, -1.9, 0.0])
			vmax_list=np.array([3.0, -0.5, 0.001])
			if use_ALE is True:
				vmin_list=np.array([0.0, -1.9, 0.0])
				vmax_list=np.array([10.0, -0.5, 0.005])
			cmap_list=np.array(['jet', 'jet', 'jet'])
			mask_grounded=np.array([True,True, True])
			#mask_open_ocean=np.array([True,False, True])
			mask_open_ocean=np.array([True, True, True])
			scale_data=True   ; 
			scale_list=np.array([1,1,1])

		#if plot_depth_spread_mass_mass is True:
		if three_fields_flag=='plot_depth_spread_mass_mass':
			field_list=np.array(['D','mass','spread_mass'])
			title_list=np.array(['Ocean Bottom','Ice Shelf Draft', 'Ice Shelf Draft (no spreading)'])
			colorbar_unit_list=np.array(['Ice Shelf Draft (m)','Ice Shelf Draft (m)','Ice Shelf Draft (m)'])
			filename_list=np.array([ocean_geometry_filename,Berg_iceberg_file,Berg_iceberg_file])
			vmin_list=np.array([0, 0.0,0.0 ])
			vmax_list=np.array([1000, 1000, 1000])
			cmap_list=np.array(['jet', 'jet', 'jet'])
			rho_ice=918.0 ; rho_sw=1025.0;
			mask_grounded=np.array([False, False, False])
			mask_open_ocean=np.array([False,False, False])
			scale_data=True   ; 
			#scale_list=np.array([1,1/rho_ice,1/rho_ice])
			scale_list=np.array([1,1/rho_sw,1/rho_sw])
		N=len(field_list)

		for k in range(N):
			field=field_list[k]
			vmin=vmin_list[k]
			vmax=vmax_list[k]
			cmap=cmap_list[k]
			plot_flag=plot_flag+'_'+ field
			filename=filename_list[k]

			ax=plt.subplot(1,N,k+1)
			data=load_and_compress_data(filename,field=field,time_slice=time_slice,time_slice_num=time_slice_num,rotated=rotated) 
			plt.plot(xvec,grounding_line, linewidth=3.0,color='black')
			if broken_shelf is False:
				plt.plot(xvec,ice_front, linewidth=3.0,color='black')
			if mask_open_ocean[k] == True:
				mask_ocean_using_bergs=True
				if mask_ocean_using_bergs is True:
					ice_data=load_and_compress_data(Berg_iceberg_file,field='spread_area',time_slice='',time_slice_num=time_slice_num,\
							rotated=rotated,direction='xy',dir_slice=None, dir_slice_num=1)
					data=mask_ocean(data,ice_data)
				else:
					data=mask_ocean(data,ice_base)
			plot_data_field((ice_base*0),x,y,-5.0, 10.0,flipped,colorbar=False,cmap='Greys',title=title,xlabel='x (km)',ylabel='',ylim_min=ylim_min,\
					ylim_max=ylim_max,return_handle=False)
			if mask_grounded[k] == True:
				data=mask_grounded_ice(data,depth,ice_base)
			if scale_data is True:
				data=data*scale_list[k]
			if args.use_title_on_figure is True:
				title=title_list[k]
			plot_colorbar=not args.just_one_colorbar
			datamap=plot_data_field(data,x,y,vmin,vmax,flipped,colorbar=plot_colorbar,cmap=cmap,title=title,\
					xlabel='x (km)',ylabel='',return_handle=True, colorbar_units=colorbar_unit_list[k],colorbar_shrink=0.5,ylim_min=ylim_min, ylim_max=ylim_max)	
			if k==0:
				plt.ylabel('y (km)')
			text(0.1,1,letter_labels[k], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)
			#plt.ylim([550., 750])
			if k>0:
				ax.set_yticks([])

			#Creating colorbar
			if args.just_one_colorbar:
				fig.subplots_adjust(right=0.85)
				cbar_ax = fig.add_axes([0.9,0.12 , 0.025, 0.75])
				cbar=fig.colorbar(datamap, cax=cbar_ax)
				cbar.set_label(colorbar_unit_list[k-1], rotation=90,fontsize=20)
				cbar.ax.tick_params(labelsize=20)


		######################data###############################################################################################
		################################  Plotting Bt stream function ########################################################
		######################################################################################################################

		#if plot_bt_stream_comparison is True:
		#	fig=plt.figure(figsize=(5,10))
		#	vmin=-1*(10**3)  ; vmax=1*(10**3)
		#	flipped=False
		#	field='barotropic_sf'
		#	plot_flag='_'+field
	
		#	data1=calculate_barotropic_streamfunction(Berg_ocean_file,depth,ice_base,time_slice='mean',time_slice_num=-1,rotated=rotated)
		#	plot_data_field(data1,x,y,vmin,vmax,flipped,colorbar=True,cmap='jet',title='Bergs',xlabel='x (km)',ylabel='y (km)')	


		###############data###############################################################################################
		################################  Plotting Cross Section        ########################################################
		######################################################################################################################

	else:
		plot_just_one_field=False
		plot_temp_temp_v=True

		#fig=plt.figure(figsize=(10,5))
		fig=plt.figure(figsize=(10,10),facecolor='grey')
		plot_anomaly=False
		#vertical_coordinate='z'
		vertical_coordinate='layers'  #'z'
		field='v'  ; vmin=-0.01 ; vmax=0.01  ;vdiff=0.01   ; vanom=0.01; plot_anomaly=False; cmap='bwr'
		#field='temp'  ; vmin=-2.0  ; vmax=1.0  ;vdiff=0.1   ; vanom=0.3
		#field='salt'  ; vmin=34  ; vmax=34.7  ;vdiff=0.05  ; vanom=0.05
		#plot_flag='_'+field
		plot_flag=''
		filename=Berg_ocean_file
		filename_init=Berg_ocean_file_init

		if args.three_fields_flag=='plot_just_one_field':
			field_list=np.array([field])
			vmin_list=np.array([vmin])
			vmax_list=np.array([vmax])
			cmap_list=np.array([cmap])
			colorbar_unit_list=np.array([''])
			time_slice_num_list=np.array([time_slice_num])
			plot_anomaly_list=np.array([plot_anomaly])
			filename_list=np.array([filename])
			time_slice_list=np.array([time_slice])
		if args.three_fields_flag=='plot_temp_temp_v':
			field_list=np.array(['temp','temp','v'])
			vmin_list=np.array([-2.0, -2.0 , -0.01])
			vmax_list=np.array([1.0, 1.0, 0.01])
			vanom_list=np.array([0.3, 0.6, 0.01])
			cmap_list=np.array(['jet', 'jet', 'jet'])
			colorbar_unit_list=np.array(['(deg C)','(deg C)','(m/s)'])
			time_slice_num_list=np.array([0,-1,-1])
			plot_anomaly_list=np.array([False, True, False])
			filename_list=np.array([Berg_ocean_file_init, Berg_ocean_file,Berg_ocean_file])
			time_slice_list=np.array(['','mean','mean'])

		
		if vertical_coordinate=='z':
			filename=filename.split('.nc')[0] + '_z.nc'
	

		N=len(field_list)
		for k in range(N):
		#for k in np.array([1]):
			field=field_list[k]
			vmin=vmin_list[k]
			vmax=vmax_list[k]
			vanom=vanom_list[k]
			filename=filename_list[k]
			time_slice_num=time_slice_num_list[k]
			cmap=cmap_list[k]
			plot_anomaly=plot_anomaly_list[k]
			time_slice=time_slice_list[k]
			if plot_anomaly==True:
				plot_anomaly=True
			if plot_anomaly == False:
				plot_anomaly=False

			plot_flag=plot_flag+'_'+ field

			ax=plt.subplot(N,1,k+1)

			data=load_and_compress_data(filename,field , time_slice, time_slice_num=time_slice_num, direction='yz',\
					dir_slice=None, dir_slice_num=20,rotated=rotated)
			data[np.where(data==0.)]=np.nan
			elevation = get_vertical_dimentions(filename,vertical_coordinate, time_slice,\
					time_slice_num=-1, direction='yz' ,dir_slice=None, dir_slice_num=20,rotated=rotated)
			(y ,z ,data) =interpolated_onto_vertical_grid(data, elevation, yvec, vertical_coordinate)
	
			if plot_anomaly is True:
				data0=load_and_compress_data(filename_init,field , time_slice=None, time_slice_num=0,\
						direction='yz' ,dir_slice=None, dir_slice_num=20,rotated=rotated)
				elevation0 = get_vertical_dimentions(filename_init,vertical_coordinate, time_slice=None,\
						time_slice_num=-1, direction='yz' ,dir_slice=None, dir_slice_num=20,rotated=rotated)
				(y0 ,z0 ,data0) =interpolated_onto_vertical_grid(data0, elevation0, yvec, vertical_coordinate)
				data=data-data0
				vmin=-vanom  ; vmax=vanom

			plot_data_field(data, y, z,  vmin, vmax, flipped=False, colorbar=True, cmap=cmap,colorbar_units=colorbar_unit_list[k])

			zoom_in_on_second_plot=True
			if zoom_in_on_second_plot is True:
				xlo=475.-320.0 ; xhi=500.-320.0
				ylo=-520.; yhi=-420.
				if k==1:
					plt.plot(np.array([xlo , xhi]) ,np.array([ylo , ylo]),'k',linewidth=3)
					plt.plot(np.array([xlo , xhi]) ,np.array([yhi , yhi]),'k',linewidth=3)
					plt.plot(np.array([xlo , xlo]) ,np.array([ylo , yhi]),'k',linewidth=3)
					plt.plot(np.array([xhi , xhi]) ,np.array([ylo , yhi]),'k',linewidth=3)
				if k==2:
					plt.xlim([xlo , xhi])
					plt.ylim([ylo , yhi])
			if k==2:
				plt.xlabel('y (km)',fontsize=20)
			plt.ylabel('Depth (m)',fontsize=20)
			text(0.1,1,letter_labels[k], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)
		
		#For plotting purposes
		field=field+'_'+ vertical_coordinate




	#plt.tight_layout()


	if save_figure==True:
		output_file='Figures/'+use_ALE_flag+'static_shelf_solo' + plot_flag +broken_path+ '.png'
		plt.savefig(output_file,dpi=300,bbox_inches='tight')
		print 'Saving ' ,output_file

	#fig.set_size_inches(9,4.5)
	plt.show()
	print 'Script complete'



if __name__ == '__main__':
	optCmdLineArgs= parseCommandLine()
	main(optCmdLineArgs)
	#sys.exit(main())














