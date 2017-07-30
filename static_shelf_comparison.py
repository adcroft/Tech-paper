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
	
	parser.add_argument('-use_Revision', type='bool', default=False,
		                        help='''When true, it uses the results of the Revision simulations (including new drag and rolling)    ''')
	
	parser.add_argument('-use_simulations_with_wind', type='bool', default=False,
		                        help='''When true, use the newer simulations with wind on from the start.   ''')
	
	#What to plot?
	parser.add_argument('-fields_to_compare', type=str, default='plot_bt_stream_comparison',
		                        help=''' This flag determine whether to plot horizontal, vertical, or special case for barotropic stream function
					(should remove this later). Options are plot_bt_stream_comparison, plot_melt_comparison, plot_cross_section   ''')
	
	parser.add_argument('-field', type=str, default='melt',
		                        help=''' Which field is plotted when horizontal comparison is used   ''')
	
	parser.add_argument('-file_type', type=str, default='ice',
		                        help=''' Which file the data is loaded from when horizontal comparison is used   ''')

	#Which file type to use
	parser.add_argument('-use_title_on_figure', type='bool', default=False,
		                        help=''' When true, figure the name of the field is written as title ''')

	parser.add_argument('-only_shelf_ALE', type='bool', default=False,
		                        help=''' When true, Figures compares Shelf ALE and Layer (must be used with use_ALE=True ''')


	parser.add_argument('-ylim_min', type=float, default=0.0,
		                        help='''Minimum y used for plotting (only applies to horizontal sections)''')

	parser.add_argument('-ylim_max', type=float, default=480.0,
		                        help='''Minimum y used for plotting (only applies to horizontal sections)''')

	parser.add_argument('-vmin', type=float, default=0.0,
		                        help='''Minimum y used for plotting (only applies to horizontal sections)''')

	parser.add_argument('-vmax', type=float, default=7.0,
		                        help='''Max value used in colorbar for plotting (only applies to horizontal sections)''')
	
	parser.add_argument('-vdiff', type=float, default=8.0,
		                        help='''Anomaly value used in colorbar for plotting (only applies to horizontal sections)''')
	
	parser.add_argument('-time_slice', type=str, default=' ',
		                        help='''Decides which time slice to use (mean, all...  ''')
	
	parser.add_argument('-time_slice_num', type=int, default=-1,
		                        help='''Decides which time slice to use (mean, all...  ''')
	
	parser.add_argument('-direction', type=str, default='xy',
		                        help='''Decides which direction to  use  ''')
	
	parser.add_argument('-dir_slice', type=str, default=' ',
		                        help='''Decides which direction slice to  use  ''')
	
	parser.add_argument('-dir_slice_num', type=int, default=0,
		                        help='''Decides which direction slice number to  use  ''')
	
	
	optCmdLineArgs = parser.parse_args()
	return optCmdLineArgs

def str2bool(string):
        if string.lower() in  ("yes", "true", "t", "1"):
                Value=True
        elif string.lower() in ("no", "false", "f", "0"):
                Value=False
        else:
                print '**********************************************************************'
                print 'The input variable ' ,str(string) ,  ' is not suitable for boolean conversion, using default'
                print '**********************************************************************'

                Value=None
                return

        return Value

def transpose_matrix(data):
	if len(data.shape)==2:
		M=data.shape
		data_new=np.zeros([M[1],M[0]])
		for i in range(M[0]):
			for j in range(M[1]):
				data_new[j,i]=data[i,j]
		#print 'After rotation' ,data.shape
	if len(data.shape)==3:
		M=data.shape
		data_new=np.zeros([M[0],M[2],M[1]])
		for i in range(M[1]):
			for j in range(M[2]):
				data_new[:,j,i]=data[:,i,j]
		#print 'After rotation' ,data.shape
	if len(data.shape)==4:
		M=data.shape
		data_new=np.zeros([M[0],M[1],M[3],M[2]])
		for i in range(M[2]):
			for j in range(M[3]):
				data_new[:,:,j,i]=data[:,:,i,j]
		#print 'After rotation' ,data.shape
	
	data_new=np.ma.masked_invalid(data_new)
	return data_new


def mask_grounded_ice(data,depth,base):
   """
   Mask regions where the ice shelf is grounded (base>=depth). Works with 2D or 3D arrays.
   """
   if len(data.shape) == 2: # 2D array
      data = np.ma.masked_where(base+1.0>=depth, data) # need +1 here      
   else: # 3D array
      NZ,NY,NX = data.shape
      base = np.resize(base,(NZ,NY,NX))
      depth = np.resize(depth,(NZ,NY,NX))
      data = np.ma.masked_where(base+1.0>=depth, data) # need +1 here

   return data

def mask_ice_old(data,ice_base):
   """
   Mask regions where the ice shelf is above ocean Works with 2D or 3D arrays.
   This is not coded so well, a bit of a hack.
   """
   mask=np.zeros([data.shape[1],data.shape[2]])+1.
   mask[np.where(ice_base>0)]=np.nan
   if len(data.shape) == 2: # 2D array
	   data[k,:,:]=data[:,:]*mask[:,:]
   if len(data.shape) == 3: # 3D array
	   for k in range(data.shape[0]):
		   data[k,:,:]=data[k,:,:]*mask[:,:]
		 
   return data

def mask_ocean(data,area,tol=0.0):
   """
   Mask open ocean. Works with 2D or 3D arrays.
   """
   if len(data.shape) == 2: # 2D array
     data = np.ma.masked_where(area<=tol,data)
     #data[np.where(area<0.5)]=np.nan

   else: # 3D array
     NZ,NY,NX = data.shape
     area=np.resize(area,(NZ,NY,NX))
     data = np.ma.masked_where(area<=tol,data)

   return  data

def mask_ice(data,area,tol=None):
	"""
	Mask open ocean. Works with 2D or 3D arrays.
	"""
	if tol is None:
		tol=0.8
	if len(data.shape) == 2: # 2D array
		data = np.ma.masked_where(area>tol,data)
		#data[np.where(area==0)]=np.nan

	else: # 3D array
		NZ,NY,NX = data.shape
		area=np.resize(area,(NZ,NY,NX))
		data = np.ma.masked_where(area==0,data)
	return  data

def get_psi2D(u,v):
    '''
    Loop in time and compute the barotropic streamfunction psi at h points.
    '''
    NT,NY,NX = u.shape
    uh = np.zeros(u.shape); vh = np.zeros(v.shape); psi = np.zeros(u.shape)
    # u and v at h points
    utmp = 0.5 * (u[:,:,0:-1] + u[:,:,1::]) #u_i = 0.5(u_(i+0.5) + u_(i-0.5))
    vtmp = 0.5 * (v[:,0:-1,:] + v[:,1::,:]) #v_j = 0.5(v_(j+0.5) + v_(j-0.5))
    uh[:,:,1::] = utmp; uh[:,:,0] = 0.5*u[:,:,0] #u_i=1 = 0.5*u_(i=3/2)
    vh[:,1::,:] = vtmp; vh[:,0,:] = 0.5*v[:,0,:] #v_j=1 = 0.5*v_(j=3/2)
    for t in range(NT):
        psi[t,:,:] = (-uh[t,:,:].cumsum(axis=0) + vh[t,:,:].cumsum(axis=1))*0.5
        
    return psi 

def squeeze_matrix_to_2D(data,time_slice,time_slice_num,direction=None,dir_slice=None, dir_slice_num=None):
	#This routine reduces the data to two dimensions. It assumes that the first dimension is time. 
	#The time_slice / time_slice_num must be provided to tell where the averages should be applied.
	#If there are three spatial dimensions, then the dirction, and dir_slice, dir_slice_num must be added

	#Trimming and squeezing the data

	if len(data.shape)>2:
		
		#Removing extra horizontal dimension
		if len(data.shape) == 4:
			if dir_slice=='mean':
				if direction=='xy':
					axis_num=1
				if direction=='xz':
					axis_num=2
				if direction=='yz':
					axis_num=3
				data=np.squeeze(np.mean(data,axis=axis_num))
			else:
				if direction=='xy':
					data=np.squeeze(data[:,dir_slice_num,:,:])
				if direction=='xz':
					data=np.squeeze(data[:,:,dir_slice_num,:])
				if direction=='yz':
					data=np.squeeze(data[:,:,:,dir_slice_num])
		
		#Removing extra time dimensions
		if time_slice!='all':
			#Reducing time dimension  - assumes that first dim is time.
			if time_slice=='mean':
				data=np.squeeze(np.mean(data,axis=0))  #Mean over first variable
			else:
				#if len(data.shape) == 3:
				data=np.squeeze(data[time_slice_num,:,:])  #Mean over first variable
				#if len(data.shape) == 4:
					#data=np.squeeze(data[time_slice_num,:,:,:])  #Mean over first variable
	return data

def load_data_from_file(filename, field, rotated):
	field_tmp=field
	if rotated is True:
		if field=='u':
			field_tmp='v'
		if field=='v':
			field_tmp='u'
		if field=='uo':
			field_tmp='vo'
		if field=='vo':
			field_tmp='uo'
		if field=='spread_uvel':
			field_tmp='spread_vvel'
		if field=='spread_vvel':
			field_tmp='spread_uvel'
		if field=='vhbt':
			field_tmp='uhbt'
		if field=='uhbt':
			field_tmp='vhbt'

	with nc.Dataset(filename) as file:
		data = file.variables[field_tmp][:]
	
	if rotated is True:
		data=transpose_matrix(data)
		if field=='u' or  field=='uo' or field=='uhbt' or field=='spread_uvel':
			data=-data
	return data

def load_and_compress_data(filename, field, time_slice, time_slice_num, direction=None ,dir_slice=None, dir_slice_num=None,\
		rotated=False,return_time=False,depth=None, ice_base=None):

	#Loading data
	if field=='barotropic_sf':  #special case for barotropic stream function.
		if (depth is None) or (ice_base is None):
			print 'Depth and ice base needed to calculate stream function'
		data=calculate_barotropic_streamfunction(filename,depth,ice_base,time_slice=time_slice,time_slice_num=time_slice_num,rotated=rotated)
	else:
		data=load_data_from_file(filename, field, rotated)
	
	#Give an error message if direction is not given for 4D matrix	
	if (len(data.shape)>3 and direction is None):
		print 'Direction must be provided for 4-dim matrix'
		return
	
	data=squeeze_matrix_to_2D(data,time_slice,time_slice_num,direction,dir_slice, dir_slice_num)

	if (return_time is True) and ((time_slice=='') or (time_slice is None)):
		time=load_data_from_file(filename, 'time', rotated=False)[time_slice_num]
		print 'time = ', time , ' days'
		return [data , time]
	else:
		return data

def load_and_compress_time_series(filename, field, time_slice, time_slice_num, direction=None ,dir_slice=None, dir_slice_num=None,\
		rotated=False,return_time=False,depth=None, ice_base=None):

	#Loading data
	if field=='barotropic_sf':  #special case for barotropic stream function.
		if (depth is None) or (ice_base is None):
			print 'Depth and ice base needed to calculate stream function'
		data=calculate_barotropic_streamfunction(filename,depth,ice_base,time_slice=time_slice,time_slice_num=time_slice_num,rotated=rotated)
	else:
		data=load_data_from_file(filename, field, rotated)
	
	#Give an error message if direction is not given for 4D matrix	
	if (len(data.shape)>3 and direction is None):
		print 'Direction must be provided for 4-dim matrix'
		return
	
	data=squeeze_matrix_to_time_series(data,time_slice,time_slice_num,direction,dir_slice, dir_slice_num)

	return data

def squeeze_matrix_to_time_series(data,time_slice,time_slice_num,direction=None,dir_slice=None, dir_slice_num=None):
	#This routine reduces the data to a time series. It assumes that the first dimension is time. 
	#The time_slice / time_slice_num must be provided to tell where the averages should be applied.
	#If there are three spatial dimensions, then the dirction, and dir_slice, dir_slice_num must be added

	if len(data.shape)==4:
		data=np.mean(data,axis=3)
	if len(data.shape)==3:
		data=np.mean(data,axis=2)
	if len(data.shape)==2:
		data=np.mean(data,axis=1)



	return data

def calculate_barotropic_streamfunction(filename,depth,ice_base,time_slice=None,time_slice_num=-1,rotated=False):
	uhbt=load_data_from_file(filename, 'uhbt', rotated)
	vhbt=load_data_from_file(filename, 'vhbt', rotated)
	# mask grouded region
	uhbt = mask_grounded_ice(uhbt,depth,ice_base)
	vhbt = mask_grounded_ice(vhbt,depth,ice_base)
	psi2D = get_psi2D(uhbt,vhbt)
	psi2D = mask_grounded_ice(psi2D,depth,ice_base)
	#saveXY(psi2D,'barotropicStreamfunction')

	#Taking only final value
	psi2D=squeeze_matrix_to_2D(psi2D,time_slice,time_slice_num)

	return psi2D
	
def get_vertical_dimentions(filename, vertical_coordinate, time_slice, time_slice_num, direction ,dir_slice, dir_slice_num,rotated=False):
	if vertical_coordinate=='z':
		print 'The filename is ....', filename
		with nc.Dataset(filename) as file:
			#z = file.variables['zt'][:]
			z = file.variables['z_l'][:]
		return z
	if vertical_coordinate=='layers':
		z=load_and_compress_data(filename, 'e' , time_slice, time_slice_num, direction ,dir_slice, dir_slice_num,rotated=rotated)

	return z

def switch_x_and_y(x,y):
	tmp=y ; y=x ; x=tmp
	
	return [x,y]

def load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated,xy_from_zero=True):
	# bedrock
	depth = Dataset(ocean_geometry_filename).variables['D'][:]
	# area under shelf 
	shelf_area = Dataset(ice_geometry_filename).variables['area'][:,:]
	# base of STATIC ice shelf, which is ssh(t=0); make it positive
	ice_base = -Dataset(ISOMIP_IC_filename).variables['ave_ssh'][0,:,:]
	ice_base[ice_base<1e-5] = 0.0
	#mask grounded ice and open ocean
	shelf_area = mask_grounded_ice(shelf_area,depth,ice_base) 
	shelf_area = mask_ocean(shelf_area,shelf_area)

	# x,y, at tracer points 
	x=Dataset(ocean_geometry_filename).variables['geolon'][:]#*1.0e3 # in m
	y=Dataset(ocean_geometry_filename).variables['geolat'][:]#*1.0e3 # in m
	xvec=Dataset(ocean_geometry_filename).variables['lonh'][:]#*1.0e3 # in m
	yvec=Dataset(ocean_geometry_filename).variables['lath'][:]#*1.0e3 # in m

	if xy_from_zero is True:
		x_start=np.min(x)-(0.5*(xvec[2]-xvec[1]));
		y_start=np.min(y)-(0.5*(yvec[2]-yvec[1]));
		print 'Subtracting y,x=', x_start, y_start
		x=x-x_start
		y=y-y_start
		xvec=xvec-x_start
		yvec=yvec-y_start

	if rotated is True:
		depth=transpose_matrix(depth)
		shelf_area=transpose_matrix(shelf_area)
		ice_base=transpose_matrix(ice_base)
		[x,y]=switch_x_and_y(x,y)		
		x=-transpose_matrix(x)
		y=transpose_matrix(y)
		[xvec,yvec]=switch_x_and_y(xvec,yvec)
		xvec=-xvec
		xvec=xvec-np.min(xvec)
		x=x-np.min(x)
	return [depth, shelf_area, ice_base, x,y, xvec, yvec]

def find_grounding_line(depth, shelf_area, ice_base, x,y, xvec, yvec):
        #Finding grounding line (assuming shelf in the South)
        M=depth.shape
        grounding_line=np.zeros(M[1])
        for i in range(M[1]):
                Flag=False
                for j in range(M[0]):
                        if (Flag is False) and shelf_area[j,i]>0.5:
                                Flag=True
                                grounding_line[i]=yvec[j]
        return grounding_line

def find_ice_front(depth,shelf_area,x,y, xvec, yvec):
        #Finding grounding line (assuming shelf in the South)
        M=depth.shape
        ice_front=np.zeros(M[1])
        for i in range(M[1]):
                Flag=False
                for j in range(M[0]-1,0,-1):
                        if (Flag is False) and shelf_area[j,i]>0.5:
                                Flag=True
                                ice_front[i]=yvec[j]
        return ice_front

def get_plot_axes_limits(x, y, xlim_min, xlim_max, ylim_min, ylim_max):
	if xlim_min is None:
		xlim_min=np.min(x)
	if xlim_max is None:
		xlim_max=np.max(x)
	if ylim_min is None:
		ylim_min=np.min(y)
	if ylim_max is None:
		ylim_max=np.max(y)
	return [xlim_min, xlim_max, ylim_min, ylim_max]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
	cmap = plt.get_cmap(cmap)
	new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
	return new_cmap

def plot_data_field(data,x,y,vmin=None,vmax=None,flipped=False,colorbar=True,cmap='jet',title='',xlabel='',ylabel='',return_handle=False,grounding_line=None, \
		xlim_min=None, xlim_max=None, ylim_min=None,ylim_max=None,colorbar_units='',colorbar_shrink=1.0): 
	if cmap=='Greys':
		cmap = truncate_colormap(cmap, 0.0, 0.75)
	if flipped is True:
		data=transpose_matrix(data)
		tmp=y ; y=x ; x=tmp
		x=transpose_matrix(x)
		y=transpose_matrix(y)
		#y=-y+(np.max(y))
		(xlabel , ylabel) = switch_x_and_y(xlabel , ylabel)
		(xlim_max , ylim_max) = switch_x_and_y(xlim_max , ylim_max)
		(xlim_min , ylim_min) = switch_x_and_y(xlim_min , ylim_min)

	
	print 'Starting to plot...'	
	if vmin==None:
		vmin=np.min(data)
	else:
		vmin=float(vmin)

	if vmax==None:
		vmax=np.max(data)
	else:
		vmax=float(vmax)

	cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
	print 'CNorm', vmin, vmax, cNorm
	datamap=plt.pcolormesh(x,y,data,norm=cNorm,cmap=cmap)
	if colorbar is True:
		cbar=plt.colorbar(datamap, cmap=cmap, norm=cNorm, shrink=colorbar_shrink)
		cbar.set_label(colorbar_units, rotation=90,fontsize=20)

	(xlim_min, xlim_max, ylim_min, ylim_max)=get_plot_axes_limits(x, y, xlim_min, xlim_max, ylim_min, ylim_max)
	plt.xlim(xlim_min,xlim_max)
	plt.ylim(ylim_min,ylim_max)
	if grounding_line is not None:
		if  flipped is False:
			plt.plot(x[0,:],grounding_line, linewidth=3.0,color='black')
		else:
			plt.plot(grounding_line,y[:,0], linewidth=3.0,color='black')

	plt.xlabel(xlabel,fontsize=20)
	plt.ylabel(ylabel,fontsize=20)
	#plt.grid(True)
	plt.title(title,fontsize=20)

	if flipped is True:
		plt.gca().invert_yaxis()

	if return_handle is True:
		return datamap



def interpolated_layers_onto_grid(data, layer_interface, x):
	representation='linear'  #'pcm'
        #representation='pcm'
        M=layer_interface.shape
        layers=range(0,M[0]-1)
        q=data[range(0,M[0]-1),:] ;q=q[:,range(0,len(x)-1) ]
        layer_interface=layer_interface[:,range(0,len(x)-1)]
        (X, Z, Q)=section2quadmesh(x, layer_interface, q, representation)
	
	return [X,Z,Q]


def interpolated_onto_vertical_grid(data, layer_interface, x, vertical_coordinate):
	print 'Beginning interpolation...'
	if vertical_coordinate=='layers':  
		if len(data.shape)==2:
			(X, Z, Q)=interpolated_layers_onto_grid(data, layer_interface, x)
		elif len(data.shape)==3:
			T=data.shape[0]
			(X_tmp, Z_tmp, Q_tmp)=interpolated_layers_onto_grid(data[0,:,:], layer_interface[0,:,:], x)
			M=X_tmp.shape
			X=np.zeros([T,X_tmp.shape[0]])
			Z=np.zeros([T,Z_tmp.shape[0],Z_tmp.shape[1]])
			Q=np.zeros([T,Q_tmp.shape[0],Q_tmp.shape[1]])
			for n in range(data.shape[0]):
				(X_tmp, Z_tmp, Q_tmp)=interpolated_layers_onto_grid(data[n,:,:], layer_interface[n,:,:], x)
				X[n,:]=X_tmp
				Z[n,:,:]=Z_tmp
				Q[n,:,:]=Q_tmp
        
	
	if vertical_coordinate=='z':
                X=x
                Z=-layer_interface
                Q=data

	return [X,Z,Q]

def create_subsampled_zero_matrix(data,New_time_step_number):
		print 'Lendth before shortening', len(data.shape)
		if len(data.shape)==2:
			data_new=np.zeros([New_time_step_number,data.shape[1]])
		if len(data.shape)==3:
			data_new=np.zeros([New_time_step_number,data.shape[1],data.shape[2]])
		if len(data.shape)==4:
			data_new=np.zeros([New_time_step_number,data.shape[1],data.shape[2],data.shape[3]])
		return data_new


def subsample_data(x, y, data,  axes_fixed, subsample_num=None):
	if subsample_num is None:
		return [x, y, data]
	else:
		Num_timesteps_old=data.shape[0]
		Num_timesteps_new=int(np.ceil(Num_timesteps_old/float(subsample_num)))
		
		#Creating empty matricies
		print data.shape
		data_new=create_subsampled_zero_matrix(data, Num_timesteps_new)
		if axes_fixed is False:
			x_new=create_subsampled_zero_matrix(x, Num_timesteps_new)
			y_new=create_subsampled_zero_matrix(y, Num_timesteps_new)
		else:
			x_new=x   ;  y_new =y
		
		#Filling in the new data
		count=-1
		for i in range(0,Num_timesteps_old,subsample_num):
			count=count+1
			data_new[count,:]=data[i,:]
			if axes_fixed is False:
				x_new[count,:]=x[i,:]
				y_new[count,:]=y[i,:]

		return [x_new, y_new, data_new]

def switch_axis_if_rotated(rotated,yvec,xvec):
	if rotated is True:
		direction='yz'
		dist=yvec
	else:
		direction='xz'
		dist=xvec
	return  [dist , direction]


##########################################################  Main Program   #########################################################################
####################################################################################################################################################

def main(args):

	#Plotting flats
	save_figure=args.save_figure
	
	#General flags
	rotated=args.rotated
	use_ALE=args.use_ALE
	use_Revision=args.use_Revision

	#What to plot?
	fields_to_compare=args.fields_to_compare


	#Defining path
	use_Wind_flag=''
	if args.use_simulations_with_wind is True:
		use_Wind_flag='Wind_'
		Path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Lagrangian_ISOMIP/'
		Folder_name= 'Static_with_Wind/' 
	else:
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
	use_Revision_flag=''
	if use_ALE is True:
		use_ALE_flag='ALE_z_'
		Folder_name='ALE_z_' +Folder_name
		if use_Revision is True:
			use_Revision_flag='Revision_'
			Folder_name='Revision_' +Folder_name
		Shelf_path=Path+'Shelf/' + Folder_name
		Berg_path=Path+'Bergs/' + Folder_name

		#Comparing Layer vs ALE
		if args.only_shelf_ALE is True:
			Berg_path=Layer_Shelf_path 
	
	#Shelf files
	Shelf_ocean_file=Shelf_path+'00010101.ocean_month.nc'
	Shelf_ice_file=Shelf_path+'00010101.ice_month.nc'
	Shelf_prog_file=Shelf_path+'00010101.prog.nc'

	#Berg files
	Berg_ocean_file=Berg_path+'00010101.ocean_month.nc'
	Berg_ice_file=Berg_path+'00010101.ice_month.nc'
	Berg_prog_file=Berg_path+'00010101.prog.nc'
	Berg_iceberg_file=Berg_path+'00010101.icebergs_month.nc'

	#Other
	letter_labels=np.array(['(a)','(b)','(c)','(d)','(e)'])



	print ice_geometry_filename
	print ocean_geometry_filename
	print ISOMIP_IC_filename
	#Load static fields
	(depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated)	
      	grounding_line=find_grounding_line(depth, shelf_area, ice_base, x,y, xvec, yvec)
	ice_front=find_ice_front(depth,shelf_area,x,y, xvec, yvec)

	
	#Defining figure characteristics
	fig=plt.figure(figsize=(15,10),facecolor='grey')
	#fig = plt.figure(facecolor='black')
	ax = fig.add_subplot(111,axisbg='gray')

	#Scale
	ylim_min=args.ylim_min
	ylim_max=args.ylim_max

	######################################################################################################################
	################################  Plotting melt comparison  ##########################################################
	######################################################################################################################
	
	if fields_to_compare=='horizontal_comparison':
		#vmin=0.0  ; vmax=3.0  ; vdiff=3.0
		#if use_ALE is True:
		vmin=args.vmin  ; vmax=args.vmax  ; vdiff=args.vdiff
		flipped=False
		field=args.field
		time_slice=args.time_slice
		time_slice_num=args.time_slice_num
		file_type=args.file_type
		direction=args.direction
		dir_slice=args.dir_slice
		dir_slice_num=args.dir_slice_num


		if file_type=='ice':
			Shelf_file=Shelf_ice_file
			Berg_file=Berg_ice_file
		if file_type=='prog':
			Shelf_file=Shelf_prog_file
			Berg_file=Berg_prog_file
		if field=='melt':
			field2='melt_m_per_year'
			Shelf_file=Shelf_ocean_file
			Berg_file=Berg_iceberg_file
		else:
			field2=field


		print Shelf_file
		print Berg_file
		data1=load_and_compress_data(Shelf_file,field=field, time_slice=time_slice,time_slice_num=time_slice_num,rotated=rotated,\
				direction=direction ,dir_slice=dir_slice, dir_slice_num=dir_slice_num )
		data2=load_and_compress_data(Berg_file, field=field2, time_slice=time_slice,time_slice_num=time_slice_num,rotated=rotated,\
				direction=direction ,dir_slice=dir_slice, dir_slice_num=dir_slice_num )

		#Masking out ocean and grounded ice
		mask_open_ocean=True
		if mask_open_ocean == True:
			mask_ocean_using_bergs=True
			if mask_ocean_using_bergs is True:
				ice_data=load_and_compress_data(Berg_iceberg_file,field='spread_area',time_slice='',time_slice_num=1,\
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

		data1=mask_ocean(data1,shelf_area)
		data2=mask_ocean(data2,shelf_area)
		#data3=mask_ocean(data1-data2,shelf_area)
		ax=plt.subplot(1,3,1)
		plot_data_field((ice_base*0),x,y,-5.0, 10.0,flipped,colorbar=False,cmap='Greys',title=title,xlabel='x (km)',ylabel='',ylim_min=ylim_min, \
				ylim_max=ylim_max,return_handle=False)
		plot_data_field(data1,x,y,vmin,vmax,flipped,colorbar=True,cmap='jet',title='Eularian',xlabel='x (km)',ylabel='y (km)', ylim_min=ylim_min, ylim_max=ylim_max)	
		text(0.1,1,letter_labels[0], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)
		ax=plt.subplot(1,3,2)
		plot_data_field((ice_base*0),x,y,-5.0, 10.0,flipped,colorbar=False,cmap='Greys',title=title,xlabel='x (km)',ylabel='',ylim_min=ylim_min, \
				ylim_max=ylim_max,return_handle=False)
		plot_data_field(data2,x,y,vmin,vmax,flipped,colorbar=True,cmap='jet',title='Lagrangian',xlabel='x (km)',ylabel='', ylim_min=ylim_min, ylim_max=ylim_max)
		text(0.1,1,letter_labels[1], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)
		ax=plt.subplot(1,3,3)
		plot_data_field((ice_base*0),x,y,-5.0, 10.0,flipped,colorbar=False,cmap='Greys',title=title,xlabel='x (km)',ylabel='',ylim_min=ylim_min, \
				ylim_max=ylim_max,return_handle=False)
		plot_data_field(data1-data2,x,y,vmin=-vdiff,vmax=vdiff,flipped=flipped,colorbar=True,cmap='bwr',title='Difference',xlabel='x (km)',ylabel='', ylim_min=ylim_min, ylim_max=ylim_max)
		text(0.1,1,letter_labels[2], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)

	######################data###############################################################################################
	################################  Plotting Bt stream function ########################################################
	######################################################################################################################

	if fields_to_compare=='plot_bt_stream_comparison':
		vmin=-3*(10**3)  ; vmax=3*(10**3)
		if use_ALE is True:
			vmin=-7*(10**3)  ; vmax=7*(10**3)
		flipped=False
		field='barotropic_sf'
		cmap='jet'

		data1=calculate_barotropic_streamfunction(Shelf_ocean_file,depth,ice_base,time_slice='mean',time_slice_num=-1,rotated=rotated)
		data2=calculate_barotropic_streamfunction(Berg_ocean_file,depth,ice_base,time_slice='mean',time_slice_num=-1,rotated=rotated)

		#Masking out grounded ice
		mask_grounded=True
		if mask_grounded == True:
			data1=mask_grounded_ice(data1,depth,ice_base)
			data2=mask_grounded_ice(data2,depth,ice_base)

		#Plotting everyone
		ax=plt.subplot(1,3,1)
		plot_data_field((ice_base*0),x,y,-5.0, 10.0,flipped,colorbar=False,cmap='Greys',title=title,xlabel='x (km)',ylabel='',ylim_min=ylim_min, \
				ylim_max=ylim_max,return_handle=False)
		plot_data_field(data2,x,y,vmin,vmax,flipped,colorbar=True,cmap=cmap,title='Lagrangian',\
				xlabel='x (km)',ylabel='y (km)',colorbar_shrink=0.5,colorbar_units='(Sv)', ylim_min=ylim_min, ylim_max=ylim_max)	
                plt.plot(xvec,grounding_line, linewidth=3.0,color='black')
                plt.plot(xvec,ice_front, linewidth=3.0,color='black')
		text(0.1,1,letter_labels[0], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)
		
		ax=plt.subplot(1,3,2)
		plot_data_field((ice_base*0),x,y,-5.0, 10.0,flipped,colorbar=False,cmap='Greys',title=title,xlabel='x (km)',ylabel='',ylim_min=ylim_min, \
				ylim_max=ylim_max,return_handle=False)
		plot_data_field(data1,x,y,vmin,vmax,flipped,colorbar=True,cmap=cmap,title='Eularian',\
				xlabel='x (km)',ylabel='',colorbar_shrink=0.5,colorbar_units='(Sv)', ylim_min=ylim_min, ylim_max=ylim_max)	
                plt.plot(xvec,grounding_line, linewidth=3.0,color='black')
                plt.plot(xvec,ice_front, linewidth=3.0,color='black')
		text(0.1,1,letter_labels[1], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)
		
		ax=plt.subplot(1,3,3)
		plot_data_field((ice_base*0),x,y,-5.0, 10.0,flipped,colorbar=False,cmap='Greys',title=title,xlabel='x (km)',ylabel='',ylim_min=ylim_min, \
				ylim_max=ylim_max,return_handle=False)
		plot_data_field(data1-data2,x,y,-vmax, vmax, flipped,colorbar=True,cmap='bwr',title='Difference',\
				xlabel='x (km)',ylabel='',colorbar_shrink=0.5,colorbar_units='(Sv)', ylim_min=ylim_min, ylim_max=ylim_max)	
                plt.plot(xvec,grounding_line, linewidth=3.0,color='black')
                plt.plot(xvec,ice_front, linewidth=3.0,color='black')
		text(0.1,1,letter_labels[2], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)

	if fields_to_compare=='plot_cross_section':
		plot_anomaly=True
		time_slice='mean'
		#vertical_coordinate='z'
		vertical_coordinate='layers'  #'z'
		#field='temp'  ; vmin=-2.0  ; vmax=1.0  ;vdiff=0.1   ; vanom=0.3
		field='salt'  ; vmin=34  ; vmax=34.7  ;vdiff=0.03  ; vanom=0.03
		#field='v'  ; vmin=-0.01  ; vmax=0.01  ;vdiff=0.01  ; vanom=0.01
		#field='v'  ; vmin=-0.01  ; vmax=0.01  ;vdiff=0.01  ; vanom=0.01
		filename1=Shelf_ocean_file
		filename2=Berg_ocean_file
		
		if vertical_coordinate=='z':
			filename1=filename1.split('.nc')[0] + '_z.nc'
			filename2=filename2.split('.nc')[0] + '_z.nc'
		if rotated is True:
                        direction='yz'
                        dist=yvec
                else:
                        direction='xz'
                        dist=xvec


		data1=load_and_compress_data(filename1,field , time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		elevation1 = get_vertical_dimentions(filename1,vertical_coordinate, time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		(y1 ,z1 ,data1) =interpolated_onto_vertical_grid(data1, elevation1, dist, vertical_coordinate)

		data2=load_and_compress_data(filename2,field , time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		elevation2 = get_vertical_dimentions(filename2,vertical_coordinate, time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
		(y2 ,z2 ,data2) =interpolated_onto_vertical_grid(data2, elevation2, dist, vertical_coordinate)

		if plot_anomaly is True:
			data0=load_and_compress_data(filename1,field , time_slice=None, time_slice_num=0, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
			elevation0 = get_vertical_dimentions(filename1,vertical_coordinate, time_slice=None,\
			  time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
			(y0 ,z0 ,data0) =interpolated_onto_vertical_grid(data0, elevation0, dist, vertical_coordinate)
			data1=data1-data0
			data2=data2-data0
			vmin=-vanom  ; vmax=vanom


		ax=plt.subplot(3,1,1)
		plot_data_field(data1, y1, z1, vmin, vmax, flipped=False, colorbar=True, cmap='jet')
		surface=np.squeeze(elevation1[0,:])   ;  plot(dist, surface,'black')
		bottom=np.squeeze(depth[:,20])   ;  plot(dist, -bottom,'black')
		text(0.1,1,letter_labels[0], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)

		ax=plt.subplot(3,1,2)
		plot_data_field(data2, y2, z2, vmin, vmax, flipped=False, colorbar=True, cmap='jet')
		surface=np.squeeze(elevation2[0,:])   ;  plot(dist, surface,'black')
		bottom=np.squeeze(depth[:,20])   ;  plot(dist, -bottom,'black')
		text(0.1,1,letter_labels[1], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)

		ax=plt.subplot(3,1,3)
		surface=np.squeeze(elevation1[0,:])   ;  plot(dist, surface,'black')
		plot_data_field(data1-data2, y1, z1, vmin=-vdiff, vmax=vdiff, flipped=False, colorbar=True, cmap='bwr')
		bottom=np.squeeze(depth[:,20])   ;  plot(dist, -bottom,'black')
		text(0.1,1,letter_labels[2], ha='right', va='bottom',transform=ax.transAxes,fontsize=20)
		
		#For plotting purposes
		field=field+'_'+ vertical_coordinate




	plt.tight_layout()


	if save_figure==True:
		output_file='Figures/'+ use_Wind_flag +  use_Revision_flag +use_ALE_flag +'static_shelf_comparison_' + field + '.png'
		plt.savefig(output_file,dpi=300,bbox_inches='tight')
		print 'Saving ' ,output_file

	#fig.set_size_inches(9,4.5)
	plt.show()
	print 'Script complete'



if __name__ == '__main__':
	optCmdLineArgs= parseCommandLine()
	main(optCmdLineArgs)

	#sys.exit(main())














