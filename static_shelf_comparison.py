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

def mask_ocean(data,area):
   """
   Mask open ocean. Works with 2D or 3D arrays.
   """
   if len(data.shape) == 2: # 2D array
     data = np.ma.masked_where(area==0,data)
     #data[np.where(area==0)]=np.nan

   else: # 3D array
     NZ,NY,NX = data.shape
     area=np.resize(area,(NZ,NY,NX))
     data = np.ma.masked_where(area==0,data)

   return  data

def mask_ice(data,area):
   """
   Mask open ocean. Works with 2D or 3D arrays.
   """
   if len(data.shape) == 2: # 2D array
     data = np.ma.masked_where(area>0.1,data)
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

def load_and_compress_data(filename, field, time_slice, time_slice_num, direction=None ,dir_slice=None, dir_slice_num=None,rotated=False,return_time=False):

	#Loading data
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
		with nc.Dataset(filename) as file:
			z = file.variables['zt'][:]
		return z
	if vertical_coordinate=='layers':
		z=load_and_compress_data(filename, 'e' , time_slice, time_slice_num, direction ,dir_slice, dir_slice_num,rotated=rotated)

	return z

def switch_x_and_y(x,y):
	tmp=y ; y=x ; x=tmp
	
	return [x,y]

def load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated):
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

def plot_data_field(data,x,y,vmin=None,vmax=None,flipped=False,colorbar=True,cmap='jet',title='',xlabel='',ylabel='',return_handle=False,grounding_line=None, \
		xlim_min=None, xlim_max=None, ylim_min=None,ylim_max=None): 
	if flipped is True:
		data=transpose_matrix(data)
		tmp=y ; y=x ; x=tmp
		x=transpose_matrix(x)
		y=transpose_matrix(y)
		y=-y+(np.max(y))
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
	datamap=plt.pcolormesh(x,y,data,norm=cNorm,cmap=cmap)
	if colorbar is True:
		plt.colorbar(datamap, cmap=cmap, norm=cNorm, shrink=0.5)
	(xlim_min, xlim_max, ylim_min, ylim_max)=get_plot_axes_limits(x, y, xlim_min, xlim_max, ylim_min, ylim_max)
	plt.xlim(xlim_min,xlim_max)
	plt.ylim(ylim_min,ylim_max)
	if grounding_line is not None:
		if  flipped is False:
			plt.plot(x[0,:],grounding_line, linewidth=3.0,color='black')
		else:
			plt.plot(grounding_line,y[:,0], linewidth=3.0,color='black')

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#plt.grid(True)
	plt.title(title)

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

def main():
	parser = argparse.ArgumentParser()
        parser.add_argument('--file1', default=None, help='The input data file1 in NetCDF format.')
        parser.add_argument('--file2', default=None, help='The input data file2 in NetCDF format.')
        parser.add_argument('--field', default=None, help='Feild to plot')
        parser.add_argument('--field2', default=None, help='Feild to plot')
        parser.add_argument('--operation',default=None, help='Operation betweeen fields')
        parser.add_argument('--vmax', default=None, help='Maximum for plotting')
        parser.add_argument('--vmin', default=None, help='Minimum for plotting')
        parser.add_argument('--rotated', default=None, help='Minimum for plotting')
	args = parser.parse_args()


	#Plotting flats
	save_figure=False
	
	#General flags
	rotated=True

	#What to plot?
	plot_melt_comparison=False
	plot_bt_stream_comparison=False
	plot_cross_section=True


	#Defining path
	Shelf_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Shelf/Melt_on_without_decay_with_spreading_trimmed_shelf/'
	Berg_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/Melt_on_without_decay_with_spreading_trimmed_shelf/'

	#Geometry files
	ocean_geometry_filename=Shelf_path +'ocean_geometry.nc'
	ice_geometry_filename=Shelf_path+'/MOM_Shelf_IC.nc'
	ISOMIP_IC_filename=Shelf_path+'ISOMIP_IC.nc'
	
	#Shelf files
	Shelf_ocean_file=Shelf_path+'00010101.ocean_month.nc'

	#Berg files
	Berg_ocean_file=Berg_path+'00010101.ocean_month.nc'
	Berg_iceberg_file=Berg_path+'00010101.icebergs_month.nc'



	#Load static fields
	(depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated)	
	
	#Defining figure characteristics
	fig=plt.figure(figsize=(15,10),facecolor='grey')
	#fig = plt.figure(facecolor='black')
	ax = fig.add_subplot(111,axisbg='gray')

	######################################################################################################################
	################################  Plotting melt comparison  ##########################################################
	######################################################################################################################
	
	if plot_melt_comparison is True:
		vmin=0.0  ; vmax=3.0
		flipped=False
		field='melt'

		data1=load_and_compress_data(Shelf_ocean_file,field='melt',time_slice='mean',time_slice_num=-1,rotated=rotated)
		data2=load_and_compress_data(Berg_iceberg_file,field='melt_m_per_year',time_slice='mean',time_slice_num=-1,rotated=rotated) #field name is difference since it comes from a diff file.

		data1=mask_ocean(data1,shelf_area)
		data2=mask_ocean(data2,shelf_area)
		#data3=mask_ocean(data1-data2,shelf_area)
		plt.subplot(1,3,1)
		plot_data_field(data1,x,y,vmin,vmax,flipped,colorbar=True,cmap='jet',title='Shelf',xlabel='x (km)',ylabel='y (km)')	
		plt.subplot(1,3,2)
		plot_data_field(data2,x,y,vmin,vmax,flipped,colorbar=True,cmap='jet',title='Bergs',xlabel='x (km)',ylabel='')	
		plt.subplot(1,3,3)
		plot_data_field(data1-data2,x,y,vmin=-3.,vmax=3.,flipped=flipped,colorbar=True,cmap='bwr',title='Difference',xlabel='x (km)',ylabel='')	

	######################data###############################################################################################
	################################  Plotting Bt stream function ########################################################
	######################################################################################################################

	if plot_bt_stream_comparison is True:
		vmin=-3*(10**3)  ; vmax=3*(10**3)
		flipped=False
		field='barotropic_sf'
		cmap='jet'

		data1=calculate_barotropic_streamfunction(Shelf_ocean_file,depth,ice_base,time_slice='mean',time_slice_num=-1,rotated=rotated)
		data2=calculate_barotropic_streamfunction(Berg_ocean_file,depth,ice_base,time_slice='mean',time_slice_num=-1,rotated=rotated)
		plt.subplot(1,3,1)
		plot_data_field(data1,x,y,vmin,vmax,flipped,colorbar=True,cmap=cmap,title='Shelf',xlabel='x (km)',ylabel='y (km)')	
		plt.subplot(1,3,2)
		plot_data_field(data2,x,y,vmin,vmax,flipped,colorbar=True,cmap=cmap,title='Bergs',xlabel='x (km)',ylabel='y (km)')	
		plt.subplot(1,3,3)
		plot_data_field(data1-data2,x,y,-vmax, vmax, flipped,colorbar=True,cmap='bwr',title='Difference',xlabel='x (km)',ylabel='')	


	if plot_cross_section is True:
		plot_anomaly=True
		time_slice='mean'
		#vertical_coordinate='z'
		vertical_coordinate='layers'  #'z'
		#field='temp'  ; vmin=-2.0  ; vmax=1.0  ;vdiff=0.1   ; vanom=0.3
		field='salt'  ; vmin=34  ; vmax=34.7  ;vdiff=0.02  ; vanom=0.02
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


		plt.subplot(3,1,1)
		plot_data_field(data1, y1, z1, vmin, vmax, flipped=False, colorbar=True, cmap='jet')
		plt.subplot(3,1,2)
		plot_data_field(data2, y2, z2, vmin, vmax, flipped=False, colorbar=True, cmap='jet')
		plt.subplot(3,1,3)
		plot_data_field(data1-data2, y1, z1, vmin=-vdiff, vmax=vdiff, flipped=False, colorbar=True, cmap='bwr')
		
		#For plotting purposes
		field=field+'_'+ vertical_coordinate




	plt.tight_layout()


	if save_figure==True:
		output_file='Figures/static_shelf_comparison_' + field + '.png'
		plt.savefig(output_file,dpi=300,bbox_inches='tight')
		print 'Saving ' ,output_file

	#fig.set_size_inches(9,4.5)
	plt.show()
	print 'Script complete'



if __name__ == '__main__':
	main()
	#sys.exit(main())














