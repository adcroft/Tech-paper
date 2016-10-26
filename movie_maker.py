#!/usr/bin/env python
import matplotlib.animation as animation
import numpy as np
from netCDF4 import Dataset
from pylab import *
from static_shelf_comparison import *


def get_nth_values(n,data,x,y,axes_fixed):
	data_n=data[n,:,:]
	
	if axes_fixed is True:
		x_n=x
		y_n=y

	else:
		if len(x.shape)==2:
			x_n=x[n,:]
		if len(x.shape)==3:
			x_n=x[n,:,:]
		if len(y.shape)==2:
			y_n=y[n,:]
		if len(y.shape)==3:
			y_n=y[n,:,:]

	return [data_n, x_n, y_n]


def ani_frame(x,y,data,output_filename,vmin,vmax,cmap,axes_fixed,fig_length=14,fig_height=6,resolution=360,xlabel='',ylabel='',frames_per_second=120, frame_interval=100, Max_frames=None,\
		grounding_line=None):
	
	#Defining Figure Properties
	fig=plt.figure(figsize=(fig_length,fig_height),facecolor='white')
	ax = fig.add_subplot(111,axisbg='gray')
	(data_n , xn ,yn)=get_nth_values(0,data,x,y,axes_fixed)
	im=plot_data_field(data_n,xn,yn,vmin,vmax,flipped=False,colorbar=True,cmap=cmap,title='',xlabel=xlabel,ylabel=ylabel,return_handle=True,grounding_line=grounding_line)
        #cbar=fig.colorbar(im, orientation="horizontal")

        tight_layout()
	#plt.show()
	#return

        def update_img(n):
		#Updating figure for each frame
		print 'Frame number' ,n ,  'writing now.' 
		fig.clf()
		#ax = fig.add_subplot(111,axisbg='gray')
		(data_n , xn ,yn)=get_nth_values(n,data,x,y,axes_fixed)
		im=plot_data_field(data_n,xn,yn,vmin,vmax,flipped=False,colorbar=True,cmap=cmap,title='',xlabel=xlabel,ylabel=ylabel,return_handle=True)
		
                return im

        #legend(loc=0)
	if Max_frames is None:
	        Number_of_images=data.shape[0]
	else:
	        Number_of_images=Max_frames
        ani = animation.FuncAnimation(fig,update_img,Number_of_images,interval=frame_interval)
        writer = animation.writers['ffmpeg'](fps=frames_per_second)

        ani.save(output_filename,writer=writer,dpi=resolution)
	print 'Movie saved: ' , output_filename
        return ani


def load_data_from_nc_file(filename,field_name):
        #Importing the data file
        nc = Dataset(filename, mode='r')

        #Use the length field to help filter the data
        field = nc.variables[field_name][:,:,:]
        x = nc.variables['xT'][:]
        y = nc.variables['yT'][:]
        return [field,x,y]


###########################################################################################################################


def main():
        input_filename='/home/aas/Iceberg_Project/iceberg_scripts/python_scripts/movies/00010101.icebergs_day.nc'
        output_filename='movies/test4.mp4'

	#General flags
	horizontal_movie=False
	cross_section_movie=True


	#Berg_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/Melt_on_without_decay_with_spreading_trimmed_shelf/'
	Berg_path='/ptmp/aas/data/fixed_speed_Moving_berg_trimmed_shelf_from_zero_small_step_u01/'
	exp_name='fixed_u01_from_zero'


	#Geometry files
        ocean_geometry_filename=Berg_path +'ocean_geometry.nc'
        ice_geometry_filename=Berg_path+'/MOM_Shelf_IC.nc'
        ISOMIP_IC_filename=Berg_path+'ISOMIP_IC.nc'

	#Berg files
	extension_name='prog_combined.nc'
	#extension_name='icebergs_month_combined.nc'
	#extension_name='ocean_month_combined.nc'

        #Berg_ocean_file=Berg_path+'00010101.ocean_month.nc'
        Berg_ocean_file=Berg_path+extension_name
	
	#General flags
	rotated=True	

        #Load static fields
        (depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated)
	grounding_line=find_grounding_line(depth, shelf_area, ice_base, x,y, xvec, yvec)



	
        ######################################################################################################################
        ################################  Plotting melt comparison  ##########################################################
        ######################################################################################################################

	if horizontal_movie is True:

		filename=Berg_ocean_file
		print filename
		#field_name='mass_berg'
		#field_name='spread_area'
		field_name='u' ;  vmin=-0.1  ;vmax=0.1 ; cmap='bwr'
		direction='xy'
		data=load_and_compress_data(filename,field=field_name,time_slice='all',time_slice_num=-1,rotated=rotated, direction=direction ,dir_slice=None, dir_slice_num=20)

		#Configuring movie specs
		print 'Staring to make a freaking movie!'
		axes_fixed=True
		output_filename='movies/' + exp_name + '_' +field_name + '_' + direction + '_' + str(data.shape[0])+ 'frames' + '.mp4'
		ani_frame(x,y,data,output_filename,vmin,vmax,cmap, axes_fixed,fig_length=6,fig_height=12,resolution=360,\
		xlabel='x (km)',ylabel='y (km)',frames_per_second=120, frame_interval=100, Max_frames=None,grounding_line=grounding_line)


        ######################################################################################################################
        ################################  Plotting Cross Section      ########################################################
        ######################################################################################################################


        if cross_section_movie is True:
                plot_anomaly=True
                time_slice='all'
                #vertical_coordinate='z'
                vertical_coordinate='layers'  #'z'
                #field_name='temp'  ; vmin=-2.0  ; vmax=1.0  ;vdiff=0.1   ; vanom=0.3
                #field_name='salt'  ; vmin=34  ; vmax=34.7  ;vdiff=0.02  ; vanom=0.02
                field_name='u'  ; vmin=-0.1  ; vmax=0.1  ;vdiff=0.02  ; vanom=0.02 ; cmap='bwr'
                #field_name='v'  ; vmin=-0.01  ; vmax=0.01  ;vdiff=0.01  ; vanom=0.01
                filename=Berg_ocean_file
                
                if vertical_coordinate=='z':
                        filename=filename.split('.nc')[0] + '_z.nc'
                if rotated is True:
                        direction='yz'
                        dist=yvec
                else:
                        direction='xz'
                        dist=xvec

                data=load_and_compress_data(filename,field_name , time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
                elevation = get_vertical_dimentions(filename,vertical_coordinate, time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
                (y ,z ,data) =interpolated_onto_vertical_grid(data, elevation, dist, vertical_coordinate)
		print 'Interpolation complete', data.shape, y.shape, z.shape


		axes_fixed=False
		print 'Staring to make a freaking movie!'
		output_filename='movies/' + exp_name + '_' +field_name + '_' + direction + '_' + str(data.shape[0])+ 'frames' + '.mp4'
		print output_filename
		ani_frame(y,z,data,output_filename,vmin,vmax,cmap, axes_fixed,fig_length=14,fig_height=6,resolution=360,\
				xlabel='y (km)',ylabel='depth (m)',frames_per_second=120, frame_interval=100, Max_frames=None)

if __name__ == '__main__':
        main()
        #sys.exit(main())

