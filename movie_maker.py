#!/usr/bin/env python
import matplotlib.animation as animation
import numpy as np
from netCDF4 import Dataset
from pylab import *
from static_shelf_comparison import *



def ani_frame(field,output_filename):

        dpi = 150
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        M=field.shape

        #im = ax.imshow(rand(M[0],M[1]),cmap='jet',interpolation='nearest')
        im = ax.imshow(field[0,:,:],cmap='jet',interpolation='nearest')
        #cbar=fig.colorbar(im, orientation="horizontal")

        vmax=np.max(field)
        vmin=np.min(field)
        im.set_clim([0,1])
        im.set_clim([vmin,vmax])
        fig.set_size_inches([15,30])

        tight_layout()


        def update_img(n):
                #tmp = rand(300,300)
                tmp =field[n,:,:]
                im.set_data(tmp)
                return im

        #legend(loc=0)
        Number_of_images=10
        fps=120
        ani = animation.FuncAnimation(fig,update_img,Number_of_images,interval=30)
        writer = animation.writers['ffmpeg'](fps=fps)

        ani.save(output_filename,writer=writer,dpi=dpi)
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
        output_filename='movies/test5.mp4'

	#General flags
	horizontal_movie=False
	cross_section_movie=True


	Berg_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/Melt_on_without_decay_with_spreading_trimmed_shelf/'
	Berg_path='/ptmp/aas/data/fixed_speed_Moving_berg_trimmed_shelf_from_zero_small_step_u01/'

	#Geometry files
        ocean_geometry_filename=Berg_path +'ocean_geometry.nc'
        ice_geometry_filename=Berg_path+'/MOM_Shelf_IC.nc'
        ISOMIP_IC_filename=Berg_path+'ISOMIP_IC.nc'

	#Berg files
	extension_name='prog_combined.nc'
	#extension_name='ocean_month_combined.nc'

        #Berg_ocean_file=Berg_path+'00010101.ocean_month.nc'
        Berg_ocean_file=Berg_path+extension_name
	
	#General flags
	rotated=True	

        #Load static fields
        (depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated)



	
        ######################################################################################################################
        ################################  Plotting melt comparison  ##########################################################
        ######################################################################################################################

	if horizontal_movie is True:

		filename=Berg_ocean_file
		print filename
		field_name='mass_berg'
		#field_name='u'
		direction='xy'
		data=load_and_compress_data(filename,field=field_name,time_slice='all',time_slice_num=-1,rotated=rotated, direction=direction ,dir_slice=None, dir_slice_num=20)
		print data.shape

		#Loading data from NC file.
		#[field,x,y]=load_data_from_nc_file(input_filename,field_name)


        ######################################################################################################################
        ################################  Plotting Cross Section      ########################################################
        ######################################################################################################################


        if cross_section_movie is True:
                plot_anomaly=True
                time_slice='all'
                #vertical_coordinate='z'
                vertical_coordinate='layers'  #'z'
                #field='temp'  ; vmin=-2.0  ; vmax=1.0  ;vdiff=0.1   ; vanom=0.3
                field='salt'  ; vmin=34  ; vmax=34.7  ;vdiff=0.02  ; vanom=0.02
                #field='v'  ; vmin=-0.01  ; vmax=0.01  ;vdiff=0.01  ; vanom=0.01
                #field='v'  ; vmin=-0.01  ; vmax=0.01  ;vdiff=0.01  ; vanom=0.01
                filename=Berg_ocean_file
                
                if vertical_coordinate=='z':
                        filename=filename.split('.nc')[0] + '_z.nc'
                if rotated is True:
                        direction='yz'
                        dist=yvec
                else:
                        direction='xz'
                        dist=xvec

                data=load_and_compress_data(filename,field , time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
                elevation = get_vertical_dimentions(filename,vertical_coordinate, time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
                (y ,z ,data) =interpolated_onto_vertical_grid(data, elevation, dist, vertical_coordinate)
		print 'Interpolation complete', data.shape



	print 'Staring to make a freaking movie!'
	ani_frame(data,output_filename)

if __name__ == '__main__':
        main()
        #sys.exit(main())

