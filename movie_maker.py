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
        output_filename='test5.mp4'

	Shelf_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Shelf/Melt_on_without_decay_with_spreading_trimmed_shelf/'
	Berg_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/Melt_on_without_decay_with_spreading_trimmed_shelf/'

	#Geometry files
        ocean_geometry_filename=Shelf_path +'ocean_geometry.nc'
        ice_geometry_filename=Shelf_path+'/MOM_Shelf_IC.nc'
        ISOMIP_IC_filename=Shelf_path+'ISOMIP_IC.nc'

	#Berg files
        Berg_ocean_file=Berg_path+'00010101.ocean_month.nc'
	
	#General flags
	rotated=True	

        #Load static fields
        (depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated)



	
        ######################################################################################################################
        ################################  Plotting melt comparison  ##########################################################
        ######################################################################################################################

	filename=Berg_ocean_file
	print filename
        field_name='mass_berg'
        #field_name='u'
	direction='xy'
	data=load_and_compress_data(filename,field=field_name,time_slice='all',time_slice_num=-1,rotated=rotated, direction=direction ,dir_slice=None, dir_slice_num=20)
	print data.shape

        #Loading data from NC file.
        #[field,x,y]=load_data_from_nc_file(input_filename,field_name)



        ani_frame(data,output_filename)

if __name__ == '__main__':
        main()
        #sys.exit(main())

