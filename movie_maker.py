#!/usr/bin/env python
import matplotlib.animation as animation
import numpy as np
from netCDF4 import Dataset
from pylab import *
import matplotlib.pyplot as plt
import scipy.io
from static_shelf_comparison import *


def get_nth_values(n,data,x,y,axes_fixed):
	data_n=data[n,:,:]
	
	if axes_fixed is True:
		x_n=x
		y_n=y

	else:
		if len(x.shape)==1:
			x_n=x   ; y_n=y
		if len(x.shape)==2:
			x_n=x[n,:]
		if len(x.shape)==3:
			x_n=x[n,:,:]
		if len(y.shape)==2:
			y_n=y[n,:]
		if len(y.shape)==3:
			y_n=y[n,:,:]

	return [data_n, x_n, y_n]


def find_grounding_line_new(depth, shelf_area, ice_base, x,y, xvec, yvec):
        #Finding grounding line (assuming shelf in the South)
        M=depth.shape
        grounding_line=np.zeros(M[1])
        for i in range(M[1]):
                Flag=False
                for j in range(M[0]):
			tol=-0.1
			condition = ( ice_base[j,i] -depth[j,i]  ) <tol
                        if (Flag is False) and condition:
                                Flag=True
                                grounding_line[i]=yvec[j]
				print("grounding_line[i]",grounding_line[i])
        return grounding_line

def ani_frame(x,y,data,output_filename,vmin,vmax,cmap,axes_fixed,fig_length=14,fig_height=6,resolution=360,xlabel='',ylabel='',frames_per_second=120, frame_interval=100, Max_frames=None,\
		grounding_line=None,flipped=False,just_a_test=False):
	
	#Defining Figure Properties
	fig=plt.figure(figsize=(fig_length,fig_height),facecolor='grey')
	ax = fig.add_subplot(111,axisbg='gray')
	(data_n , xn ,yn)=get_nth_values(0,data,x,y,axes_fixed)
	print xn.shape, yn.shape
	im=plot_data_field(data_n,xn,yn,vmin,vmax,flipped=flipped,colorbar=True,cmap=cmap,title='',xlabel=xlabel,ylabel=ylabel,return_handle=True,grounding_line=grounding_line)
        #cbar=fig.colorbar(im, orientation="horizontal")
        tight_layout()

	if just_a_test is True:
		fig.clf()
		ax = fig.add_subplot(111,axisbg='gray')
		(data_n , xn ,yn)=get_nth_values(-1,data,x,y,axes_fixed)
		im=plot_data_field(data_n,xn,yn,vmin,vmax,flipped=flipped,colorbar=True,cmap=cmap,title='',xlabel=xlabel,ylabel=ylabel,return_handle=True,grounding_line=grounding_line)
		plt.show()
		return

        def update_img(n):
		#Updating figure for each frame
		print 'Frame number' ,n ,  'writing now.' 
		fig.clf()
		ax = fig.add_subplot(111,axisbg='gray')
		(data_n , xn ,yn)=get_nth_values(n,data,x,y,axes_fixed)
		im=plot_data_field(data_n,xn,yn,vmin,vmax,flipped=flipped,colorbar=True,cmap=cmap,title='',xlabel=xlabel,ylabel=ylabel,return_handle=True,grounding_line=grounding_line)
		
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

def save_pngs_from_data(x,y,data,output_folder,vmin,vmax,cmap,axes_fixed,fig_length=14,fig_height=6,resolution=360,xlabel='',ylabel='', grounding_line=None, Max_frames=None,flipped=False):
	
	#Defining Figure Properties
	fig=plt.figure(figsize=(fig_length,fig_height),facecolor='grey')
	if Max_frames is None:
	        Number_of_images=data.shape[0]
	else:
	        Number_of_images=Max_frames

	for n in range(Number_of_images):
		if n>1705:
			#Updating figure for each frame
			if n<10:
				frame_number='000'+str(n)
			elif n<100:
				frame_number='00'+str(n)
			elif n<1000:
				frame_number='0'+str(n)
			else:
				frame_number=str(n)
			print 'Frame number' ,frame_number ,  'writing now.' 
			fig.clf()
			ax = fig.add_subplot(111,axisbg='gray')
			(data_n , xn ,yn)=get_nth_values(n,data,x,y,axes_fixed)
			im=plot_data_field(data_n,xn,yn,vmin,vmax,flipped=flipped,colorbar=True,cmap=cmap,title='',xlabel=xlabel,ylabel=ylabel,return_handle=True,grounding_line=grounding_line)
			tight_layout()
			#plt.show()
			#return
			
			output_file=output_folder + '/image_' + frame_number + '.png'
			plt.savefig(output_file,dpi=300,bbox_inches='tight')
			print 'Saving ' ,output_file


	#ax = fig.add_subplot(111,axisbg='gray')
	#(data_n , xn ,yn)=get_nth_values(0,data,x,y,axes_fixed)
	#im=plot_data_field(data_n,xn,yn,vmin,vmax,flipped=False,colorbar=True,cmap=cmap,title='',xlabel=xlabel,ylabel=ylabel,return_handle=True,grounding_line=grounding_line)
        #cbar=fig.colorbar(im, orientation="horizontal")

	print 'Finished saving images. Have a great day: ' , output_folder
	plt.show()


		#Upload the data
def adjust_initial(t, m, melt, e,t_z,  init_ocean_file, init_ocean_file_z, init_Berg_file):
	print("init_ocean_file", init_ocean_file)
	print("init_ocean_file_z", init_ocean_file_z)
	print("init_Berg_file", init_Berg_file)
	onc = scipy.io.netcdf_file(init_ocean_file)
	onc_z = scipy.io.netcdf_file(init_ocean_file_z)
	snc = scipy.io.netcdf_file(init_Berg_file)
		
	#Pull the variables
	t_init = onc.variables['temp'][:,:,:,:]
	m_init = snc.variables['spread_mass'][:,:,:]
	melt_init = snc.variables['melt_m_per_year'][:,:,:]
	e_init = onc.variables['e'][:,:,:,:]
	t_z_init = onc_z.variables['temp'][:,:,:,:]

	M=shape(t)
	t_new=np.zeros((M[0],M[1],M[2],M[3]))
	t_new[:,:,:,:]=t[:,:,:,:]
	t_new[0,:,:,:]=t_init[0,:,:,:]

	M=shape(t_z)
	t_z_new=np.zeros((M[0],M[1],M[2],M[3]))
	t_z_new[:,:,:,:]=t_z[:,:,:,:]
	t_z_new[0,:,:,:]=t_z_init[0,:,:,:]

	M=shape(e)
	e_new=np.zeros((M[0],M[1],M[2],M[3]))
	e_new[:,:,:,:]=e[:,:,:,:]
	e_new[0,:,:,:]=e_init[0,:,:,:]

	M=shape(m)
	m_new=np.zeros((M[0],M[1],M[2]))
	m_new[:,:,:]=m[:,:,:]
	m_new[0,:,:]=m_init[0,:,:]

	#print("m", m)
	M=shape(melt)
	melt_new=np.zeros((M[0],M[1],M[2]))
	melt_new[:,:,:]=melt[:,:,:]
	melt_new[0,:,:]=melt_init[0,:,:]

	return (t_new, m_new , melt_new, e_new, t_z_new) 



def load_data_from_nc_file(filename,field_name):
        #Importing the data file
        nc = Dataset(filename, mode='r')

        #Use the length field to help filter the data
        field = nc.variables[field_name][:,:,:]
        x = nc.variables['xT'][:]
        y = nc.variables['yT'][:]
        return [field,x,y]

def make_data_anomolous(data,vanom,field_name):
	for k in range(data.shape[0]-1,-1,-1):
		data[k,]=data[k,:]-data[0,:]
	vmin=-vanom  ; vmax=vanom
	field_name=field_name+'_anom'
	cmap='bwr'
	return [data,cmap,field_name,vmin,vmax]

def create_e_with_correct_form(x, y ,time,  onc):
	z = -onc.variables['zw'][:]
	e = np.zeros((len(time), len(z), len(y), len(x)))
	for i in range(e.shape[0]):
		for j in range(e.shape[2]):
			for k in range(e.shape[3]):
				e[i,:,j,k]=z
	return e

def create_double_image(n,e,t,x,y,time,m,e_z, t_z,melt =None ,grounding_line=None, depth=None, ymin=600.0, ymax =780.0 , plot_anom=False, x_num=None, plot_four=False ):
	ymin=450.0 ; ymax =780.0 ; plot_anom=True ; x_num=20  ;y_num=149 ;  plot_four=True
	num_col=1 ;
	if plot_four is True:
		num_col=2


	#im=plt.figure(figsize=(10,6), facecolor='grey')
	nt = e.shape[0];
	#for n in range(nt):
	#for n in range(nt):
	i =y_num
	if x_num is None:
		j=max(8, int( 20 - (50.*n)/nt))
	else:
		j=x_num


	#display.display(plt.gcf())
	plt.clf();
	im=plt.subplot(2,num_col,1);
	sst = np.ma.array(t[n,0], mask=m[n,:,:]>1e4)
	
	if plot_anom is False:
		sst=np.ma.array(t[n,0], mask=(m[n,:,:])>1e4)
		vmin =-1.8 ; vmax=-1.2
		cmap= 'jet'
	else:
		#t_m=np.ma.array(t_z[n,:,j,:]-t_z[0,:,j,:], mask=( abs(t_z[n,:,j,:]) +  abs(t_z[0,:,j,:]))  >1e4)
		sst=np.ma.array(t[n,0]-t[0,0], mask=(m[n,:,:])>1e4)
		vmin =-0.1 ; vmax=0.1
		cmap= 'bwr'


	plt.pcolormesh(x,y,e[n,0], cmap='Greys'); plt.xlim(ymin,ymax); plt.clim(-150,50);# plt.colorbar(); plt.title(r'$\eta$ (m)');
	plt.pcolormesh(x, y, sst,cmap=cmap); plt.xlim(ymin,ymax); plt.clim(vmin,vmax); plt.colorbar(); plt.title(r'SST ($^\degree$C)')
	plt.ylabel('X (km)');
	plt.xlabel('Y (km)');
	#plt.gca().set_xticklabels([]);
	plt.text(740,90,'Time = %.1f days'%(time[n]-time[0]));
	plt.plot([x[0],x[-1]],[y[j],y[j]],'k--');
	plt.plot([x[i],x[i]],[y[0],y[-1]],'k--');
	plt.plot(grounding_line,y, 'k')


	#########################################
	if plot_four is False:
		plt.subplot(2,num_col, 2);
	else:
		plt.subplot(2,num_col, 3);
	plot_anom = True
	if plot_anom is False:
		t_m=np.ma.array(t[n,:,j,:], mask=abs(t[n,:,j,:])>1e4)
		e_m = e
		vmin =-1.8 ; vmax=0
		cmap= 'jet'
	else:
		t_m=np.ma.array(t_z[n,:,j,:]-t_z[0,:,j,:], mask=( abs(t_z[n,:,j,:]) +  abs(t_z[0,:,j,:]))  >1e4)
		e_m = e_z
		vmin =-0.1 ; vmax=0.1
		cmap= 'bwr'

	#plt.pcolormesh(x, e[n,:,j,:], t[n,:,j,:]); plt.xlim(600,780); plt.ylim(-600,2); plt.colorbar();
	plt.pcolormesh(x, e_m[n,:,j,:], t_m,cmap=cmap); plt.xlim(ymin,780); plt.ylim(-760,2); plt.colorbar();

	plt.plot(x, e[n,0,j,:], color='black' )
	if plot_anom is True:
		plt.plot(x, e[0,0,j,:], color='black',linestyle=':' )

	plt.clim(vmin,vmax); 
	#plt.clim(-1.8,0);
	plt.xlabel('Y (km)'); plt.ylabel('Z (m)'); plt.title(r'$\theta$ ($^\degree$C)');
	plt.plot([x[i],x[i]],[-720,0],'k--');
	plt.plot(x,-depth[:,j],'k');



	#########################################################		
	if plot_four is True:
		plt.subplot(2,num_col, 4);
		if plot_anom is False:
			t_m=np.ma.array(t[n,:,:,i], mask=abs(t[n,:,:,i])>1e4)
			e_m = e
			vmin =-1.8 ; vmax=0
			cmap= 'jet'
		else:
			t_m=np.ma.array(t_z[n,:,:,i]-t_z[0,:,:,i], mask=( abs(t_z[n,:,:,i]) +  abs(t_z[0,:,:,i]))  >1e4)
			e_m = e_z
			vmin =-0.1 ; vmax=0.1
			cmap= 'bwr'

		plt.pcolormesh(y, e_m[n,:,:,i], t_m,cmap=cmap); plt.xlim(0.0,80.0); plt.ylim(-760,2); plt.colorbar();
		plt.plot(y, e[n,0,:,i], color='black' )
		if plot_anom is True:
			plt.plot(y, e[0,0,:,i], color='black',linestyle=':' )
		plt.clim(vmin,vmax); 
		plt.plot([y[j],y[j]],[-720,0],'k--');
		plt.xlabel('X (km)'); plt.ylabel('Z (m)'); plt.title(r'$\theta$ ($^\degree$C)');
		plt.plot(y,-depth[i,:],'k');



		plt.subplot(2,num_col,2);
		if plot_anom  is False:
			melt_m=np.ma.array(melt[n,:,:], mask=abs(t[n,:,:])==0)
			#melt_m =melt[n, :, :]
		else:
			#melt_m =melt[n, :, :] - melt[0, :, :]
			melt_m=np.ma.array(melt[n,:,:]-melt[0,:,:], mask=( (melt[n,:,:]==0)+(melt[0,:,:]==0)>0.5 ))

		#plt.pcolormesh(x,y,e[n,0], cmap='Greys'); plt.xlim(ymin,ymax); plt.clim(-150,50);# plt.colorbar(); plt.title(r'$\eta$ (m)');
		#plt.pcolormesh(x, y, sst); plt.xlim(ymin,ymax); plt.clim(-0.5,0.5); plt.colorbar(); plt.title(r'SST ($^\degree$C)')
		plt.pcolormesh(x, y, melt_m, cmap=cmap); plt.xlim(ymin,ymax); plt.clim(-0.5,0.5); plt.colorbar(); plt.title("melt (m/yr)")
		plt.ylabel('Y (km)');
		plt.xlabel('X (km)');
		#plt.gca().set_xticklabels([]);
		plt.text(740,90,'Time = %.1f days'%(time[n]-time[0]));
		plt.plot([x[0],x[-1]],[y[j],y[j]],'k--');
		plt.plot(grounding_line,y, 'k')
		plt.plot([x[i],x[i]],[y[0],y[-1]],'k--');
	#plt.show()

	return im
			
	    #plt.savefig('figs/img_%3.3i.png'%(n+1))
	    #display.clear_output(wait=True)

###########################################################################################################################


def main():

	#General flags
	horizontal_movie=False
	cross_section_movie=False
	Alistair_double_movie=True

	make_a_movie=True
	save_pngs=False
	test_movie=False
	
	#General parameters
	subsample_num=None

	#Berg_path='/lustre/f1/unswept/Alon.Stern/MOM6-examples_Alon/ice_ocean_SIS2/Tech_ISOMIP/Bergs/Melt_on_without_decay_with_spreading_trimmed_shelf/'
	Base_path='/ptmp/aas/data/'

	#Choosing_an_experiment
	#exp_name='fixed_u01_from_zero'
	#exp_name='Bonds_kd2_from_zero'
	#exp_name='drift_from_zero'
	#exp_name='Collapse_from_zero'
	#exp_name='After_melt_fixed_u01'
	#exp_name='Melt_on_without_decay'
	#exp_name='ALE_z_After_melt_Collapse_diag_Strong_Wind'
	#exp_name='ALE_z_After_melt_drift_Strong_Wind'
	#exp_name='ALE_z_After_melt_Collapse_diag_Strong_Wind_Splitting'
	exp_name='Lag_After_Collapse'


	if exp_name=='fixed_u01_from_zero':
		experiment_path='fixed_speed_Moving_berg_trimmed_shelf_from_zero_small_step_u01/'    
	if exp_name=='Bonds_kd2_from_zero':
		experiment_path='Bond_drifting_Moving_berg_trimmed_shelf_small_step_kd2/'    
	if exp_name=='drift_from_zero':
		experiment_path='drifting_Moving_berg_trimmed_shelf_small_step/'    
	if exp_name=='Collapse_from_zero':
		experiment_path='Collapse_with_Bond_drifting_Moving_berg_trimmed_shelf_small_step_kd2/'    
	if exp_name=='After_melt_fixed_u01':
		experiment_path='After_melt_fixed_speed_small_step_u01/'    
	if exp_name=='Melt_on_without_decay':
		experiment_path='Melt_on_without_decay_with_spreading_trimmed_shelf/'    
	if exp_name=='ALE_z_After_melt_Collapse_diag_Strong_Wind':
		experiment_path='ALE_z_After_melt_Collapse_diag_Strong_Wind/'    
	if exp_name=='ALE_z_After_melt_drift_Strong_Wind':
		experiment_path='ALE_z_After_melt_drift_Strong_Wind/'    
	if exp_name=='ALE_z_After_melt_Collapse_diag_Strong_Wind_Splitting':
		experiment_path='ALE_z_After_melt_Collapse_diag_Strong_Wind_Splitting/'    
	if exp_name=='Lag_After_Collapse':
		experiment_path='Lag_After_Collapse/'    
		init_path='Lag_After_Static/'    
	Berg_path=Base_path + experiment_path


	#Geometry files
        ocean_geometry_filename=Berg_path +'ocean_geometry.nc'
        ice_geometry_filename=Berg_path+'/MOM_Shelf_IC.nc'
        ISOMIP_IC_filename=Berg_path+'ISOMIP_IC.nc'

	#Berg files
	#extension_name='prog.nc'; subsample_num=2
	#extension_name='icebergs_month_combined.nc'
	extension_name='icebergs_month.nc'
	#extension_name='ocean_month.nc'  ;  subsample_num=10
	#extension_name='ocean_month_z.nc' ;  subsample_num=10
	extension_name='ocean_month.nc'  ; subsample_num=10
	#extension_name='ocean_month_z.nc'  ; subsample_num=2




        #Berg_ocean_file=Berg_path+'00010101.ocean_month.nc'
        Berg_ocean_file=Berg_path+extension_name
	print Berg_ocean_file


	#These two are used for the Alistair movie maker script. I should clean this up later (so the script has less repitition)
	berg_extension='icebergs_month.nc'
	ocean_extension='ocean_month.nc'
	#ocean_extension='ocean_month_zold.nc'
	ocean_z_extension='ocean_month_zold.nc'
	#ocean_zold_extension='ocean_month_zold.nc'
        Berg_file=Berg_path+berg_extension
	ocean_file=Berg_path+ocean_extension
	ocean_file_z=Berg_path+ocean_z_extension

	#Specifying initial file
	replace_init=True
	if replace_init is True:
		Berg_path_init=Base_path + init_path
        	init_Berg_file=Berg_path_init+berg_extension
		init_ocean_file=Berg_path_init+ocean_extension
		init_ocean_file_z=Berg_path_init+ocean_z_extension

	#General flags
	rotated=True	

        #Load static fields
	#if Alistair_double_movie is False:
	(depth, shelf_area, ice_base, x,y, xvec, yvec)=load_static_variables(ocean_geometry_filename,ice_geometry_filename,ISOMIP_IC_filename,rotated,xy_from_zero=False)
	grounding_line=find_grounding_line_new(depth, shelf_area, ice_base, x,y, xvec, yvec)

	print("Grounding", grounding_line)
	#return


	
        ######################################################################################################################
        ################################  Plotting melt comparison  ##########################################################
        ######################################################################################################################

	if horizontal_movie is True:
                plot_anomaly=False
		filename=Berg_ocean_file
		print filename
		dir_slice_num=1
		#flipped=False ; flip_flag=''  ; fig_length=6 ; fig_height=12
		flipped=True  ; flip_flag='_flipped_'  ; fig_length=18; fig_height=4.5

		#field_name='mass_berg'  ; vmin=-0.0  ;vmax=200000. ; cmap='jet'
		#field_name='area_berg'  ; vmin=-0.0  ;vmax=1.0 ; cmap='jet'
		#field_name='spread_area'
		field_name='v' ;  vmin=-0.1  ;vmax=0.1 ; cmap='bwr'
		#field_name='u' ;  vmin=-0.1  ;vmax=0.1 ; cmap='bwr'  ;vanom=0.02
                #field_name='salt'  ; vmin=33.5  ; vmax=33.9  ;vdiff=0.02  ; vanom=0.02 ; cmap='jet'
                #field_name='temp'  ; vmin=-2.1  ; vmax=-1.1  ;vdiff=0.1   ; vanom=0.5  ;cmap='jet'  #vmax=-1.1
		direction='xy'
		print 'Starting to load the horizontal data!'
		data=load_and_compress_data(filename,field=field_name,time_slice='all',time_slice_num=-1,rotated=rotated, direction=direction ,dir_slice=None, dir_slice_num=dir_slice_num)
		print data.shape, shelf_area.shape
		#data=mask_ice(data,ice_base)

		#Configuring movie specs
		print 'Staring to make a freaking movie!'
		axes_fixed=True
		(x, y, data) = subsample_data(x, y, data,  axes_fixed, subsample_num=subsample_num )
		data[np.where(abs(data)>10**20.)]=np.nan
		if plot_anomaly is True:
			(data,cmap,field_name,vmin,vmax)=make_data_anomolous(data,vanom,field_name)

		frames_per_second=20
		output_filename='movies/' + exp_name + '/'  + field_name + '_' + direction + '_' +extension_name.split('.nc')[0]  + '_' + str(data.shape[0])+ flip_flag + 'frames' + '.mp4'
		ani_frame(x,y,data,output_filename,vmin,vmax,cmap, axes_fixed,fig_length=fig_length,fig_height=fig_height,resolution=200,\
		xlabel='x (km)',ylabel='y (km)',frames_per_second=frames_per_second, frame_interval=30, Max_frames=None,grounding_line=grounding_line,flipped=flipped, just_a_test=test_movie)


        ######################################################################################################################
        ################################  Plotting Cross Section      ########################################################
        ######################################################################################################################


        if cross_section_movie is True:
		axes_fixed=False
                plot_anomaly=False
                time_slice='all'
                #vertical_coordinate='z'  ;axes_fixed=True
                vertical_coordinate='layers'  #'z'
                #field_name='temp'  ; vmin=-2.0  ; vmax=1.0  ;vdiff=0.1   ; vanom=0.5  ; cmap='jet'
                #field_name='salt'  ; vmin=34  ; vmax=34.7  ;vdiff=0.02  ; vanom=0.02 ; cmap='jet' 
                #field_name='u'  ; vmin=-0.1  ; vmax=0.1  ;vdiff=0.02  ; vanom=0.02 ; cmap='bwr'
                field_name='v'  ; vmin=-0.1  ; vmax=0.1  ;vdiff=0.1  ; vanom=0.1 ; cmap='bwr'
                filename=Berg_ocean_file
                
                if vertical_coordinate=='z':
			if filename[-4]!='z':
	                        filename=filename.split('.nc')[0] + '_z.nc'
                if rotated is True:
                        direction='yz'
                        dist=yvec
                else:
                        direction='xz'
                        dist=xvec

		print 'Starting to load the vertical data!'
                data=load_and_compress_data(filename,field_name , time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
                elevation = get_vertical_dimentions(filename,vertical_coordinate, time_slice, time_slice_num=-1, direction=direction ,dir_slice=None, dir_slice_num=20,rotated=rotated)
                (y ,z ,data) =interpolated_onto_vertical_grid(data, elevation, dist, vertical_coordinate)
		print 'Interpolation complete', data.shape, y.shape, z.shape

		(y, z, data) = subsample_data(y, z, data,  axes_fixed, subsample_num=subsample_num)
		data[np.where(abs(data)>10**20.)]=np.nan

		if plot_anomaly is True:
			(data,cmap,field_name,vmin,vmax)=make_data_anomolous(data,vanom,field_name)

		if make_a_movie is True:
			frames_per_second=22;#*60
			print 'Staring to make a freaking movie!', 
			output_filename='movies/'  + exp_name + '/' +  field_name + '_' + direction + '_' + str(data.shape[0])+ 'frames' + '.mp4'
			ani_frame(y,z,data,output_filename,vmin,vmax,cmap, axes_fixed,fig_length=14,fig_height=6,resolution=200,\
				xlabel='y (km)',ylabel='depth (m)',frames_per_second=frames_per_second, frame_interval=30, Max_frames=None,just_a_test=test_movie)
		
		if save_pngs is True:
			output_folder= Berg_path + '/png_folder/' + field_name + '_' + direction +'_' +extension_name.split('.nc')[0] +  '_' + str(data.shape[0])+ 'frames' 
			print 'Staring to save images!', output_folder
			save_pngs_from_data(y,z,data,output_folder,vmin,vmax,cmap, axes_fixed,fig_length=14,fig_height=6,resolution=360, xlabel='y (km)',ylabel='depth (m)', Max_frames=None)
	


	if Alistair_double_movie is True:
		print 'Making movies in a whole new way!'
		print ocean_file
		
		#Upload the data
		onc = scipy.io.netcdf_file(ocean_file)
		onc_z = scipy.io.netcdf_file(ocean_file_z)
		snc = scipy.io.netcdf_file(Berg_file)
		
		#Pull the variables
		t = onc.variables['temp'][:,:,:,:]
		x = onc.variables['xh'][:]
		y = onc.variables['yh'][:]
		time = onc.variables['time']
		m = snc.variables['spread_mass'][:,:,:]
		melt = snc.variables['melt_m_per_year'][:,:,:]
		e = onc.variables['e'][:,:,:,:]
		
		if  ocean_file_z is not None:
			t_z = onc_z.variables['temp']
			e_z = create_e_with_correct_form(x, y ,time[:],  onc_z)
		else:
			t_z = None ; e_z = None

		if replace_init is True:
			(t, m , melt, e, t_z) = adjust_initial(t, m, melt, e,t_z,  init_ocean_file, init_ocean_file_z, init_Berg_file)

		#movie parameter:
		frame_interval=30
		frames_per_second=10
		resolution=300
		
		#fig=plt.figure(figsize=(fig_length,fig_height),facecolor='grey')
		#fig=plt.figure(figsize=(10,6))
		fig=plt.figure(figsize=(15,10))
		im = create_double_image(0,e,t,x,y,time,m,e_z, t_z, melt, grounding_line,depth)



		Number_of_images = e.shape[0];
		#for n in range(number_of_images):
		
		def update_img(n):
			#Updating figure for each frame
			print 'Frame number' ,n ,  'writing now.' 
			fig.clf()
			ax = fig.add_subplot(111,axisbg='gray')
			#(data_n , xn ,yn)=get_nth_values(n,data,x,y,axes_fixed)
			#im=plot_data_field(data_n,xn,yn,vmin,vmax,flipped=flipped,colorbar=True,cmap=cmap,title='',xlabel=xlabel,ylabel=ylabel,return_handle=True,grounding_line=grounding_line)
			im = create_double_image(n,e,t,x,y,time,m,e_z, t_z,melt, grounding_line, depth)
			
			return im


		ani = animation.FuncAnimation(fig,update_img,Number_of_images,interval=frame_interval)
		writer = animation.writers['ffmpeg'](fps=frames_per_second)

		output_filename='movies/' + exp_name + '/'  + 'Iceberg_movie' + '.mp4'
		ani.save(output_filename,writer=writer,dpi=resolution)
		#plt.show()
		print 'Movie saved: ' , output_filename



	

if __name__ == '__main__':
        main()
        #sys.exit(main())

