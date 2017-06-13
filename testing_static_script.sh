#Meridional Velocity
#./static_shelf_comparison.py -fields_to_compare=horizontal_comparison -field=VO -vmin=-0.1 -vmax=0.1 -vdiff=0.02 -time_slice=mean -time_slice_num=-1 -save_figure=True

#Zonal Velocity, UO
#./static_shelf_comparison.py -fields_to_compare=horizontal_comparison -field=UO -vmin=-0.1 -vmax=0.1 -vdiff=0.02 -time_slice=mean -time_slice_num=-1 -use_ALE=True -save_figure=True

#Sea Surface Temp
#./static_shelf_comparison.py -fields_to_compare=horizontal_comparison -field=SST -vmin=-2 -vmax=0 -vdiff=0.3 -time_slice=mean -time_slice_num=-1 -save_figure=True

#Sea Surface Salinity
#./static_shelf_comparison.py -fields_to_compare=horizontal_comparison -field=SSS -vmin=33.6 -vmax=34.2 -vdiff=0.3 -time_slice=mean -time_slice_num=-1 -save_figure=True

#Sea Surface Salinity
#./static_shelf_comparison.py -fields_to_compare=horizontal_comparison -field=SSS -vmin=33.6 -vmax=34.2 -vdiff=0.3 -time_slice=mean -time_slice_num=-1 -save_figure=True

#Rho mixed layer
#./static_shelf_comparison.py -fields_to_compare=horizontal_comparison -field=Rml -vmin=1026.8 -vmax=1027.6 -vdiff=0.1 -time_slice=mean -time_slice_num=-1 -file_type=prog -save_figure=True

#Mixed layer depth
./static_shelf_comparison.py -fields_to_compare=horizontal_comparison -field=e -vmin=0 -vmax=1 -vdiff=0.5 -time_slice=mean -time_slice_num=-1 -file_type=prog -save_figure=True
