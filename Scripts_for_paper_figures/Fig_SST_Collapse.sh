cd ../
./snapshots.py -extension=ocean_month.nc -plot_horizontal_field=True -use_ALE=True  -time_ind1=14 -time_ind2=29 -time_ind3=59 -cmap=jet -field=temp -vmin=-1.8 -vmax=-1.4  -use_days_title=False -mask_using_bergs=True -dir_slice_num=1 -colorbar_units='(deg C)' -use_Mixed_Melt=False -dashed_num=12 -plot_second_colorbar=True -second_colorbar_units='(m)' -use_Revision=True -save_figure=True
