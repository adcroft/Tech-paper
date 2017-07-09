cd ../
./snapshots.py -extension=ocean_month.nc -plot_horizontal_field=True -use_ALE=True  -time_ind1=14 -time_ind2=29 -time_ind3=59 -cmap=jet -field=temp -vmin=-1.8 -vmax=-1.3  -use_days_title=False -mask_using_bergs=True -dir_slice_num=1 -colorbar_units='(deg C)' -simulation='Drift' -use_Mixed_Melt=True -plot_second_colorbar=True -use_Revision=True -second_colorbar_units='(m)' -save_figure=True
