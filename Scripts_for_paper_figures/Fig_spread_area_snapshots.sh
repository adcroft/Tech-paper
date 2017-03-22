cd ../
./snapshots.py -extension=icebergs_month.nc -plot_horizontal_field=True -use_ALE=True -time_ind1=14 -time_ind2=29 -time_ind3=59 -cmap=Blues -field=spread_area -vmin=0.0 -vmax=1.0  -use_days_title=False -mask_using_bergs=False -dir_slice_num=1 -colorbar_units='(non dim)' -use_Mixed_Melt=True -save_figure=True
