cd ../
./snapshots.py -extension=ocean_month.nc -plot_horizontal_field=False -use_ALE=True  -time_ind1=14 -time_ind2=29 -time_ind3=59 -cmap=jet -field=temp -vmin=-2.0 -vmax=1.0  -use_days_title=False -mask_using_bergs=True -plot_anomaly=False -vanom=0.3 -vertical_coordinate=layers -xmin=130.0 -xmax=430.0 -dir_slice_num=12 -colorbar_units='(deg C)' -use_Mixed_Melt=True -save_figure=True
