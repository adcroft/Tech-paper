cd ../
./snapshots.py -extension=ocean_month.nc -plot_horizontal_field=False -use_ALE=True  -time_ind1=14 -time_ind2=29 -time_ind3=59 -cmap=jet -field=temp -vmin=-2.0 -vmax=1.0  -use_days_title=False -mask_using_bergs=True -plot_anomaly=False -vanom=0.3 -vertical_coordinate=layers -xmin=450.0 -xmax=750.0 -dir_slice_num=10 -colorbar_units='(deg C)' -save_figure=True
