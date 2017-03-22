function hexagon_xy(x0,y0,H,color_str,linewidth)


berg_count=1;
hold on
%plot(x0,y0,'o','linewidth',2 )
%plot(x0+(H.*cos(circ_ind)),y0+(H.*sin(circ_ind)),'k','linewidth',2 )

circ_ind=[0:0.1:2*pi];
S=(2/sqrt(3))*H;   %side of hexigon

if color_str=='grey'
    color_str=[0.4,0.4,0.4];
end
%elseif color_str=='light_grey'
%    color_str=[0.9,0.9,0.9];
%end


%line 1
plot([x0+S x0+H/sqrt(3)],[y0 y0+H],'Color',color_str,'linewidth',linewidth)
%line 2
plot([x0+H/sqrt(3) x0+H/sqrt(3)-S],[y0+H y0+H],'Color',color_str,'linewidth',linewidth)
%line 3
plot([x0+H/sqrt(3)-S x0-S],[y0+H y0],'Color',color_str,'linewidth',linewidth)
%line 4
plot([x0-S x0+H/sqrt(3)-S],[y0 y0-H],'Color',color_str,'linewidth',linewidth)
%line 5
plot([x0+H/sqrt(3)-S x0+H/sqrt(3)],[y0-H y0-H],'Color',color_str,'linewidth',linewidth)
%line 6
plot([x0+H/sqrt(3) x0+S],[y0-H y0],'Color',color_str,'linewidth',linewidth)






