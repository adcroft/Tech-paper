clear all
close all

%Parameters
filename='Figures/P11B.jpg'
fontsize=20;

%Read in the image
img = imread(filename);



% set the range of the axes
% The image will be stretched to this.
L=6;
min_x = -L;
max_x = L;
min_y = -L;
max_y = L;

% Create data to plot over the image
Num_dots=5




L=6;
dx=0.01;
%Initialize grid
x=[-L:dx:L];
y=x;
circ_ind=[0:0.1:2*pi];


C1=0;C2=0;C3=0;C4=0;

H=0.4  %Height of hexigon
x0=0.1;  %x position of center of hexigon
y0=0.15;  %y position of center of hexigon
S=(2/sqrt(3))*H;   %side of hexigon

N=length(x);
Matrix2=zeros(N,N);

berg_count=0;

outer_max=7;
inner_max=4;
berg_pos_x=zeros(outer_max*inner_max,1);
berg_pos_y=zeros(outer_max*inner_max,1);

y_ini=0;
x_ini=0;
R=H;
%for outer=0:outer_max-1
for outer=-outer_max-1:outer_max+1
    x_start=y_ini + (sqrt(3)*R*outer);
    y_start=mod(outer,2)*R;
    
    
    %    for inner = 0 : inner_max-1
    for inner = -inner_max : inner_max-1
        berg_count=berg_count+1;
        berg_pos_y(berg_count)=y_start+(inner*2*R);
        berg_pos_x(berg_count)=x_start;
        
    end
end


%Deciding which ones to plot
bad_list=[];
bad_list=[16 135 134 130 128 129 121 122 126 119 111 112 113 114 105 104 95 96 97 96 88 89 80 81 72 73 64 65 24 16 17 9 10 1 2 3 5 6 7 8 18 25 49 57 66 82 90 98 106 32 120 48 127 136 131 115];
new_count=0
Max_berg_count=berg_count
for berg_count = 1:Max_berg_count
    in_bad_list=0
    for k=bad_list
        if k==berg_count
            in_bad_list=1
        end
    end
    if in_bad_list==0
        new_count=new_count+1
        new_berg_pos_x(new_count)=berg_pos_x(berg_count);
        new_berg_pos_y(new_count)=berg_pos_y(berg_count);
        new_berg_pos_num(new_count)=berg_count
    end
end
berg_pos_x=new_berg_pos_x;
berg_pos_y=new_berg_pos_y;
Max_berg_count=length(berg_pos_x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Plotting


figure;
%subplot(1,2,2)
% Flip the image upside down before showing it
imagesc([min_x max_x], [min_y max_y], flipud(img));


hold on;
%plot(x,y,'b-*','linewidth',1.5);
grey=[0.8,0.8,0.8];
light_grey=[0.6,0.6,0.6];

for berg_count = 1:Max_berg_count
    %hline11(i,j)=plot((x(i)/1000)+(((dx/2)/1000).*cos(circ_ind)),(y(j)/1000)+(((dx/2)/1000).*sin(circ_ind)),'k','linewidth',4);
    plot(berg_pos_x(berg_count)+(H.*cos(circ_ind)),berg_pos_y(berg_count)+(H.*sin(circ_ind)),'Color', 'k','linewidth',1 )
    hexagon_xy(berg_pos_x(berg_count),berg_pos_y(berg_count),H,'grey',1.5)
    %plot(berg_pos_x(berg_count),berg_pos_y(berg_count),'o','linewidth',5)
   % text(berg_pos_x(berg_count),berg_pos_y(berg_count),num2str(berg_count),'color','r')
   % text(berg_pos_x(berg_count),berg_pos_y(berg_count),num2str(new_berg_pos_num(berg_count)),'color','r')
    new_berg_pos_num(new_count)
    mult=0.15*H;
    fill((berg_pos_x(berg_count)+mult.*cos(circ_ind)),(berg_pos_y(berg_count)+mult.*sin(circ_ind)),'b');
    
    if berg_count>1
        for inner_count = 1:berg_count-1
            if sqrt(((berg_pos_y(berg_count)-berg_pos_y(inner_count))^2.) +((berg_pos_x(berg_count)-berg_pos_x(inner_count))^2.))<(2.1*H)
                plot([berg_pos_x(berg_count) berg_pos_x(inner_count)], [berg_pos_y(berg_count) berg_pos_y(inner_count)],'Color','m','linestyle',':','linewidth',1.5)
            end
        end
    end
end

% set the y-axis back to normal.
set(gca,'ydir','normal');
set(gca,'xtick',[],'ytick',[])

%imagesc(img);
%xlabel('Raster Column');
%ylabel('Raster Row');
%colormap(gray);


ylim([-4 4.5])


% subplot(1,2,1)
% %imagesc([min_x max_x], [min_y max_y], img);
% imagesc([min_x max_x], [min_y max_y], flipud(img));
% ylim([-4 4.5])
% set(gca,'ydir','normal');
% set(gca,'xtick',[],'ytick',[])

axes('Position',[.75 .75 .16 .16])
box on
hexagon_intersection2