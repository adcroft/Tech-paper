clear all
close all

%Flags
plot_hexagon_intersection=1;

if plot_hexagon_intersection==1
    hexagon_intersection
    clear all
    plot_hexagon_intersection=1;
end

fontsize=20;

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

outer_max=6;
inner_max=6;
berg_pos_x=zeros(outer_max*inner_max,1);
berg_pos_y=zeros(outer_max*inner_max,1);

y_ini=0;
x_ini=0;
R=H;
for outer=0:outer_max-1
    x_start=y_ini + (sqrt(3)*R*outer);
    y_start=mod(outer,2)*R;
    
    
    for inner = 0 : inner_max-1
        berg_count=berg_count+1;
        berg_pos_y(berg_count)=y_start+(inner*2*R);
        berg_pos_x(berg_count)=x_start;
        y0=y_start+(inner*2*R);
        x0=x_start;
        
        
        %Initialize matrix
        Matrix=zeros(N,N);
        
        
        %Define axes
        Ax=0*ones(length(x),1);
        Ay=0*ones(length(y),1);
        
        %Define 6 lines for linear programing (plus axes) - in order anticlockwise
        L1=-sqrt(3)*x + (y0+(sqrt(3)*(S+x0)));    %upper right
        L2= (y0+H)*ones(length(x),1);  %Top
        L3=sqrt(3)*x + (y0+(sqrt(3)*(S-x0)));  %Upper left
        
        L4=sqrt(3)*x + (y0-(sqrt(3)*(S+x0)));  %Lower right
        L5= (y0-H)*ones(length(x),1);  %Bottom
        L6=-sqrt(3)*x + (y0-(sqrt(3)*(S-x0)));   %Lower left
        
        
        A_hex=0;
        A1=0;
        A2=0;
        A3=0;
        A4=0;
        
        %Color in hexigon
        for i=1:N
            for j=1:N
                Condition1=(y(j)< L1(i)) & (y(j)<L2(i)) & (y(j)< L3(i)) & (y(j)> L4(i)) & (y(j)>L5(i)) & (y(j)> L6(i)); %in the hexagon
                First_Q= (y(j)>0) & (x(i)>0);
                Second_Q= (y(j)>0) & (x(i)<0);
                Third_Q= (y(j)<0) & (x(i)<0);
                Forth_Q= (y(j)<0) & (x(i)>0);
                if Condition1
                    Matrix(i,j)=1;
                    A_hex=A_hex+(dx^2);
                end
                if Condition1 & First_Q
                    Matrix(i,j)=1;
                    A1=A1+(dx^2);
                end
                if Condition1 & Second_Q
                    Matrix(i,j)=2;
                    A2=A2+(dx^2);
                    
                end
                if Condition1 & Third_Q
                    Matrix(i,j)=3;
                    A3=A3+(dx^2);
                end
                if Condition1 & Forth_Q
                    Matrix(i,j)=4;
                    A4=A4+(dx^2);
                end
            end
        end
        
        Matrix(find(Matrix==1))=A1/(A1+A2+A3+A4);
        Matrix(find(Matrix==2))=A2/(A1+A2+A3+A4);
        Matrix(find(Matrix==3))=A3/(A1+A2+A3+A4);
        Matrix(find(Matrix==4))=A4/(A1+A2+A3+A4);
        Matrix(find(Matrix==0))=NaN;
        Matrix2(find(Matrix>0))=mod(berg_count,4)+1;
        
        
        
    end
end

%%%%%%%%%%%%% Starting to plot  %%%%%%%%%%%%%%%%%
hold on
if plot_hexagon_intersection==1
    subplot(1,2,1)
end
colormap(jet)
h1=pcolor(x,y,Matrix2');
hold on
%plot(x,Ax)
%plot(x,Ax+1,'k')
%plot(x,Ax-1,'k')
%plot(Ay,y,'k')
%plot(Ay+1,y,'k')
%plot(Ay-1,y,'k')
set(h1, 'EdgeColor','none')
L=2;
axis([-S ((2*(inner_max))-4)*S -H 2*(outer_max)*H])
text(3.5,5,'(a)','fontsize',fontsize)
set(gca,'fontsize',fontsize)
xlabel('x (km)', 'fontsize',fontsize)
ylabel('y (km)', 'fontsize',fontsize)


%colorbar
%caxis([0 1])

Max_berg_count=berg_count
for berg_count = 1:Max_berg_count
    %hline11(i,j)=plot((x(i)/1000)+(((dx/2)/1000).*cos(circ_ind)),(y(j)/1000)+(((dx/2)/1000).*sin(circ_ind)),'k','linewidth',4);
    plot(berg_pos_x(berg_count)+(H.*cos(circ_ind)),berg_pos_y(berg_count)+(H.*sin(circ_ind)),'k','linewidth',2 )
    
    if berg_count>1
        for inner_count = 1:berg_count-1
            if sqrt(((berg_pos_y(berg_count)-berg_pos_y(inner_count))^2.) +((berg_pos_x(berg_count)-berg_pos_x(inner_count))^2.))<(2.1*H)
                plot([berg_pos_x(berg_count) berg_pos_x(inner_count)], [berg_pos_y(berg_count) berg_pos_y(inner_count)],':m','linewidth',2)
            end
        end
    end
end


