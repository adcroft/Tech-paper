clear all
close all

%Flags
plot_hexagon_intersection=1;


L=2;
dx=0.01;
%Initialize grid
x=[-L:dx:L];
y=x;


C1=0;C2=0;C3=0;C4=0;

H=0.4  %Height of hexigon

x0=0.1;  %x position of center of hexigon
y0=0.15;  %y position of center of hexigon
S=(2/sqrt(3))*H;   %side of hexigon

N=length(x);

for k=1:2
    
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
    
    
    
    if plot_hexagon_intersection==1
        hold on
        subplot(1,2,2)
        colormap(jet)
        h1=pcolor(x,y,Matrix');
        hold on
        plot(x,Ax,'k')
        plot(x,Ax+1,'k')
        plot(x,Ax-1,'k')
        plot(Ay,y,'k')
        plot(Ay+1,y,'k')
        plot(Ay-1,y,'k')
        set(h1, 'EdgeColor','none')
        L=2
        axis([-1 L -1 2])
        colorbar
        caxis([0 1])
        fontsize=20
        text(1.8,2.1,'(b)','fontsize',fontsize)
        set(gca,'fontsize',fontsize)
        xlabel('x (km)', 'fontsize',fontsize)
        ylabel('y (km)', 'fontsize',fontsize)
    end
    
end
