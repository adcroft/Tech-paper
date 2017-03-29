clear all
close all

plot_second_plot=1;


if plot_second_plot==1
    FigHandle=figure
    set(FigHandle, 'Position', [50, 50, 1200, 500]);
end


L=2;
dx=0.01;
%Initialize grid
x=[-L:dx:L];
y=x;

%Circle 1
xi_vec=[-0.05 -0.05 ]
yi_vec=[-0.05 -0.05]
%H=0.4
Di_vec=[0.75 0.75];

%Circle 2
%xj_vec=[0.9 1.3]
%yj_vec=[0.6 1.3]
R_vec=[1.1 1.8]
theta=40*pi/180;
xj_vec=[xi_vec(1)+R_vec(1)*cos(theta) xi_vec(1)+R_vec(2)*cos(theta)]
yj_vec=[yi_vec(1)+R_vec(1)*sin(theta) yi_vec(1)+R_vec(2)*sin(theta)]

%H=0.4
Dj_vec=[0.9 0.6] ;

font_seq=['L_{ij}- d_{ij}';'d_{ij}- L_{ij}'];
letters=['(a)';'(b)'];
colorvec=['rb']

for k=[1 2]
    F1=subplot(1,2,k)
    %Circle 1
    xi=xi_vec(k)
    yi=yi_vec(k)
    Di=Di_vec(k)
    
    %Circle 2
    xj=xj_vec(k)
    yj=yj_vec(k)
    Dj=Dj_vec(k);
    
    
    
    circ_ind=[0:0.1:2*pi];
    circ_ind2=[0:0.1:3*pi];
    
    hold on
    %Plotting circle 1
    fill(xi+(Di/20.*cos(circ_ind)),yi+(Di/20.*sin(circ_ind)),'m' )
    plot(xi+(Di.*cos(circ_ind2)),yi+(Di.*sin(circ_ind2)),'k','linewidth',2 )
    
    %Plotting circle 2
    fill(xj+(Dj/20.*cos(circ_ind)),yj+(Dj/20.*sin(circ_ind)),'m' )
    plot(xj+(Dj.*cos(circ_ind2)),yj+(Dj.*sin(circ_ind2)),'k','linewidth',2 )
    
    
    %Line through intersection
    theta_ij=atan((yi-yj)/(xi-xj))
    %plot([xi+Di*cos(theta_ij)], [yi+Di*sin(theta_ij)],'*')
    %plot([xj-Dj*cos(theta_ij)], [yj-Dj*sin(theta_ij)],'r*')
    plot([xj-Dj*cos(theta_ij) xi+Di*cos(theta_ij)],[yj-Dj*sin(theta_ij) yi+Di*sin(theta_ij)],'b','linewidth',2)
    
    
    
    
    %Plotting radius 1
    %theta_i=140*pi/180;
    theta_i=theta_ij+(pi/2)%140*pi/180;
    plot([xi xi+Di*cos(theta_i)],[yi yi+Di*sin(theta_i)],'b','linewidth',2)
 
    
    %Plotting radius 2
    %theta_j=140*pi/180;
    theta_j=theta_ij+(pi/2)%140*pi/180;
    plot([xj xj+Dj*cos(theta_j)],[yj yj+Dj*sin(theta_j)],'b','linewidth',2)
    
    
    %Text i
    eps8a=0.06;eps8b=0.1;eps8c=0.15
    text(xi-eps8c, yi+0*eps8b ,'i', 'fontsize',20,'color','m')
    %Text j
    text(xj+eps8a, yj+eps8b ,'j', 'fontsize',20,'color','m')
    
    
    eps=0.02;
    %Text i
    text(xi+Di/2*cos(theta_i)+eps, yi+Di/2*sin(theta_i) ,'R_{i}', 'fontsize',20)
    %Text j
    text(xj+Dj/2*cos(theta_j)+eps, yj+Dj/2*sin(theta_j) ,'R_{j}', 'fontsize',20)
    
    %Text ij
    eps2a=0.06
    eps2b=0.17
    %text(0.5*((xj-Dj*cos(theta_ij)) +(xi+Di*cos(theta_ij)))-eps2a, 0.5*(yj-Dj*sin(theta_ij)+ yi+Di*sin(theta_ij))-eps2b ,'|L_{ij}-d_{ij}|', 'fontsize',20)
    text(0.5*((xj-Dj*cos(theta_ij)) +(xi+Di*cos(theta_ij)))-eps2a, 0.5*(yj-Dj*sin(theta_ij)+ yi+Di*sin(theta_ij))-eps2b ,font_seq(k,:), 'fontsize',20)
    
    %Distance between elements (dij)
    eps3=0.1
    D0=(max(Di,Dj)+eps3);
    %plot([xi+Di*cos(theta_i) xj+Dj*cos(theta_j)],[yi+Di*sin(theta_i)   yj+Dj*sin(theta_j)],'b','linewidth',2)
    plot([xi+D0*cos(theta_i) xj+D0*cos(theta_j)],[yi+D0*sin(theta_i)   yj+D0*sin(theta_j)],'b','linewidth',2)
    
    %Text dij
    eps4=0.2
    text(0.5*(xi+D0*cos(theta_i)+ xj+D0*cos(theta_j)), 0.5*(yi+D0*sin(theta_i)  + yj+D0*sin(theta_j))+eps4 ,'d_{ij}', 'fontsize',20)

    
    
    %Connecting centers to d_ij line
     plot([xi xi+D0*cos(theta_i)],[yi yi+D0*sin(theta_i)],'b:','linewidth',1)
     plot([xj xj+D0*cos(theta_j)],[yj yj+D0*sin(theta_j)],'b:','linewidth',1)

     
     %Arrow
     eps5=0.1;
     A=0.3;
     %annotation('arrow', [xi-(eps5*cos(theta_i))  xi-(eps5*cos(theta_i)) - A*cos(theta_ij)] ,  [ yi-(eps5*sin(theta_i)) yi -(eps5*sin(theta_i))- A*sin(theta_ij) ])
     %annotation('arrow', [xi-(eps5*cos(theta_i))  xi-(eps5*cos(theta_i)) - A*cos(theta_ij)] ,  [ yi-(eps5*sin(theta_i)) yi -(eps5*sin(theta_i))- A*sin(theta_ij) ])
     p1=xi-(eps5*cos(theta_i))
     q1=yi-(eps5*sin(theta_i))
     p2=xj-(eps5*cos(theta_j))
     q2=yj-(eps5*sin(theta_j))

     
    R=.6
    if k==2
        R=-R; 
    end
    quiver(p1,q1, -R*cos(theta_ij),-R*sin(theta_ij),'MaxHeadSize',3,'linewidth',3,'color',colorvec(k))
    quiver(p2,q2, R*cos(theta_ij),R*sin(theta_ij),'MaxHeadSize',3,'linewidth',3,'color',colorvec(k))
  
     
    %Text for forces
    eps6a=0.1;    eps6b=0.35; R=.4;
    if k==1
    text(xi-eps6a, yi -eps6b ,'(F_{e})_{ij}', 'fontsize',20,'color',colorvec(k))
    text(xj- eps6a + R*cos(theta_ij), yj -eps6b+ R*sin(theta_ij) ,'(F_{e})_{ji}', 'fontsize',20,'color',colorvec(k))
    end
    if k==2
        text(xi-eps6a + R*cos(theta_ij), yi -eps6b+ R*sin(theta_ij) ,'(F_{e})_{ij}', 'fontsize',20,'color',colorvec(k))
        text(xj- eps6a , yj -eps6b ,'(F_{e})_{ji}', 'fontsize',20,'color',colorvec(k))
    end
       
     
     
  
    %Plotting the gird
    
    %Define axes
    Ax=0*ones(length(x),1);
    Ay=0*ones(length(y),1);
  %  plot(x,Ax,'k:','linewidth',2)
  %  plot(x,Ax+1,'k:','linewidth',2)
    plot(x,Ax+2,'k','linewidth',2)
    plot(x,Ax-1,'k','linewidth',2)
  %  plot(Ay,y,'k:','linewidth',2)
  %  plot(Ay+1,y,'k:','linewidth',2)
    plot(Ay+2,y,'k','linewidth',2)
    plot(Ay-1,y,'k','linewidth',2)
    axis('off')
    axis([-1 L -1 2])
    caxis([0 1])
    %text(L-L/20,2.1,letters(k,:),'fontsize',20)
    text(-1,2.1,letters(k,:),'fontsize',20)

    
end
