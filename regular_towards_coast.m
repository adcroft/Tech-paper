clear all
close all
load 'blue_white.mat'
path_name='/Users/alon/Desktop/files/Icebergs_clusters/Matlab_scripts/Box_experiment';
cd(path_name)

%filename='Viscous_spring_contact'
%filename='Different_sizes_Viscous_spring_contact'
%filename='Including_big_Bergs_with_Jacobi_5';
%filename='All_bonds_Including_big_Bergs_with_Jacobi_5';
%filename='Half_circle_exp_with_bergs_Jacobi_5';
%filename='Half_circle_exp_with_little_bergs_Jacobi_5';
%filename='Two_interacting_bergs_springs_for_land_bonds_interaction';
%filename='Test_using_L_eff';
%filename='Test_using_L_eff_alpha_max_10';
%filename='Box_using_L_eff_alpha_max_10';
%filename='Conc_on_Box_using_L_eff_alpha_max_10';
%filename='Convex__Conc_on_alpha_max_10';
%filename='Convex__Conc_on_alpha_mu_0';
%filename='Convex__Conc_on_alpha_mu_1';
%filename='Convex__Conc_on_alpha_mu_3';
%filename='Convex_interaction_factor_mu_3';
%filename='Convex_interaction_factor_mu_3_gamma_0_3';
%filename='Convex_interaction_factor_mu_3_gamma_0_3_take2';
%filename='Convex_interaction_factor_mu_3_gamma_0_3_take3';
%filename='Convex_interaction_factor_mu_5_gamma_1_take4';
%filename='Convex_interaction_factor_mu_5_gamma_1_take5';
%filename='Reverse_wind_dir';
%filename='Circular_wind_dir';
%filename='Sea_surface_slope';
%filename='Gridded_bergs_towards_wall';
% %filename='Corriolis_Gridded_bergs_towards_wall';
% %filename='Gridded_bergs_towards_wall_again';
% filename='Gridded_bergs_towards_wall_again2';
% filename='Gridded_bergs_towards_wall_again_stiffer';
% filename='Gridded_bergs_towards_wall_again_stiffer_2';
% %filename='Gridded_bergs_towards_wall_again_stiffer_Jacobi';
% filename='Gridded_bergs_towards_wall_again_stiffer_Jacobi';
% filename='Gridded_bergs_towards_wall_again_stiffer_Jacobi_verlet';
% filename='Gridded_bergs_towards_wall_again_stiffer_Jacobi_verlet_with_contact_correction';
% filename='Two_particle_spring';
% filename='Two_particle_Jacobi';
% filename='Big_and_small_bergs_curved_coast_again';
filename='New_test_curved_coast';
% filename='New_test_curved_coast_up_pia';
%filename='New_test_curved_coast_up_pia_and_q_ia';
%filename='From_saved_test';
%filename='Footloose_test1_Linit_100km'


circ_ind=[0:0.1:2*pi];
fontsize=25;
init_time=1;
time_step=80;
final_time_ind=0;   %final_time=0 to automatically do until end of list

plot_the_area=0;
plot_dots_and_circ=1;
plot_the_bonds=1;
plot_land_each_time=0;
plot_L_eff=1;
make_a_movie=0;
plot_tails=1;

%Defining the grid
Lx=500*1000;%100*1000; %Length of the domain
Ly=500*1000;%120*1000 %Width of the domain
dx=10*1000;  %ocean grid spacing in x
dy=dx; %ocean grid spacing in y
Nx=(Lx/dx)+1; %number of positions in the x grid
Ny=(Ly/dy)+1; %number of positions in the y grid
x=([1:Nx]-1).*dx;
y=([1:Ny]-1).*dy;
fontsize=14;


Hov_Conc=zeros(302,101);

cd data
cd(filename)

load 'iceberg_mat_data.mat'  %berg_pos_x berg_pos_y berg_pos_L berg_pos_time
load 'environmental_variables.mat'  %Land eta U_w V_w x y Num_bergs
load 'iceberg_time_ind.mat'
load 'Parameters.mat'

Land(Nx-1,:)=0;


%Defining the grid for concentration
dx=2*1000;  %ocean grid spacing in x
dy=dx;
Nx2=(Lx/dx)+1; %number of positions in the x grid
Ny2=(Ly/dy)+1; %number of positions in the y grid
x2=([1:Nx2]-1).*dx;
y2=([1:Ny2]-1).*dy;
W=berg_pos_L(:,1);

if final_time_ind==0;
    final_time_ind=length(berg_pos_time);
end

%time_ind_list=[1 80 160 240];  %'Convex_interaction_factor_mu_5_gamma_1_take5'
time_ind_list=[1 200 400 600];  %'Big_and_small_bergs_curved_coast_again'
%time_ind_list=[1 150 300 450];%'Gridded_bergs_towards_wall_again_stiffer_Jacobi_verlet_with_contact_correction';

%time_ind_list=[1 10 20 30 40 50]; %'Footloose_test1_Linit_100km'
%time_ind_list=[1 10 25 40]; %'Footloose_test1_Linit_100km'


cd ../
cd ../


figure;
colormap(blue_white)

letter_list=['(a)';'(b)';'(c)';'(d)'];
letter_list=[' ';' ';' ';' '];
hold on
count=0;
plot_count=0;
for time_count=time_ind_list;
    time_count_temp=time_count;
    if time_count==1
        time_count_temp=0;
    end
    plot_count=plot_count+1;
    time=berg_pos_time(time_count);
    
    subplot(2,2,plot_count)
    hold on
    
    pcolor(x/1000,y/1000,eta');shading('interp');%colorbar;
    set(gca,'color','k','fontsize',fontsize)
    caxis([-0 1]);
    xlabel('x axis (km)');
    ylabel('y axis (km)');
    axis([0 Lx/1000 0 Ly/1000]);drawnow
    
    %axis([10 600 200 480]);drawnow
    %axis([10 550 200 400]);drawnow
    box on
    grid off
    title(['Day ' num2str(round(time/60/60/24))],'fontsize',fontsize);
    xlabel('x (km)','fontsize',fontsize)
    ylabel('y (km)','fontsize',fontsize)
    set(gca,'fontsize',fontsize)
    text(500,505,letter_list(plot_count,:),'fontsize',fontsize)
    
    %Plotting the land
    hold on
    for i=1:Nx
        for j=1:Ny
            if Land(i,j)==1
                count=count+1;
                %hline10(i,j)=plot(x(i)/1000,y(j)/1000,'k*');
                mult=5;
                hline11(i,j)=fill((x(i)/1000)+(((mult*dx/2)/1000).*cos(circ_ind)),(y(j)/1000)+(((mult*dx/2)/1000).*sin(circ_ind)),'k');
                %hline11(i,j)=plot((x(i)/1000)+(((dx/2)/1000).*cos(circ_ind)),(y(j)/1000)+(((dx/2)/1000).*sin(circ_ind)),'k','linewidth',4);
            end
        end
    end
    
    
    if plot_the_area==1
        Conc=zeros(Nx2,Ny2);
        for i=1:Nx2
            for j=1:Ny2
                alpha=3;
                R=sqrt(  ((berg_pos_x(:,time_count)-x2(i)).^2) +  ((berg_pos_y(:,time_count)-y2(j)).^2));
                Conc(i,j)=sum(pi.*((berg_pos_L(:,time_count)./2).^(2)).*Kernal(R,berg_pos_L_eff(:,time_count),alpha));
                % R=sqrt(  ((berg_pos_x(:,time_count)-x(i)).^2) +  ((berg_pos_y(:,time_count)-y(j)).^2));
                % Conc(i,j)=sum(pi.*(L(:,1)./2).^(2).*Kernal(R,L,alpha));
            end
        end
        
        
        if max(max(Conc))>1.1
            ['There is a very large Conc at time = ' num2str(round(10*time/60/60/24)/10) ' days']
        end
        %Conc(find(Land==1))=NaN;
        hline5=pcolor(x2/1000,y2/1000,Conc');shading('interp');caxis([0. 1]);colorbar
        
        
        
        
        
        
        
    end
    
    if plot_dots_and_circ==1
        for berg_count=1:Num_bergs
            %    hline(:,berg_count)=plot(berg_pos_x(berg_count,time_count)/1000,berg_pos_y(berg_count,time_count)/1000,'*w');
            %hline2(:,berg_count)=plot((berg_pos_x(berg_count,time_count)/1000)+(((berg_pos_L(berg_count,time_count)/2)/1000).*cos(circ_ind)),(berg_pos_y(berg_count,time_count)/1000)+(((berg_pos_L(berg_count,time_count)/2)/1000).*sin(circ_ind)),'g');
            hline2(:,berg_count)=fill((berg_pos_x(berg_count,time_count)/1000)+(((berg_pos_L(berg_count,time_count)/2)/1000).*cos(circ_ind)),(berg_pos_y(berg_count,time_count)/1000)+(((berg_pos_L(berg_count,time_count)/2)/1000).*sin(circ_ind)),'w');
            
            if plot_tails==1
                N_tail=25;
                tail_count=0;
                if time_count-N_tail>0
                    x_tail=zeros(N_tail,1);
                    for k_tail=1:1:N_tail
                        tail_count=tail_count+1;
                        x_tail(tail_count,1)=berg_pos_x(berg_count,time_count-k_tail)/1000;
                        y_tail(tail_count,1)=berg_pos_y(berg_count,time_count-k_tail)/1000;
                        tail_mult=1/10;
                    end
                    hline62(:,berg_count)=plot(x_tail,y_tail,'w','linewidth',0.5);
                    %hline62(:,berg_count)=fill((berg_pos_x(berg_count,time_count-k_tail)/1000)+(((tail_mult*berg_pos_L(berg_count,time_count-k_tail)/2)/1000).*cos(circ_ind)),(berg_pos_y(berg_count,time_count)/1000)+(((tail_mult*berg_pos_L(berg_count,time_count)/2)/1000).*sin(circ_ind)),'w');
                    %hline62(:,berg_count)=plot((berg_pos_x(berg_count,time_count-k_tail)/1000)+(((tail_mult*berg_pos_L(berg_count,time_count-k_tail)/2)/1000).*cos(circ_ind)),(berg_pos_y(berg_count,time_count)/1000)+(((tail_mult*berg_pos_L(berg_count,time_count)/2)/1000).*sin(circ_ind)),'w');
                end
            end
        
        if plot_L_eff==1
            hline8(:,berg_count)=plot((berg_pos_x(berg_count,time_count)/1000)+(((berg_pos_L_eff(berg_count,time_count)/2)/1000).*cos(circ_ind)),(berg_pos_y(berg_count,time_count)/1000)+(((berg_pos_L_eff(berg_count,time_count)/2)/1000).*sin(circ_ind)),'w');
        end
    end
end

if plot_the_bonds==1
    %Plotting the bonds
    for berg_count=1:Num_bergs
        for pair_berg=1:berg_count
            if (Bond(pair_berg,berg_count)==1)
                hline4(:,berg_count,pair_berg)=plot([berg_pos_x(berg_count,time_count)/1000 berg_pos_x(pair_berg,time_count)/1000],[berg_pos_y(berg_count,time_count)/1000 berg_pos_y(pair_berg,time_count)/1000],'m','linewidth',2);
            end
        end
        if (L_Bond(pair_berg,berg_count)==1)
            hline5(:,berg_count,pair_berg)=plot([berg_pos_x(berg_count,time_count)/1000 berg_pos_x(pair_berg,time_count)/1000],[berg_pos_y(berg_count,time_count)/1000 berg_pos_y(pair_berg,time_count)/1000],'r','linewidth',3);
        end
    end
end

%   hline3=text((Lx/2/1000)-(Lx/20/1000),(Ly/1000)+(Ly/20/1000),['time = ' num2str(round(10*time/60/60/24)/10) ' days'],'fontsize',fontsize);
drawnow


end

'About to save figure',
saveas(gcf,'/Users/alon/Desktop/files/Icebergs_clusters/Towards_Publication/Tech_paper/Github_stuff/Tech-paper/Figures/Rregular_towards_coast.png')





