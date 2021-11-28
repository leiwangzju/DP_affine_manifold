
%%%Average Consensus with Manifold Dependency%%%%
%%%Trejectories In the Mean and Mean square sense %%%%%%%%%%%
%%% 10 agents%%%
%%% Laplace Mechanism %%%%

close all
clear all
clc

n=10; %% number of nodes;

%%define the neighbor sets
S_N=zeros(n,4);
S_N(1,:) = [2;10;0;0];
S_N(2,:) = [1;3;5;10];
S_N(3,:) = [2;4;0;0];
S_N(4,:) = [3;5;0;0];
S_N(5,:) = [2;4;6;7];
S_N(6,:) = [5;7;0;0];
S_N(7,:) = [5;6;8;10];
S_N(8,:) = [7;9;0;0];
S_N(9,:) = [8;10;0;0];
S_N(10,:) = [1;2;7;9];


%%% Each edge is assigned with the same weight 1/4
w = 1/4;

%%%Define Laplacian matrix
L = zeros(n,n);
for i=1:n
    for j=1:4
        if S_N(i,j) ~= 0
            L(i,S_N(i,j)) = -w;
            L(i,i) = L(i,i) - L(i,S_N(i,j));
        end
    end
end

%%%private dataset%%%
d = [10;100;20;-30;-20;-60;70;0;80;-20];

average = 15;

T=30; %%Running steps

S_num = 500; %%Sampling time

x_mse = zeros(n,T+1); %mean square error
e_m = zeros(n,T+1); %mean error

g_max = 3;
Error_MS = zeros(T+1,g_max);

for g=1:g_max
    %%%Parameter Design %%%%
    %%privacy budgets %%%
    epsilon = 10^(1-g); %%try different values 0.001,0.01, 0.1
    
    
    mu = 1;
    

    sigma_gamma = mu/epsilon;
    
    for s=1:S_num
        %----Generate standard Gaussian noise--------
        etab = zeros(n,1);
        for i=1:n
            etab(i) = laprnd(1, 1, 0, 1);
        end
        
        %----Average Consensus Update---
        y = zeros(n,T+1); %Communication message
        x = zeros(n,T+1); %Node states
        x_error = zeros(n,T+1); %Computation errors
        
        
        
        x(:,1) = d; %Initial states
        y(:,1) = x(:,1) + sigma_gamma*etab; %Initial communication message
        x_error(:,1) = x(:,1) - average*ones(n,1); %Initial computation error
        x_mse(:,1) = x_mse(:,1) + x_error(:,1).*x_error(:,1); %mean square error
        e_m(:,1) = e_m(:,1) + x_error(:,1); %mean error
        
        for t=1:T
            for i=1:n
                x(i,t+1) = x(i,t) - L(i,:)*y(:,t); %state update
                y(i,t+1) = x(i,t+1) + sigma_gamma*etab(i); %communication message update
                x_error(i,t+1) = x(i,t+1)- average; %Computation error update
                x_mse(i,t+1) = x_mse(i,t+1) + x_error(i,t+1)*x_error(i,t+1); %mean square error
                e_m(i,t+1) = e_m(i,t+1) + x_error(i,t+1); %mean error
            end
        end
    end
    
   %Total computation error in the mean square 
    for i=1:n
        Error_MS(:,g) = Error_MS(:,g) + x_mse(i,:)'/S_num;
    end
    
    %plot computation error in the mean sqaure
    figure(g)
    grid on
    T_vec = 0:T;
    for i=1:n
        plot(T_vec,e_m(i,:)/S_num,'LineWidth',1)
        hold on
    end
    
    xlabel('$t$','interpreter','latex')
    ylabel('$\mathrm{E}\, x_i(t)-x^{\star}$','interpreter','latex')
    %legend('\epsilon=10^{1-3}','\epsilon=10^{-2}','\epsilon=10^{-1}')
    grid on
    
    
    %plot computation error in the mean sqaure
    
%     figure(g_max+g)
%     grid on
%     T_vec = 0:T;
%     for i=1:n
%         plot(T_vec,x_mse(i,:)/S_num,'LineWidth',1)
%         hold on
%     end
%     
%     xlabel('$t$','interpreter','latex')
%     ylabel('$\mathrm{E}\,|x_i(t)-x^{\star}|^2$','interpreter','latex')
%     %legend('\epsilon=10^{1-3}','\epsilon=10^{-2}','\epsilon=10^{-1}')
%     grid on
    
    

    
end

MS_Upper_Bound = [n*1*ones(T+1,1),n*10^2*ones(T+1,1),n*100^2*ones(T+1,1)];
figure(7)
for g=1:g_max
    plot(T_vec,Error_MS(:,g),'LineWidth',1)
    hold on
end
xlabel('$t$','interpreter','latex')
ylabel('$\mathrm{E}\,\|\mathbf{x}(t)-\mathbf{1}_n\otimes x^{\star}\|^2$','interpreter','latex')
legend('Privacy budget 1: $10^{0}$','Privacy budget 2: $10^{-1}$','Privacy budget 3: $10^{-2}$','interpreter','latex')
grid on

for g=1:g_max
    hold on
    plot(T_vec,MS_Upper_Bound(:,g),'LineWidth',1)
end