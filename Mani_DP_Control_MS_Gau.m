
%%%Cloud-based control with Manifold Dependency for vehicles%%%%
%%%Mean-square errors with 500 samples %%%%%%%%%%%
%%% Gaussian Mechanism %%%%

close all
clear all
clc

%%--- define samples number ---%%
S_num = 500;

%%---define samping period T_s---%%
T_s=0.1;

%%---define system dimension n---%%
n=2;

%%---define running time T---%%
T = 200;

%%---define system matrix---%%
A= [1, T_s; 0,1];
B = [T_s^2/2;T_s];
C = [1,0];


%%---define closed-loop signals---%%
x = zeros(n,T+1); %plant state
x_r= zeros(n,T+1); %reference
x_hat= zeros(n,T+1); %observe state



%%---Define random noise for privacy preservation---%%

%%---Define correlation model D*x_1 + b = 0---%%
D = zeros(T-1,T);
for t=1:T-1
    D(t,t) = 1;
    D(t,t+1) = -1;
end

%%---Define D_bot satisfying D*D_bot = 0 and rank(D;D_bot')=rows---%%
D_bot = ones(T,1);

%%--- verify D_bot ---%%
D*D_bot
rank([D;D_bot'])

%%--- define the manifold M(x) = F*x + gamma ---%%
F = eye(T);

%%--- Design following Algorithm 1 ---%%
r = 1;

Lambda_bar  = F*D_bot;

%% ---compute Moore–Penrose inverse of Lambda_bar---%%
Lambda_bar_inv = inv(Lambda_bar'*Lambda_bar)*Lambda_bar';


%% --- compute O_ij ---%%
O_ij = F*ones(T,1);

%% --- compute sensitivity ---
S = Lambda_bar_inv*O_ij;

Delta_G = abs(S);

for g=1:3
    %%%Parameter Design %%%%
    %%privacy budgets %%%
    epsilon = 10^(1-g); %%try different values 0.001,0.01, 0.1
    
    delta = 0.01; %%delta is fixed
    
    mu = 1;
    
    
    %%%---compute the function R(epsilon,delta)---%%%
    syms y
    
    Q_inv = solve((1/(sqrt(2*pi)))*int(exp(-0.5*y*y),y,y,Inf)==delta);  %%the value of inverse of Q with delta
    
    %Phi_inv = solve((1/(sqrt(2*pi)))*int(exp(-0.5*y*y),y,-Inf,y)==delta);
    
    kappa_ep = (double(Q_inv) + sqrt(double(Q_inv)*double(Q_inv)+2*epsilon))/(2*epsilon); %%return the value of function kappa with values epsilon and delta.
    
    sigma = mu*Delta_G*kappa_ep;
    
    
    
    
    %----initialization
    x(:,1) = [-6;0];
    x_r(:,1) = [tanh(1);1-abs(tanh(-10))];
    
    %---Compute the control gain K and observer gain L
    Q = 20*eye(2);
    R = 1;
    
    [K,S,CLP] = dlqr(A,B,Q,R);
    
   % eig(A-B*K)
    
    Q2 = 2*eye(2);
    R2 = 1;
    [L,S2,CLP2] = dlqr(A',C',Q2,R2);
    
   % eig(A-L'*C)
    
    %%---define mean and mean-square errors---%%
    error_mean = zeros(2,T+1);
    error_MS = zeros(2,T+1);
    
    for s=1:S_num
        %----Generate standard Gaussian noise--------
    etab = normrnd(0,1);
        
    lambda = sigma*Lambda_bar*etab;
    
    %%system evolution
    for t=1:T
        x_r(:,t+1) = [tanh(t+1);1-abs(tanh(t-9))];
        x(:,t+1) = A*x(:,t) - B*K*(x_hat(:,t)-x_r(:,t));
        x_hat(:,t+1) = A*x_hat(:,t) - B*K*(x_hat(:,t)-x_r(:,t)) + L'*(x(1,t)+ lambda(t)-x_hat(1,t));
    end
    error_mean = error_mean + x-x_r;
    error_MS = error_MS + (x-x_r).^2;
    end
    %plot computation error
    figure(g)
    grid on
    T_vec = 0:T;
    for i=1:n
        plot(T_vec,error_mean(i,:)/S_num,'-','LineWidth',1)
        hold on
    end
    
    figure(4)
    for i=1:n
        plot(T_vec,error_MS(i,:)/S_num,'-','LineWidth',1)
        hold on
    end
    
    xlabel('$t$','interpreter','latex')
  %  ylabel('$x_i(t)$','interpreter','latex')
  %  legend('E$[p(t)-p_r(t)]$','E$[v(t)-v_r(t)]$','E$|p(t)-p_r(t)|^2$','E$|v(t)-v_r(t)|^2$','interpreter','latex')
    grid on
    
end



