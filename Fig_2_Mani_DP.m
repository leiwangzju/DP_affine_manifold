
%%% --Cloud-based control with Manifold Dependency for vehicles-- %%%%
%%% --Figure 2-- %%%%%%%%%%%
%%% --Gaussian Mechanism-- %%%%

close all
clear all
clc

%%--- define samples number ---%%
S_num = 100;

%%---define samping period T_s---%%
T_s=0.1;

%%---define system dimension n---%%
n=2;

%%---define running time T---%%
T = 100;

%%---define system matrix---%%
A= [1, T_s; 0,1];
B = [T_s^2/2;T_s];
C = [1,0];


%%---define closed-loop signals---%%
x = zeros(n,T+1,S_num); %plant state
x_r= zeros(n,T+1,S_num); %reference
x_hat= zeros(n,T+1,S_num); %observe state

%%---Define correlation model D*x_1 + b = 0---%%
D = zeros(T-1,T);
for t=1:T-1
    D(t,t) = 1;
    D(t,t+1) = -1;
end

%%---Define D_bot satisfying D*D_bot = 0 and rank(D;D_bot')=rows---%%
D_bot = ones(T,1);

%%--- verify D_bot ---%%
% D*D_bot;
% rank([D;D_bot']);

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
Delta_G = abs(Lambda_bar_inv*O_ij);


%%privacy levels %%%
epsilon = 1;
delta = 0.01;
mu = 1;


%%%---compute the sigma---%%%
syms y

%%--Define an inverse of Phi(s) and find the s such that Phi(s)=delta--%%
Phi_inv = solve((1/(sqrt(2*pi)))*int(exp(-0.5*y*y),y,-Inf,y)==delta);

sigma = mu*Delta_G/(double(Phi_inv) + sqrt(double(Phi_inv)*double(Phi_inv)+2*epsilon));


%----initialization
x(:,1) = [-20;0];
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

%%---define total errors of all samples for computing the mean-square error---%%
total_error = zeros(2,T+1);

for s=1:S_num
    %----Generate standard Gaussian noise--------
    etab = normrnd(0,1); %%generate a gaussian noise
    lambda = sigma*Lambda_bar*etab;  %%added noise
    
    total_error(:,1) = total_error(:,1) + (x(:,1,s)-x_r(:,1)).^2; %%the total  error of all samples at initial time
    
    %%system evolution
    for t=1:T
        x_r(:,t+1) = [tanh(t+1);1-abs(tanh(t-9))];
        x(:,t+1,s) = A*x(:,t,s) - B*K*(x_hat(:,t,s)-x_r(:,t)); %%evolution of system plant
        x_hat(:,t+1,s) = A*x_hat(:,t,s) - B*K*(x_hat(:,t,s)-x_r(:,t)) + L'*(x(1,t,s)+ lambda(t)-x_hat(1,t,s)); %%evolution of observer
        total_error(:,t+1) = total_error(:,t+1) + (x(:,t+1,s)-x_r(:,t+1)).^2; %%the total  error of all samples at t+1
    end
    
    
end

%%---compute the mean-square error--%%
error_MS = total_error./S_num;

%%---Define matrices for plotting boxplot of position and velocity trajectories---%%
position_samples = zeros(S_num,20); %%20 time slots from the position trajectory
velocity_samples = zeros(S_num,20); %%20 time slots from the velocity trajectory
pos = (5:5:T)'; %%positions of these 20 time slots

for k=1:20
    for s=1:S_num
        position_samples(s,k) = x(1,5*k,s)-x_r(1,5*k);
        velocity_samples(s,k) = x(2,5*k,s)-x_r(2,5*k);
    end
end

T_vec = 0:T; %%define time vector




figure(1)
title('\epsilon=1')
subplot(2,1,1)
boxplot(position_samples,pos,'notch', 'on', 'colors', 'r')
xlabel('$t$','interpreter','latex')
ylabel('$p(t)-p_r(t)$','interpreter','latex')
subplot(2,1,2)
plot(T_vec,error_MS(1,:),'r','LineWidth',1);
ylabel('E$|p(t)-p_r(t)|^2$','interpreter','latex')
xlabel('$t$','interpreter','latex')

figure(2)
subplot(2,1,1)
boxplot(velocity_samples,pos,'notch', 'on', 'colors', 'b')
xlabel('$t$','interpreter','latex')
ylabel('$v(t)-v_r(t)$','interpreter','latex')
subplot(2,1,2)
plot(T_vec,error_MS(2,:),'b','LineWidth',1);
ylabel('E$|v(t)-v_r(t)|^2$','interpreter','latex')
xlabel('$t$','interpreter','latex')


