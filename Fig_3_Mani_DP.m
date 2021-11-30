
%%% --Cloud-based control with Manifold Dependency for vehicles-- %%%%
%%% --Figure 3-- %%%%%%%%%%%
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
x_r= zeros(n,T+1); %reference

x = zeros(n,T+1,S_num); %plant state for case r=1
x_hat= zeros(n,T+1,S_num); %observe state for case r=1

xT = zeros(n,T+1,S_num); %plant state for case r=T
xT_hat= zeros(n,T+1,S_num); %observe state for case r=T

%---Compute the control gain K and observer gain L
Q = 20*eye(2);
R = 1;

[K,S,CLP] = dlqr(A,B,Q,R); %%compute the control gain K%%

% eig(A-B*K)

Q2 = 2*eye(2);
R2 = 1;
[L,S2,CLP2] = dlqr(A',C',Q2,R2); %%compute the observer gain L%%



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
Lambda_bar  = F*D_bot; %%r = 1

Lambda_barT  = eye(T,T); %%r = T, i.e., we use iid noises


%% ---compute Moore–Penrose inverse of Lambda_bar---%%
Lambda_bar_inv = inv(Lambda_bar'*Lambda_bar)*Lambda_bar'; %%r = 1
Lambda_barT_inv = inv(Lambda_barT'*Lambda_barT)*Lambda_barT'; %%r = T

%% --- compute O_ij ---%%
O_ij = F*ones(T,1);

%% --- compute sensitivity ---
Delta_G = sqrt((Lambda_bar_inv*O_ij)'*(Lambda_bar_inv*O_ij)); %%r = 1
Delta_GT = sqrt((Lambda_barT_inv*O_ij)'*(Lambda_barT_inv*O_ij)); %%r = T

%% --number of privacy requriemt-- %%
privacy_vec = [10^(-2),5*10^(-2),10^(-1),5*10^(-1),10^0,5,10];
num_p = length(privacy_vec);


%% --Define matrices for total errors and mean-square accuracy
total_error_MS = zeros(2,num_p); %%total error of all samples at T for r = 1
accuracy_MS = zeros(2,num_p); %%mean-square accuracy for r = 1

total_error_MST = zeros(2,num_p); %%total error of all samples at T for r = T
accuracy_MST = zeros(2,num_p); %%mean-square accuracy for r = T


for g=1:num_p
    %%privacy budgets %%%
    epsilon = privacy_vec(g); %%try different values [10^(-2),10^(-1),10^0,10,10^2]
    
    delta = 0.01; %%delta is fixed
    
    mu = 1;
    
    
    %%%---compute the sigma---%%%
    syms y
    
    %%--Define an inverse of Phi(s) and find the s such that Phi(s)=delta--%%
    Phi_inv = solve((1/(sqrt(2*pi)))*int(exp(-0.5*y*y),y,-Inf,y)==delta);
    
    sigma = mu*Delta_G/(double(Phi_inv) + sqrt(double(Phi_inv)*double(Phi_inv)+2*epsilon)); %%r = 1
    sigmaT = mu*Delta_GT/(double(Phi_inv) + sqrt(double(Phi_inv)*double(Phi_inv)+2*epsilon)); %%r = T
    
    %----initialization---
    x_r(:,1) = [tanh(1);1-abs(tanh(-10))];
    x(:,1) = [-20;0];
    xT(:,1) = [-20;0];
    
    
    for s=1:S_num
        %----Generate standard Gaussian noise--------
        etab = normrnd(0,1);
        lambda = sigma.*Lambda_bar*etab;
        
        etabT = normrnd(0,1,T,1);  %%T iid gaussian noises
        lambdaT = sigmaT.*Lambda_barT*etabT;
        
        %%system evolution
        for t=1:T
            x_r(:,t+1) = [tanh(t+1);1-abs(tanh(t-9))];
            x(:,t+1,s) = A*x(:,t,s) - B*K*(x_hat(:,t,s)-x_r(:,t)); %system evolution for r=1
            x_hat(:,t+1,s) = A*x_hat(:,t,s) - B*K*(x_hat(:,t,s)-x_r(:,t)) + L'*(x(1,t,s)+ lambda(t)-x_hat(1,t,s)); %observer evolution for r=1
            
            xT(:,t+1,s) = A*xT(:,t,s) - B*K*(xT_hat(:,t,s)-x_r(:,t)); %system evolution for r=T
            xT_hat(:,t+1,s) = A*xT_hat(:,t,s) - B*K*(xT_hat(:,t,s)-x_r(:,t)) + L'*(xT(1,t,s)+ lambdaT(t)-xT_hat(1,t,s)); %observer evolution for r=T
        end
        total_error_MS(:,g) = total_error_MS(:,g) + (x(:,T+1,s)-x_r(:,T+1)).^2; %total error of all samples at T for r=1
        total_error_MST(:,g) = total_error_MST(:,g) + (xT(:,T+1,s)-x_r(:,T+1)).^2; %total error of all samples at T total accuracy r=T
    end
    
    accuracy_MS(:,g) = total_error_MS(:,g)./S_num; %r=1
    accuracy_MST(:,g) = total_error_MST(:,g)./S_num; %r=T
end



%%%--plot--%%%
figure(1) %%plot for position
plot(privacy_vec,accuracy_MS(1,:),'r-*','LineWidth',1)
hold on
plot(privacy_vec,accuracy_MST(1,:),'b-o','LineWidth',1)
hold on
ylabel('E$|p(t)-p_r(t)|^2$','interpreter','latex')
xlabel('$\epsilon$','interpreter','latex')
legend('$r=1$','$r=T$','interpreter','latex')


figure(2) %%plot for velocity
plot(privacy_vec,accuracy_MS(2,:),'r-*','LineWidth',1)
hold on
plot(privacy_vec,accuracy_MST(2,:),'b-o','LineWidth',1)
hold on
ylabel('$E|v(t)-v_r(t)|^2$','interpreter','latex')
xlabel('$\epsilon$','interpreter','latex')
legend('$r=1$','$r=T$','interpreter','latex')

%%%%%%%


