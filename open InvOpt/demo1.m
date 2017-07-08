% This is a demo for figure1_code.m
clc
clearvars
rng('shuffle')

addpath('./model/')

n = 10;
m = 10;
N = 10;
N_test = 100;
delta = 1;
radius = 1;
eta = 1;
run_count = 20;
epsilon = [(1:9:9)*1e-5 (1:9:9)*1e-4 (1:9:9)*1e-3 (1:9:10)*1e-2];
param.solver = 'sedumi';
param(1:run_count) = struct('W',[],'H',[],'h',[],'C',zeros(1,m),'d',0, ...
                            'pnorm',1,'alpha',0.9,'epsilon',epsilon,   ...
                            'delta',delta,'set_theta',struct('center',[], ...
                            'radius',radius,'pnorm',inf));
data(run_count) = struct('x',[],'s',[]);
Suboptimality = zeros(length(epsilon),run_count);
Predictability = zeros(length(epsilon),run_count);
Identifiability = zeros(length(epsilon),run_count);

parfor r = 1 : run_count  
    fprintf('Running iteration %d ... \n',r);
    
    %======================== Setting Parameters =========================%
    param(r).W = [2 * rand(m,n) - 1; eye(n); -eye(n)];
    param(r).H = [eye(m); zeros(n,m); zeros(n,m)];
    param(r).h = [zeros(m,1); -ones(n,1); -ones(n,1)];
    center = radius + 4 * radius * rand(n,1);
    sgn = randi(2,[n,1])-1;
    center(sgn == 0) = -center(sgn == 0);
    param(r).set_theta.center = center;
    theta_star = center + 2 * radius * rand(n,1) - radius;    
    
    %========================= Generate Dataset ==========================%
    random_x = 2*rand(n,N) - 1;
    s = param(r).W(1:m,:) * random_x;
    s = s - eta * rand(size(s));
    suboptimal = SubLinear_Model_YALMIP(param(r),theta_star,s);
    suboptimal_x = [suboptimal.x];
        
    %==================== Solve the Inverse Problem ======================%
    data(r).x   = suboptimal_x;
    data(r).s   = s;
    opt_inv     = Linear_Inverse_YALMIP(param(r),data(r));
    theta       = [opt_inv.theta];
    
    %================= Evaluate the Model on Test Data ===================%
    random_x = 2*rand(n,N_test) - 1;
    s = param(r).W(1:m,:) * random_x;
    s = s - eta * rand(size(s));
    optimal = Linear_Model_YALMIP(param(r),theta_star,s);
    optimal_x = [optimal.x];
    tmp_sub     = zeros(size(theta,2),1);
    tmp_pre     = zeros(size(theta,2),1);
    for j = 1 : size(theta,2)
        opt_model   = Linear_Model_YALMIP(param(r),theta(:,j),s);
        tmp_sub(j)  = mean( max(theta(:,j)'* (optimal_x - [opt_model.x]), 0) );
        tmp_pre(j)  = mean( sqrt(sum((optimal_x - [opt_model.x]).^2,1)) );
    end
    Suboptimality(:,r)  = tmp_sub;
    Predictability(:,r) = tmp_pre;
    Identifiability(:,r)= sqrt(sum((theta - repmat(theta_star,[1,size(theta,2)])).^2,1))';
end
%%
figure;
font_size = 14;
[hAx,hLine1,hLine2] = plotyy(epsilon,mean(Suboptimality,2),epsilon,mean(Predictability,2),'semilogx','semilogx');
xlabel('$\varepsilon$','Interpreter','latex','FontSize',font_size,'LineWidth',3);
ylabel(hAx(1),'\delta-Suboptimality','FontSize',font_size) % left y-axis
ylabel(hAx(2),'Predictability','FontSize',font_size) % right y-axis
set(hAx,'FontSize', font_size, {'ycolor'},{[0, 0.447, 0.741];[0.85, 0.325, 0.098]});
set(hLine1,'linewidth',3,'color',[0, 0.447, 0.741])
set(hLine2,'linewidth',3,'color',[0.85, 0.325, 0.098],'linestyle','-.')