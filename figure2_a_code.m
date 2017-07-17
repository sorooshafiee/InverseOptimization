clc
clearvars
rng('shuffle')

addpath('./model/')

n = 50;
m = 50;
N = 10;
N_test = 1000;
delta = 1;
radius = 1;
eta = 0;
run_count = 100;
epsilon = [(1:9)*1e-5 (1:9)*1e-4 (1:9)*1e-3 (1:10)*1e-2];
param(1:run_count) = struct('W',[],'H',[],'h',[],'C',zeros(1,m),'d',0, ...
                            'pnorm',1,'alpha',1,'epsilon',epsilon,   ...
                            'delta',delta,'set_theta',struct('center',[], ...
                            'radius',radius,'pnorm',inf));
data(run_count) = struct('x',[],'s',[]);
Suboptimality = zeros(length(epsilon),run_count);
Predictability = zeros(length(epsilon),run_count);

for r = 1 : run_count
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
    s = s - eta;
    suboptimal = SubLinear_Model(param(r),theta_star,s);
    suboptimal_x = [suboptimal.x];
        
    %==================== Solve the Inverse Problem ======================%
    data(r).x   = suboptimal_x;
    data(r).s   = s;
    opt_inv     = Linear_Inverse(param(r),data(r));
    theta       = [opt_inv.theta];
    
    %================= Evaluate the Model on Test Data ===================%
    random_x = 2*rand(n,N_test) - 1;
    s = param(r).W(1:m,:) * random_x;
    s = s - eta;
    suboptimal = SubLinear_Model(param(r),theta_star,s);
    suboptimal_x = [suboptimal.x];
    tmp_sub     = zeros(size(theta,2),1);
    tmp_pre     = zeros(size(theta,2),1);
    for j = 1 : size(theta,2)
        opt_model   = Linear_Model(param(r),theta(:,j),s);
        tmp_sub(j)  = mean( max(theta(:,j)'* (suboptimal_x - [opt_model.x]) - delta, 0) );
        tmp_pre(j)  = mean( sqrt(sum((suboptimal_x - [opt_model.x]).^2,1)) );
    end
    Suboptimality(:,r)  = tmp_sub;
    Predictability(:,r) = tmp_pre;
end
%%
fig = figure;
set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
font_size = 18;

yyaxis left
semilogx(epsilon,mean(Suboptimality,2),'linewidth',4,'color',[0, 0.447, 0.741]);
set(gca, 'FontSize', font_size);
xlabel('$\varepsilon$','Interpreter','latex', 'FontSize',26);
ylabel('\delta-Suboptimality','FontSize',font_size) % left y-axis

yyaxis right
semilogx(epsilon,mean(Predictability,2),'linewidth',4,'color',[0.85, 0.325, 0.098],'linestyle','-.');
ylabel('Predictability','FontSize',font_size) % right y-axis

cd figs
saveas(fig,'fig2-a','png')
cd ..