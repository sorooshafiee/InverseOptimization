clc
clear
rng('shuffle')

addpath('./model/')

N           = 20;
N_test      = 1000;
n           = 10;
m           = n;
M           = 100;
epsilon2    = [0 (1:9)*1e-4 (1:9)*1e-3 (1:9)*1e-2 (1:10)*1e-1];
ne          = length(epsilon2);
run_count   = 100;

%======================== Setting Parameters ===========================%
param.epsilon2   = epsilon2; 
param.C          = zeros(1,m);
param.d          = 0;
param.W          = [eye(n);-eye(n)];
param.H          = [zeros(n,m);zeros(n,m)];
param.h          = [zeros(n,1);-5*ones(n,1)];
param.solver     = 'mosek';
param.alpha      = 1;
param.tol        = 0;
param.delta      = 0.2;
param.isfeasible = 0;
param.set_theta  = '[Q_xx >= 0, Q_xs == eye(n,m)]';

data(run_count) = struct('x',[],'s',[]);
Suboptimality   = zeros(length(epsilon2),run_count);
Predictability  = zeros(length(epsilon2),run_count);
Identifiability = zeros(length(epsilon2),run_count);

for r = 1 : run_count
    fprintf('Running iteration %d ... \n',r);
    
    %========================= Generate Dataset ==========================%
    s       = rand(m,N);    
    U       = randn(n);
    [U,~]   = eig(U+U');
    d       = 0.2 + 0.8 * rand(n,1); 
    phi     = -U'*diag(d)*U; 
    psi     = 2 * rand(n,1); 
    subopt  = SubQuadratic_Model(param,-phi,eye(n,m),-psi,s);
    x_sub   = [subopt.x]; 
    
    %==================== Solve the Inverse Problem ======================%
    data(r).x   = x_sub;
    data(r).s   = s;
    opt_inv     = Quadratic_Inverse(param, data(r));
    Q_xx        = [opt_inv.Q_xx]; 
    Q_xx        = reshape(Q_xx,n,n,[]);
    Q_xs        = [opt_inv.Q_xs];
    Q_xs        = reshape(Q_xs,n,m,[]);
    q           = [opt_inv.q]; 
    q           = reshape(q,n,1,[]);
    
    %================= Evaluate the Model on Test Data ===================%
    s           = rand(m,N_test);     
    optimal     = Quadratic_Model(param,-phi,eye(n,m),-psi,s);
    x_star      = [optimal.x]; 
    tmp_sub     = zeros(ne,1);
    tmp_pre     = zeros(ne,1);
    for j = 1 : ne
        opt_model   = Quadratic_Model(param,Q_xx(:,:,j),Q_xs(:,:,j),q(:,:,j),s);
        obj         = [opt_model.objective];
        tmp_sub(j)  = mean(diag(x_star'*Q_xx(:,:,j)*x_star)' + diag(x_star'*Q_xs(:,:,j)*s)' + q(:,:,j)'*x_star - obj);
        tmp_pre(j)  = mean(sqrt(sum((x_star - [opt_model.x]).^2,1)));
    end
    Suboptimality(:,r)  = tmp_sub;
    Predictability(:,r) = tmp_pre;
    
end
%%
fig = figure;
set(fig, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
font_size = 18;

yyaxis left
semilogx(epsilon2,mean(Suboptimality,2),'linewidth',4,'color',[0, 0.447, 0.741]);
set(gca, 'FontSize', font_size);
xlabel('$\varepsilon^2$','Interpreter','latex', 'FontSize',26);
ylabel('Suboptimality','FontSize',font_size) % left y-axis

yyaxis right
semilogx(epsilon2,mean(Predictability,2),'linewidth',4,'color',[0.85, 0.325, 0.098],'linestyle','-.');
ylabel('Predictability','FontSize',font_size) % right y-axis

cd figs
saveas(fig,'fig3-a','png')
cd ..