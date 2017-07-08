clc
clear
rng('shuffle')

addpath('./model/')

N           = 20;
N_test      = 100;
n           = 10;
m           = n;
M           = 100;
epsilon2    = [0 (1:9:9)*1e-4 (1:9:9)*1e-3 (1:9:9)*1e-2 (1:9:10)*1e-1];
ne          = length(epsilon2);
run_count   = 20;

%======================== Setting Parameters ===========================%
param.epsilon2   = epsilon2; 
param.C          = zeros(1,m);
param.d          = 0;
param.W          = [eye(n);-eye(n)];
param.H          = [zeros(n,m);zeros(n,m)];
param.h          = [zeros(n,1);-5*ones(n,1)];
param.alpha      = 0.9;
param.tol        = 0;
param.delta      = 0.2;
param.isfeasible = 0;
param.solver     = 'sedumi';
param.set_theta  = '[Q_xx >= 0, Q_xs == eye(n,m)]';

data(run_count) = struct('x',[],'s',[]);
Suboptimality   = zeros(length(epsilon2),run_count);
Predictability  = zeros(length(epsilon2),run_count);
Identifiability = zeros(length(epsilon2),run_count);

parfor r = 1 : run_count
    fprintf('Running iteration %d ... \n',r);
    
    %========================= Generate Dataset ==========================%
    s       = rand(m,N);    
    U       = randn(n);
    [U,~]   = eig(U+U');
    d       = 0.2 + 0.8 * rand(n,1); 
    phi     = -U'*diag(d)*U; 
    psi     = 2 * rand(n,1); 
    subopt  = SubQuadratic_Model_YALMIP(param,-phi,eye(n,m),-psi,s);
    x_sub   = [subopt.x]; 
    
    %==================== Solve the Inverse Problem ======================%
    data(r).x   = x_sub;
    data(r).s   = s;
    opt_inv     = Quadratic_Inverse_YALMIP(param, data(r));
    Q_xx        = [opt_inv.Q_xx]; 
    Q_xx        = reshape(Q_xx,n,n,[]);
    Q_xs        = [opt_inv.Q_xs];
    Q_xs        = reshape(Q_xs,n,m,[]);
    q           = [opt_inv.q]; 
    q           = reshape(q,n,1,[]);
    
    %================= Evaluate the Model on Test Data ===================%
    s           = rand(m,N_test);     
    optimal     = Quadratic_Model_YALMIP(param,-phi,eye(n,m),-psi,s);
    x_star      = [optimal.x]; 
    tmp_sub     = zeros(ne,1);
    tmp_pre     = zeros(ne,1);
    for j = 1 : ne
        opt_model   = Quadratic_Model_YALMIP(param,Q_xx(:,:,j),Q_xs(:,:,j),q(:,:,j),s);
        obj         = [opt_model.objective];
        tmp_sub(j)  = mean(diag(x_star'*Q_xx(:,:,j)*x_star)' + diag(x_star'*Q_xs(:,:,j)*s)' + q(:,:,j)'*x_star - obj);
        tmp_pre(j)  = mean(sqrt(sum((x_star - [opt_model.x]).^2,1)));
    end
    Suboptimality(:,r)  = tmp_sub;
    Predictability(:,r) = tmp_pre;
    
end
%%
figure;
font_size = 14;
[hAx,hLine1,hLine2] = plotyy(epsilon2,mean(Suboptimality,2),epsilon2,mean(Predictability,2),'semilogx','semilogx');
xlabel('$\varepsilon^2$','Interpreter','latex','FontSize',font_size,'LineWidth',3);
ylabel(hAx(1),'Suboptimality','FontSize',font_size) % left y-axis
ylabel(hAx(2),'Predictability','FontSize',font_size) % right y-axis
set(hAx,'FontSize', font_size, {'ycolor'},{[0, 0.447, 0.741];[0.85, 0.325, 0.098]});
set(hLine1,'linewidth',3,'color',[0, 0.447, 0.741])
set(hLine2,'linewidth',3,'color',[0.85, 0.325, 0.098],'linestyle','-.')