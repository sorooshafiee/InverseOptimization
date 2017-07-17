clc
clearvars
rng('shuffle')

addpath('./model/')

N           = 20;
N_test      = 1000;
n           = 10;
m           = n;
epsilon2    = [0 1e-4 1e-3 1e-2 1e-1 1];
sigma       = [3 6 9 12];
kernels     = {'poly2', 'poly3'};
kappa       = [1e-1 1e-2 1e-3];
ne          = length(epsilon2);
run_count   = 100;
kfold       = 5;

param(1:run_count) = struct('W',[eye(n);-eye(n)],'H',[zeros(n,m);zeros(n,m)], ...
                            'h',[zeros(n,1);-5*ones(n,1)],'C',zeros(1,m), ...
                            'd',0,'alpha',1,'solver','mosek','tol',0, ...
                            'epsilon2',epsilon2,'delta',0.2,'pnorm',1, ...
                            'kernel', [], 'sigma', [], 'kappa', kappa, ...
                            'set_theta','[Q_xx >= 0, Q_xs == eye(n,m)]');
data(run_count) = struct('x',[],'s',[]);
Suboptimality1  = zeros(3,run_count);
Predictability1 = zeros(3,run_count);
Suboptimality2  = zeros(3,run_count);
Predictability2 = zeros(3,run_count);
Suboptimality3  = zeros(5,run_count);
Predictability3 = zeros(5,run_count);


% Noisy training
for r = 1 : run_count  
    fprintf('Running iteration %d ... \n',r);
    tmp1 = zeros(3,1);
    tmp2 = zeros(3,1);
    param(r).epsilon2 = epsilon2;

    %========================= Generate Dataset ==========================%
    s       = rand(m,N+N_test);    
    U       = randn(n);
    [U,~]   = eig(U+U');
    d       = 0.2 + 0.8 * rand(n,1); 
    phi     = -U'*diag(d)*U; 
    psi     = 2 * rand(n,1); 
    optimal = Quadratic_Model(param(r),-phi,eye(n,m),-psi,s);
    x_star  = [optimal.x]; 

    %========= Select the Best Model Based on Validation Data ========%
    x_n = x_star(:,1:N) + 0.1 * (2*rand(n,N) - 1);
    tmp_sub = NaN(length(epsilon2),kfold);
    tmp_pre = NaN(length(epsilon2),kfold);
    pw = fix(N/kfold);
    for k = 1 : kfold
        s_tr = s(:,1:N);
        x_tr = x_n;
        s_tr(:,(k-1)*pw+1:k*pw) = [];
        x_tr(:,(k-1)*pw+1:k*pw) = [];
        s_v = s(:,(k-1)*pw+1:k*pw);
        x_v = x_n(:,(k-1)*pw+1:k*pw);    
        data(r).x = x_tr;
        data(r).s = s_tr;
        opt_inv = Quadratic_Inverse(param(r), data(r));
        diagnosis = [opt_inv.diagnosis];
        feas_ind = find([diagnosis.problem] == 0);
        Q_xx = [opt_inv.Q_xx]; 
        Q_xx = reshape(Q_xx,n,n,[]);
        Q_xs = [opt_inv.Q_xs];
        Q_xs = reshape(Q_xs,n,m,[]);
        q = [opt_inv.q]; 
        q = reshape(q,n,1,[]);
        for j = feas_ind               
            opt_model = Quadratic_Model(param(r),Q_xx(:,:,j),Q_xs(:,:,j),q(:,:,j),s_v);
            obj = [opt_model.objective];
            tmp_sub(j,k) = mean(diag(x_v'*Q_xx(:,:,j)*x_v)' + diag(x_v'*Q_xs(:,:,j)*s)' + q(:,:,j)'*x_v - obj);
            tmp_pre(j,k) = mean(sqrt(sum((x_v - [opt_model.x]).^2,1)));
        end            
    end   
    tmp = (kfold-1)/kfold * mean(tmp_sub,2) + 1/kfold * std(tmp_sub,[],2);
    index = find(round(tmp,4) == min(round(tmp,4)));
    ind = max(index);
    param(r).epsilon2 = [0 epsilon2(ind)];   

    %================== Solve the Inverse Problem ====================%
    data(r).x = x_n;
    data(r).s = s(:,1:N);
    opt_inv = Quadratic_Inverse(param(r), data(r));    
    Q_xx = [opt_inv.Q_xx]; 
    Q_xx = reshape(Q_xx,n,n,[]);
    Q_xs = [opt_inv.Q_xs];
    Q_xs = reshape(Q_xs,n,m,[]);
    q = [opt_inv.q]; 
    q = reshape(q,n,1,[]);
    opt_inv_g = Gupta_Quadratic_Inverse(param(r), data(r));
    Q_xx_g = opt_inv_g.Q_xx;
    if min(eig(Q_xx_g)) < 0
        Q_xx_g = Q_xx_g + 1.1*abs(min(eig(Q_xx_g)))*eye(n);
    end
    Q_xs_g = opt_inv_g.Q_xs;
    q_g = opt_inv_g.q;

    %=============== Evaluate the Model on Test Data =================%
    x_star = x_star(:,N+1:end);
    s = s(:,N+1:end);
    
    % Check Vishal Gupta's Solution (parametric)
    opt_gupta = Quadratic_Model(param(r),Q_xx_g,Q_xs_g,q_g,s);
    tmp1(1) = mean(diag(x_star'*Q_xx_g*x_star)' + diag(x_star'*Q_xs_g*s)' + q_g'*x_star - [opt_gupta.objective]);
    tmp2(1) = mean( sqrt(sum((x_star - [opt_gupta.x]).^2,1)) );
    % Check ERM Solution
    opt_ERM = Quadratic_Model(param(r),Q_xx(:,:,1),Q_xs(:,:,1),q(:,:,1),s);
    tmp1(2) = mean(diag(x_star'*Q_xx(:,:,1)*x_star)' + diag(x_star'*Q_xs(:,:,1)*s)' + q(:,:,1)'*x_star - [opt_ERM.objective]);
    tmp2(2) = mean( sqrt(sum((x_star - [opt_ERM.x]).^2,1)) );
    % Check DRO Solution
    opt_DRO = Quadratic_Model(param(r),Q_xx(:,:,2),Q_xs(:,:,2),q(:,:,2),s);
    tmp1(3) = mean(diag(x_star'*Q_xx(:,:,2)*x_star)' + diag(x_star'*Q_xs(:,:,2)*s)' + q(:,:,2)'*x_star - [opt_DRO.objective]);
    tmp2(3) = mean( sqrt(sum((x_star - [opt_DRO.x]).^2,1)) );   
    
    Suboptimality1(:,r) = tmp1;
    Predictability1(:,r) = tmp2;
end

% Suboptimal training
for r = 1 : run_count  
    fprintf('Running iteration %d ... \n',r);
    tmp1 = zeros(3,1);
    tmp2 = zeros(3,1);
    param(r).epsilon2 = epsilon2;

    %========================= Generate Dataset ==========================%
    s       = rand(m,N);    
    U       = randn(n);
    [U,~]   = eig(U+U');
    d       = 0.2 + 0.8 * rand(n,1); 
    phi     = -U'*diag(d)*U; 
    psi     = 2 * rand(n,1); 
    subopt  = SubQuadratic_Model(param(r),-phi,eye(n,m),-psi,s);
    x_sub   = [subopt.x]; 

    %========= Select the Best Model Based on Validation Data ========%
    tmp_sub = zeros(length(epsilon2),kfold);
    tmp_pre = zeros(length(epsilon2),kfold);
    pw = fix(N/kfold);
    for k = 1 : kfold
        s_tr = s;
        x_tr = x_sub;
        s_tr(:,(k-1)*pw+1:k*pw) = [];
        x_tr(:,(k-1)*pw+1:k*pw) = [];
        s_v = s(:,(k-1)*pw+1:k*pw);
        x_v = x_sub(:,(k-1)*pw+1:k*pw);    
        data(r).x = x_tr;
        data(r).s = s_tr;
        opt_inv = Quadratic_Inverse(param(r), data(r));
        Q_xx = [opt_inv.Q_xx]; 
        Q_xx = reshape(Q_xx,n,n,[]);
        Q_xs = [opt_inv.Q_xs];
        Q_xs = reshape(Q_xs,n,m,[]);
        q = [opt_inv.q]; 
        q = reshape(q,n,1,[]);
        for j = 1 : length(epsilon2)                
            opt_model = Quadratic_Model(param(r),Q_xx(:,:,j),Q_xs(:,:,j),q(:,:,j),s_v);
            obj = [opt_model.objective];
            tmp_sub(j,k) = mean(diag(x_v'*Q_xx(:,:,j)*x_v)' + diag(x_v'*Q_xs(:,:,j)*s)' + q(:,:,j)'*x_v - obj);
            tmp_pre(j,k) = mean(sqrt(sum((x_v - [opt_model.x]).^2,1)));
        end            
    end   
    tmp = (kfold-1)/kfold * mean(tmp_sub,2) + 1/kfold * std(tmp_sub,[],2);
    index = find(round(tmp,4) == min(round(tmp,4)));
    ind = max(index);
    param(r).epsilon2 = [0 epsilon2(ind)];   

    %================== Solve the Inverse Problem ====================%
    data(r).x = x_sub;
    data(r).s = s;
    opt_inv = Quadratic_Inverse(param(r), data(r));    
    Q_xx = [opt_inv.Q_xx]; 
    Q_xx = reshape(Q_xx,n,n,[]);
    Q_xs = [opt_inv.Q_xs];
    Q_xs = reshape(Q_xs,n,m,[]);
    q = [opt_inv.q]; 
    q = reshape(q,n,1,[]);
    opt_inv_g = Gupta_Quadratic_Inverse(param(r), data(r));
    Q_xx_g = opt_inv_g.Q_xx;
    if min(eig(Q_xx_g)) < 0
        Q_xx_g = Q_xx_g + 1.1*abs(min(eig(Q_xx_g)))*eye(n);
    end
    Q_xs_g = opt_inv_g.Q_xs;
    q_g = opt_inv_g.q;

    %=============== Evaluate the Model on Test Data =================%
    s = rand(m,N_test);     
    optimal = Quadratic_Model(param(r),-phi,eye(n,m),-psi,s);
    x_star = [optimal.x]; 
    
    % Check Vishal Gupta's Solution (parametric)
    opt_gupta = Quadratic_Model(param(r),Q_xx_g,Q_xs_g,q_g,s);
    tmp1(1) = mean(diag(x_star'*Q_xx_g*x_star)' + diag(x_star'*Q_xs_g*s)' + q_g'*x_star - [opt_gupta.objective]);
    tmp2(1) = mean( sqrt(sum((x_star - [opt_gupta.x]).^2,1)) );
    % Check ERM Solution
    opt_ERM = Quadratic_Model(param(r),Q_xx(:,:,1),Q_xs(:,:,1),q(:,:,1),s);
    tmp1(2) = mean(diag(x_star'*Q_xx(:,:,1)*x_star)' + diag(x_star'*Q_xs(:,:,1)*s)' + q(:,:,1)'*x_star - [opt_ERM.objective]);
    tmp2(2) = mean( sqrt(sum((x_star - [opt_ERM.x]).^2,1)) );
    % Check DRO Solution
    opt_DRO = Quadratic_Model(param(r),Q_xx(:,:,2),Q_xs(:,:,2),q(:,:,2),s);
    tmp1(3) = mean(diag(x_star'*Q_xx(:,:,2)*x_star)' + diag(x_star'*Q_xs(:,:,2)*s)' + q(:,:,2)'*x_star - [opt_DRO.objective]);
    tmp2(3) = mean( sqrt(sum((x_star - [opt_DRO.x]).^2,1)) );   
    
    Suboptimality2(:,r) = tmp1;
    Predictability2(:,r) = tmp2;
end

% Model Mismatch
for r = 1 : run_count  
    fprintf('Running iteration %d ... \n',r);
    tmp1 = zeros(4,1);
    tmp2 = zeros(4,1);
    param(r).epsilon2 = epsilon2;

    %========================= Generate Dataset ==========================%
    s           = rand(m,N+N_test);    
    A           = diag(0.5 + 0.5 * rand(n,1)); 
    b           = -0.25 * rand(n,1); 
    optimal     = Utility_Model_Sqrt(param(r),A,b,s);
    x_star      = [optimal.x]; 

    %========= Select the Best Model Based on Validation Data ========%
    % Best parameter for DRO
    tmp_sub = zeros(length(epsilon2),kfold);
    pw = fix(N/kfold);
    for k = 1 : kfold
        s_tr = s(:,1:N);
        x_tr = x_star(:,1:N);
        s_tr(:,(k-1)*pw+1:k*pw) = [];
        x_tr(:,(k-1)*pw+1:k*pw) = [];
        s_v = s(:,(k-1)*pw+1:k*pw);
        x_v = x_star(:,(k-1)*pw+1:k*pw);    
        data(r).x = x_tr;
        data(r).s = s_tr;
        opt_inv = Quadratic_Inverse(param(r), data(r));
        Q_xx = [opt_inv.Q_xx]; 
        Q_xx = reshape(Q_xx,n,n,[]);
        Q_xs = [opt_inv.Q_xs];
        Q_xs = reshape(Q_xs,n,m,[]);
        q = [opt_inv.q]; 
        q = reshape(q,n,1,[]);
        for j = 1 : length(epsilon2)                
            opt_model = Quadratic_Model(param(r),Q_xx(:,:,j),Q_xs(:,:,j),q(:,:,j),s_v);
            obj = [opt_model.objective];
            tmp_sub(j,k) = mean(diag(x_v'*Q_xx(:,:,j)*x_v)' + diag(x_v'*Q_xs(:,:,j)*s)' + q(:,:,j)'*x_v - obj);
        end            
    end   
    tmp_1 = (kfold-1)/kfold * mean(tmp_sub,2) + 1/kfold * std(tmp_sub,[],2);
    index = find(round(tmp_1,4) == min(round(tmp_1,4)));
    ind = max(index);
    param(r).epsilon2 = [0 epsilon2(ind)];  
    % Best parameter for Gupta semi-parametric
    selected_sigma = [];
    selected_kappa = [];
    for kernel = kernels
        param(r).kernel = kernel{1};
        if strcmp('poly',kernel{1}(1:4))
            used_sigma = sigma;
        else
            used_sigma = sigma;
        end
        tmp_pre = zeros(length(used_sigma),length(kappa),kfold);
        pw = fix(N/kfold);
        for k = 1 : kfold
            param(r).sigma = used_sigma;            
            s_tr = s(:,1:N);
            x_tr = x_star(:,1:N);
            s_tr(:,(k-1)*pw+1:k*pw) = [];
            x_tr(:,(k-1)*pw+1:k*pw) = [];
            s_v = s(:,(k-1)*pw+1:k*pw);
            x_v = x_star(:,(k-1)*pw+1:k*pw);    
            data(r).x = x_tr;
            data(r).s = s_tr;
            opt_inv_ng = Gupta_Semi_Parametric_Inverse(param(r), data(r));
            for i = 1 : length(used_sigma)
                for j = 1 : length(kappa)                
                    param(r).sigma = used_sigma(i);
                    opt_gupta_n = Steepest_Descent_For_Gupta_Matlab(param(r),opt_inv_ng(i,j).alpha,x_tr,s_v);
                    tmp_pre(i,j,k) = mean(sqrt(sum((x_v - opt_gupta_n.x).^2,1)));
                end
            end
        end
        tmp_2 = (kfold-1)/kfold * mean(tmp_pre,3) + 1/kfold * std(tmp_pre,[],3);
        [i,j] = find(round(tmp_2,4) == min(min(round(tmp_2,4))));
        selected_sigma(end+1) = used_sigma(i(end));
        selected_kappa(end+1) = kappa(j(end));
    end

    %================== Solve the Inverse Problem ====================%
    x_tr = x_star(:,1:N);
    s_tr = s(:,1:N);
    data(r).x = x_tr;
    data(r).s = s_tr;
    opt_inv = Quadratic_Inverse(param(r), data(r));    
    Q_xx = [opt_inv.Q_xx]; 
    Q_xx = reshape(Q_xx,n,n,[]);
    Q_xs = [opt_inv.Q_xs];
    Q_xs = reshape(Q_xs,n,m,[]);
    q = [opt_inv.q]; 
    q = reshape(q,n,1,[]);
    opt_inv_g = Gupta_Quadratic_Inverse(param(r), data(r));
    Q_xx_g = opt_inv_g.Q_xx;
    if min(eig(Q_xx_g)) < 0
        Q_xx_g = Q_xx_g + 1.1*abs(min(eig(Q_xx_g)))*eye(n);
    end
    Q_xs_g = opt_inv_g.Q_xs;
    q_g = opt_inv_g.q;
    alpha_ng = zeros(n,N,length(kernels));
    cnt = 0;
    for kernel = kernels
        cnt = cnt + 1;
        param(r).kernel = kernel{1};
        param(r).sigma = selected_sigma(cnt);
        param(r).kappa = selected_kappa(cnt);
        opt_inv_ng = Gupta_Semi_Parametric_Inverse(param(r), data(r));
        alpha_ng(:,:,cnt) = opt_inv_ng.alpha;
    end

    %=============== Evaluate the Model on Test Data =================%
    x_star = x_star(:,N+1:end);
    s = s(:,N+1:end);
    
    % Check Vishal Gupta's Solution (parametric)
    opt_gupta = Quadratic_Model(param(r),Q_xx_g,Q_xs_g,q_g,s);
    tmp1(1) = mean(diag(x_star'*Q_xx_g*x_star)' + diag(x_star'*Q_xs_g*s)' + q_g'*x_star - [opt_gupta.objective]);
    tmp2(1) = mean( sqrt(sum((x_star - [opt_gupta.x]).^2,1)) );
    % Check Vishal Gupta's Solution (semi parametric)
    % The vectorized Matlab implementation of steepest descent is much
    % faster than it's cpp implementation
    cnt = 1;
    for kernel = kernels
        param(r).kernel = kernel{1};
        param(r).sigma = selected_sigma(cnt);
        param(r).kappa = selected_kappa(cnt);      
        opt_gupta_n = Steepest_Descent_For_Gupta_Matlab(param(r),alpha_ng(:,:,cnt),x_tr,s);
        cnt = cnt + 1;
        if strcmp('poly',kernel{1}(1:4))
            tmp1(cnt) = mean(sum(s.*(x_star-opt_gupta_n.x),1)' ...
                             +int_poly(x_star,x_tr,alpha_ng(:,:,cnt-1),1/selected_sigma(cnt-1)^2,cnt) ...
                             -int_poly(opt_gupta_n.x,x_tr,alpha_ng(:,:,cnt-1),1/selected_sigma(cnt-1)^2,cnt));
        end
        tmp2(cnt) = mean( sqrt(sum((x_star - opt_gupta_n.x).^2,1)) );
    end
    % Check ERM Solution
    opt_ERM = Quadratic_Model(param(r),Q_xx(:,:,1),Q_xs(:,:,1),q(:,:,1),s);
    tmp1(4) = mean(diag(x_star'*Q_xx(:,:,1)*x_star)' + diag(x_star'*Q_xs(:,:,1)*s)' + q(:,:,1)'*x_star - [opt_ERM.objective]);
    tmp2(4) = mean( sqrt(sum((x_star - [opt_ERM.x]).^2,1)) );
    % Check DRO Solution
    opt_DRO = Quadratic_Model(param(r),Q_xx(:,:,2),Q_xs(:,:,2),q(:,:,2),s);
    tmp1(5) = mean(diag(x_star'*Q_xx(:,:,2)*x_star)' + diag(x_star'*Q_xs(:,:,2)*s)' + q(:,:,2)'*x_star - [opt_DRO.objective]);
    tmp2(5) = mean( sqrt(sum((x_star - [opt_DRO.x]).^2,1)) ); 
    
    Suboptimality3(:,r) = tmp1;
    Predictability3(:,r) = tmp2;
end
%% Table in Latex
fileID = fopen('.\tables\table5.tex','w');
ts = tinv([0.1  0.9],run_count-1); % 80% confidence interval
fprintf(fileID,'\\documentclass{article}\n\\usepackage{multirow,rotating}\n\\begin{document}\n\\begin{table}\n\\centering \n');
caption = 'Comparison';
fprintf(fileID,'\\caption{%s} \n',caption);
fprintf(fileID,'\\begin{tabular}{llcc}\\hline \n');

fprintf(fileID,' &  Methods & Suboptimality & Predictability \\\\ \\hline');

fprintf(fileID,'\\multirow{3}{*}{Noisy training } &  VI (Parametric)');
str1 = value2latex(mean(Suboptimality1(1,:)));
str2 = value2latex(mean(Predictability1(1,:)));
fprintf(fileID,'& $%s$ & $%s$ \\\\ \n',str1,str2);
fprintf(fileID,' & ERM ');
str1 = '-';
str2 = '-';
fprintf(fileID,'& $%s$ & $%s$ \\\\ \n',str1,str2);
fprintf(fileID,' & DRO ');
str1 = value2latex(mean(Suboptimality1(3,:)));
str2 = value2latex(mean(Predictability1(3,:)));
fprintf(fileID,'& $%s$ & $%s$ \\\\ \\hline \n',str1,str2);

fprintf(fileID,'\\multirow{3}{*}{Suboptimal training } & VI (Parametric)');
str1 = value2latex(mean(Suboptimality2(1,:)));
str2 = value2latex(mean(Predictability2(1,:)));
fprintf(fileID,'& $%s$ & $%s$ \\\\ \n',str1,str2);
fprintf(fileID,' & ERM ');
str1 = value2latex(mean(Suboptimality2(2,:)));
str2 = value2latex(mean(Predictability2(2,:)));
fprintf(fileID,'& $%s$ & $%s$ \\\\ \n',str1,str2);
fprintf(fileID,' & DRO ');
str1 = value2latex(mean(Suboptimality2(3,:)));
str2 = value2latex(mean(Predictability2(3,:)));
fprintf(fileID,'& $%s$ & $%s$ \\\\ \\hline \n',str1,str2);

fprintf(fileID,'\\multirow{5}{*}{Model mismatch } & VI (Parametric)');
str1 = value2latex(mean(Suboptimality3(1,:)));
str2 = value2latex(mean(Predictability3(1,:)));
fprintf(fileID,'& $%s$ & $%s$ \\\\ \n',str1,str2);
fprintf(fileID,' & VI (Semi-Parametric Polynomial Degree 2) ');
str1 = value2latex(mean(Suboptimality3(2,:)));
str2 = value2latex(mean(Predictability3(2,:)));
fprintf(fileID,'& $%s$ & $%s$ \\\\ \n',str1,str2);
fprintf(fileID,' & VI (Semi-Parametric  Polynomial Degree 3) ');
str1 = value2latex(mean(Suboptimality3(3,:)));
str2 = value2latex(mean(Predictability3(3,:)));
fprintf(fileID,'& $%s$ & $%s$ \\\\ \n',str1,str2);
fprintf(fileID,' & ERM ');
str1 = value2latex(mean(Suboptimality3(4,:)));
str2 = value2latex(mean(Predictability3(4,:)));
fprintf(fileID,'& $%s$ & $%s$ \\\\ \n',str1,str2);
fprintf(fileID,' & DRO ');
str1 = value2latex(mean(Suboptimality3(5,:)));
str2 = value2latex(mean(Predictability3(5,:)));
fprintf(fileID,'& $%s$ & $%s$ \\\\ \\hline \n',str1,str2);

fprintf(fileID,'\\end{tabular} \n\\end{table}\n\n\n\\end{document}\n');
fclose(fileID);
cd .\tables\
command3 = 'pdflatex table5.tex';
[status3,cmdout3] = system(command3);
cd ..
