clc
clearvars
rng('shuffle')

addpath('./model/')

n = 10;
m = 10;
all_N = [2:10, 20:10:100];
N_test = 1000;
delta = 1;
radius = 1;
eta = 0;
run_count = 100;
epsilon = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1];
param(1:run_count) = struct('W',[],'H',[],'h',[],'C',zeros(1,m),'d',0, ...
                            'pnorm',1,'alpha',1,'epsilon',[],   ...
                            'delta',[],'set_theta',struct('center',[], ...
                            'radius',radius,'pnorm',inf));
data(run_count) = struct('x',[],'s',[]);
Suboptimality = zeros(4,length(all_N),run_count);
Predictability = zeros(4,length(all_N),run_count);
Identifiability = zeros(4,length(all_N),run_count);

for r = 1 : run_count  
    fprintf('Running iteration %d ... \n',r);
    
    %========================= Setting Parameters =========================%
    param(r).W = [2 * rand(m,n) - 1; eye(n); -eye(n)];
    param(r).H = [eye(m); zeros(n,m); zeros(n,m)];
    param(r).h = [zeros(m,1); -ones(n,1); -ones(n,1)];
    param(r).epsilon = epsilon;
    center = radius + 4 * radius * rand(n,1);
    sgn = randi(2,[n,1])-1;
    center(sgn == 0) = -center(sgn == 0);
    param(r).set_theta.center = center;
    theta_star = center + 2 * radius * rand(n,1) - radius;    
    
    %========================== Generate Dataset ==========================%
    random_x = 2*rand(n,max(all_N)+N_test) - 1;
    s = param(r).W(1:m,:) * random_x;
    s = s - eta;
    param(r).delta = delta;
    suboptimal = SubLinear_Model(param(r),theta_star,s(:,1:max(all_N)));
    suboptimal_x = [suboptimal.x];
    param(r).delta = 0;
    optimal = Linear_Model(param(r),theta_star,s(:,max(all_N)+1:end));
    s_test = s(:,max(all_N)+1:end);
    x_test = [optimal.x];
    
    tmp1 = NaN(4,length(all_N));
    tmp2 = NaN(4,length(all_N));
    tmp3 = NaN(4,length(all_N));
    flg = true;
    for N = all_N
        if N <= 10
            kfold = N;
        end
        param(r).epsilon = epsilon;
        ind_N = find(N == all_N);        
        %======= Select the Best Model Based on Validation Process =======%
        tmp_sub = zeros(length(epsilon),kfold);
        tmp_pre = zeros(length(epsilon),kfold);
        s_N = s(:,1:N);
        x_N = suboptimal_x(:,1:N);
        pw = fix(N/kfold);
        for k = 1 : kfold
            s_tr = s_N;
            x_tr = x_N;
            s_tr(:,(k-1)*pw+1:k*pw) = [];
            x_tr(:,(k-1)*pw+1:k*pw) = [];
            s_v = s_N(:,(k-1)*pw+1:k*pw);
            x_v = x_N(:,(k-1)*pw+1:k*pw);    
            data(r).x = x_tr;
            data(r).s = s_tr;
            opt_inv = Linear_Inverse(param(r),data(r));
            theta = [opt_inv.theta];
            for j = 1 : length(epsilon)                
                opt_model = Linear_Model(param(r),theta(:,j),s_v);
                tmp_sub(j,k) = mean( max(theta(:,j)'* (x_v - [opt_model.x]), 0) );
                tmp_pre(j,k) = mean( sqrt(sum((x_v - [opt_model.x]).^2,1)) );
            end            
        end   
        tmp = (kfold-1)/kfold * mean(tmp_sub,2) + 1/kfold * std(tmp_sub,[],2);
        index = find(round(tmp,4) == min(round(tmp,4)));
        ind = max(index);
        param(r).epsilon = [0 epsilon(ind)];
        
        %================== Solve the Inverse Problem ====================%
        data(r).x    = suboptimal_x(:,1:N);
        data(r).s    = s(:,1:N);
        opt_inv      = Linear_Inverse(param(r),data(r));
        theta        = [opt_inv.theta];
        opt_inv_g    = Gupta_Linear_Inverse(param(r),data(r));
        theta_gupta  = opt_inv_g.theta;
        theta_aswani = NaN(n,1);
        if flg
            opt_inv_a    = Aswani_Linear_Inverse_YALMIP(param(r),data(r));       
            theta_aswani = opt_inv_a.theta;
            if sum(isnan(theta_aswani)) ~= 0
                flg = false;
            end
        end
        
        %=============== Evaluate the Model on Test Data =================%        
        % Check Anil Aswani's Solution
        if sum(isnan(theta_aswani)) == 0
            opt_aswani = Linear_Model(param(r),theta_aswani,s_test);
            tmp1(1,ind_N) = mean( max(theta_aswani'* (x_test - [opt_aswani.x]), 0) );
            tmp2(1,ind_N) = mean( sqrt(sum((x_test - [opt_aswani.x]).^2,1)) );
            tmp3(1,ind_N) = norm(theta_aswani - theta_star)/norm(theta_star);
        end
        % Check Vishal Gupta's Solution        
        opt_gupta = Linear_Model(param(r),theta_gupta,s_test);
        tmp1(2,ind_N) = mean( max(theta_gupta'* (x_test - [opt_gupta.x]), 0) );
        tmp2(2,ind_N) = mean( sqrt(sum((x_test - [opt_gupta.x]).^2,1)) );
        tmp3(2,ind_N) = norm(theta_gupta - theta_star)/norm(theta_star);
        % Check SAA Solution
        opt_SAA = Linear_Model(param(r),theta(:,1),s_test);
        tmp1(3,ind_N) = mean( max(theta(:,1)'* (x_test - [opt_SAA.x]), 0) );
        tmp2(3,ind_N) = mean( sqrt(sum((x_test - [opt_SAA.x]).^2,1)) );
        tmp3(3,ind_N) = norm(theta(:,1) - theta_star)/norm(theta_star);
        % Check DRO Solution
        opt_DRO = Linear_Model(param(r),theta(:,2),s_test);
        tmp1(4,ind_N) = mean( max(theta(:,2)'* (x_test - [opt_DRO.x]), 0) );
        tmp2(4,ind_N) = mean( sqrt(sum((x_test - [opt_DRO.x]).^2,1)) );
        tmp3(4,ind_N) = norm(theta(:,2) - theta_star)/norm(theta_star);
    end   
    Suboptimality(:,:,r) = tmp1;
    Predictability(:,:,r) = tmp2;
    Identifiability(:,:,r) = tmp3;
end
%%
shaded = true;
fig1 = figure;
set(fig1, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
font_size = 18;
solve_num = sum(~isnan(squeeze(Suboptimality(1,:,:))),2);
Suboptimality(1,solve_num <= 50,:) = NaN;
Predictability(1,solve_num <= 50,:) = NaN;
h1 = semilogx(all_N, nanmean(Suboptimality(1,:,:),3), ':', 'linewidth', 4);
hold on
h2 = semilogx(all_N, mean(Suboptimality(4,:,:),3), 'linewidth', 4);
h3 = semilogx(all_N, mean(Suboptimality(2,:,:),3), '-.', 'color', [0.47,0.67,0.19], 'linewidth', 4);
xlabel('$N$','Interpreter','latex','FontSize',font_size); 
ylabel('Suboptimality','FontSize',font_size);
set(gca, 'FontSize', font_size);
if shaded
    prc = 25;
    alphaa = 0.1;
    all_N2 = [all_N, flip(all_N)];
    BP1 = [prctile(Suboptimality(1,:,:),prc,3),flip(prctile(Suboptimality(1,:,:),100-prc,3))];
    fill(all_N2(~isnan(BP1)), BP1(~isnan(BP1)), ...
         [0,0.44,0.74], 'LineStyle','none')
    fill(all_N2, [prctile(Suboptimality(4,:,:),prc,3),flip(prctile(Suboptimality(4,:,:),100-prc,3))], ...
         [0.85,0.32,0.10],'LineStyle','none')
    fill(all_N2, [prctile(Suboptimality(2,:,:),prc,3),flip(prctile(Suboptimality(2,:,:),100-prc,3))], ...
         [0.47,0.67,0.19],'LineStyle','none')
    alpha(alphaa)
end
legend([h1 h2 h3],{'BP','DRO','VI'});
cd figs
saveas(gcf,'fig1-b','png')
cd ..

fig2 = figure;
set(fig2, 'Units', 'normalized', 'Position', [0.35, 0.25, 0.4, 0.55])
h1 = semilogx(all_N, nanmean(Predictability(1,:,:),3), ':', 'linewidth', 4);
hold on
h2 = semilogx(all_N, mean(Predictability(4,:,:),3), 'linewidth', 4);
h3 = semilogx(all_N, mean(Predictability(2,:,:),3), '-.', 'color', [0.47,0.67,0.19], 'linewidth', 4);
xlabel('$N$','Interpreter','latex','FontSize',font_size); 
ylabel('Predictability','FontSize',font_size);
set(gca, 'FontSize', font_size);
if shaded
    prc = 25;
    alphaa = 0.1;
    all_N2 = [all_N, flip(all_N)];
    BP1 = [prctile(Predictability(1,:,:),prc,3),flip(prctile(Predictability(1,:,:),100-prc,3))];
    fill(all_N2(~isnan(BP1)), BP1(~isnan(BP1)), ...
         [0,0.44,0.74], 'LineStyle','none')
    fill(all_N2, [prctile(Predictability(4,:,:),prc,3),flip(prctile(Predictability(4,:,:),100-prc,3))], ...
         [0.85,0.32,0.10],'LineStyle','none')
    fill(all_N2, [prctile(Predictability(2,:,:),prc,3),flip(prctile(Predictability(2,:,:),100-prc,3))], ...
         [0.47,0.67,0.19],'LineStyle','none')
    alpha(alphaa)
end
legend([h1 h2 h3],{'BP','DRO','VI'});
cd figs
saveas(gcf,'fig1-c','png')
cd ..