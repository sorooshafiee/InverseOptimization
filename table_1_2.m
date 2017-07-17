clc
clearvars
rng('shuffle')

addpath('./model/')

all_n = [10, 20, 30, 40, 50];
all_m = [10, 20, 30, 40, 50];
N = 10;
kfold = 5;
N_test = 1000;
radius = 1;
run_count = 100;
epsilon = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1];
param(1:run_count) = struct('W',[],'H',[],'h',[],'C',[],'d',[], ...
                            'pnorm',1,'alpha',0.9,'epsilon',[],   ...
                            'delta',[],'set_theta',struct('center',[], ...
                            'radius',radius,'pnorm',inf));
delta = 1;
eta = 0;
data(run_count) = struct('x',[],'s',[]);
Suboptimality = zeros(3,length(all_n),length(all_m),run_count);
Predictability = zeros(3,length(all_n),length(all_m),run_count);
Identifiability = zeros(3,length(all_n),length(all_m),run_count);

for r = 1 : run_count  
    fprintf('Running iteration %d ... \n',r);
    tmp1 = zeros(3,length(all_n),length(all_m));
    tmp2 = zeros(3,length(all_n),length(all_m));
    tmp3 = zeros(3,length(all_n),length(all_m));
    for n = all_n
        for m = all_m
            ind_n = find(n == all_n);
            ind_m = find(m == all_m);
            %========================= Setting Parameters =========================%
            param(r).W = [2 * rand(m,n) - 1; eye(n); -eye(n)];
            param(r).H = [eye(m); zeros(n,m); zeros(n,m)];
            param(r).h = [zeros(m,1); -ones(n,1); -ones(n,1)];
            param(r).C = zeros(1,m);
            param(r).d = 0;
            param(r).delta = delta;
            param(r).epsilon = epsilon;
            center = radius + 4 * radius * rand(n,1);
            sgn = randi(2,[n,1])-1;
            center(sgn == 0) = -center(sgn == 0);
            param(r).set_theta.center = center;
            theta_star = center + 2 * radius * rand(n,1) - radius;    

            %========================== Generate Dataset ==========================%
            random_x = 2*rand(n,N) - 1;
            s = param(r).W(1:m,:) * random_x;
            s = s - eta;
            suboptimal = SubLinear_Model(param(r),theta_star,s);
            suboptimal_x = [suboptimal.x];  
            param(r).delta = 0;
            
            %================== Solve the Inverse Problem ====================%
            data(r).x    = suboptimal_x;
            data(r).s    = s;
            opt_inv_g    = Gupta_Linear_Inverse(param(r),data(r));
            theta_gupta  = opt_inv_g.theta;

            %========= Select the Best Model Based on Validation Data ========%
            tmp_sub = zeros(length(epsilon),kfold);
            tmp_pre = zeros(length(epsilon),kfold);
            pw = fix(N/kfold);
            for k = 1 : kfold
                s_tr = s;
                x_tr = suboptimal_x;
                s_tr(:,(k-1)*pw+1:k*pw) = [];
                x_tr(:,(k-1)*pw+1:k*pw) = [];
                s_v = s(:,(k-1)*pw+1:k*pw);
                x_v = suboptimal_x(:,(k-1)*pw+1:k*pw);    
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
            opt_inv = Linear_Inverse(param(r),data(r));
            theta = [opt_inv.theta];

            %=============== Evaluate the Model on Test Data =================%
            random_x = 2*rand(n,N_test) - 1;
            s = param(r).W(1:m,:) * random_x;
            s = s - eta;
            optimal = Linear_Model(param(r),theta_star,s);
            x = [optimal.x];      
            
            % Check Vishal Gupta's Solution
            opt_gupta = Linear_Model(param(r),theta_gupta,s);
            tmp1(1,ind_n,ind_m) = mean( max(theta_gupta'* (x - [opt_gupta.x]), 0) );
            tmp2(1,ind_n,ind_m) = mean( sqrt(sum((x - [opt_gupta.x]).^2,1)) );
            tmp3(1,ind_n,ind_m) = norm(theta_gupta - theta_star)/norm(theta_star);
            % Check SAA Solution
            opt_SAA = Linear_Model(param(r),theta(:,1),s);
            tmp1(2,ind_n,ind_m) = mean( max(theta(:,1)'* (x - [opt_SAA.x]), 0) );
            tmp2(2,ind_n,ind_m) = mean( sqrt(sum((x - [opt_SAA.x]).^2,1)) );
            tmp3(2,ind_n,ind_m) = norm(theta(:,1) - theta_star)/norm(theta_star);
            % Check DRO Solution
            opt_DRO = Linear_Model(param(r),theta(:,2),s);
            tmp1(3,ind_n,ind_m) = mean( max(theta(:,2)'* (x - [opt_DRO.x]), 0) );
            tmp2(3,ind_n,ind_m) = mean( sqrt(sum((x - [opt_DRO.x]).^2,1)) );
            tmp3(3,ind_n,ind_m) = norm(theta(:,2) - theta_star)/norm(theta_star);       
        end
    end
    Suboptimality(:,:,:,r) = tmp1;
    Predictability(:,:,:,r) = tmp2;
    Identifiability(:,:,:,r) = tmp3;
end
%% Table in Latex
fileID = fopen('.\tables\table_1_2.tex','w');

tmp_sub = mean(Suboptimality,4);
[~,ind] = min(tmp_sub,[],1);
ind = squeeze(ind);
fprintf(fileID,'\\documentclass{article}\n\\usepackage[table]{xcolor}\n\\usepackage{multirow}\n\\begin{document}\n\\begin{table}\n\\centering \n');
caption = 'The average of supotimality';
fprintf(fileID,'\\caption{%s} \n',caption);
fprintf(fileID,'\\begin{tabular}{%s}\\hline \n',repmat('c',[1,length(all_m)+2]));
fprintf(fileID,'& & \\multicolumn{%d}{c}{$m$} \\\\ \\cline{3-%d} \n',length(all_m),length(all_m)+2);
ltx_m = latex(sym(all_m(:)'));
fprintf(fileID,'$n$ & methods & %s \\\\ \\noalign{\\vskip 1pt} \\hline \\noalign{\\vskip 1pt} \n',ltx_m(28:end-18));
for n = all_n
    ind_n = find(n == all_n);
    fprintf(fileID,'\\multirow{3}{*}{%d} & VI ',n);
    for m = all_m
        ind_m = find(m == all_m);
        tmp = squeeze(Suboptimality(1,ind_n,ind_m,:))';
        str1 = value2latex(mean(tmp));
        if ind(ind_n,ind_m) == 1
            fprintf(fileID,'& \\cellcolor{gray!25} {$%s$}',str1);
        else
            fprintf(fileID,'& $%s $',str1);
        end
    end
    fprintf(fileID,' \\\\ \n');
    fprintf(fileID,' & SAA ');
    for m = all_m
        ind_m = find(m == all_m);
        tmp = squeeze(Suboptimality(2,ind_n,ind_m,:))';
        str1 = value2latex(mean(tmp));
        if ind(ind_n,ind_m) == 2
            fprintf(fileID,'& \\cellcolor{gray!25} {$%s$}',str1);
        else
            fprintf(fileID,'& $%s $',str1);
        end
    end
    fprintf(fileID,' \\\\ \n');
    fprintf(fileID,' & DRO ');
    for m = all_m
        ind_m = find(m == all_m);
        tmp = squeeze(Suboptimality(3,ind_n,ind_m,:))';
        str1 = value2latex(mean(tmp));
        if ind(ind_n,ind_m) == 3
            fprintf(fileID,'& \\cellcolor{gray!25} {$%s$}',str1);
        else
            fprintf(fileID,'& $%s $',str1);
        end
    end
    fprintf(fileID,' \\\\ \\noalign{\\vskip 1pt} \\hline \\noalign{\\vskip 1pt} \n');
end
fprintf(fileID,'\\end{tabular} \n\\end{table}\n\n\n');

tmp_opt = mean(Predictability,4);
[~,ind] = min(tmp_opt,[],1);
ind = squeeze(ind);
fprintf(fileID,'\\begin{table}\n\\centering \n');
caption = 'The average of predictability';
fprintf(fileID,'\\caption{%s} \n',caption);
fprintf(fileID,'\\begin{tabular}{%s}\\noalign{\\vskip 1pt} \\hline \\noalign{\\vskip 1pt} \n',repmat('c',[1,length(all_m)+2]));
fprintf(fileID,'& & \\multicolumn{%d}{c}{$m$} \\\\ \\cline{3-%d} \n',length(all_m),length(all_m)+2);
ltx_m = latex(sym(all_m(:)'));
fprintf(fileID,'$n$ & methods & %s \\\\ \\noalign{\\vskip 1pt} \\hline \\noalign{\\vskip 1pt} \n',ltx_m(28:end-18));
for n = all_n
    ind_n = find(n == all_n);
    fprintf(fileID,'\\multirow{3}{*}{%d} & VI ',n);
    for m = all_m
        ind_m = find(m == all_m);
        tmp = squeeze(Predictability(1,ind_n,ind_m,:))';
        str1 = value2latex(mean(tmp));
        if ind(ind_n,ind_m) == 1
            fprintf(fileID,'& \\cellcolor{gray!25} {$%s$}',str1);
        else
            fprintf(fileID,'& $%s $',str1);
        end
    end
    fprintf(fileID,' \\\\ \n');
    fprintf(fileID,' & SAA ');
    for m = all_m
        ind_m = find(m == all_m);
        tmp = squeeze(Predictability(2,ind_n,ind_m,:))';
        str1 = value2latex(mean(tmp));
        if ind(ind_n,ind_m) == 2
            fprintf(fileID,'& \\cellcolor{gray!25} {$%s$}',str1);
        else
            fprintf(fileID,'& $%s $',str1);
        end
    end
    fprintf(fileID,' \\\\ \n');
    fprintf(fileID,' & DRO ');
    for m = all_m
        ind_m = find(m == all_m);
        tmp = squeeze(Predictability(3,ind_n,ind_m,:))';
        str1 = value2latex(mean(tmp));
        if ind(ind_n,ind_m) == 3
            fprintf(fileID,'& \\cellcolor{gray!25} {$%s$}',str1);
        else
            fprintf(fileID,'& $%s $',str1);
        end
    end
    fprintf(fileID,' \\\\ \\noalign{\\vskip 1pt} \\hline \\noalign{\\vskip 1pt} \n');
end
fprintf(fileID,'\\end{tabular} \n\\end{table}\n\n\n\\end{document}\n');
fclose(fileID);
cd .\tables\
command = 'pdflatex table1.tex';
[status,cmdout] = system(command);
cd ..