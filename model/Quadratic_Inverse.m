function optimal = Quadratic_Inverse(param, data)
    % Define Variables
    s         = data.s;
    x         = data.x;   
    epsilon2  = param.epsilon2;
    solver    = param.solver;
    d         = param.d;
    C         = param.C;
    h         = param.h;
    W         = param.W;
    H         = param.H;
    set_theta = param.set_theta;
    if isfield(param,'isfeasible')
        isfeasible = param.isfeasible;
    else
        isfeasible = 0;
    end
    if isfield(param,'isdiag')
        isdiag = param.isdiag;
    else
        isdiag = 0;
    end
    [n, N] = size(x);
    m = size(s,1);
    nh = size(h,1);
    nd = size(d,1);
    ne = length(epsilon2);
    tol = param.tol;
    alpha = param.alpha;
    
    % Initialization
    optimal(1:ne) = struct('Q_xx',[],'Q_xs',[],'q',[],'objective',[],'diagnosis',[]);
    ops = sdpsettings('solver',solver,'verbose',0);
    
    for j = 1 : ne    
        % Define Decision Variables
        lambda = sdpvar(1,1);
        r = sdpvar(1,N); 
        mu_1 = sdpvar(nh,N,'full');
        mu_2 = sdpvar(nh,N,'full');
        gamma = sdpvar(nh,N,'full');
        tau = sdpvar(1,1);
        phi_1 = sdpvar(nd,N,'full');
        phi_2 = sdpvar(nd,N,'full');
        if isdiag
            Q_xx = diag(sdpvar(n,1));
        else
            Q_xx = sdpvar(n,n);
        end
        Q_xs = sdpvar(n,m,'full');
        q = sdpvar(n,1);  
        
        % Declare objective function        
        objective = tau + 1 / alpha * (lambda*epsilon2(j) + 1/N*sum(r)); 

        % Declare constraints
        constraint = cell(2*N+2,1);
        cnt = 1;
        constraint{cnt} = [lambda >= 0, mu_1 >= 0, mu_2 >= 0, phi_1 >= 0, phi_2 >= 0, gamma >= 0];
        cnt = cnt + 1;
        for i = 1 : N 
            chi_1 = 0.5*( -C'*phi_1(:,i) + H'*( mu_1(:,i) + gamma(:,i)) - 2*lambda*s(:,i) );
            zeta_1 = 0.5*( -q - W'*mu_1(:,i) - 2*lambda*x(:,i));
            eta_1 = 0.5*( q - W'*gamma(:,i) );
            rho_1 = tau + r(i) + lambda*(x(:,i)'*x(:,i) + s(:,i)'*s(:,i)) + d'*phi_1(:,i) + h'*(mu_1(:,i)+gamma(:,i));
            constraint{cnt} = [lambda*eye(m), -0.5*Q_xs', 0.5*Q_xs', chi_1; ...
                               -0.5*Q_xs, lambda*eye(n)-Q_xx, zeros(n), zeta_1; ...
                               0.5*Q_xs, zeros(n), Q_xx, eta_1; ...
                               chi_1', zeta_1', eta_1', rho_1] >= -tol*eye(m+2*n+1);
            cnt = cnt + 1;
            if isfeasible
                constraint{cnt} = r(i) >= 0;
            else                
                chi_2 = 0.5*( -C'*phi_2(:,i) + H'*mu_2(:,i) - 2*lambda*s(:,i) );
                zeta_2 = 0.5*( -W'*mu_2(:,i) - 2*lambda*x(:,i));
                rho_2 = r(i) + lambda*(x(:,i)'*x(:,i) + s(:,i)'*s(:,i)) + d'*phi_2(:,i) + h'*mu_2(:,i);
                constraint{cnt} = [lambda*eye(m), zeros(m,n), chi_2; ...
                                    zeros(m,n), lambda*eye(n), zeta_2; ...
                                    chi_2', zeta_2', rho_2] >= -tol*eye(m+n+1);
            end
            cnt = cnt + 1;
        end
        % Constraint for the set \Theta
        constraint{cnt} = eval(set_theta);
        % Solving the Optimization Problem           
        diagnosis = optimize([constraint{:}], objective, ops);
        optimal(j).Q_xx = value(Q_xx); 
        optimal(j).Q_xs = value(Q_xs); 
        optimal(j).q = value(q);
        optimal(j).objective = value(objective);
        optimal(j).diagnosis = diagnosis;
    end
    clearvars -except optimal
end