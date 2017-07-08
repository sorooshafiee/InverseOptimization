function optimal = Linear_Inverse_YALMIP(param, data)
    % Define Variables
    s           = data.s;
    x           = data.x;   
    C           = param.C;
    d           = param.d;
    h           = param.h;
    W           = param.W;
    H           = param.H; 
    epsilon     = param.epsilon;
    alpha       = param.alpha;
    pnorm       = param.pnorm;
    set_theta   = param.set_theta;
    center      = set_theta.center;
    radius      = set_theta.radius;
    pnorm_theta = set_theta.pnorm;
    solver      = param.solver;
    if isfield(param, 'delta')
        delta  = param.delta;
    else
        delta = 0;
    end
    
    [n, N]    = size(x);
    m         = size(s,1);
    nh        = length(h);
    nd        = length(d);
    ne        = length(epsilon);

    % Initialization
    ops = sdpsettings('solver',solver,'verbose',0);
    optimal(1:ne) = struct('theta',[],'objective',[],'diagnosis',[]);
    
    for j = 1 : ne
        % Define Decision Variables
        lambda = sdpvar(1,1);
        tau = sdpvar(1,1);
        r = sdpvar(1,N);
        theta = sdpvar(n,1);
        gamma = sdpvar(nh,N,'full');
        mu_1 = sdpvar(nh,N,'full');
        mu_2 = sdpvar(nh,N,'full');
        phi_1 = sdpvar(nd,N,'full');
        phi_2 = sdpvar(nd,N,'full');

        % Declare objective function
        objective = tau + 1 / alpha * (lambda*epsilon(j) + 1/N*sum(r));

        % Declare constraints        
        constraint{1} = [norm(theta-center,pnorm_theta) <= radius];
        constraint{2} = [gamma >= 0, mu_1 >= 0, mu_2 >= 0, phi_1 >= 0, phi_2 >=0, ...
                         sum( ( W*x - H*s - h(:,ones(N,1)) ) .* (mu_1 + gamma), 1) <= r + tau(:,ones(N,1)) + delta(:,ones(N,1)), ...
                         sum( ( W*x - H*s - h(:,ones(N,1)) ) .* mu_2, 1) <= r, ...
                         theta(:,ones(N,1)) == W' * gamma];
        switch pnorm
            case 1
                s1 = sdpvar(m+n,N,'full');
                s2 = sdpvar(m+n,N,'full');
                constraint{3} = [ [C' * phi_1 - H' * (mu_1 + gamma); W' * (mu_1 + gamma)] <= s1, ...
                                 -[C' * phi_1 - H' * (mu_1 + gamma); W' * (mu_1 + gamma)] <= s1, ...
                                  [C' * phi_2 - H' * mu_2; W' * mu_2] <= s2, ...
                                 -[C' * phi_2 - H' * mu_2; W' * mu_2] <= s2, ...
                                  sum(s1,1) <= lambda , ...
                                  sum(s2,1) <= lambda];
            case 2
                constraint{3} = [sqrt(sum([C' * phi_1 - H' * (mu_1 + gamma); W' * (mu_1 + gamma)].^2,1)) <= lambda , ...
                                 sqrt(sum([C' * phi_2 - H' * mu_2; W' * mu_2].^2,1)) <= lambda];
            case inf
                constraint{3} = [ C' * phi_1 - H' * (mu_1 + gamma) <= lambda, ...
                                 -C' * phi_1 + H' * (mu_1 + gamma) <= lambda, ...
                                  W' * (mu_1 + gamma) <= lambda , ...
                                 -W' * (mu_1 + gamma) <= lambda, ...
                                  C' * phi_2 - H' * mu_2 <= lambda, ...
                                 -C' * phi_2 + H' * mu_2 <= lambda, ...
                                  W' * mu_2 <= lambda , ...
                                 -W' * mu_2 <= lambda];
            otherwise
                error('The norm should be L_1, L_2, or L_infty');
        end
                  
        % Solving the Optimization Problem
        diagnosis = optimize([constraint{:}], objective, ops);
        optimal(j).theta = value(theta);
        optimal(j).objective = value(objective);
        optimal(j).diagnosis = diagnosis;
    end
    clearvars -except optimal
end