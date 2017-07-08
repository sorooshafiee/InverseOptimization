function optimal = Gupta_Quadratic_Inverse(param, data)
    % Define Variables
    s       = data.s;
    x       = data.x;   
    h       = param.h;
    W       = param.W;
    H       = param.H;
    pnorm   = param.pnorm;
    solver  = param.solver;
    set_theta = param.set_theta;
    
    [n, N] = size(x);
    m = size(s,1);
    nh = length(h);

    % Initialization
    ops = sdpsettings('solver',solver,'verbose',0);
    optimal = struct('Q_xx',[],'Q_xs',[],'q',[],'objective',[],'diagnosis',[]);
    
    % Define Decision Variables
    r = sdpvar(1,N);
    mu = sdpvar(nh,N,'full');
    Q_xx = sdpvar(n,n);
    Q_xs = sdpvar(n,m,'full');
    q = sdpvar(n,1);

    % Declare objective function
    objective = norm(r,pnorm);

    % Declare constraints
    constraint{1} = [mu >= 0, sum(( W*x - H*s - repmat(h,1,N) ) .* mu, 1) <= r, 2*Q_xx*x + Q_xs*s + repmat(q,1,N) == W'*mu];
    constraint{2} = eval(set_theta);
    % Solving the Optimization Problem
    diagnosis = optimize([constraint{:}], objective, ops);
    optimal.Q_xx = value(Q_xx);
    optimal.Q_xs = value(Q_xs);
    optimal.q = value(q);
    optimal.objective = value(objective);
    optimal.diagnosis = diagnosis;
    clearvars -except optimal
end