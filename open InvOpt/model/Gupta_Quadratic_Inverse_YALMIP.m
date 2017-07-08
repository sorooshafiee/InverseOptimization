function optimal = Gupta_Quadratic_Inverse_YALMIP(param, data)
    % Define Variables
    s         = data.s;
    x         = data.x;   
    h         = param.h;
    W         = param.W;
    H         = param.H;
    pnorm     = param.pnorm;
    set_theta = param.set_theta;
    solver    = param.solver;
    
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
    constraint{1} = [mu >= 0, sum(( W*x - H*s - h(:,ones(N,1)) ) .* mu, 1) <= r, 2*Q_xx*x + Q_xs*s + q(:,ones(N,1)) == W'*mu];
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