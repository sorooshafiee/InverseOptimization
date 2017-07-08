function optimal = Gupta_Linear_Inverse_YALMIP(param, data)
    % Define Variables
    s           = data.s;
    x           = data.x;   
    h           = param.h;
    W           = param.W;
    H           = param.H; 
    pnorm       = param.pnorm;
    set_theta   = param.set_theta;
    center      = set_theta.center;
    radius      = set_theta.radius;
    pnorm_theta = set_theta.pnorm;
    solver = param.solver;
    [n, N]      = size(x);
    nh          = length(h);

    % Initialization
    ops = sdpsettings('solver',solver,'verbose',0,'saveduals',0);
    optimal = struct('theta',[],'objective',[],'diagnosis',[]);
    
    % Define Decision Variables
    theta   = sdpvar(n,1);
    r       = sdpvar(1,N);
    mu      = sdpvar(nh,N,'full');

    % Declare objective function
    objective = norm(r,pnorm);

    % Declare constraints
    constraint = [mu >= 0, sum((W*x - H*s - h(:,ones(N,1))) .* mu, 1) <= r, ...
                  theta(:,ones(N,1)) == W'*mu, norm(theta-center,pnorm_theta) <= radius];
        
    % Solving the Optimization Problem
    diagnosis = optimize(constraint, objective, ops);
    optimal.theta = value(theta);
    optimal.objective = value(objective);
    optimal.diagnosis = diagnosis;
    clearvars -except optimal
end