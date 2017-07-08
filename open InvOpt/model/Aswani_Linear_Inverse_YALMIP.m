function optimal = Aswani_Linear_Inverse_YALMIP(param, data)
    % Define Variables
    s           = data.s;
    x           = data.x;   
    h           = param.h;
    W           = param.W;
    H           = param.H; 
    set_theta   = param.set_theta;
    center      = set_theta.center;
    radius      = set_theta.radius;
    pnorm_theta = set_theta.pnorm;
    solver = param.solver;
    [n, N]    = size(x);

    % Initialization
    ops = sdpsettings('bilevel.maxiter',2000,'bilevel.outersolver','CBC', ...
                      'bilevel.innersolver','CBC','verbose',0,'saveduals',0);
    optimal = struct('theta',[],'objective',[],'diagnosis',[]);
    
    % Define Decision Variables
    theta   = sdpvar(n,1);
    y       = sdpvar(n,N,'full');

    % Declare objective function
    outer_objective = sum(sum(y.^2)) - 2 * sum(sum(y.*x));
    inner_objective = sum(theta' * y);

    % Declare constraints
    outer_constraint = norm(theta-center,pnorm_theta) <= radius;
    inner_constraint = W*y >= H*s + h(:,ones(N,1));
        
    % Solving the Optimization Problem
    diagnosis = solvebilevel(outer_constraint,outer_objective,inner_constraint,inner_objective,y, ops); 
    optimal.theta = value(theta);
    optimal.objective = value(outer_objective);
    optimal.diagnosis = diagnosis;
    clearvars -except optimal
end