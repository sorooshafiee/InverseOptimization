function optimal = Aswani_Linear_Inverse_YALMIP(param, data)
    % Define Variables
    s         = data.s;
    x         = data.x;   
    h         = param.h;
    W         = param.W;
    H         = param.H; 
    set_theta = param.set_theta;
    center    = set_theta.center;
    radius    = set_theta.radius;
    pnorm     = set_theta.pnorm;
    [n, N]    = size(x);

    % Initialization
    ops = sdpsettings('bilevel.maxiter',2000,'verbose',0,'saveduals',0);
    optimal = struct('theta',[],'objective',[],'diagnosis',[]);
    
    % Define Decision Variables
    theta   = sdpvar(n,1);
    y       = sdpvar(n,N,'full');

    % Declare objective function
    outer_objective = sum(sum(y.^2)) - 2 * sum(sum(y.*x));
    inner_objective = sum(theta' * y);

    % Declare constraints
    if isinf(pnorm)
        outer_constraint = [theta - center <= radius, -theta + center <= radius];
    elseif pnorm == 1
        sp = sdpvar(n,1);
        outer_constraint = [theta - center <= sp, -theta + center <= sp, sum(sp) <= radius];
    elseif pnorm == 2
        outer_constraint = dot(theta - center, theta - center) <= radius^2;
    else
        outer_constraint = norm(theta - center, pnorm) <= radius;
    end
    inner_constraint = W*y >= H*s + repmat(h,[1,N]);
        
    % Solving the Optimization Problem
    diagnosis = solvebilevel(outer_constraint,outer_objective,inner_constraint,inner_objective,y, ops); 
    optimal.theta = value(theta);
    optimal.objective = value(outer_objective);
    optimal.diagnosis = diagnosis;
    clearvars -except optimal
end