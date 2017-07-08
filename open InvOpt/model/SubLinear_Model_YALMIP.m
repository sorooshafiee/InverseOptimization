function optimal = SubLinear_Model_YALMIP(param, theta, s)
    % Define Variables
    h         = param.h;
    W         = param.W;
    H         = param.H;  
    delta     = param.delta;  
    solver    = param.solver;
    n         = length(theta);
    N         = size(s,2);
    nh        = length(h);
    
    % Initialization
    optimal(1:N) = struct('x',[],'objective',[],'diagnosis',[]);
    ops = sdpsettings('solver',solver,'verbose',0);
    
    for k = 1 : N    
        % Define Decision Variables
        x = sdpvar(n,1);
        lambda = sdpvar(nh,1);
        
        % Declare objective function  
        random_theta = sign(theta) .* rand(size(theta));
        objective = dot(x, random_theta);

        % Declare constraints        
        constraint = [lambda >= 0, W*x >= H*s(:,k) + h, W' * lambda == theta, ...
                      dot(x,theta) - dot(s(:,k),H'*lambda) - dot(h,lambda) <= delta*rand];

        % Solving the Optimization Problem           
        diagnosis = optimize(constraint, objective, ops);
        optimal(k).x = value(x); 
        optimal(k).objective = value(objective);
        optimal(k).diagnosis = diagnosis;
    end
    clearvars -except optimal
end