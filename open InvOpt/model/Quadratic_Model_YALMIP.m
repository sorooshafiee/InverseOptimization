function optimal = Quadratic_Model_YALMIP(param, Q_xx, Q_xs, q, s)
    % Define Variables
    h         = param.h;
    W         = param.W;
    H         = param.H;  
    solver    = param.solver;
    n         = length(q);
    N         = size(s,2);
    
    % Initialization
    optimal(1:N) = struct('x',[],'objective',[],'diagnosis',[]);
    ops = sdpsettings('solver',solver,'verbose',0);
    
    for k = 1 : N    
        % Define Decision Variables
        x = sdpvar(n,1);
        
        % Declare objective function        
        objective = dot(x, Q_xx * x + Q_xs * s(:,k) + q);

        % Declare constraints        
        constraint = [W*x >= H*s(:,k) + h];

        % Solving the Optimization Problem           
        diagnosis = optimize(constraint, objective, ops);
        optimal(k).x = value(x); 
        optimal(k).objective = value(objective);
        optimal(k).diagnosis = diagnosis;
    end
    clearvars -except optimal
end