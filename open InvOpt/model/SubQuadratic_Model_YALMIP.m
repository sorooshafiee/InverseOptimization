function optimal = SubQuadratic_Model_YALMIP(param, Q_xx, Q_xs, q, s)
    % Define Variables
    h         = param.h;
    W         = param.W;
    H         = param.H;  
    delta     = param.delta;  
    solver    = param.solver;
    n         = length(q);
    N         = size(s,2);
    
    % Initialization
    optimal(1:N) = struct('x',[],'objective',[],'diagnosis',[]);
    ops = sdpsettings('solver',solver,'verbose',0);
    
    for k = 1 : N    
        % first model
        x1 = sdpvar(n,1);

        objective1 = dot(x1, Q_xx * x1 + Q_xs * s(:,k) + q);

        constraint1 = [W*x1 >= H*s(:,k) + h];

        diagnosis1 = optimize(constraint1, objective1, ops);

        x2 = sdpvar(n,1);

        Q2 = diag(rand(n,1));
        q2 = rand(n,1);
        objective2 = dot(x2, Q2 * x2 - q2);

        constraint2 = [W*x2 >= H*s(:,k) + h, dot(x2, Q_xx * x2 + Q_xs * s(:,k) + q) <= value(objective1) + delta * rand];

        diagnosis2 = optimize(constraint2, objective2, ops);

        optimal(k).x = value(x2); 
        optimal(k).objective = value(objective2);
        optimal(k).diagnosis = diagnosis2;
    end
    clearvars -except optimal
end