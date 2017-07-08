function optimal = Gupta_Semi_Parametric_Inverse_YALMIP(param, data)
    % Define Variables
    s       = data.s;
    x       = data.x;   
    h       = param.h;
    W       = param.W;
    H       = param.H;
    pnorm   = param.pnorm;
    kernel_ = param.kernel;
    sigma   = param.sigma;
    kappa   = param.kappa;    
    solver  = param.solver;
    
    [n, N] = size(x);
    nh = length(h);
    nk = length(kappa);
    ns = length(sigma);

    % Initialization
    ops = sdpsettings('solver',solver,'verbose',0,'saveduals',0);
    optimal(1:ns,1:nk) = struct('alpha',[],'objective',[],'diagnosis',[]);    
    
    for i = 1 : ns
        K = Kernel_Function(x',x',kernel_,sigma(i));
        for j = 1 : nk
            % Define Decision Variables
            r = sdpvar(1,N);
            mu = sdpvar(nh,N,'full');
            alpha = sdpvar(n,N,'full');

            % Declare objective function
            objective = sum(sum((alpha*K) .* alpha, 2), 1);

            % Declare constraints
            constraint = [mu >= 0, r >= 0, sum((W*x - H*s - h(:,ones(N,1))) .* mu, 1) <= r, s + alpha*K == W'*mu, norm(r,pnorm) <= kappa(j)];
            % Solving the Optimization Problem
            diagnosis = optimize(constraint, objective, ops);
            optimal(i,j).alpha = value(alpha);
            optimal(i,j).objective = value(objective);
            optimal(i,j).diagnosis = diagnosis;
        end
    end
    clearvars -except optimal
end

function K = Kernel_Function (X1,X2,kernel_,lengthScale)
    if length(kernel_)>=4 && strcmp(kernel_(1:4),'poly')
        p		= str2double(kernel_(5:end));
        kernel_	= 'poly';
    end
    eta	= 1/lengthScale^2;
    switch lower(kernel_)
        case 'gauss',
            K	= exp(-eta*distSqrd(X1,X2));
        case 'poly',
            K	= (X1*X2' + 1).^p;
        case 'linear',
            K	= X1*X2';
        otherwise,
            error('Unrecognised kernel function type: %s', kernel_)
    end
end

function D2 = distSqrd(X,Y)
    nx	= size(X,1);
    ny	= size(Y,1);
    D2	= sum(X.^2,2)*ones(1,ny) + ones(nx,1)*sum(Y.^2,2)' - 2*(X*Y');
end