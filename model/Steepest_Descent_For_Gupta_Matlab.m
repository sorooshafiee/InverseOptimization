function optimal = Steepest_Descent_For_Gupta_Matlab(param,alpha,x_tr,s,x_0)
    % Define Variables    
    kernel_ = param.kernel;
    sigma   = param.sigma;
    
    N = size(s,2);
    n = size(x_tr,1);

    % Initialization
    optimal = struct('x',[]);    
    
    if nargin == 4
        x_0 = 5*rand(n,N);
    end
    
    x = x_0;
    count = 0;
    while true
        count = count + 1;
        x = x + 1e-3 * (- s - alpha * Kernel_Function(x_tr',x',kernel_,sigma));
        x(x<0) = 0;
        x(x>5) = 5;
        if count >= 1e5
            break
        end
    end        
    optimal.x = x;
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
            K	= (X1*(eta*X2)' + 1).^p;
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