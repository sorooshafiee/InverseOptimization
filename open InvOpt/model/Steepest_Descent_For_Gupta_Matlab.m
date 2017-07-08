function optimal = Steepest_Descent_For_Gupta_Matlab(param,alpha,x_tr,s)
    % Define Variables    
    kernel_ = param.kernel;
    sigma   = param.sigma;
    
    N = size(s,2);
    n = size(x_tr,1);

    % Initialization
    optimal = struct('x',[]);    
    
    x = 5*rand(n,N);
    count = 0;
    while true
        count = count + 1;
        K = Kernel_Function(x_tr',x',kernel_,sigma);
        d = - s - alpha * K;
        x_k = x + 1e-3*d;
        x_k(x_k<0) = 0;
        x_k(x_k>5) = 5;
        x = x_k;
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