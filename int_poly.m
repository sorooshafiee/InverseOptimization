function int_value = int_poly(x,x_tr,alpha,gamma,p)
    [n,Q] = size(x);
    N = size(x_tr,2);
    x = repmat(reshape(x,[n,1,Q]),[1,N,1]);
    x_tr = repmat(x_tr,[1,1,Q]);
    alpha = repmat(alpha,[1,1,Q]);
    int_value = (sum(alpha.*x,1)./(gamma*(p+1)*sum(x_tr.*x,1))).*((gamma*sum(x_tr.*x,1)+1).^(p+1)-1);
    int_value = sum(int_value,2);
    int_value = int_value(:);
    int_value(isnan(int_value)) = 0;
end