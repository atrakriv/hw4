function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"


Y = 1;                    % output size
d = size(data,2);         % number of features
weights = zeros(d,Y);   % 1*(d+1) ?

for it = 1 : maxiter
    phi = data;
    t = labels;
    y = (sigmf(weights'*phi',[1 0]))';
    R = diag(y.*(1-y));
    R_inv = diag(1/(y.*(1-y)));
    z = phi*weights - R_inv*(y-t);
    weights = inv(phi'*R*phi)*phi'*R*z;
    
    new_y = (sigmf(weights'*phi',[1 0]))';
    diff = mean(abs(y - new_y));
    if (abs(diff) < epsilon)
        break;
    end
end
end