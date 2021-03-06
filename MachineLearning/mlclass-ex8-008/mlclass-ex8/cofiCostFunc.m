function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% Calculate the estimation
estimation = X * Theta';

% Calculate the errors
errors = (estimation - Y);

% Calculate the cost
cost = 1/2 * sum(sum((errors.*R ).^ 2));

% Compute the regularization for Theta and X
Theta_reg = lambda/2*(sum(sum(Theta.^2)));
X_reg = lambda/2*(sum(sum(X.^2)));

regularization = Theta_reg + X_reg;

% Add the regularization to the cost
J = cost + regularization;

% Only use the valid errors
valid_errors = errors .* R;

% Compute the gradient for X
X_grad = valid_errors* Theta + lambda * X;

% Compute the gradient for Theta
Theta_grad = valid_errors' * X + lambda * Theta;


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
