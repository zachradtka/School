function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Don't penalize theta0, in octave theta(1) is theta0
theta_reg = [0; theta([2:length(theta)],:)];


% Compute the perdictions
predictions = X * theta;

% Compute the errors
errors = (predictions - y);

cost_non_reg = 1/(2*m) * errors' * errors;

% Compute the regularization
cost_reg = lambda / (2 * m) * (theta_reg' * theta_reg);

% The cost is the sum of the regularized version and the non regularized version
J = cost_non_reg + cost_reg;

% Compute the non regularized gradient descent
grad_non_reg = (1/m) * X' * errors;

% Compute the regularized term for grad descent
grad_reg = (lambda / m) * (theta_reg);

% Add the regularized term to the non regularized term to produce gradient descent
grad = grad_non_reg + grad_reg;



% =========================================================================

grad = grad(:);

end
