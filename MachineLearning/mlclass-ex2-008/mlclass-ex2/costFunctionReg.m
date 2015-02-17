function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute the cost and gradient 
[cost_non_reg, grad_non_reg] = costFunction(theta, X, y);

% Don't penalize theta0, in octave theta(1) is theta0
theta_reg = [0; theta([2:length(theta)],:)];

% Compute the regularization
reg = lambda / (2 * m) * (theta_reg' * theta_reg);

% Add the regularization terms to the cost
J = cost_non_reg + reg;

grad_reg = (lambda / m) * (theta_reg);

grad = grad_non_reg + grad_reg;

% =============================================================

end
