function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Create a y matrix with K columns
y_matrix = eye(num_labels)(y,:);

% Regularize the thetas by removing the bias unit column
theta1_reg = Theta1(:, [2:end]);
theta2_reg = Theta2(:, [2:end]);

% Add ones to the X data matrix
a1 = [ones(m, 1) X];

% Calculate z2
z2 = a1 * Theta1';

% Calculate a2
a2 = sigmoid(z2);

% Add 1's to a2
a2 = [ones(size(a2, 1), 1) a2];

% Calculate z3
z3 = a2 * Theta2';

% Calculate a3 == h(x)
a3 = sigmoid(z3);

% Compute cost of y = 1
% remember to do element wise multiplication
y1 = sum(-y_matrix .* log(a3));

% Compute the cost of y = 0
% remember to do element wise multiplication
y0 = sum((1 .- y_matrix) .* log(1 .- a3));


% Compute the regularization for the cost
reg_cost = lambda / (2 * m) * (sum(sum(theta1_reg.^2)) + sum(sum(theta2_reg.^2)) );

% Sum the results and add in the regularization
J = 1/(m) * sum(y1-y0) + reg_cost;


% Compute the delta in layer 3
d3 = a3 - y_matrix;

% Compute the delta in layer 2
d2 =  d3 * theta2_reg .*  sigmoidGradient(z2);

% Compute the gradient for d2
delta_2 = d3' * a2;

% Compute the gradient for d3
delta_1 = d2' * a1;

% Compute the gradients for theta
Theta1_grad = (1/m) * delta_1 + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, (2:end))];
Theta2_grad = (1/m) * delta_2 + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, (2:end))];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
