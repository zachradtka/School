function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


C_vec     = [0.01 0.03 0.1 0.3 1 3 10]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10]';


min_error = 1;
best_params = [1 1];

x1 = [1 2 1]; x2 = [0 4 -1];


for i = 1:length(C_vec)
    for j = 1:length(sigma_vec)
        model= svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));

        fprintf('C: %f\tsigma: %f\terror: %f\n', C_vec(i), sigma_vec(j), error);

        if error < min_error
            min_error = error;
            best_params = [i j];
        end
    end
end
fprintf('Values being used are\n');
fprintf('C is: %f\n', C_vec(best_params(1)));
fprintf('sigma is: %f\n', sigma_vec(best_params(2)));
fprintf('Error is: %f\n', min_error);


C = C_vec(best_params(1));
sigma = sigma_vec(best_params(2));



% =========================================================================

end
