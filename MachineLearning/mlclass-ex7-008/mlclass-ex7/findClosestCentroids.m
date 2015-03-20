function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Find the number of elements in X
m = size(X,1);

% Find the number of centroids
K = size(centroids, 1);

% Loop through all of the elements in X finding the closest centroid
for i = 1:m

    % Expand The current point in X
    curr_example = repmat(X(i,:), K, 1);

    % Calculate the distance from each centroid
    distance = sum((curr_example - centroids).^2, 2);
 
    % Find the index of the minimum distance
    [x, xi] = min(distance);

    % Set the index of the closest centroid
    idx(i) = xi;
end


% =============================================================

end

