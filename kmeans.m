% Load the dataset with original column headers preserved
data = readtable('Mall_Customers.csv', 'VariableNamingRule', 'preserve');

% Display the column names to check the headers
data.Properties.VariableNames

% Extract relevant features for clustering (use correct column names)
X = data{:, {'Age', 'Annual Income (k$)', 'Spending Score (1-100)'}};

% Step 1: Determine the optimal number of clusters using the Elbow Method
max_k = 10;  % Maximum number of clusters to evaluate
inertia_values = zeros(max_k, 1);  % Store inertia for each k

for k = 1:max_k
    % Initialize centroids randomly from the dataset
    centroids = X(randperm(size(X, 1), k), :);
    num_samples = size(X, 1);
    cluster_assignment = zeros(num_samples, 1);
    
    % K-means loop
    max_iters = 100;
    threshold = 1e-4; % Threshold for centroid movement to check convergence
    prev_centroids = centroids;
    
    for iter = 1:max_iters
        % Step 2: Assign points to the nearest centroid
        for i = 1:num_samples
            distances = sqrt(sum((X(i, :) - centroids) .^ 2, 2));
            [~, cluster_assignment(i)] = min(distances);
        end

        % Step 3: Update the centroids
        for j = 1:k
            points_in_cluster = X(cluster_assignment == j, :);
            if ~isempty(points_in_cluster)
                centroids(j, :) = mean(points_in_cluster, 1);
            end
        end

        % Check if centroids have converged
        if max(max(abs(centroids - prev_centroids))) < threshold
            break;
        end
        prev_centroids = centroids;
    end
    
    % Calculate inertia (Sum of squared distances)
    inertia = 0;
    for i = 1:num_samples
        centroid = centroids(cluster_assignment(i), :);
        inertia = inertia + sum((X(i, :) - centroid) .^ 2);
    end
    inertia_values(k) = inertia;  % Store inertia for the current k
end

% Step 2: Plot the Elbow Curve
figure;
plot(1:max_k, inertia_values, '-o');
title('Elbow Method for Optimal k');
xlabel('Number of Clusters (k)');
ylabel('Inertia (Sum of Squared Errors)');
grid on;

% Step 3: Automate detection of the elbow (optimal k)
diff1 = diff(inertia_values);  % First derivative (rate of decrease)
diff2 = diff(diff1);           % Second derivative (rate of change of decrease)

% Find the point with the maximum change in curvature (i.e., where diff2 is smallest)
[~, optimal_k] = min(diff2);

% Adding 1 to the optimal_k because the diff operations reduce dimensionality
optimal_k = optimal_k + 1;

disp(['Optimal number of clusters: ', num2str(optimal_k)]);

% Step 4: Run K-means with the optimal number of clusters
centroids = X(randperm(size(X, 1), optimal_k), :);
cluster_assignment = zeros(num_samples, 1);

% K-means loop for the optimal number of clusters
for iter = 1:max_iters
    % Assign each point to the nearest centroid
    for i = 1:num_samples
        distances = sqrt(sum((X(i, :) - centroids) .^ 2, 2));
        [~, cluster_assignment(i)] = min(distances);
    end
    
    % Update the centroids
    new_centroids = zeros(optimal_k, size(X, 2));
    for j = 1:optimal_k
        points_in_cluster = X(cluster_assignment == j, :);
        if ~isempty(points_in_cluster)
            new_centroids(j, :) = mean(points_in_cluster, 1);
        end
    end
    
    % Check for convergence
    if max(max(abs(new_centroids - centroids))) < threshold
        disp(['Converged in ', num2str(iter), ' iterations.']);
        break;
    end
    centroids = new_centroids;
end

% Visualization of the final clusters and centroids
figure;
gscatter(X(:, 1), X(:, 3), cluster_assignment);
hold on;
scatter(centroids(:, 1), centroids(:, 3), 100, 'k', 'filled');
title(['Customer Segmentation using K-means with k = ', num2str(optimal_k)]);
xlabel('Age');
ylabel('Spending Score');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Centroids');
hold off;

% Step 5: Evaluation - Silhouette Score
figure;
silhouette(X, cluster_assignment);
title('Silhouette Plot for K-means Clustering');
