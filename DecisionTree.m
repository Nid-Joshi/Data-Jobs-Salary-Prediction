clear
clc

% Load the dataset
s_data = readtable('Data_salaries.csv');
% Print the loaded data
s_data

%% 

% Pre-process the data to handle missing data and remove duplicates if any.

% Count null values
null_count = sum(ismissing(s_data));

% Display counts
fprintf('Null values count:\n');
disp(null_count);

% Remove rows with missing values
s_data = rmmissing(s_data,1);

% Remove duplicates
s_data = unique(s_data, 'rows');


s_data

%%
% re-arrange the numerical variables on the right end along with
% the target variable - work_year, remote_ratio, salary_in_usd; 
% salary_in_usd being the target variable

s_data = s_data(:,[2 3 6 8 10 11 1 9 7]) 
fprintf('Re-arranged table with required variables:\n');
head(s_data,5)

%%

% Specify the names of the columns you want to convert to categorical
cat_variables = {'experience_level', 'employment_type', 'salary_currency', 'employee_residence', 'company_location'};

% Since company_size is ordinal in nature, so let's assign numerical values
% to the variable accordingly
s_data.company_size = categorical(s_data.company_size, {'S', 'M', 'L'}, 'Ordinal', true);
size_mapping = containers.Map({'S', 'M', 'L'}, {0, 1, 2});

% REF: https://uk.mathworks.com/help/matlab/ref/cellfun.html

% Convert 'size' column to numerical values based on the mapping
s_data.company_size = cellfun(@(x) size_mapping(x), cellstr(s_data.company_size));


% -- Ref: https://uk.mathworks.com/help/matlab/ref/table.convertvars.html
% Convert remaining specified columns to categorical
s_data = convertvars(s_data, cat_variables, 'categorical');
fprintf('With converted categorical values:\n');
s_data


%% Check for outliers in salary_in_usd and remove them

subplot(2, 2, 1);
histogram(s_data.salary_in_usd)
title('Salary in USD')

% Check for outliers
idx = isoutlier(s_data.salary_in_usd);

% Remove any outliers and plot the histogram again
s_data(idx,:) = [];
subplot(2, 2, 2);
histogram(s_data.salary_in_usd)
title('Salary in USD after removing outliers')


%% Split the data into training and test sets

rng (12); % Set random seed for reproducibility 

% -- Ref - https://uk.mathworks.com/help/stats/cvpartition.html
% Create a random partition of training and testing set i.e. Holdout
% specifying 0.2 fraction of the data as the test set
cv_dtm = cvpartition(height(s_data), 'Holdout', 0.2);

%% Define the Feature and Target variables 

% Select all columns from start to second last as features
features = s_data(:, 1:end-1);
head(features)

% Select the last column as target i.e. salary_in_usd
target = s_data(:, end);
head(target)

%% Assign features and target for training regression model

X_train = features(training(cv_dtm),:);
y_train = target(training(cv_dtm),:);
X_test = features(test(cv_dtm),:);
y_test = target(test(cv_dtm),:);

% Save the test set separately to be used later for prediction
idx = test(cv_dtm);

% Extract the test dataset from the original data
test_set = s_data(idx, :);

% Save the test dataset to a CSV file
writetable(test_set, 'test_dtree.csv');


%% Train a decision tree regression model

dtree_model = fitrtree(X_train, y_train);

% Save the base trained Decision Tree model to a .mat file
save('base_DecisionTree.mat', 'dtree_model');


%% Load the base trained model and make predictions on the test set

% Load the base trained model and Predict using the Decision tree
load('base_DecisionTree.mat')

y_pred_dtree = predict(dtree_model, X_test);


%%

% Evaluate model performance
rmse_tree = sqrt(mean((y_test - y_pred_dtree).^2));
rmse_tree


%%

% Plot results
figure;

subplot(2, 2, 1);
scatter(y_test.salary_in_usd, y_pred_dtree);
hold on;
plot([min(y_test.salary_in_usd), max(y_test.salary_in_usd)], [min(y_test.salary_in_usd), max(y_test.salary_in_usd)], 'r--');
hold off;
title('True vs. Predicted');
xlabel('True Salary (USD)');
ylabel('Predicted Salary (USD)');

subplot(2, 2, 2);
histogram(y_test.salary_in_usd - y_pred_dtree, 20);
title('Decision Tree Residuals');
xlabel('Residuals');
ylabel('Frequency');

%% Display the decision tree

view(dtree_model, 'Mode', 'graph');

%% Hyperparameter tuning:

test_leaf_size = [1, 5, 10, 20, 50];

% Define variable to assign results for different leaf sizes
mse = zeros(size(test_leaf_size));

% Perform grid search with cross-validation
for i = 1:length(test_leaf_size)
    test_model = fitrtree(X_train, y_train, 'MinLeafSize', test_leaf_size(i), 'CrossVal', 'on');
    mse(i) = kfoldLoss(test_model);
end

% Find the best leaf size
best_size = test_leaf_size(mse == min(mse));

fprintf('Leaf Size to be taken: %d\n', best_size);

% Train the model with the best leaf size
dtree_model = fitrtree(X_train, y_train, 'MinLeafSize', best_size);

%%
% We can also prune the decision tree to make the prediction better

dtree_prune = prune(dtree_model, 'Level', 5); 

% Save the final trained Decision Tree model to a .mat file
save('final_DecisionTree.mat', 'dtree_prune');

dtree_time_taken = toc % Print the time taken to train through decision tree

%% RUN CODE FROM HERE

test = readtable('test_dtree.csv');

X_test = test(:, 1:end-1);

% Select the last column as target
y_test = test(:, end);

%% Make final predictions using the pruned tree

% Load the final trained model and Predict using the Decision tree
load('final_DecisionTree.mat')

y_pred_pruned = predict(dtree_prune, X_test);

%% Evaluate the pruned model performance

% Calculate the RSquared value
Rsquared_dtree = 1 - ((sum((y_test.salary_in_usd - y_pred_pruned).^2)) / (sum((y_test.salary_in_usd - mean(y_test.salary_in_usd)).^2)));
Rsquared_dtree

% Calculate RMSE value
rmse_tree = sqrt(mean((y_test - y_pred_pruned).^2));
rmse_tree

% Find the Mean Absolute Error
mae_tree = mean(abs(y_test.salary_in_usd - y_pred_pruned)); 
mae_tree

% Even by pruning, the RMSE gets only marginally decreased

%% Plot results

figure;

subplot(2, 2, 1);
scatter(y_test.salary_in_usd, y_pred_pruned);
hold on;
plot([min(y_test.salary_in_usd), max(y_test.salary_in_usd)], [min(y_test.salary_in_usd), max(y_test.salary_in_usd)], 'r--');
hold off;
title('True vs. Predicted Salaries');
xlabel('True Values (Salary)');
ylabel('Predicted Values (Salary)');

subplot(2, 2, 2);
histogram(y_test.salary_in_usd - y_pred_pruned, 20);
title('Decision Tree Residuals');
xlabel('Residuals');
ylabel('Frequency');