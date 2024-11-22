clear 
clc

% Load the dataset
s_data = readtable('Data_salaries.csv');
% Print the loaded data
s_data

%% Pre-process the data to handle missing data and remove duplicates if any.

% Remove rows with missing values
s_data = rmmissing(s_data,1);

% Remove duplicates
s_data = unique(s_data, 'rows');


s_data

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


% REF: https://uk.mathworks.com/help/matlab/ref/table.convertvars.html

% Convert remaining specified columns to categorical
s_data = convertvars(s_data, cat_variables, 'categorical');
fprintf('With converted categorical values:\n');
s_data


%%
  
% re-arrange the numerical variables on the right end along with
% the target variable - work_year, remote_ratio, salary_in_usd; 
% salary_in_usd being the target variable

s_data = s_data(:,[2 3 6 8 10 11 1 9 7]) 
fprintf('Re-arranged table with required variables:\n');
head(s_data,5)

% Not considered job_title & salary in the table since on evaluation,
% they were found to have no relation to our target variable

%% 

% Convert all data to numerical data through one hot encoding method
encoded_data = table();

% Initialize the encoded_data with the non-categorical columns
encoded_data = s_data(:, ~ismember(s_data.Properties.VariableNames, cat_variables));

% REF: https://uk.mathworks.com/help/deeplearning/ref/onehotencode.html
% One-hot encode categorical columns and rename the columns
for i = 1:length(cat_variables)
    encoded_columns = onehotencode(s_data(:,(cat_variables{i})));
    encoded_columns.Properties.VariableNames = strcat(cat_variables{i}, '_', encoded_columns.Properties.VariableNames);
    encoded_data = [encoded_columns encoded_data];
end

% display the one hot encoded data

encoded_data

%% Data Splitting

% Split the data into training and test sets

rng (12); % Set random seed for reproducibility 

% -- Ref - https://uk.mathworks.com/help/stats/cvpartition.html
% Create a random partition of training and testing set i.e. Holdout
% specifying 0.2 fraction of the data as the test set
cv = cvpartition(height(encoded_data), 'Holdout', 0.2);

%% Prepare the train data on the whole dataset

train_data = encoded_data(training(cv),:);


%% Define the Feature and Target variables 

% Select all columns from start to second last as FEATURES
X = encoded_data(:, 1:end-1);

% Select the last column as target i.e. salary_in_usd
y = encoded_data(:, end);

%% Prepare the train and test data on Features and Target separately

X_train = X(training(cv),:);  % Features(X) in the train data (80 percent)

y_train = y(training(cv),:);  % Target(y) in the train data (80 percent)

X_test = X(test(cv),:);  % Features(X) in the test data (20 percent)

y_test = y(test(cv),:);  % Target(y) in the test data (20 percent) 

%% Save the test set separately to be used later for prediction
idx = test(cv);

% Extract the test dataset from the original data
test_set = encoded_data(idx, :);

% Save the test dataset to a CSV file
writetable(test_set, 'test_lm.csv');
test_set


%% FORMULA TO PASS NUMEROUS HOT-ENCODED COLUMN NAMES TO FIT THE LM MODEL

% Accessing the column names from the encoded table
col_names = encoded_data.Properties.VariableNames;

% Assign the predictor and target column names
predictor_cols = col_names(1:end-1);
target_col = col_names{end};

% Create a formula for linear regression
formula_str = sprintf('%s ~ %s', target_col, strjoin(predictor_cols, ' + '));

%% Train the linear regression model

lm_model = fitlm(train_data, formula_str);

% Save the base trained Linear Regression model to a .mat file
save('base_LinearRegression.mat', 'lm_model');

% Evaluate the model by coefficients
lm_model.Coefficients

% After displaying the coefficients, we see that p-values for all other
% columns except experience_level, company size, work year and remote ratio
% are greater than 0.05, hence not significant enough
%% Plot the fitted model

subplot(2, 2, 1);
plot(lm_model)

% Plot the residuals from the fitted model

subplot(2, 2, 2);
plotResiduals(lm_model)

% Find the Rsquared and adjusted Rsquared values
lm_model.Rsquared

%% Make predictions on the test set through linear regression model

% Load the saved model
load('base_LinearRegression.mat');

y_pred_lm = predict(lm_model, X_test); % Pass the features from the test data
y_pred_lm  % should display the 20 percent of the predictions - test data

%%

% REF: https://uk.mathworks.com/help/stats/train-linear-regression-model.html

% From the plot we see that there are a few residuals more than 2*10^5,
% find those outliers
outliers = find(lm_model.Residuals.Raw > 2*10^5)
outliers


%% Evaluate model performance through Root Mean Squared Error

rmse_lm = sqrt(mean((y_test.salary_in_usd - y_pred_lm).^2));  
rmse_lm

%% Plot results

figure;

subplot(2, 2, 1);
scatter(y_test.salary_in_usd, y_pred_lm);
hold on;
plot(y_test.salary_in_usd, y_test.salary_in_usd, 'r--');  % Add a red dashed line for reference
hold off;

xlabel('True Values (Salary)');
ylabel('Predicted Values (Salary)');
title('True vs. Predicted Salaries');
grid on; 

subplot(2, 2, 2);
histogram(y_test.salary_in_usd - y_pred_lm, 20);  % y_test vs. y_pred_lm
title('Linear Regression Residuals');
xlabel('Residuals');
ylabel('Frequency');

%% HYPERPARAMTER TUNING: Check for outliers in salary_in_usd and remove them

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

%% Train the linear regression model again after removing outliers

lm_model = fitlm(train_data, formula_str);

lm_time_taken = toc % Print the time taken to train through linear regression

% Save the base trained Linear Regression model to a .mat file
save('final_LinearRegression.mat', 'lm_model');


%% RUN CODE FROM HERE

test = readtable('test_lm.csv');

X_test = test(:, 1:end-1);

% Select the last column as target
y_test = test(:, end);

%% Make predictions on the test set through final model

% Load the saved model
load('final_LinearRegression.mat');
X_test
y_pred_lm = predict(lm_model, X_test); % Pass the features from the test data
y_pred_lm  % should display the 20 percent of the predictions - test data

%% Evaluate model performance through RSquared and RMSE values

% Calculate the RSquared value
Rsquared_lm = 1 - ((sum((y_test.salary_in_usd - y_pred_lm).^2)) / (sum((y_test.salary_in_usd - mean(y_test.salary_in_usd)).^2)));
Rsquared_lm

% Find the RMSE value of the final model
rmse_lm = sqrt(mean((y_test.salary_in_usd - y_pred_lm).^2));  
rmse_lm

% Find the Mean Absolute Error
mae_lm = mean(abs(y_test.salary_in_usd - y_pred_lm)); 
mae_lm

%% Plot results

figure;

subplot(2, 2, 1);
scatter(y_test.salary_in_usd, y_pred_lm);
hold on;
plot(y_test.salary_in_usd, y_test.salary_in_usd, 'r--');  % Add a red dashed line for reference
hold off;

xlabel('True Values (Salary)');
ylabel('Predicted Values (Salary)');
title('True vs. Predicted Salaries');
grid on; 

subplot(2, 2, 2);
histogram(y_test.salary_in_usd - y_pred_lm, 20);  % y_test vs. y_pred_lm
title('Linear Regression Residuals');
xlabel('Residuals');
ylabel('Frequency');