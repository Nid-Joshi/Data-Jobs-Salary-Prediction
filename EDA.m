% PRE-PROCESS DATA AND EVALUATE DIFFERENT FEATURES AGAINST THE TARGET VARIABLE THROUGH PLOTS

% Clear the command window
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


%% IDENTIFYING CATEGORICAL AND NUMERICAL DATA

% Specify the names of the categorical columns
cat_variables = {'experience_level', 'employment_type', 'salary_currency', 'employee_residence', 'company_location', 'company_size'};
% Convert remaining specified columns to categorical
s_data = convertvars(s_data, cat_variables, 'categorical');


% Specify the names of the numerical columns
num_variables = {'work_year', 'remote_ratio'};

%% ASSESS RELATIONSHIP BETWEEN EACH VARIABLE

% Plot boxplots to assess the relationship between 'salary_in_usd' with each categorical variable

% REF: https://uk.mathworks.com/help/stats/boxplot.html

for i = 1:width(cat_variables)
    figure;
    boxplot(s_data.salary_in_usd, s_data.(cat_variables{i}), 'Labels', categories(s_data.(cat_variables{i})));
    title(['Salary vs. ' cat_variables{i}]);
    xlabel(cat_variables{i});
    ylabel('Salary (USD)');
end
%% Scatter plot for 'salary_in_usd against every other numerical variable

% REF: https://uk.mathworks.com/help/matlab/ref/scatter.html?searchHighlight=scatter%20plots&s_tid=srchtitle_support_results_1_scatter%20plots

for j = 1:width(num_variables)
    figure;
    scatter(s_data.(num_variables{j}), s_data.salary_in_usd);
    title(['Salary vs. ' num_variables{j}]);
    ylabel('Salary (USD)');
    xlabel(num_variables{j});
end

%% Re-arrange all the numerical variables on the right end along with
% the target variable i.e. work_year, remote_ratio, salary_in_usd; 
% salary_in_usd being the target variable

s_data = s_data(:,[2 3 6 8 10 11 1 9 7]) 
fprintf('Re-arranged table with required variables:\n');
head(s_data,5)
