clear;close all, clc

% NOTATIONS:
% n = number of features
% m = number of traning exemples

file = 'ex1data2.txt';
data = load(file);
X = data(:,1:size(data,2)-1); % #(m_X_n)#
y = data(:,size(data,2)); % #(m_X_1)# the value to predict must be at the last column
m = length(y);
alpha = 0.3; % Learning rate
num_iters = 400;

function [X_norm, mu, sigma] = featureNormalize(X)

    X_norm = X;
    sigma = std(X); #(n_X_1)#
    mu = mean(X); #(n_X_1)#
    X_norm = (X-mu)./sigma; #(m_X_n)#
end

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters

        theta = theta - alpha*(1/(2*m))*(X'*((X*theta)-y)); % #(n+1_X_1)#
        J_history(iter) = (1/(2*m))*((X*theta-y)'*(X*theta-y)); % Track error values to future analysis
    end
end


[X,mu,sigma] = featureNormalize(X);
X = [ones(m, 1) X]; % #(m_X_n+1)# Add the "x_0" column that will multiply the theta_0


theta = zeros(size(X,2), 1); % #(n+1_X_1)#
[theta,J_history] = gradientDescentMulti(X,y,theta,alpha,num_iters);

theta

