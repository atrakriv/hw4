clc; clear; close all;
load('data.txt')

n_vec = [200, 500, 800, 1000, 1500, 2000];

n_cnt = 1;
for n = n_vec
    m0 = 2001;
    m1 = length(data);
    m = m1 - m0 + 1;
    
    training_data = [data(1:n,:) ones(n,1)];
    testing_data = [data(m0:m1,:) ones(m,1)];
    
    load('labels.txt')
    training_labels = labels(1:n,:);
    testing_labels = labels(m0:m1,:);
    
    maxiter = 1000;
    epsilon = 1e-5;
    
    [w] = logistic_train(training_data, training_labels, epsilon, maxiter);
    
    %% Testing
    y_test = (sigmf(w'*testing_data',[1 0]))';
    yp = 1*heaviside(y_test - 0.5);  % step function to map y values to labels 0 and 1
    cnt = 0;
    for i = 1 : m
        if testing_labels(i) == yp(i)
            cnt = cnt+1;
        end
    end
    accuracy(n_cnt) = cnt/m;
    
    X = sprintf('n = %s , accuracy = %s', num2str(n), num2str(accuracy(n_cnt)));
    disp(X)
    n_cnt = n_cnt + 1;
end

%%
plot(n_vec,accuracy, '-o', 'MarkerFaceColor','b')
xlabel('n')
ylabel('accuracy')
title('training size vs. accuracy')