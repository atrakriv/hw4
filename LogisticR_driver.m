clc; clear; close all;

load('data.txt')
n = 50;             % training set size
m0 = 2001;          % pick data from 2001 to end as a testing
m1 = length(data);  % dataset size
m = m1 - m0 + 1;    % testing set size
d = size(data,2);   % feature space size

training_data = data(1:n,:);               % training data without bias
testing_data = [data(m0:m1,:) ones(m,1)];  % testing data with bias

load('labels.txt')
training_labels = labels(1:n,:);
testing_labels = labels(m0:m1,:);

A = training_data;                          % training data
y = 2*training_labels - 1;                  % map labels from 0,1 to -1,1
z_vec = [0.01 0.02 0.05 0.1 0.15 0.2]; % vec of legularization parameter

cnt_z = 1;
for z = z_vec
    opts = [];                       % opts has been defined in sll_opts function
    [x, c, funVal, ValueL] = LogisticR(A, y, z, opts);
    
    %% testing
    w = [x; c];                      % putting together weights and bias weight
    y_test = (sigmf(w'*testing_data',[1 0]))';
    yp = 1*heaviside(y_test - 0.5);  % step function to map y values to 0,1
    
    cnt_c = 0;                       % counting the correct recognition
    t_test = testing_labels;
    for i = 1 : m
        if t_test(i) == yp(i)
            cnt_c = cnt_c+1;
        end
    end
    accuracy(cnt_z) = cnt_c/m;
    
    cnt_f(cnt_z) = 0;                % counting the number of selected feature
    for i = 1 : d
        if x(i) ~= 0
            cnt_f(cnt_z) = cnt_f(cnt_z) + 1;
        end
    end
    %%
    X = sprintf('for z = %s',z);
    disp(X)
    X = sprintf('accuracy = %s', num2str(accuracy(cnt_z)));
    disp(X)
    X = sprintf('number of selected features in x: %s',num2str(cnt_f(cnt_z)));
    disp(X)
    disp('------------------------------------------------------------')
    cnt_z = cnt_z + 1;
end

%%
plot(z_vec, cnt_f, '-o', 'MarkerFaceColor','b')
xlabel('l1')
ylabel('number of selected features')
title('legularization parameter vs. number of selected features')