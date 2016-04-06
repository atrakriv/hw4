clc; clear; close all;

load('ad_data')
n = size(X_train,1);       % training set size
m = size(X_test,1);        % testing set size
d = size(X_train,2);       % feature space size

training_data = X_train;               % training data without bias
%testing_data = [X_test ones(m,1)];     % testing data with bias

training_labels = y_train;
testing_labels = y_test;

A = training_data;                    % training data
y = training_labels;                  % training labels
% vec of legularization parameter
%z_vec = [0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]; 
z_vec = [0.1];

% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations.

%%
cnt_z = 1;
for z = z_vec
%     opts = [];                       % opts has been defined in sll_opts function
    [x, c, funVal, ValueL] = LogisticR(A, y, z, opts);
    
    %% testing
    %w = [x; c];                      % putting together weights and bias weight
    %out_test = (sigmf(w'*testing_data',[1 0]))';
    %yp = 2*heaviside(out_test - 0.5)-1;  % step function to map y values to 0,1 and then to -1,1
    %p_p = 1./(1+ exp(-1*(x' * X_test' + c) ) ); 
    %p_n = 1./(1+ exp(1*(x' * X_test' + c) ) );
    
    sign = x' * X_test' + c;
    
%     out = -1*ones(m,1);
%     for i = 1:m
%         if p_p(i)>p_n(i)
%             out(i,1) = 1;
%         end
%     end

    test_out = -1*ones(m,1);
    for i = 1:m
        if sign(i)> 0
            test_out(i,1) = 1;
        end
    end
       
    [X,Y,T,AUC] = perfcurve(y_test,test_out,1);
%     plot(X,Y)
    
   
    cnt_c = 0;                       % counting the correct recognition
    t_test = testing_labels;
    for i = 1 : m
        %if t_test(i) == yp(i)
        if t_test(i)*sign(i)>0
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
    S = sprintf('for z = %s',z);
    disp(S)
    S = sprintf('AUC = %s',AUC);
    disp(S)
    S = sprintf('accuracy = %s', num2str(accuracy(cnt_z)));
    disp(S)
    S = sprintf('number of selected features in x: %s',num2str(cnt_f(cnt_z)));
    disp(S)
    disp('------------------------------------------------------------')
    cnt_z = cnt_z + 1;
end

%%
% plot(z_vec, cnt_f, '-o', 'MarkerFaceColor','b')
% xlabel('l1')
% ylabel('number of selected features')
% title('legularization parameter vs. number of selected features')