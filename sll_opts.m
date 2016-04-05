function opts=sll_opts(opts)
opts.nFlag = 0;        % normalization flag
opts.init = 2;         % initialize a starting point
opts.mFlag = 0;        %??
opts.lFlag = 0;        %??
opts.rFlag = 1;        % range of par within [0, 1]
opts.tol = 1e-6;       % optimization precision
opts.tFlag = 4;        % termination options.
opts.maxIter = 5000;   % maximum iterations.
end
