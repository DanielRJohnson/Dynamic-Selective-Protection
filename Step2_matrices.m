function Step2_matrices

% This function computes the preconditioner matrix and the row-2 norms of
% the matrices. It is a preprocessing step before other functions are
% called. The preconditioner matrix and row-2 norms are saved in the same
% directory as the original  matrix. 

% matrices = {'cvxbqp1', 'thermal1', 'nd6k', ...
%    'bcsstk18', 'bodyy5', 'cbuckle', 'Pres_Poisson', 'bcsstk36', 'ct20stif', 'gyro_m', 't2dah_e', 'm_t1', 'msc23052', '2cubes_sphere', 'pwtk', 'G2_circuit', 'raefsky4', ...
%    'Trefethen_20000', 'vanbody','wathen100'};
matrices = {'cbuckle', 'bcsstk18'};
num_matrices = length(matrices);

opt_type = 'ict'; % 'nofill'
opt_droptol = 1e-3; % 1e-1
opt_michol = 'off'; % 'on'

for m = 1:num_matrices
    matrixname = matrices{m};
    disp(matrixname);
    matrixfile = ['./matrices/raw/', matrixname, '.mat'];
    
    %% load matrix
    load(matrixfile);
    A = Problem.A;
    [N1, N2] = size(A);
    if N1 ~= N2
        disp('matrix is not squared');
        return;
    end
    N = N1;
    disp('Matrix loaded');
    drawnow('update');

    %% compute preconditioner
    precond_filename = ['./matrices/preconditioners/', matrixname, '_precond.mat'];
    if exist(precond_filename, 'file') < 1 % file does not exist 
        try
           L = ichol(A,struct('type',opt_type, 'droptol',opt_droptol, 'michol',opt_michol));
           save(precond_filename, 'L');
           disp('Done incomplete Cholesky factorization');
           drawnow('update');
        catch ME
           %disp(ME.identifier); 
           %if (strcmp(ME.identifier,'Encountered nonpositive pivot'))
           alpha = max(sum(abs(A),2)./diag(A))-2;
           L = ichol(A,struct('type',opt_type, 'droptol',opt_droptol, 'michol',opt_michol, 'diagcomp',alpha));
           save(precond_filename, 'L');
           disp('Done incomplete Cholesky factorization (with diagcomp)');
           drawnow('update');
           %end
        end 
    end
    
    %% compute row 2-norms
    norms_filename = ['./matrices/2norms/', matrixname, '_norms.mat'];
    if exist(norms_filename, 'file') < 1 % file does not exist 
        A_row_2norm = zeros(N, 1);
        for n = 1:N
            A_row_2norm(n) = norm(A(n, :), 2);
            if mod(n*100, N) == 0
                disp(['row ', num2str(n), '/', num2str(N)]);
                drawnow('update');
            end
        end
        save(norms_filename, 'A_row_2norm');
        disp('Done computing row 2-norms of matrix A');
        drawnow('update');
    end
end 
    
end