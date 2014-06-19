%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - Active Sample Selection for Matrix Completion -
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  - Sebastian Grubb - sg3510@ic.ac.uk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% - Description -
% This allows a dataset and selection criteria to be chosen allowing the
% active sampling algorithm to be tested.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Setup

% Clear Environment
clc
clear all
close all

% Add File Paths
addpath('data')
addpath('mat2tikz')
addpath('matrixfactoriser')
addpath('routines')

% Load Data
load('hiva_subset.mat')
R_full = R_train + R_test + R_val;
discrete = 1; % 1 - Discrete, 0 - Continuous
% Min and Max if discrete
min_d = 0;max_d = 1;

% Initialise Main Variables
samples = 500; % Ensure not larger than total set
[x,y] = size(R_train);
mf_type = 4; % 1 - PMF, 2 - BPMF, 3 - KPMF,  4 - PPMF

num_feat = 15;

% PMF Settings
lambda_pmf = 0.01; % Regularisation
mu_pmf = 0.005; % Learning rate
iter_pmf = 100; % Iterations

% BPMF Settings
beta_bpmf = 3.5; % Precision
iter_bpmf = 15; % Iterations

% KMPF settings
iter_kmpf = 75; % Iterations
epsilon_kpmf = 0.005; % Learning rate
diffu_kmpf = 0.1; %Diffusion value if using dissuion kernel
%% Setup for Active Sampling Process

% Initialise sampling variables
freq_count = 0;

% Initialise for Random Sampling
z_train_rand = z_train; z_test_rand = z_test;
R_train_rand = R_train; R_test_rand = R_test;
sample_rand = zeros(x,y);
err_rand = zeros(1,samples);

% Initialise for targeted Sampling
z_train_targ = z_train; z_test_targ = z_test;
R_train_targ = R_train; R_test_targ = R_test;
sample_targ = zeros(x,y);
err_targ = zeros(1,samples);
U_targ = randn(x,num_feat); V_targ = randn(y,num_feat);

% Choose Targeting
% 1 - MKS, 2 - CKS, 3 - MNVar, 4 - MCMCVar, 5 - Hybrid
targeting_type = 4;
% How many samples to wait until knowledge recalculation
% For MCMCVar CKS hybrid this is the MCMCVar reculculation period
% 50 is agood setting
targeting_freq = 10; 
select_min = 0; % 0 - Select Max, 1 - Select Min
randomise = 1; % 1 - Randomise variants, 0 - Deterministic
batch_req = 100; % How many samples to request in one sampling
% process, 1 for true active sampling basis

%% Start Active Sampling
for req = 1:samples
    %_______________________________
    % Random Sampling Stage
    %_______________________________
    
    fprintf('Getting Random Sample(s)\n');
    % Get random sample from test set
    for samp_req = 1:batch_req
        [R_train_rand,z_train_rand, R_test_rand, z_test_rand,a,b] = random_sample(R_train_rand,z_train_rand, R_test_rand, z_test_rand);
        sample_rand(a,b) = req;
    end
    
    fprintf('Trainging with Random Sample(s)\n');
    % Train
    switch mf_type
        case 1 % PMF
            [U_rand,V_rand,~ ] = pmf_func_val(R_train_rand, z_train_rand,R_val,z_val, iter_pmf,num_feat,lambda_pmf,mu_pmf);
            R_pred_rand = U_rand*V_rand';
        case 2 % BPMF
            %Train first for PMF
            [U_rand,V_rand,~ ] = pmf_func_val(R_train_rand, z_train_rand,R_val,z_val, 50,num_feat,lambda_pmf,mu_pmf);
            [U_rand,V_rand,~,mv_rand] = bpmf_func(R_train_rand,z_train_rand,U_rand,V_rand,iter_bpmf,num_feat,beta_bpmf);
            R_pred_rand = U_rand*V_rand' + mv_rand;
        case 3 % KPMF
            [U_rand,V_rand,~] = kpmf_func(R_train_rand,z_train_rand,iter_kmpf,num_feat,epsilon_kpmf,diffu_kmpf);
            R_pred_rand = U_rand*V_rand';
        case 4 % PPMF
            [ U_rand,V_rand,mv_rand ] = run_ppmf(R_train_rand, z_train_rand,num_feat, R_val,z_val);
            R_pred_rand = U_rand*V_rand' + mv_rand;
        otherwise
            disp('wrong type of Matrix Factorisation selected'); break;
    end
    
    if discrete
        R_pred_rand = min_max_round(R_pred_rand,min_d,max_d);
    end
    
    % RMSE calculation
    err_rand(req) = rmse_calc( R_pred_rand, R_test_rand, z_test_rand);
    
    
    %_______________________________
    % Active Sampling Stage
    %_______________________________
    
    fprintf('Getting Targeting Criteria\n');
    % Criteria for targeted
    switch targeting_type
        case 1 % MKS
            knowledge =  mean(z_train_targ,2)*mean(z_train_targ,1);
        case 2 % CKS
            knowledge  = CKS_UV_cluster(U_targ,V_targ, z_train_targ);
        case 3 % MNVar
            if ((mod(freq_count,targeting_freq) == 0)|| freq_count == 1)
                knowledge = MN_V(U_targ,V_targ,1-(z_train_targ+z_val));
                knowledge = knowledge - min(knowledge(:));
                knowledge = knowledge/max(knowledge(:));
            end
            freq_count = freq_count + 1;
        case 4 % MCMCVar
            if ((mod(freq_count,targeting_freq) == 0) || freq_count == 1)
                [U_targ,V_targ,~,~] = GS_U_V(R_train_targ,z_train_targ,U_targ,V_targ);
                knowledge = MN_V(U_targ,V_targ,1-(z_train_targ+z_val));
                knowledge = knowledge - min(knowledge(:));
                knowledge = knowledge/max(knowledge(:));
            end
            freq_count = freq_count + 1;
        case 5 %MCMVar and CKS hybrid
            if ((mod(freq_count,targeting_freq) == 0) || freq_count == 1)
                [U_b,V_b,e,mean_rating] = GS_U_V(R_train_targ,z_train_targ,U_targ,V_targ);
                var_know = MN_V(U_b,V_b,1-(z_train_targ+z_val));
                var_know = var_know - min(var_know(:));
                var_know = var_know/max(var_know(:));
            end
            [a,b] = find(knowledge==max(knowledge(:)));%a = a(j);b = b(j);
            knowledge(:,:) = 0;
            knowledge(a,b) = var_know(a,b);
        otherwise
            disp('Wrong type of Active Sampling selected'); break;
    end
    
    fprintf('Getting Targeted Sample(s)\n');
    %Request Samples in batch
    for samp_req = 1:batch_req
        [ R_train_targ,z_train_targ, R_test_targ, z_test_targ,a,b] = select_sample( R_train_targ,z_train_targ, R_test_targ, z_test_targ, knowledge, select_min,randomise);
        sample_targ(a,b) = req;
    end
    
    fprintf('Training with Targeted Samples\n');
    % Train
    switch mf_type
        case 1 % PMF
            [U_targ,V_targ,~ ] = pmf_func_val(R_train_targ, z_train_targ,R_val,z_val, iter_pmf,num_feat,lambda_pmf,mu_pmf);
            R_pred_targ = U_targ*V_targ';
        case 2 % BPMF
            %Train first for PMF
            [U_targ,V_targ,~ ] = pmf_func_val(R_train_targ, z_train_targ,R_val,z_val, 50,num_feat,lambda_pmf,mu_pmf);
            [U_targ,V_targ,~,mv_targ] = bpmf_func(R_train_targ,z_train_targ,U_targ,V_targ,iter_bpmf,num_feat,beta_bpmf);
            R_pred_targ = U_targ*V_targ' + mv_targ;
        case 3 % KPMF
            [U_targ,V_targ,~] = kpmf_func(R_train_targ,z_train_targ,iter_kmpf,num_feat,epsilon_kpmf,diffu_kmpf);
            R_pred_targ = U_targ*V_targ';
        case 4 % PPMF
            [ U_targ,V_targ,mv_targ ] = run_ppmf(R_train_targ, z_train_targ,num_feat, R_val,z_val);
            R_pred_targ = U_targ*V_targ' + mv_targ;
        otherwise
            disp('wrong type of Matrix Factorisation selected'); break;
    end
    
    if discrete
        R_pred_targ = min_max_round(R_pred_targ,min_d,max_d);
    end
    
    % RMSE calculation
    err_targ(req) = rmse_calc( R_pred_targ, R_test_targ, z_test_targ);
    
    %_______________________________
    % Draw
    %_______________________________
    subplot(2,3,1);
    plot([smooth(err_rand(1:req),30) smooth(err_targ(1:req),30)])
    legend('Random','Targeted')
    title(sprintf('Target Advantage:%1.3f',sum(err_rand)/sum(err_targ)))
    ylabel('RMSE')
    subplot(2,3,2);
    
    image(sample_rand/req*50)
    title('Random Request')
    subplot(2,3,3);
    image(R_pred_rand*50)
    title('Random Prediction')
    
    
    subplot(2,3,4);
    image(R_full/max(R_full(:))*50)
    title('Original')
    subplot(2,3,5);
    
    image(sample_targ/req*50)
    title('Active Request')
    subplot(2,3,6);
    image(R_pred_targ*50)
    title('Targeted Prediction')
    drawnow
    
end