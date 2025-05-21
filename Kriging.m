% 读取数据
data = readmatrix('数据集.xlsx');

% 划分输入输出
X = data(:, 1:12);       % 补水量输入
Y = data(:, 13:end);     % 水位埋深输出

% 划分训练集与测试集
X_train = X(1:100, :);
X_test = X(101:end, :);
Y_train = Y(1:100, :);
Y_test = Y(101:end, :);

% 初始化预测结果矩阵和误差指标
Y_pred = zeros(size(Y_test));
rmse = zeros(1, size(Y,2));
r2 = zeros(1, size(Y,2));

% 初始化保存超参数的单元格
hyperparams = cell(size(Y_train,2), 5); % 输出点、回归函数、相关函数、Sigma、length scales

for i = 1:size(Y_train, 2)
    fprintf('正在训练第 %d 个 Kriging 模型（自动超参数优化）...\n', i);

    y_train = Y_train(:, i);
    y_test = Y_test(:, i);

    % 自动超参数优化
    gprMdl = fitrgp(X_train, y_train, ...
        'BasisFunction', 'constant', ...
        'KernelFunction', 'ardsquaredexponential', ...
        'Standardize', true, ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', struct('ShowPlots', false, 'Verbose', 0));

    y_pred = predict(gprMdl, X_test);
    Y_pred(:, i) = y_pred;

    % 误差计算
    rmse(i) = sqrt(mean((y_pred - y_test).^2));
    ss_res = sum((y_test - y_pred).^2);
    ss_tot = sum((y_test - mean(y_test)).^2);
    r2(i) = 1 - ss_res / ss_tot;

    % 记录超参数
    hyperparams{i,1} = i;
    hyperparams{i,2} = gprMdl.BasisFunction;
    hyperparams{i,3} = gprMdl.KernelFunction;
    hyperparams{i,4} = gprMdl.Sigma;
    hyperparams{i,5} = gprMdl.KernelInformation.KernelParameters'; % 行向量
end

% 平均指标输出
avg_rmse = mean(rmse);
avg_r2 = mean(r2);

fprintf('\n=== 模型整体评估 ===\n');
fprintf('平均 RMSE: %.4f\n', avg_rmse);
fprintf('平均 R²  : %.4f\n', avg_r2);

% 输出所有输出点的超参数表
param_table = cell2table(hyperparams, ...
    'VariableNames', {'OutputIndex', 'BasisFunction', 'KernelFunction', 'Sigma', 'LengthScales'});
disp(param_table);

% 可选：写入Excel表
writetable(param_table, 'Kriging_Hyperparameters.xlsx');
