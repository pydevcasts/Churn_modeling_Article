function DNN_K_FOLD()
    % Load and preprocess the data
    [X, y] = preprocess_churn_data();

    % Parameters
    num_folds = 10;
    num_samples = size(X, 1);
    indices = randperm(num_samples);
    batch_size = 7; % Define the batch size
    % Separate positive and negative indices
    positive_indices = find(y == 1);
    negative_indices = find(y == 0);

    % Determine folds for each class
    positive_folds = ceil((1:length(positive_indices)) / (length(positive_indices) / num_folds));
    negative_folds = ceil((1:length(negative_indices)) / (length(negative_indices) / num_folds));
    fold_indices = cell(num_folds, 1);

    for fold = 1:num_folds
        fold_indices{fold} = [positive_indices(positive_folds == fold); negative_indices(negative_folds == fold)];
    end

    % Store results
    accuracy_scores = zeros(num_folds, 1);
    precision_scores = zeros(num_folds, 1);
    recall_scores = zeros(num_folds, 1);
    f1_scores = zeros(num_folds, 1);
    auc_scores = zeros(num_folds, 1);

    % Network settings
    input_size = size(X, 2);
    hidden1 = 10;
    hidden2 = 6;
    output_size = 1;
    epochs = 30;
    lr = 0.001;

    for fold = 1:num_folds
        test_idx = fold_indices{fold};
        train_idx = setdiff(indices, test_idx);

        X_train = X(train_idx, :);
        y_train = y(train_idx);
        X_test = X(test_idx, :);
        y_test = y(test_idx);

        % Over-sampling on training data
        [X_train, y_train] = oversample_minority(X_train, y_train, 1);

        fprintf("Fold %d - Train Pos: %d, Neg: %d\n", fold, sum(y_train), length(y_train) - sum(y_train));

        % Initialize weights and biases
        W1 = randn(input_size, hidden1) * 0.01;
        b1 = zeros(1, hidden1);
        W2 = randn(hidden1, hidden2) * 0.01;
        b2 = zeros(1, hidden2);
        W3 = randn(hidden2, output_size) * 0.01;
        b3 = zeros(1, output_size);

        % Train the network
        for epoch = 1:epochs
            Z1 = X_train * W1 + b1;
            A1 = relu(Z1);
            Z2 = A1 * W2 + b2;
            A2 = relu(Z2);
            Z3 = A2 * W3 + b3;
            A3 = sigmoid(Z3);

            loss = -mean(y_train .* log(A3 + eps) + (1 - y_train) .* log(1 - A3 + eps));

            dZ3 = A3 - y_train;
            dW3 = (1 / size(X_train, 1)) * (A2' * dZ3);
            db3 = mean(dZ3);

            dZ2 = (dZ3 * W3') .* relu_derivative(Z2);
            dW2 = (1 / size(X_train, 1)) * (A1' * dZ2);
            db2 = mean(dZ2);

            dZ1 = (dZ2 * W2') .* relu_derivative(Z1);
            dW1 = (1 / size(X_train, 1)) * (X_train' * dZ1);
            db1 = mean(dZ1);

            W1 = W1 - lr * dW1;
            b1 = b1 - lr * db1;
            W2 = W2 - lr * dW2;
            b2 = b2 - lr * db2;
            W3 = W3 - lr * dW3;
            b3 = b3 - lr * db3;
        end

        % Prediction
        y_pred_prob = predict(X_test, W1, b1, W2, b2, W3, b3);
        y_pred = y_pred_prob >= 0.5;

        % Calculate metrics
        accuracy_scores(fold) = mean(y_pred == y_test);
        tp = sum(y_pred & y_test);
        fp = sum(y_pred & ~y_test);
        fn = sum(~y_pred & y_test);

        precision_scores(fold) = tp / max(tp + fp, eps);
        recall_scores(fold) = tp / max(tp + fn, eps);
        f1_scores(fold) = 2 * (precision_scores(fold) * recall_scores(fold)) / max(precision_scores(fold) + recall_scores(fold), eps);
        auc_scores(fold) = calculate_auc(y_test, y_pred_prob);
    end

    % Display overall results
    fprintf("\n=== Average Results (10-Fold CV) ===\n");
    fprintf("Accuracy: %.2f%%\n", mean(accuracy_scores) * 100);
    fprintf("Precision: %.2f%%\n", mean(precision_scores) * 100);
    fprintf("Recall: %.2f%%\n", mean(recall_scores) * 100);
    fprintf("F1 Score: %.2f%%\n", mean(f1_scores) * 100);
    fprintf("AUC-ROC: %.2f\n", mean(auc_scores));
end

function [X, y] = preprocess_churn_data()
    % Read data and remove header row
    data = csvread("Churn_Modelling.csv", 1, 0);

    % Remove columns: RowNumber (1), CustomerId (2), Surname (3)
    data(:, [1,2,3]) = [];

    % Label encoding for Geography and Gender columns:
    Geography = data(:,1);
    Gender = data(:,2);
    Geography = map_geography(Geography);
    Gender = map_gender(Gender);

    data(:,1) = Geography;
    data(:,2) = Gender;

    % Extract main variables
    CreditScore = data(:,3);
    Age = data(:,4);
    Tenure = data(:,5);
    Balance = data(:,6);
    NumOfProducts = data(:,7);
    HasCrCard = data(:,8);
    IsActiveMember = data(:,9);
    EstimatedSalary = data(:,10);
    y = data(:,11);  % Target variable Exited

    % Create new features
    BalanceSalary = Balance ./ (EstimatedSalary + eps);
    TenureAge = Tenure ./ (Age + eps);
    ScoreAge = CreditScore ./ (Age + eps);
    tenure_age = Tenure ./ ((Age - 17) + eps);
    tenure_salary = Tenure ./ (EstimatedSalary + eps);
    score_age = CreditScore ./ ((Age - 17) + eps);
    score_salary = CreditScore ./ (EstimatedSalary + eps);
    newAge = Age - Tenure;
    score_balance = Balance ./ (CreditScore + eps);
    age_balance = Balance ./ (Age + eps);
    balance_salary = Balance ./ (EstimatedSalary + eps);
    age_hascrcard = HasCrCard ./ (Age + eps);
    product_util_rate_year = NumOfProducts ./ (Tenure + (Tenure == 0));
    product_util_rate_salary = NumOfProducts ./ (EstimatedSalary + eps);
    monthly_salary = EstimatedSalary ./ 12;

    geo_salary_norm = zeros(length(Geography),1);
    for i = 1:length(Geography)
        if Geography(i) == 3
            geo_salary_norm(i) = monthly_salary(i) / 4740;
        elseif Geography(i) == 1
            geo_salary_norm(i) = monthly_salary(i) / 3696;
        elseif Geography(i) == 2
            geo_salary_norm(i) = monthly_salary(i) / 2257;
        else
            geo_salary_norm(i) = 0;
        end
    end

    X_raw = [Geography, Gender, CreditScore, Age, Tenure, Balance, ...
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, ...
        BalanceSalary, TenureAge, ScoreAge, tenure_age, ...
        tenure_salary, score_age, score_salary, newAge, ...
        score_balance, age_balance, balance_salary, age_hascrcard, ...
        product_util_rate_year, product_util_rate_salary, geo_salary_norm];

    % Normalization
    mu = mean(X_raw);
    sigma = std(X_raw);
    X = (X_raw - mu) ./ (sigma + eps);

    fprintf("✅ Data has been prepared: %d rows, %d features\n", size(X));
end

function geo_num = map_geography(column)
    geo_num = zeros(size(column));
    for i = 1:length(column)
        geo_text = column(i);
        if geo_text == double('G')
            geo_num(i) = 3;
        elseif geo_text == double('S')
            geo_num(i) = 2;
        else
            geo_num(i) = 1;
        end
    end
end

function gen_num = map_gender(column)
    gen_num = zeros(size(column));
    for i = 1:length(column)
        if column(i) == double('M')
            gen_num(i) = 1;
        else
            gen_num(i) = 0;
        end
    end
end

function A = relu(Z)
    A = max(0, Z);
end

function D = relu_derivative(Z)
    D = Z > 0;
end

function S = sigmoid(Z)
    S = 1 ./ (1 + exp(-Z));
end

function output = predict(X, W1, b1, W2, b2, W3, b3)
    A1 = relu(X * W1 + b1);
    A2 = relu(A1 * W2 + b2);
    output = sigmoid(A2 * W3 + b3);
end

function [X_aug, y_aug] = oversample_minority(X, y, ratio)
    pos_idx = find(y == 1);
    neg_idx = find(y == 0);

    X_pos = X(pos_idx, :);
    X_neg = X(neg_idx, :);

    n_pos = size(X_pos, 1);
    n_neg = size(X_neg, 1);
    target_pos = round(n_neg * ratio);
    n_new = target_pos - n_pos;

    if n_new <= 0
        X_aug = X;
        y_aug = y;
        return;
    end

    idx = randi([1 n_pos], n_new, 1);
    X_synthetic = X_pos(idx, :) + 0.01 * randn(n_new, size(X, 2));
    y_synthetic = ones(n_new, 1);

    X_aug = [X; X_synthetic];
    y_aug = [y; y_synthetic];

    mix = randperm(length(y_aug));
    X_aug = X_aug(mix, :);
    y_aug = y_aug(mix, :);
end

function auc = calculate_auc(y_true, y_score)
    [~, order] = sort(y_score, 'descend');
    y_true = y_true(order);
    tp = cumsum(y_true == 1);
    fp = cumsum(y_true == 0);
    tp_rate = tp / max(tp(end), eps);
    fp_rate = fp / max(fp(end), eps);
    auc = trapz(fp_rate, tp_rate);
end
