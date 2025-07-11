clear;

filename = 'Churn_Modelling.csv';

% Step 1: Read CSV
fid = fopen(filename);
if fid == -1, error('❌ File not found'); end
header = strsplit(fgetl(fid), ',');
raw = {};
while ~feof(fid)
    line = fgetl(fid);
    row = strsplit(line, ',');
    raw = [raw; row];
end
fclose(fid);

% Step 2: Remove unnecessary columns
drop = {'RowNumber', 'CustomerId', 'Surname'};
keep = ~ismember(header, drop);
header = header(keep);
raw = raw(:, keep);

% Step 3: Extract label
label_idx = find(strcmp(header, 'Exited'));
y = str2double(raw(:, label_idx));
header(label_idx) = [];
raw(:, label_idx) = [];

% Step 4: Encode + Normalize
X = [];
for i = 1:length(header)
    col = raw(:, i);
    nums = str2double(col);
    if all(~isnan(nums))
        mu = mean(nums); sigma = std(nums);
        X = [X, (nums - mu) ./ sigma];
    else
        values = unique(col);
        for j = 1:length(values)
            X = [X, strcmp(col, values{j})];
        end
    end
end

% Step 5: Oversampling minority class
X0 = X(y == 0, :);  y0 = y(y == 0);
X1 = X(y == 1, :);  y1 = y(y == 1);
repeat = floor(size(X0,1) / size(X1,1));
X1_os = repmat(X1, repeat, 1);
y1_os = repmat(y1, repeat, 1);
X_bal = [X0; X1_os];
y_bal = [y0; y1_os];

% Step 6: Shuffle and Split
m = size(X_bal, 1);
idx = randperm(m);
X_bal = X_bal(idx, :);
y_bal = y_bal(idx);
split = round(0.8 * m);
X_train = X_bal(1:split, :);
y_train = y_bal(1:split);
X_test = X_bal(split+1:end, :);
y_test = y_bal(split+1:end);

% Step 7: Parameters
in = size(X_train, 2); h1 = 6; h2 = 6; out = 1;
alpha = 0.001; rho = 0.9; eps = 1e-8;
epochs = 50; batch_size = 10;
relu = @(x) max(0,x); drelu = @(x) x>0; sig = @(x) 1./(1+exp(-x));

% Step 8: Ensemble
num_models = 3;
pred_all = zeros(length(y_test), num_models);
for model_num = 1:num_models
    W1 = randn(h1, in) * 0.1; b1 = zeros(h1, 1);
    W2 = randn(h2, h1) * 0.1; b2 = zeros(h2, 1);
    W3 = randn(out, h2) * 0.1; b3 = zeros(out, 1);
    G1 = zeros(size(W1)); Gb1 = zeros(size(b1));
    G2 = zeros(size(W2)); Gb2 = zeros(size(b2));
    G3 = zeros(size(W3)); Gb3 = zeros(size(b3));

    for ep = 1:epochs
        inds = randperm(size(X_train, 1));
        for i = 1:batch_size:size(X_train, 1)
            id = inds(i:min(i+batch_size-1, end));
            x = X_train(id,:)'; t = y_train(id)';
            z1 = W1*x + b1; a1 = relu(z1);
            z2 = W2*a1 + b2; a2 = relu(z2);
            z3 = W3*a2 + b3; a3 = sig(z3);

            dz3 = a3 - t; dW3 = dz3*a2'; db3 = sum(dz3,2);
            dz2 = (W3'*dz3).*drelu(z2); dW2 = dz2*a1'; db2 = sum(dz2,2);
            dz1 = (W2'*dz2).*drelu(z1); dW1 = dz1*x'; db1 = sum(dz1,2);

            G3 = rho*G3 + (1-rho)*(dW3.^2); Gb3 = rho*Gb3 + (1-rho)*(db3.^2);
            G2 = rho*G2 + (1-rho)*(dW2.^2); Gb2 = rho*Gb2 + (1-rho)*(db2.^2);
            G1 = rho*G1 + (1-rho)*(dW1.^2); Gb1 = rho*Gb1 + (1-rho)*(db1.^2);

            W3 -= alpha * dW3 ./ sqrt(G3 + eps); b3 -= alpha * db3 ./ sqrt(Gb3 + eps);
            W2 -= alpha * dW2 ./ sqrt(G2 + eps); b2 -= alpha * db2 ./ sqrt(Gb2 + eps);
            W1 -= alpha * dW1 ./ sqrt(G1 + eps); b1 -= alpha * db1 ./ sqrt(Gb1 + eps);
        end
    end

    % Predict test set
    z1 = W1 * X_test'; a1 = relu(z1);
    z2 = W2 * a1;       a2 = relu(z2);
    z3 = W3 * a2;       a3 = sig(z3);
    pred_all(:, model_num) = a3';
end

% Step 9: Soft Voting
y_prob = mean(pred_all, 2);
y_pred = y_prob > 0.5;
acc = sum(y_pred == y_test) / length(y_test);
TP = sum((y_pred == 1) & (y_test == 1));
FP = sum((y_pred == 1) & (y_test == 0));
FN = sum((y_pred == 0) & (y_test == 1));
TN = sum((y_pred == 0) & (y_test == 0));
precision = TP / (TP + FP);
recall = TP / (TP + FN);

fprintf('\n✅ Ensemble Accuracy: %.2f%%\n', acc*100);
fprintf('Precision: %.2f | Recall: %.2f\n', precision, recall);
fprintf('Confusion Matrix:\n');
disp([TP, FP; FN, TN]);

% Step 10: ROC Curve + AUC
t = linspace(0,1,100);
TPR = zeros(size(t)); FPR = zeros(size(t));
for i = 1:length(t)
    yp = y_prob > t(i);
    TP = sum((yp == 1) & (y_test == 1));
    FP = sum((yp == 1) & (y_test == 0));
    FN = sum((yp == 0) & (y_test == 1));
    TN = sum((yp == 0) & (y_test == 0));
    TPR(i) = TP / (TP + FN);
    FPR(i) = FP / (FP + TN);
end
[~, sidx] = sort(FPR); FPR = FPR(sidx); TPR = TPR(sidx);
auc = trapz(FPR, TPR);
figure;
plot(FPR, TPR, 'b', 'LineWidth', 2); hold on;
plot([0 1],[0 1], 'r--');
title(sprintf('ROC Curve (AUC = %.2f)', auc));
xlabel('False Positive Rate'); ylabel('True Positive Rate');
grid on;
