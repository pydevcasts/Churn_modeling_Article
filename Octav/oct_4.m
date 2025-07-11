clear;
filename = 'Churn_Modelling.csv';

% --- 1. Read CSV file manually ---
fid = fopen(filename);
if fid == -1
    error('❌ Unable to open file: %s', filename);
end
header_line = fgetl(fid);
headers = strsplit(header_line, ',');
data = {};
while ~feof(fid)
    line = fgetl(fid);
    row = strsplit(line, ',');
    data = [data; row];
end
fclose(fid);

% --- 2. Remove unnecessary columns ---
remove_cols = {'RowNumber', 'CustomerId', 'Surname'};
keep_idx = ~ismember(headers, remove_cols);
headers = headers(keep_idx);
data = data(:, keep_idx);

% --- 3. Extract target column (Exited) ---
y_idx = find(strcmp(headers, 'Exited'));
y = str2double(data(:, y_idx));
headers(y_idx) = [];
data(:, y_idx) = [];

% --- 4. Process features: normalization and encoding ---
X = [];
for i = 1:length(headers)
    col = data(:, i);
    nums = str2double(col);
    if all(~isnan(nums))
        mu = mean(nums);
        sigma = std(nums);
        nums = (nums - mu) ./ sigma;
        X = [X, nums];
    else
        categories = unique(col);
        for j = 1:length(categories)
            X = [X, strcmp(col, categories{j})];
        end
    end
end

% --- 5. Train/Test Split ---
m = size(X,1);
rand_indices = randperm(m);
split = round(0.8 * m);
X_train = X(rand_indices(1:split), :);
y_train = y(rand_indices(1:split));
X_test  = X(rand_indices(split+1:end), :);
y_test  = y(rand_indices(split+1:end));

% --- 6. Network Architecture Parameters ---
input_size = size(X_train,2);
hidden1 = 6;
hidden2 = 6;
output_size = 1;

% --- 7. Initialize weights and biases ---
W1 = randn(hidden1, input_size) * 0.1;
b1 = zeros(hidden1, 1);
W2 = randn(hidden2, hidden1) * 0.1;
b2 = zeros(hidden2, 1);
W3 = randn(output_size, hidden2) * 0.1;
b3 = zeros(output_size, 1);

% --- 8. RMSProp Settings ---
alpha = 0.001;     % learning rate
rho = 0.9;         % decay rate
eps = 1e-8;
batch_size = 7;
epochs = 32;

% RMSProp gradient memory
G_W1 = zeros(size(W1));
G_b1 = zeros(size(b1));
G_W2 = zeros(size(W2));
G_b2 = zeros(size(b2));
G_W3 = zeros(size(W3));
G_b3 = zeros(size(b3));

% --- 9. Activation functions ---
relu = @(x) max(0, x);
drelu = @(x) x > 0;
sigmoid = @(x) 1 ./ (1 + exp(-x));

% --- 10. Train the network ---
for epoch = 1:epochs
    indices = randperm(size(X_train,1));
    for i = 1:batch_size:size(X_train,1)
        idx = indices(i:min(i+batch_size-1, end));
        x = X_train(idx,:)';
        t = y_train(idx)';

        % Forward propagation
        z1 = W1 * x + b1;
        a1 = relu(z1);
        z2 = W2 * a1 + b2;
        a2 = relu(z2);
        z3 = W3 * a2 + b3;
        a3 = sigmoid(z3);

        % Backpropagation
        dz3 = a3 - t;
        dW3 = dz3 * a2';
        db3 = sum(dz3, 2);

        dz2 = (W3' * dz3) .* drelu(z2);
        dW2 = dz2 * a1';
        db2 = sum(dz2, 2);

        dz1 = (W2' * dz2) .* drelu(z1);
        dW1 = dz1 * x';
        db1 = sum(dz1, 2);

        % RMSProp update
        G_W3 = rho * G_W3 + (1 - rho) * (dW3.^2);
        G_b3 = rho * G_b3 + (1 - rho) * (db3.^2);
        G_W2 = rho * G_W2 + (1 - rho) * (dW2.^2);
        G_b2 = rho * G_b2 + (1 - rho) * (db2.^2);
        G_W1 = rho * G_W1 + (1 - rho) * (dW1.^2);
        G_b1 = rho * G_b1 + (1 - rho) * (db1.^2);

        W3 -= alpha * dW3 ./ sqrt(G_W3 + eps);
        b3 -= alpha * db3 ./ sqrt(G_b3 + eps);
        W2 -= alpha * dW2 ./ sqrt(G_W2 + eps);
        b2 -= alpha * db2 ./ sqrt(G_b2 + eps);
        W1 -= alpha * dW1 ./ sqrt(G_W1 + eps);
        b1 -= alpha * db1 ./ sqrt(G_b1 + eps);
    end
end

% --- 11. Final Testing ---
z1 = W1 * X_test';
a1 = relu(z1);
z2 = W2 * a1;
a2 = relu(z2);
z3 = W3 * a2;
a3 = sigmoid(z3);

y_pred = a3 > 0.5;
accuracy = sum(y_pred' == y_test) / length(y_test);

fprintf('✅ Test Accuracy: %.2f%%\n', accuracy * 100);

% --- 12. Evaluation Metrics ---
TP = sum((y_pred' == 1) & (y_test == 1));
FP = sum((y_pred' == 1) & (y_test == 0));
FN = sum((y_pred' == 0) & (y_test == 1));
TN = sum((y_pred' == 0) & (y_test == 0));

precision = TP / (TP + FP);
recall = TP / (TP + FN);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('Confusion Matrix:\n');
disp([TP, FP; FN, TN]);


% --- Plot Confusion Matrix ---
figure;
imagesc([TP, FP; FN, TN]);
colormap('bone');
colorbar;
xlabel('Predicted Label');
ylabel('True Label');
title('Confusion Matrix');
set(gca, 'XTick', 1:2, 'XTickLabel', {'Negative', 'Positive'});
set(gca, 'YTick', 1:2, 'YTickLabel', {'Negative', 'Positive'});


% --- Compute ROC Curve & AUC ---
thresholds = linspace(0, 1, 100);
TPR = zeros(size(thresholds));
FPR = zeros(size(thresholds));

for i = 1:length(thresholds)
    th = thresholds(i);
    y_tmp = a3 > th;

    TP = sum((y_tmp' == 1) & (y_test == 1));
    FP = sum((y_tmp' == 1) & (y_test == 0));
    FN = sum((y_tmp' == 0) & (y_test == 1));
    TN = sum((y_tmp' == 0) & (y_test == 0));

    TPR(i) = TP / (TP + FN);
    FPR(i) = FP / (FP + TN);
end

% Sort by FPR for plotting and AUC calculation
[FPR, sort_idx] = sort(FPR);
TPR = TPR(sort_idx);

% Plot ROC Curve
figure;
plot(FPR, TPR, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'r--');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
legend('Model ROC', 'Random Guess');
grid on;

% Compute AUC (using Trapezoidal rule)
auc = trapz(FPR, TPR);
fprintf('AUC: %.4f\n', auc);

