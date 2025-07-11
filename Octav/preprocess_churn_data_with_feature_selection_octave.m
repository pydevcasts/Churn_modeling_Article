function [X, y] = preprocess_churn_data_with_feature_selection_octave()
    % بازکردن فایل CSV به صورت متنی
    fid = fopen('Churn_Modelling.csv', 'r');
    header = fgetl(fid); % خواندن هدر و حذف آن
    
    % خواندن داده‌ها
    data_cell = textscan(fid, '%d %d %s %s %f %f %d %f %d %d %f %d %d', ...
                'Delimiter', ',', 'CollectOutput', false);
    fclose(fid);
    
    % data_cell شامل ستون‌ها به صورت سلولی است
    % ترتیب ستون‌ها:
    % 1: RowNumber (int)
    % 2: CustomerId (int)
    % 3: Surname (string)
    % 4: Geography (string)
    % 5: Gender (string)
    % 6: CreditScore (float)
    % 7: Age (float)
    % 8: Tenure (int)
    % 9: Balance (float)
    % 10: NumOfProducts (int)
    % 11: HasCrCard (int)
    % 12: IsActiveMember (int)
    % 13: EstimatedSalary (float)
    % 14: Exited (int)
    
    % تبدیل ستون‌های متنی به عددی
    Geography_str = data_cell{3};
    Gender_str = data_cell{4};
    
    Geography = zeros(length(Geography_str),1);
    for i=1:length(Geography_str)
        if strcmp(Geography_str{i}, 'Germany')
            Geography(i) = 3;
        elseif strcmp(Geography_str{i}, 'Spain')
            Geography(i) = 2;
        else
            Geography(i) = 1; % France
        end
    end
    
    Gender = zeros(length(Gender_str),1);
    for i=1:length(Gender_str)
        if strcmp(Gender_str{i}, 'Male')
            Gender(i) = 1;
        else
            Gender(i) = 0; % Female
        end
    end
    
    % استخراج بقیه ستون‌ها (اعدادی)
    CreditScore = data_cell{5};
    Age = data_cell{6};
    Tenure = data_cell{7};
    Balance = data_cell{8};
    NumOfProducts = data_cell{9};
    HasCrCard = data_cell{10};
    IsActiveMember = data_cell{11};
    EstimatedSalary = data_cell{12};
    y = data_cell{13};
    
    % حذف داده‌های نویزی مثل قبل
    valid_idx = (CreditScore > 359) & (Age < 71) & (NumOfProducts < 4);
    
    Geography = Geography(valid_idx);
    Gender = Gender(valid_idx);
    CreditScore = CreditScore(valid_idx);
    Age = Age(valid_idx);
    Tenure = Tenure(valid_idx);
    Balance = Balance(valid_idx);
    NumOfProducts = NumOfProducts(valid_idx);
    HasCrCard = HasCrCard(valid_idx);
    IsActiveMember = IsActiveMember(valid_idx);
    EstimatedSalary = EstimatedSalary(valid_idx);
    y = y(valid_idx);
    
    % مهندسی ویژگی‌ها
    BalanceSalary = Balance ./ (EstimatedSalary + eps);
    TenureAge = Tenure ./ (Age + eps);
    ScoreAge = CreditScore ./ (Age + eps);
    tenure_age = Tenure ./ (Age - 17 + eps);
    tenure_salary = Tenure ./ (EstimatedSalary + eps);
    score_age = CreditScore ./ (Age - 17 + eps);
    score_salary = CreditScore ./ (EstimatedSalary + eps);
    newAge = Age - Tenure;
    score_balance = Balance ./ (CreditScore + eps);
    age_balance = Balance ./ (Age + eps);
    balance_salary = Balance ./ (EstimatedSalary + eps);
    age_hascrcard = HasCrCard ./ (Age + eps);
    
    n = length(Geography);
    product_util_rate_year = zeros(n,1);
    product_util_rate_salary = zeros(n,1);
    geo_salary_norm = zeros(n,1);
    
    for i=1:n
        if NumOfProducts(i) == 0
            product_util_rate_year(i) = 0;
        elseif Tenure(i) == 0
            product_util_rate_year(i) = NumOfProducts(i);
        else
            product_util_rate_year(i) = NumOfProducts(i) / Tenure(i);
        end
        
        if NumOfProducts(i) == 0
            product_util_rate_salary(i) = 0;
        else
            product_util_rate_salary(i) = NumOfProducts(i) / (EstimatedSalary(i) + eps);
        end
        
        monthly_salary = EstimatedSalary(i) / 12;
        if Geography(i) == 3
            geo_salary_norm(i) = monthly_salary / 4740;
        elseif Geography(i) == 1
            geo_salary_norm(i) = monthly_salary / 3696;
        elseif Geography(i) == 2
            geo_salary_norm(i) = monthly_salary / 2257;
        else
            geo_salary_norm(i) = 0;
        end
    end
    
    % ماتریس نهایی ویژگی‌ها
    X_raw = [Geography, Gender, CreditScore, Age, Tenure, Balance, ...
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, ...
        BalanceSalary, TenureAge, ScoreAge, tenure_age, tenure_salary, ...
        score_age, score_salary, newAge, score_balance, age_balance, ...
        balance_salary, age_hascrcard, product_util_rate_year, product_util_rate_salary, geo_salary_norm];
    
    % نرمال‌سازی
    mu = mean(X_raw);
    sigma = std(X_raw);
    X = (X_raw - mu) ./ (sigma + eps);
    
    fprintf("✅ داده‌ها آماده‌سازی شدند: %d سطر، %d ویژگی\n", size(X));
end
