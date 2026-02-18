%% Load Data with Dimension Checks
% Define the folder containing the data 
dataFolder = 'C:\Users\user\Downloads\CW-Data\CW-Data'; 
users = 1:10;  
featureFiles = {'Acc_FreqD_FDay', 'Acc_FreqD_MDay', 'Acc_TimeD_FDay', 'Acc_TimeD_MDay', 'Acc_TimeD_FreqD_FDay', 'Acc_TimeD_FreqD_MDay'};
data = []; 
labels = []; 

% Load data files
for user = users
    for i = 1:length(featureFiles)
        filename = sprintf('%s/U%02d_%s.mat', dataFolder, user, featureFiles{i});
        if exist(filename, 'file')
            loadedData = load(filename);
            field = fieldnames(loadedData);
            userData = loadedData.(field{1}); 
            if isempty(data)
                data = userData;
                labels = user * ones(size(userData, 1), 1);
            elseif size(userData, 2) == size(data, 2)
                data = [data; userData];
                labels = [labels; user * ones(size(userData, 1), 1)];
            else
                warning('Skipping %s due to inconsistent dimensions: %d columns.', filename, size(userData, 2));
            end
        else
            warning('File not found: %s', filename);
        end
    end
end

%% Descriptive Statistics and Visualizations
%  Display the calculated mean and standard deviation
meanFeatures = mean(data);
stdFeatures = std(data);


fprintf('Mean of Features:\n');
disp(meanFeatures);
fprintf('Standard Deviation of Features:\n');
disp(stdFeatures);

% Visualize data distribution
figure;
histogram(data(:,1));
xlabel('Feature Values');
ylabel('Frequency');
title('Histogram of Feature 1');
grid on;


%% Calculate intra-user and inter-user variance for each feature
intraUserVariances = zeros(10, size(data, 2));  
for user = 1:10
    for featureIdx = 1:size(data, 2)
        userData = data(labels == user, featureIdx);
        intraUserVariances(user, featureIdx) = var(userData);
    end
end
% Calculate inter-user variance and variance ratios 
interUserVariances = var(intraUserVariances, 0, 1);
varianceRatios = interUserVariances ./ mean(intraUserVariances, 1); 
disp('Variance Ratios for Each Feature:');
disp(varianceRatios);

%% Split Data into Training and Testing Sets

cv = cvpartition(labels, 'HoldOut', 0.2); % create partition object

trainIdx = training(cv);  % Training data indices
testIdx = test(cv);      % Testing data indices


XTrain = data(trainIdx, :);  % Training feature data
YTrain = labels(trainIdx);   % Training labels
XTest = data(testIdx, :);    % Testing feature data
YTest = labels(testIdx);     % Testing labels

%% Feature Scaling (Standardization)

[XTrain, mu, sigma] = zscore(XTrain); 
XTest = (XTest - mu) ./ sigma; 

%% Define and Train Feedforward Neural Network

hiddenLayerSize = 100;
net = feedforwardnet(hiddenLayerSize);

net.trainFcn = 'trainlm'; 
net.performFcn = 'mse'; 
net.trainParam.epochs = 500; 

[net, tr] = train(net, XTrain', dummyvar(categorical(YTrain))');

%% Evaluate the Model

YTestPred = net(XTest'); 
[~, YPredLabels] = max(YTestPred, [], 1); 
YPredLabels = YPredLabels'; 

% Calculate and display the accuracy of the predictions
accuracy = sum(YPredLabels == YTest) / length(YTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

%% Classification Report
fprintf('Classification Report:\n');

confMat = confusionmat(YTest, YPredLabels);
disp('Confusion Matrix:');
disp(confMat);

% Calculate and display precision, recall, and F1-score for each user
for i = 1:length(users)
    precision = confMat(i, i) / sum(confMat(:, i));
    recall = confMat(i, i) / sum(confMat(i, :));
    f1 = 2 * (precision * recall) / (precision + recall);
    fprintf('User %d - Precision: %.2f, Recall: %.2f, F1-Score: %.2f\n', ...
        users(i), precision, recall, f1);
end

%% Plot Confusion Matrix
figure;
confusionchart(YTest, YPredLabels); % Create a confusion chart
title('Confusion Matrix for User Authentication'); % Add a title to the plot

%% Feature Selection
numFeatures = size(data, 2);
corrCoeffs = zeros(1, numFeatures);
for i = 1:numFeatures
    corrCoeffs(i) = corr(data(:, i), labels, 'type', 'Spearman'); 
end

corrThreshold = 0.1;
selectedFeatures = abs(corrCoeffs) > corrThreshold;

dataSelected = data(:, selectedFeatures);

fprintf('Number of selected features: %d\n', sum(selectedFeatures));

%% Split Data into Training and Testing Sets (Using Selected Features)
cv = cvpartition(labels, 'HoldOut', 0.2);

trainIdx = training(cv);
testIdx = test(cv);

% Create training and testing sets with selected features
XTrain = dataSelected(trainIdx, :);
YTrain = labels(trainIdx);
XTest = dataSelected(testIdx, :);
YTest = labels(testIdx);

[XTrain, mu, sigma] = zscore(XTrain);
XTest = (XTest - mu) ./ sigma;

%% Hyperparameter Tuning: Neurons and Learning Rate
hiddenLayerSizes = [50, 100, 150, 200]; 
learningRates = [0.001, 0.01, 0.1]; 

bestAccuracy = 0;
bestConfig = struct('hiddenLayerSize', 0, 'learningRate', 0);

% Iterate over all combinations of hyperparameters
for h = 1:length(hiddenLayerSizes)
    for lr = 1:length(learningRates)
        
        hiddenLayerSize = hiddenLayerSizes(h);
        learningRate = learningRates(lr);
        
        net = feedforwardnet(hiddenLayerSize); 
        net.trainFcn = 'trainlm'; 
        net.performFcn = 'mse'; 
        net.trainParam.epochs = 500; 
        net.trainParam.lr = learningRate; 
        
        % Train the network
        try
            [net, tr] = train(net, XTrain', dummyvar(categorical(YTrain))');
            
            % Evaluate the network on the test set
            YTestPred = net(XTest'); 
            [~, YPredLabels] = max(YTestPred, [], 1); 
            accuracy = sum(YPredLabels' == YTest) / length(YTest); % Accuracy
            
            
            if accuracy > bestAccuracy
                bestAccuracy = accuracy;
                bestConfig.hiddenLayerSize = hiddenLayerSize;
                bestConfig.learningRate = learningRate;
            end
            
            % Display progress
            fprintf('Hidden Layer: %d, Learning Rate: %.4f, Accuracy: %.2f%%\n', ...
                hiddenLayerSize, learningRate, accuracy * 100);
        catch ME
            fprintf('Error training with Hidden Layer: %d, Learning Rate: %.4f\n', ...
                hiddenLayerSize, learningRate);
            disp(ME.message);
        end
    end
end

% Display the best configuration
fprintf('Best Configuration: Hidden Layer: %d, Learning Rate: %.4f, Accuracy: %.2f%%\n', ...
    bestConfig.hiddenLayerSize, bestConfig.learningRate, bestAccuracy * 100);

