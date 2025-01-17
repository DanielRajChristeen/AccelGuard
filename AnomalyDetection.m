% Complete Anomaly Detection for LIS3DHTR 3-Axis Accelerometer Data
% Visualizes Similarity Scores for the Entire Testing Dataset

% Step 1: Load and Verify Data
normalData = readtable('normal.csv');     % Load normal data
abnormalData = readtable('abnormal.csv'); % Load abnormal data
testData = readtable('testing.csv');      % Load test data

% Display column names for debugging
disp('Column names in normal.csv:');
disp(normalData.Properties.VariableNames);
disp('Column names in abnormal.csv:');
disp(abnormalData.Properties.VariableNames);
disp('Column names in testing.csv:');
disp(testData.Properties.VariableNames);

% Map columns to accelerometer axes (assuming Var1 = x, Var2 = y, Var3 = z)
normalX = normalData.Var1; normalY = normalData.Var2; normalZ = normalData.Var3; % For normal data
abnormalX = abnormalData.Var1; abnormalY = abnormalData.Var2; abnormalZ = abnormalData.Var3; % For abnormal data
testX = testData.Var1; testY = testData.Var2; testZ = testData.Var3; % For test data

% Step 2: Define Signal Processing Functions
blockSize = 50; % Size of each block for Max Pooling

% Max Pooling Function
maxPooling = @(signal, blockSize) arrayfun(@(i) ...
    max(signal(i:min(i+blockSize-1, end))), 1:blockSize:length(signal))';

% Absolute Value Function
absoluteValue = @(signal) abs(signal);

% Step 3: Process Data for Training
% Process Normal Data
normalFeatures = [maxPooling(normalX, blockSize), ...
                  maxPooling(normalY, blockSize), ...
                  maxPooling(normalZ, blockSize)];
normalFeatures = absoluteValue(normalFeatures);

% Process Abnormal Data
abnormalFeatures = [maxPooling(abnormalX, blockSize), ...
                    maxPooling(abnormalY, blockSize), ...
                    maxPooling(abnormalZ, blockSize)];
abnormalFeatures = absoluteValue(abnormalFeatures);

% Combine Normal and Abnormal Data for Training
features = [normalFeatures; abnormalFeatures];
labels = [zeros(size(normalFeatures, 1), 1); ones(size(abnormalFeatures, 1), 1)]; % 0 = normal, 1 = abnormal

% Step 4: Train ML Model (Logistic Regression)
mdl = fitglm(features, labels, 'Distribution', 'binomial', 'Link', 'logit');
disp('Acknowledgment: ML Model successfully trained. ML is ready for classification.');

% Step 5: Process Entire Test Data
% Note: Test data is processed in blocks of 50 samples for all rows
testFeaturesX = maxPooling(testX, blockSize);
testFeaturesY = maxPooling(testY, blockSize);
testFeaturesZ = maxPooling(testZ, blockSize);

% Combine the features from all axes
testFeatures = [testFeaturesX, testFeaturesY, testFeaturesZ];
testFeatures = absoluteValue(testFeatures);

% Predict similarity scores for the entire test dataset
similarityScores = 1 - predict(mdl, testFeatures); % Higher similarity indicates normal behavior
similarityScores = similarityScores * 100; % Convert to percentage

% Step 6: Visualization of Entire Testing Data
sampleNumber = 1:length(similarityScores); % Sample indices
threshold = 90; % Threshold for similarity score (lower scores are anomalies)

% Plot similarity scores
figure;
hold on;
plot(sampleNumber, similarityScores, 'b', 'LineWidth', 1.5); % Plot all similarity scores

% Highlight points below the threshold in red
belowThreshold = similarityScores < threshold; % Logical index for points below threshold
scatter(sampleNumber(belowThreshold), similarityScores(belowThreshold), ...
        50, 'r', 'filled'); % Highlight anomalous samples in red

% Plot the threshold line
yline(threshold, '--k', 'LineWidth', 1); % Add horizontal line for threshold

% Add labels, title, and legend
title('Similarity Score vs Sample Number (Entire Testing Data)');
xlabel('Sample Number');
ylabel('Similarity Score (%)');
legend('Similarity Score', 'Below Threshold', 'Threshold = 90%', 'Location', 'Best');
grid on;

% Display Summary of Anomalous Cases
disp('Anomalous Samples (Similarity Score Below Threshold):');
disp(find(belowThreshold));

% Save the plot
saveas(gcf, 'resultGraph.png'); % Save the plot as an image
disp('Acknowledgment: Results visualization saved as "SimilarityScorePlot_EntireTestingData.png".');
