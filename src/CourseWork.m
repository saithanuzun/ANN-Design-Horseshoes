clear; clc; close all;

%% HyperParameters
KFold = 4;     

hiddenLayerSize = [20];     
trainFcn = 'trainbr';          
transferFcn = 'tansig';        
normRange = [-1, 1];           
                 
showTrainingWindow = true;            
showPlotResults = true;   
%% -------------------

% Data
run("DataProcessing");

inputsTranspose = inputs';   
targetsTranspose = targets';

cv = cvpartition(size(targets,1), 'KFold', KFold);
R2_scores = zeros(KFold,1);

bestR2 = 0;
bestNet = [];

for i = 1:KFold

    trainIndices = cv.training(i);
    testIndices = cv.test(i);
    
    inputTrain = inputsTranspose(:, trainIndices);
    targetTrain = targetsTranspose(:, trainIndices);
    inputTest  = inputsTranspose(:, testIndices);
    targetTest = targetsTranspose(:, testIndices);

    %% Noise augmentation loop but disabled
    inputAug = inputTrain;
    targetAug = targetTrain;


    % Normalize  
    [inputTrainNorm, inputSettings] = mapminmax(inputAug, normRange(1), normRange(2));
    [targetTrainNorm, targetSettings] = mapminmax(targetAug, normRange(1), normRange(2));


    
    % Create and train network with 
    net = fitnet(hiddenLayerSize, trainFcn);
    net.divideFcn = 'dividetrain';
    net.trainParam.showWindow = showTrainingWindow;
    net.layers{1}.transferFcn = transferFcn;

    [net, tr] = train(net, inputTrainNorm, targetTrainNorm);

    % Normalize test data 
    inputTestNorm = mapminmax('apply', inputTest, inputSettings);
    
    % Predict
    outputTestNorm = net(inputTestNorm);
    
    % Denormalize predictions
    outputTest = mapminmax('reverse', outputTestNorm, targetSettings);
    
    % Calculate R2 
    SS_res = sum((targetTest - outputTest).^2, 2);       
    SS_tot = sum((targetTest - mean(targetTest,2)).^2, 2);
    R2 = 1 - (SS_res ./ SS_tot);

    R2_scores(i) = R2;

    if (R2 > bestR2)
        bestR2 = R2;
        bestNet = net;
        bestInputSettings = inputSettings;
        bestTargetSettings = targetSettings;
       
    end

    if showPlotResults
        figure;
        plot(targetTest, outputTest, 'o');
        hold on;
        xlabel('Target Values');
        ylabel('Predicted Values');
        title(['Fold ' num2str(i) ' R2 = ' num2str(R2, '%.3f')]);
        grid on;
    end
end

meanR2 = mean(R2_scores)
R2_scores

save('trainedNet.mat', 'bestNet', 'bestInputSettings', 'bestTargetSettings');

%optional testing on all data
run("Testing.m");