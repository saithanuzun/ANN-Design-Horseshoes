%% --------------------------------------------------------------
%% This is an extra script to see performance on all data 
%% net can be trained with all data with seleceted paramaters for deployment purpose
%% --------------------------------------------------------------


clear;

data = readtable('HorseShoe.csv');
data.Properties.VariableNames = {'InternalCurve','Width','Cord','Curve','ExternalCurve'};

data.Width = data.Width / 10;

% Original features
realInputs  = [data.InternalCurve, data.Width, data.Cord, data.Curve];
realTargets = data.ExternalCurve;

% Remove Cord (3rd column)
realInputs(:,3) = [];

% Create derived features
IC_test = realInputs(:,1);
W_test  = realInputs(:,2);
C_test  = realInputs(:,3);

f1_test = IC_test ./ W_test;
f2_test = IC_test - W_test;
f3_test = IC_test .* C_test;

realInputs = [realInputs, f1_test, f2_test, f3_test];

% Transpose for NN input
testInputs  = realInputs';       
testTargets = realTargets';      

% Load saved network
load('trainedNet.mat');  % bestNet, inputSettings, targetSettings

% Normalize test 
testInputsNorm = mapminmax('apply', testInputs, bestInputSettings);

% Predict
outputNorm = bestNet(testInputsNorm);
output = mapminmax('reverse', outputNorm, bestTargetSettings);

% R2
SS_res = sum((testTargets - output).^2);
SS_tot = sum((testTargets - mean(testTargets)).^2);
R2 = 1 - (SS_res / SS_tot);
disp(['R2 on real test data = ' num2str(R2, '%.3f')]);

% Plot predictions
if true
figure;
plot(testTargets, output, 'o');
xlabel('Target Values');
ylabel('Predicted Values');
title(['NN Performance on Real Data, R2 = ' num2str(R2, '%.3f')]);
grid on;
end