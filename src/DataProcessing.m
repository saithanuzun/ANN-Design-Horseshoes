data = readtable('HorseShoe.csv');

data.Properties.VariableNames = {'InternalCurve','Width','Cord','Curve','ExternalCurve'};

featureNames = {'InternalCurve','Width','Cord','Curve'};

data.Width = data.Width / 10;
 
inputs  = [data.InternalCurve, data.Width, data.Cord, data.Curve];
targets = data.ExternalCurve;


%box plot to distribution
figure;
boxplot([inputs,targets],[featureNames,"ExternalCurve"]);
title("data distribution");


cordOutliersRows  = [46];                
widthOutliersRows = [23, 28, 81]; 

rowsToRemove = [cordOutliersRows,widthOutliersRows];

inputs(rowsToRemove, :)  = [];
targets(rowsToRemove, :) = [];

figure;
gplotmatrix([inputs,targets],[],[],"b","o",[],"on","none",[featureNames,"ExternalCurve"],[]);
title("Pairwise Relationships")

%correlation of all features
corrMatrix = corr([inputs,targets]);

%cord is removed from our inputs
inputs(:,3) = [];



%% Data Augmentation disabled
augmentation = false;
augmentationFactor = 10;   
noiseLevel = 0.02;        

inputAug  = inputs;
targetAug = targets;

if augmentation

    for n = 1:augmentationFactor 

    noise = noiseLevel * (std(inputs) .* randn(size(inputs)));
    newInputs = inputs + noise;

    inputAug  = [inputAug; newInputs];
    targetAug = [targetAug; targets];
    end

end

inputs  = inputAug;
targets = targetAug;


%% feature engineering
IC = inputs(:,1);
W  = inputs(:,2);
C  = inputs(:,3);

f1 = IC ./ W;
f2 = IC - W;
f3 = IC .* C;

inputs = [inputs,f1,f2,f3];
