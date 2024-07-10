clc
close all
clear all


XTrain = xlsread('dataset.xlsx',1,'A1:L100');
YTrain = xlsread('dataset.xlsx',1,'M1:AZ100');

XTest = xlsread('dataset.xlsx',1,'A101:L150');
YTest = xlsread('dataset.xlsx',1,'M101:AZ150');



x = XTrain;
y = YTrain;

[xnorm,xopt] = mapminmax(x',0,1);
[ynorm,yopt] = mapminmax(y',0,1);
x = x';




for i = 1:length(ynorm)

    Train_xNorm{i} = reshape(xnorm(:,i),12,1,1);
    Train_yNorm(:,i) = ynorm(:,i);
    Train_y(i,:) = y(i,:);
end
Train_yNorm= Train_yNorm';


xtest = XTest;
ytest = YTest;
[xtestnorm] = mapminmax('apply', xtest',xopt);
[ytestnorm] = mapminmax('apply',ytest',yopt);
xtest = xtest';
for i = 1:length(ytestnorm)
   Test_xNorm{i} = reshape(xtestnorm(:,i),12,1,1);
   Test_yNorm(:,i) = ytestnorm(:,i);
   Test_y(i,:) = ytest(i,:);
end
Test_yNorm = Test_yNorm';



inputSize = 12;   
outputSize = 40; 
numhidden_units1=200;
numhidden_units2= 200;



opts = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'GradientThreshold',1,...
    'ExecutionEnvironment','cpu',...
    'InitialLearnRate',0.002, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',100, ...  
    'LearnRateDropFactor',0.5, ...
    'L2Regularization',1e-6,...
    'Shuffle','once',... 
    'SequenceLength',1,...
    'MiniBatchSize',100,...
    'Verbose',1,...
    'Plots','training-progress');


layers = [ ...
    
    sequenceInputLayer([inputSize,1,1],'name','input')   
    sequenceFoldingLayer('name','fold')
    convolution2dLayer([2,1],10,'Stride',[1,1],'name','conv1')
    batchNormalizationLayer('name','batchnorm1')
    reluLayer('name','relu1')
    maxPooling2dLayer([1,3],'Stride',1,'Padding','same','name','maxpool')
    sequenceUnfoldingLayer('name','unfold')
    flattenLayer('name','flatten')
    lstmLayer(numhidden_units1,'Outputmode','sequence','name','hidden1') 
    dropoutLayer(0.3,'name','dropout_1')
    lstmLayer(numhidden_units2,'Outputmode','last','name','hidden2') 
    dropoutLayer(0.3,'name','drdiopout_2')
    fullyConnectedLayer(outputSize,'name','fullconnect')   
    tanhLayer('name','softmax')
    regressionLayer('name','output')];

lgraph = layerGraph(layers)
lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');




tic

net = trainNetwork(Train_xNorm,Train_yNorm,lgraph,opts);





close all


Predict_Ynorm_Train = net.predict(Train_xNorm);
Predict_Y_Train  = mapminmax('reverse',Predict_Ynorm_Train',yopt);
Predict_Y_Train = Predict_Y_Train';


for i=1:40
figure
plot(Predict_Y_Train(:,i),'r-*','LineWidth',1.5)
hold on 
plot(Train_y(:,i),'b-o','LineWidth',1.5);    
xlabel('Training sample number')
legend('Predictive value','Actual value')
end

for i =1:40
 ae= abs(Predict_Y_Train(:,i) -Train_y(:,i));
 rmse = (mean(ae.^2)).^0.5;
 mse = mean(ae.^2);
 mae = mean(ae);
 mape = mean(ae./Train_y(:,i));


 y=Predict_Y_Train(:,i)
 disp(['RMSE = ', num2str(rmse)])
 disp(['MSE  = ', num2str(mse)])
 disp(['MAE  = ', num2str(mae)])
 disp(['MAPE = ', num2str(mape)])
 disp('================================')
 clear rmse mse mae mape
end



Predict_Ynorm = net.predict(Test_xNorm);
Predict_Y  = mapminmax('reverse',Predict_Ynorm',yopt);
Predict_Y = Predict_Y';

for i=1:40
figure
plot(Predict_Y(:,i),'r-*','LineWidth',1.5)
hold on 
plot(Test_y(:,i),'b-o','LineWidth',1.5);    
xlabel('Training sample number')
legend('Predictive value','Actual value')
end

for i =1:40
 ae= abs(Predict_Y(:,i) -Test_y(:,i));
 rmse = (mean(ae.^2)).^0.5;
 mse = mean(ae.^2);
 mae = mean(ae);
 mape = mean(ae./Test_y(:,i));


 y=Predict_Y(:,i)
 disp(['RMSE = ', num2str(rmse)])
 disp(['MSE  = ', num2str(mse)])
 disp(['MAE  = ', num2str(mae)])
 disp(['MAPE = ', num2str(mape)])
 disp('================================')
 clear rmse mse mae mape
end





