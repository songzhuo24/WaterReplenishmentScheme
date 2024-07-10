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
    'InitialLearnRate',0.005, ...
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

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');




tic

net = trainNetwork(Train_xNorm,Train_yNorm,lgraph,opts);





close all


Predict_Ynorm_Train = net.predict(Train_xNorm,'ExecutionEnvironment','cpu');
Predict_Y_Train  = mapminmax('reverse',Predict_Ynorm_Train',yopt);
Predict_Y_Train = Predict_Y_Train';



Predict_Ynorm = net.predict(Test_xNorm,'ExecutionEnvironment','cpu');
Predict_Y  = mapminmax('reverse',Predict_Ynorm',yopt);
Predict_Y = Predict_Y';

dimension=12;Size=50;
Tmax=100;
c1=1.5;c2=1.5;
w_max = 0.9;                     
w_min = 0.2;

Xmax=[1000 500 1000 500 1000 500 1000 500 1000 500 1000 500];
Xmin=[722 289 872 348 705 280 488 192 670 267 686 274];
k=0.5;
Vmax= Xmax*k;Vmin=-Vmax;   

for i=1:Size
    for j=1:dimension
X(j,i) = Xmin(j)+(Xmax(j)-Xmin(j))*rand ;   
V(j,i) = Vmin(j)+(Vmax(j)-Vmin(j))*rand;   
    end
end

Pi=X; 
Pg=zeros(dimension,1);
for j=1:Size
    f_x(j)=Fitness_Function(X(:,j)',xopt,net,yopt);
    f_xbest(j)=f_x(j);           
end
[f_g,I]=min(f_xbest);     
Pg=X(:,I); 


for t=1:Tmax
    time(t)=t;
    fprintf('Current evolution generation£º%d\n',t)
     w = w_max-(w_max-w_min)*t/Tmax;£©
     r1=rand(1);r2=rand(1);
     for j=1:Size
         V(:,j) = w*V(:,j) + c1*r1*(Pi(:,j)-X(:,j)) + c2*r2*(Pg-X(:,j)); 
     end

for j=1:Size
    for i=1:dimension
        if V(i,j)>Vmax(i) 
            V(i,j)=Vmax(i);
        elseif V(i,j)<Vmin (i)
            V(i,j)=Vmin(i);
        else
        end
    end
end
 X= X+V;

for j=1:Size
    for i=1:dimension
        if X(i,j)>Xmax (i)
            X(i,j)=Xmax(i);
        elseif X(i,j)<Xmin (i)
            X(i,j)=Xmin(i);
        else
        end
    end
end

 for j=1:Size
     f_x(j)=Fitness_Function(X(:,j)',xopt,net,yopt);
     if f_x(j)<f_xbest(j)
       f_xbest(j)=f_x(j);
       Pi(:,j)=X(:,j);
     end
     if f_xbest(j)<f_g
             Pg= Pi(:,j);
            f_g=f_xbest(j);
     end
 end
  f_gg(t)=f_g;
  PGG{t}=Pg;
end

[~,index]=min(f_gg);
best_x=PGG{index(1)};

figure
plot(time,-f_gg);
xlabel('The number of iterations');ylabel('Target');    
[~,YY,flag1,flag2]=Fitness_Function(best_x',xopt,net,yopt);


