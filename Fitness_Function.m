function [y,Predict_Y_Train,flag1,flag2]=Fitness_Function(x,xopt,net,yopt)

[xtestnorm] = mapminmax('apply', x',xopt);
x0{1}=xtestnorm;
Predict_Ynorm_Train = net.predict(x0,'ExecutionEnvironment','cpu');
Predict_Y_Train  = mapminmax('reverse',Predict_Ynorm_Train',yopt);
y=-sum(x);
flag1=0;flag2=0;
Qmax=1500;dmin=2;
for i=1:6
Q(i,1:2)=x(1:2);x(1:2)=[];
end
QQ=sum(double(Q),2);
if ~isempty(find(QQ>Qmax))
flag1=1;
end
if ~isempty(find(Predict_Y_Train([14 15 16 19 20 21 22 25 26 27 28 31 32 33 37 38])<2))
flag2=1;
end
flag3=0;
if ~isempty(find(Predict_Y_Train([1 2 3 4 5 6 7 8 9 10 11 12 13 17 18 23 24 29 30 34 35 36 39 40])<2.5))
flag3=1;
end
if flag1
y=y+1e9;    
end
if flag2||flag3
y=y+1e9;    
end
end      
