%% ------------------------The RBF training algorithm----------------------
%% ---------------------------------reset----------------------------------
clear;close all;clc;
%% -------------------------------initialize-------------------------------
Data=mackeyglass(); %makeyglass Data input
start_train=1;
end_train=25000;
start_test=25000;
end_test=30000;
clusterN=25; %number of clusters
past=2; %number of past value for prediction
pre=2; %Number of forward value prediction
eta=0.01;
%
train1=Data(start_train:end_train-pre,2);
train2=Data(start_train+pre:end_train,2);
index_train=Data(start_train+pre:end_train,1);
%
test1=Data(start_test:end_test-pre,2);
test2=Data(start_test+pre:end_test,2);
index_test=Data(start_test+pre:end_test,2);
%% ---------------------------------train----------------------------------
[Data_label, center, width]=kmeans(train1,clusterN);
N=size(train1,1);
K=size(center,1);
phi=zeros(N,K); 
weight=randn(1,clusterN); % weight
b=1; % bias
temp=zeros(1,past);
q=-1:2/(size(train1,1)-1):1;
train2=0.3*sin(q*pi)'+train2; %add sin(x) noise to train data
%
for i=1:N
    temp(1:end-1)=temp(2:end);
    temp(end)=train1(i); 
    for j=1:K
        phi(i,j)=exp(-0.5*((norm(temp-center(j,:))))^2/width(j,:).^2);
    end
    train_out(i)=weight*phi(i,:)'+b;
    e(i)=train2(i)-train_out(i);
    weight=weight+eta*e(i)*phi(i,:);
end
%% ---------------------------------test-----------------------------------
N=size(test1,1);
temp=zeros(1,past);
for i=1:N
    temp(1:end-1)=temp(2:end);
    temp(end)=test1(i); 
    for j=1:K
        phi(i,j)=exp(-0.5*((norm(temp-center(j,:))))^2/width(j,:).^2);
    end
    test_out(i)=weight*phi(i,:)'+b;
    err(i)=abs(test_out(i)-test2(i));
end
total_err=sum(err,2);
mean_err=mean(err,2);
%% ---------------------------------result---------------------------------

figure
plot(index_train,train2,'k');
title(sprintf('Train data(Mackey-Glass time series with noise)'));
%
figure
plot(test2,'k');
hold on;
plot(test_out,'b-');
title(sprintf('Test results(Prediction values=%d | Clusters = %d)',pre,clusterN));
legend('Actual Value',sprintf('GRBF Predicted (Average Error = %d)',mean_err),'Location','Best');
grid minor

