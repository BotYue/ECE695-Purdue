close all
clear all
clc

n0 = 50;
k = 2^10;
figure (1)
colormap gray

phantomData = phantom(n0);

phantomData = double(phantomData - min(phantomData(:))); % set min of image to zero

% pad the image with zeros so we don't lose anything when we rotate.
padDims = ceil(norm(size(phantomData)) - size(phantomData));
P       = padarray(phantomData,padDims);

subplot(2,2,1)

imagesc(P);
[c,s]=wavedec2(P,5,'haar');
n = size(c,2); % signal size
m = floor(n/5); % measurement
sigma = 0.05; % noise level

A = randn(m,n);
x0 = c';
e = randn(m,1);
b = A*x0 + sigma*e;


% parameter setting
tau = 0.005*max(abs(A'*b));
first_tau_factor = 0.8*(max(abs(A'*b))/tau);
steps = 5;
debias = 0;
stopCri=3;
tolA=10.e-5;

% fitting
[x_Basic,x_debias_Basic,obj_Basic,...
    times_Basic,debias_start_Basic,mse_Basic]= ...
         GPSR_Basic(b,A,tau,...
         'Debias',debias,...
         'Initialization',0,...
         'MaxiterA',10000,...
         'True_x',c',...
         'StopCriterion',stopCri,...
       	 'ToleranceA',tolA,...
         'Verbose',0);
     
[x_Basic_cont,x_debias_Basic_cont,obj_Basic_cont,...
    times_Basic_cont,debias_start_Basic,mse_Basic_cont]= ...
         GPSR_Basic(b,A,tau,...
         'Debias',debias,...
         'Continuation',1,...
         'ContinuationSteps',steps,...
         'FirstTauFactor',first_tau_factor,...
         'Initialization',0,...
         'True_x',c',...
         'StopCriterion',stopCri,...
       	 'ToleranceA',tolA,...
         'Verbose',0);
t_Basic_cont = times_Basic_cont(end);


X_new = waverec2(x_Basic',s,'haar');
title('Raw')
subplot(2,2,2)

imagesc(X_new)
title('GPSR')


% figure (2)
subplot(2,2,[3,4])
plot(times_Basic,mse_Basic,'LineWidth',2)
hold on
plot(times_Basic_cont,mse_Basic_cont,'k:','LineWidth',2)
legend('GPSR-Basic','GPSR-Basic with continuation')
set(gca,'FontName','Times','FontSize',16)
title('Error over GPU Time(sec)')
xlim([0 20])

disp ('Numbers of non-zero elements')
disp (nnz(x_Basic))
disp (nnz(x_Basic_cont))