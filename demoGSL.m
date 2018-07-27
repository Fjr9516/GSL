function demoGSL()
clear;close all;clc;
addpath('liblinear-2.1/matlab');

% Set algorithm parameters
kernel_type = 'primal';     % kernel type, options: primal,linear  ...
options.ReducedDim = 10;    % equal to c
options.T = 5;              % iterations of inloop

gamma=1;            % the parameter for rbf kernel
options.alpha=0.1;         
options.beta =17;        
options.lamda=10;   % the parameter for NGSL        

% Choose S and T, 4DA C-W tasks for example
src = 'Caltech10';
tgt = 'webcam';
fprintf('%s vs %s ', src, tgt);

load(['data\' src '_SURF_L10.mat']); 
Xs = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xs = zscore(Xs);
Xs = normr(Xs)';
Xs_label = labels;
clear fts;
clear labels;

load(['data\' tgt '_SURF_L10.mat']); 
Xt = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xt = zscore(Xt);
Xt = normr(Xt)';
Xt_label = labels;
clear fts;
clear labels;
% ------------------------------------------
%             Transfer Learning
% ------------------------------------------
Pt = eye(size(Xt, 1));            
P=[];
Ptp = ones(size(Xs, 1), 10);
if strcmp(kernel_type,'primal')
    for outloop=1:10
        if outloop > 1, Ptp = Pt;end
        %labeling target data
        P = labeling_target(Xs, Xt, Xs_label, Xt_label, Pt, P);
        Xt_fake_label=P.Yt;
        XtC=P.Xt;
        X_label=[Xs_label;Xt_fake_label];
        X = [Xs,XtC];
        X = X*diag(sparse(1./sqrt(sum(X.^2))));
        %do GSL
        Pt = GSL(Xs,XtC,X,X_label,options);
        deltaPt=norm(Pt - Ptp, 'fro') / norm(Pt, 'fro')
    end
else
    for outloop=1:10
        if outloop > 1, Ptp = Pt;end
        %labeling target data
        P = labeling_target( Xs, Xt, Xs_label, Xt_label, Pt, P);
        Xt_fake_label=P.Yt;
        XtC=P.Xt;            
        X_label=[Xs_label;Xt_fake_label];
        X = [Xs,XtC];
        X = X*diag(sparse(1./sqrt(sum(X.^2))));
        % Calculated initial Ps
        [Ps0,~] = PCA1(Xs', options);            
        Ps1=X\Ps0;
        %Structured kernel matrix
        K = My_kernel(kernel_type,X,[],gamma);
        Ks= My_kernel(kernel_type,X,Xs,gamma);
        Kt= My_kernel(kernel_type,X,XtC,gamma);
        %do GSL_kernel
        Pt = NGSL(K,Ks,Kt,Ps1,X,X_label,options);
        deltaPt=norm(Pt - Ptp, 'fro') / norm(Pt, 'fro')
    end
end
end


function P = labeling_target(Xs, Xt, Xs_label, Xt_label, Pt, P)
C = [0.001 0.01 0.1 1.0 10 100 1000 10000];  
for chsvm = 1 :length(C)
    tmd = ['-s 3 -c ' num2str(C(chsvm)) ' -B 1 -q'];
    model(chsvm) = train(Xs_label, sparse(double(Xs'*Pt)),tmd);
    [~,acc, ~] = predict(Xt_label, sparse(double(Xt'*Pt)), model(chsvm), '-q');
    acc1(chsvm)=acc(1);
end	
[acc,bestsvm_id]=max(acc1);
fprintf(' svm acc=%2.2f %%\n',acc);
model=model(bestsvm_id);
c=C(bestsvm_id);
score = model.w * [Pt'*Xt; ones(1, size(Xt, 2))];

th = mean(score, 2)';
[confidence, C] = max(score, [], 1);
idxpos = confidence > th(C);

P.id = idxpos;
P.Xt = Xt(:, idxpos);
P.Yt = C(idxpos)';
 
P.Xt = Xt;
P.Yt = C';
end

function K = My_kernel(ker,X,X2,gamma)

    switch ker
        case 'linear'

            if isempty(X2)
                K = X'*X;
            else
                K = X'*X2;
            end

        case 'rbf'

            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K = exp(-gamma*D); 
        otherwise
            error(['Unsupported kernel ' ker])
    end
end