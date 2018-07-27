function  Pt = GSL(Xs,Xt,X,X_label,options)
alpha = options.alpha;
beta = options.beta;
max_iter = options.T;
Class = options.ReducedDim;

Y = Construct_Y(X_label,length(X_label),Class); 

[m,n1] = size(Xs); n2 = size(Xt,2);
n=size(X,2);
max_mu = 10^6;
mu = 0.1;
rho = 1.01;
convergence = 10^-6;
[Ps1,~] = PCA1(Xs', options);
% ----------------------------------------------
%               Initialization
% ----------------------------------------------
M = ones(Class,n);
Pt = zeros(m,Class);

Z = zeros(n1,n2);
L = zeros(n1,n2);

Y1 = zeros(n1,n2);
% ------------------------------------------------
%                   Main Loop
% ------------------------------------------------
conv(1)=100;
for iter = 1:max_iter     
    % updating Ps
    if (iter == 1)
        Ps = Ps1;
    else
        Ps = (2*Xs*Z*Z'*Xs'+2*beta*eye(m))\(2*Xs*Z*Xt'*Pt+2*beta*Pt);
    end       
    
    % updating Pt 
    A1 = Y.*M;
    Pt = (2*Xt*Xt'+X*X'+2*beta*eye(m))\(X*A1'+2*Xt*(Ps'*Xs*Z)'+2*beta*Ps);
    
    % updating M
    A3 = Pt'*X;
    A4 = Y.*A3;
    [p,q] = size(A4);
    for k1 = 1:p
        for k2 = 1:q
            M(k1,k2) = max(A4(k1,k2),0);
        end
    end    
    
    % updating Z
    A2 = L-Y1/mu;%G5
    Z = (mu*eye(n1)+2*Xs'*Ps*Ps'*Xs)\(mu*A2+2*Xs'*Ps*Pt'*Xt);   
    
    % updating  L
    ta = alpha/mu;
    temp_L = Z+Y1/mu;
    [U01,S01,V01] = svd(temp_L,'econ');
    S01 = diag(S01);
    svp = length(find(S01>ta));
    if svp >= 1
        S01 = S01(1:svp)-ta;
    else
        svp = 1;
        S01 = 0;    
    end
    L = U01(:,1:svp)*diag(S01)*V01(:,1:svp)';
    
    % updating Y1
    Y1 = Y1+mu*(Z-L);
    
    % updating mu
    mu = min(rho*mu,max_mu);
    
    % checking convergence
    leq1(iter) = norm(Z-L,Inf);
    conv(iter+1)=beta*norm(Ps-Pt,'fro')+alpha*nuclear_norm(L)+norm(Pt'*Xt-Ps'*Xs*Z,'fro')+0.5*norm(Pt'*X-A1,'fro');
    if iter > 2
         if leq1(iter)<convergence || abs((conv(iter)-conv(iter+1)))<convergence
              break
         end
    end    
end
end

function a=nuclear_norm(Y)
[U,S,V]=svd(Y);
S=diag(S);
a=sum(S);
end

function Y = Construct_Y(gnd,num_l,nClass)
Y = zeros(nClass,length(gnd));
for i = 1:num_l
    for j = 1:nClass
        if j == gnd(i)
            Y(j,i) = 1;
        else
            Y(j,i) = -1;
        end  
    end
end
end


