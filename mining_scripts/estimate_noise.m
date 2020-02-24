function Sigma=estimate_noise(I)

[H, W]=size(I);
I=double(I);

% compute sum of absolute values of Laplacian
M=[1 -2 1; -2 4 -2; 1 -2 1];
Sigma=sum(sum(abs(conv2(I, M))));

% scale sigma with proposed coefficients
Sigma=Sigma*sqrt(0.5*pi)./(6*(W-2)*(H-2));

end