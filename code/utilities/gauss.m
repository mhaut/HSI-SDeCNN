function [sigma] = gauss(B, eta, beta)

sigma = zeros(1,B);
num = zeros(1,B);
num_1 = zeros(1,B);
num_2 = zeros(1,B);

for n =1:B
    %num(n) = exp((-(((n-1) - B/2)^2))/(2*(eta^2)));
    num_1(n) = (n-B/2)^2;
    num_2(n) = -num_1(n)/(2*eta^2);
    num(n) = exp(num_2(n));
end

den = sum(num);
sigma = beta * sqrt(num/den)/255.;

% plot(sigma);
end
% 
% function [sigma] = gauss(B, mean, std)
% % sigma = zeros(1,B);
% num = zeros(1,B);
% den = zeros(1,B);
% 
% for n =1:B
%   %aaaa = exp(-((n-1) - B/2)^2 / 2*(std^2))
%   %num(n) = exp((-((n-1) - B/2)^2) / (2*(std^2)));
%   %num = aaaa;
%   num(n) = exp((-(n-B/2)^2) / (2*(std^2)));
% end
% 
% for i =1:B
%   den(i) = exp((-(i-B/2)^2) / (2*(std^2)));
% end
% den = sum(den);
% sigma = beta * sqrt(num/den)/255.;
% 
% figure, plot(sigma);
% end
