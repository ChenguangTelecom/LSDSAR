close all
clear all
clc

% A sentinel 1 image of Saclay area provided by ESA;
load image/real_SAR/saclay
I=saclay;

% weight parameter in Gradient by Ratio.
alpha=4;% alpha can be only 1, 2, 3, 4 or 5
%Precomputed transition probabilities of the first order Markov chain
[p11,p01,p11_2,p01_2,p11_4,p01_4]=precompute_transition(alpha);
I=double(I);%input image should be in double format
%threshold for NFA, epsilon
eps=1/1;
%Density threshold for the candidate rectangle
density=0.4;
% number of pixels in the maximum rectangle considered, rectangles having
% more pixels than this will be accepted directly
[M,N]=size(I);
sizenum=sqrt(M^2+N^2)*5;
% if sizenum is too large, use 10000, otherwise matlab could be killed
if sizenum>10^4
    sizenum=10^4;
end
%angle tolerance for region growing
angth=22.5;
inputv=[alpha,eps,density,sizenum,angth,p11,p01,p11_2,p01_2,p11_4,p01_4];
%line segment detection with LSDSAR
%for each line segment, there are seven components:
%(x1,y1,x2,y2,width,angle tolerance, NFA)
%x1,y1,x2,y2: the coordinates of the starting and ending points.
%width: width of the line segment
%angle tolerance: the angle tolerance used during region growing
%NFA: Here it is not exact the NFA, but the -log10(NFA)
lines=mexlsdsar(I,inputv);
size(lines)
lines(:,1:4)=lines(:,1:4)+1;%in C code index starts from 0.
%plot line segments 
figure,sarimshow(I,'nsig',0.1),hold on
max_len=0;
for k=1:size(lines,1)
xy=[lines(k,2),lines(k,1);lines(k,4),lines(k,3)];
plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
%plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
len=norm(lines(k,1:2)-lines(k,3:4));
if(len>max_len)
    max_len=len;
    xy_long=xy;
end
end
 

