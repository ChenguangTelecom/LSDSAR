function [p11,p01,p11_2,p01_2,p11_4,p01_4]=precompute_transition(alpha)
%Precomputed transition probabilities of the first order Markov chain with
%3 angle tolerances 22.5, 22.5/2 and 22.5/4. The transition probabilities
%are estimated in gradient magnitude computed by Gradient by Ratio in pure
%speckle noise image considering all pixel pairs along horizontal vertical
%lines.
%input: 
%alpha, the weight parameter in Gradient by Ratio
%output:
%p11, p(1|1) with angle tolerance 22.5 degrees.
%p01, p(1|0), with angle tolerance 22.5 degrees.
%p11_2, p(1|1) with angle tolerance 22.5/2 degrees
%p01_2, p(1|0) with angle tolerance 22.5/2 degrees
%p11_4, p(1|1) with angle tolerance 22.5/4 degrees
%p01_4, p(1|0) with angle tolerance 22.5/4 degrees
if alpha==1
p11=0.2444;
p01=0.1076;
elseif alpha==2
p11=0.4078;
p01=0.0846;
elseif alpha==3
p11=0.5144;
p01=0.0694;
elseif alpha==4
p11=0.5874;
p01=0.0590;
elseif alpha==5
    p11=0.6356;
    p01=0.0521;
end
% angle tolerance 11.25
 if alpha==1
     p11_2=0.1314;
     p01_2=0.0577;
 elseif alpha==2
     p11_2=0.2550;
     p01_2=0.0497;
 elseif alpha==3
     p11_2=0.3610;
     p01_2=0.0427;
 elseif alpha==4
     p11_2=0.4465;
     p01_2=0.0369;
 elseif alpha==5
     p11_2=0.5094;
     p01_2=0.0327;
 end
 if alpha==1
     p11_4=0.0672;
     p01_4=0.0300;
 elseif alpha==2
     p11_4=0.1391;
     p01_4=0.0278;
 elseif alpha==3
     p11_4=0.2116;
     p01_4=0.0254;
 elseif alpha==4
     p11_4=0.2811;
     p01_4=0.0232;
 elseif alpha==5
     p11_4=0.3396;
     p01_4=0.0213;
 end