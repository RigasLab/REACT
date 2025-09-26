function [ X, Y ] = EvalCoP( pData )
%EvalCoP Fuction to avaluate the normalised CoP on the Ahmed body from a 
%vector of ESP pressure data
%   Detailed explanation goes here


% Check the dimensions of the ESP data
dims = size(pData);
if dims(1)<64
    disp('At least 64 tappings required.')
    return
else
    pData = pData(1:64,:);
end

% Setup the arrays of coordinates (in mm)
x = (-7:2:7)/7*(90.51);
y = (7:-2:-7)/7*(67.06);
[X,Y] = meshgrid(x,y);
X64 = reshape(X',[1,64]);
Y64 = reshape(Y',[1,64]);

% Evaluate the positions and normalise by width and height
X = X64*pData./sum(pData,1);
X = X/216;
Y = Y64*pData./sum(pData,1);
Y = Y/160;

end

