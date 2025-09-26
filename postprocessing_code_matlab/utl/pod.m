function [U, S, V] = pod(X)
%POD - Proper Orthogonal Decomposition
%   By Fernando Zigunov, 2019
%   [U, S, V] = POD(X) returns the proper orthogonal decomposition
%   of the data matrix X whose first dimension is time/snapshots. X can have any
%   number of additional spatial dimensions or variable indices.
%POD has several names (POD, SVD, PCA, ...) and is a mathematical tool that
%highlights the principal modes of a random series of variables (i.e.,
%financial measurements, fluid flow fields, structural vibrations, neuron
%firing patterns, video frames, and basically anything imaginable!)
%The POD tool is extremely useful in data analysis to identify the most
%energetic modes of a complex system by simply performing measurements when
%its operating. (But if you're already here you already know all of this.)
%This function is a nice wrapper for Matlab's built-in SVD function, which
%only operates in 2D data sets. The wrapper just reshapes the matrices
%before and after applying SVD such that the input series X can be
%n-dimensional and ensures the output modes U are also n-dimensional.
%============Inputs:=============
%X - Time series or snapshots to be analyzed. Can be a n-dimensional matrix
%(n>=2). First dimension is time/snapshot, and all other dimensions are
%conserved for output.
%============Outputs:=============
%U - Mode matrix, has the same size of X. First dimension of X was
%"time/snapshot", therefore first dimension of U will be "mode". Modes are
%automatically organized from most energetic U(1,:,:...) to least
%energetic.
%S - Mode eigenvalues. S.^2 gives the mode energies. S is a diagonal
%matrix, so returns just the vector S to save one line of post-processing.
%V - Mode time-series or snapshot coefficients - V contains the
%contribution of each mode to a specific snapshot of X. U*S*V gives back X
%(given U and X are reshaped into 2D matrices again).
dims=size(X);
%Removes mean. Note: Not removing the mean biases the modes as the
%data points centroid is shifted. If one wants to capture only the
%oscillations around the mean, the mean MUST be removed.
X=X-repmat(mean(X,1),[dims(1) ones(1,length(dims)-1)]);
%Reshapes X
X=reshape(X,dims(1),prod(dims(2:end)));
%Performs SVD
[U, S, V]=svd(X.','econ');
%Reshapes U back
U=U.';
dims_min=dims; dims_min(1)=min(dims(1),prod(dims(2:end))); %Redefines dims, if there are more snapshots than sensors. (Required because the 'econ' tag in svd will make U square in this case)
U=reshape(U,dims_min);
S=diag(S); %Zero entries in S are pointless.