% The Model Loss Function takes as input the parameters (weights) the features (X),
% the adjacency matrix (A), and the onehotencoded label data (L) and returns the loss
% and the gradients of the loss according to the parameters of that iteration
function [loss,gradients] = modelLoss(parameters,X,A,L)

Y = model(parameters,X,A); %Predictions
loss = crossentropy(Y,L,DataFormat="BC"); %Value of loss
gradients = dlgradient(loss, parameters); %Value of the mean of all gradients

end