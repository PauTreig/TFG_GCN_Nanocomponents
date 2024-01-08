%The model function executes the formula of the model: ZL+1 = σL(D-1/2 * Â * D-1/2 * ZL * WL) + ZL
%and returns the probabilistic distribution for each node, where:
%Z1 = X (features)
%L = processing layer starting at 0
%σ = activation function
%W = weights
%Â = A + IN, A being the adjacency matrix and IN the identity matrix
%D = degree of the matrix Â
%
%For layers 1 and 2, the activation function is the ReLU function, and for
%the layer 3 the activation function is softmax

function Y = model(parameters,X,A)

ANorm = normalizeAdjacency(A);

Z1 = X;

Z2 = ANorm * Z1 * parameters.layer1.Weights(1,:);
Z2 = relu(Z2) + Z1;

Z3 = ANorm * Z2 * parameters.layer2.Weights;
Z3 = relu(Z3) + Z2;

Z4 = ANorm * Z3 * parameters.layer3.Weights;
Y = softmax(Z4,DataFormat="BC"); %probability of each class per each node (must sum 1)

end