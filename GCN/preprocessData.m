% Preprocess Data Function
% This function preprocesses the data from arguments
% For the adjacency matrices, builds a sparse block-diagonal matrix of the
% adjacency matrices of the different graph instances so that each block in the matrix corresponds to 
% the adjacency matrix of one graph instance
% For the features, it transposes the vector so that the format is correct
% Finally, for the labels it converts the data to categorical, meaning that
% the value can only be one from a limited range of classes

function [adjacency,features,labels] = preprocessData(adjacencyData,featureData,atomData)

adjacency = sparse([]);
features = [];

for i = 1:size(adjacencyData, 3)
    numNodes = find(any(adjacencyData(:,:,i)),1,"last");

    A = adjacencyData(1:numNodes,1:numNodes,i);
    X = featureData(i,1:numNodes);

    %Transpose X
    X = X';

    %Append extracted data using the blkdiag function
    adjacency = blkdiag(adjacency,A);
    features = [features; X];
end
features(:,1) = features;

labels = [];

%Convert the unique atom labels to categorical labels
for i = 1:size(adjacencyData,3)
    T = nonzeros(atomData(i,:));
    labels = [labels; T];
end
atomicNumbers = unique(labels);
atomNames =  atomicSymbol(atomicNumbers);
labels = categorical(labels, atomicNumbers, atomNames);

end