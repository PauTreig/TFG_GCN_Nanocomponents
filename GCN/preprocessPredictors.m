function [adjacency,features] = preprocessPredictors(adjacencyData,featureData)

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
end