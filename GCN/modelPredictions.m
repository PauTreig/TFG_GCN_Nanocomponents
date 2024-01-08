%Function to make some predictions on a trained dataset
function predictions = modelPredictions(parameters,featureData,adjacencyData,mu,sigsq,classes)

predictions = {};
numObservations = size(featureData,1);

for i = 1:numObservations
    numNodes = find(any(adjacencyData(:,:,i)),1,"last");
    A = adjacencyData(1:numNodes,1:numNodes,i);
    X = featureData(i,1:numNodes);

    %Preprocess data
    [A,X] = preprocessPredictors(A,X);
    X = (X - mu)./sqrt(sigsq);
    X = dlarray(X);

    %Make predictions
    Y = model(parameters,X,A);
    Y = onehotdecode(Y,classes,2);
    predictions{end+1} = Y;
end

end