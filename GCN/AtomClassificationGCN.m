%----------------------------------------------------------------
% Atom classification model using a Graph Convolutional Network |
% Author: Pau Treig Solé                                        |
% Director: Francesc Serratosa                                  |
% Universitat Rovira i Virgili 2022                             |
%----------------------------------------------------------------

%Load all mol2 file names and initialize arrays of data
cd 02-ligands-coordinates\
mat = dir('*.mol2');
MAX_ATOMS = 63; %Max atoms per molecule
featureData = zeros(length(mat),MAX_ATOMS);
atomData = zeros(length(mat),MAX_ATOMS);
adjacencyData = zeros(MAX_ATOMS,MAX_ATOMS,length(mat));
positionData = zeros(length(mat),MAX_ATOMS,3);

%Read and convert mol2 data into graphs and save the data into the arrays
for i = 1:length(mat)
    [error,graphl]=mol2graph(mat(i).name);
    for j = 1:graphl.numnodes
        atomData(i,j) = graphl.Nodes(j).atom;
        %featureData(i,j) = sqrt((graphl.Nodes(j).position(1)^2)+(graphl.Nodes(j).position(2)^2)+(graphl.Nodes(j).position(3)^2));
        featureData(i,j) = graphl.Nodes(j).feature;
        positionData(i,j,1) = graphl.Nodes(j).position(1);
        positionData(i,j,2) = graphl.Nodes(j).position(2);
        positionData(i,j,3) = graphl.Nodes(j).position(3);
    end
    graphl.Edges(MAX_ATOMS,MAX_ATOMS) = 0;
    adjacencyData(:,:,i) = graphl.Edges;
end
cd ..

%Show first 4 molecules as a preview
figure
tiledlayout("flow")
for i = 1:4
    %Erase zeros from adjacency matrix
    atomicNumbers = nonzeros(atomData(i,:));
    numNodes = numel(atomicNumbers);
    A = adjacencyData(1:numNodes,1:numNodes,i);
    %Convert adjacency matrix to graph
    G = graph(A);
    %Convert atomic numbers to symbols using atomicSymbol function
    symbols = atomicSymbol(atomicNumbers);
    %Show the graphs using the plot function
    nexttile
    plot(G,NodeLabel=symbols,Layout="force")
    title("Molecule " + i)
end

%Show atom frequency among all molecules using a histogram
figure
histogram(categorical(atomicSymbol(atomData)))
xlabel("Node Label")
ylabel("Frequency")
title("Label Counts")

%Create partitions for cross-validation
numObservations = size(adjacencyData,3);
[idxTrain,idxValidation,idxTest] = trainingPartitions(numObservations,[0.8 0.1 0.1]);

adjacencyDataTrain = adjacencyData(:,:,idxTrain);
adjacencyDataValidation = adjacencyData(:,:,idxValidation);
adjacencyDataTest = adjacencyData(:,:,idxTest);

featureDataTrain = featureData(idxTrain,:);
featureDataValidation = featureData(idxValidation,:);
featureDataTest = featureData(idxTest,:);

atomDataTrain = atomData(idxTrain,:);
atomDataValidation = atomData(idxValidation,:);
atomDataTest = atomData(idxTest,:);

%Preprocess the training and validation data using the preprocessData function
[ATrain,XTrain,atomSymbolTrain] = preprocessData(adjacencyDataTrain,featureDataTrain,atomDataTrain);
[AValidation,XValidation,atomSymbolValidation] = preprocessData(adjacencyDataValidation,featureDataValidation,atomDataValidation);

%Normalize the features for training and validation using the mean and variance of the training features.
meanXTrain = mean(XTrain);
varXTrain = var(XTrain,1);

XTrain = (XTrain - meanXTrain)./sqrt(varXTrain);
XValidation = (XValidation - meanXTrain)./sqrt(varXTrain);

%Definition of the weights for each of the 3 layers
parameters = struct;

%Initialize the first layer of weights with an output size of 32 and 1 feature
numHiddenFeatureMaps = 32;
numInputFeatures = 1;

sz = [numInputFeatures numHiddenFeatureMaps];
numOut = numHiddenFeatureMaps;
numIn = numInputFeatures;
parameters.layer1.Weights = initializeWeights(sz,numOut,numIn,"double");

%Initialize the second layer of weights with the same input size as the output
%size of the previous layer. The output size will 32 too
sz = [numHiddenFeatureMaps numHiddenFeatureMaps];
numOut = numHiddenFeatureMaps;
numIn = numHiddenFeatureMaps;
parameters.layer2.Weights = initializeWeights(sz,numOut,numIn,"double");

%Initialize the third and last layer of weights with the same input size as the
%output size of the previous layer. The output size will be the number of classes
classes = categories(atomSymbolTrain);
numClasses = numel(classes);

sz = [numHiddenFeatureMaps numClasses];
numOut = numClasses;
numIn = numHiddenFeatureMaps;
parameters.layer3.Weights = initializeWeights(sz,numOut,numIn,"double");

%Definition of the prediction model parameters
%Setting the number of iterations (epochs) and the learning rate for the adam update function
numEpochs = 500;
learningRate = 0.02;

%Choose frequency of validation (each n epochs)
validationFrequency = 100;

%Initialize the plot for the training progress
figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
lineLossValidation = animatedline(LineStyle="--",Marker="o",MarkerFaceColor="black");
ylim([0 inf])
xlabel("Epoch")
ylabel("Loss")
grid on

%Parameters for the adam update function
trailingAvg = [];
trailingAvgSq = [];

%Convert the feature data to a dlarray object for the dl matlab functions
XTrain = dlarray(XTrain);
XValidation = dlarray(XValidation);

%Convert the arrays of training and validation with the symbols of each node to a
%onehotencoded vector using the onehotencode function
LTrain = onehotencode(atomSymbolTrain,2,ClassNames=classes);
LValidation = onehotencode(atomSymbolValidation,2,ClassNames=classes);

%The training uses full-batch gradient descent
%Evalute the loss using predefined dlfeval and modelLoss
%Update network using backpropagation with adamupdate
%Update the graphic
%Validate the network if necessary

start = tic;

for epoch = 1:numEpochs
    %Evaluate the model loss and gradients using the dlfeval function from MATLAB
    [loss,gradients] = dlfeval(@modelLoss,parameters,XTrain,ATrain,LTrain);

    %Update the parameters using the adamupdate function
    [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients,trailingAvg,trailingAvgSq,epoch,learningRate);

    %Update the training progress plot
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    title("Epoch: " + epoch + ", Elapsed: " + string(D))
    loss = extractdata(loss);
    addpoints(lineLossTrain,epoch,loss)
    drawnow

    %If it's a validation epoch, enter the condition and validate model
    if epoch == 1 || mod(epoch,validationFrequency) == 0
        YValidation = model(parameters,XValidation,AValidation);
        lossValidation = crossentropy(YValidation,LValidation,DataFormat="BC");
        lossValidation = extractdata(lossValidation);
        addpoints(lineLossValidation,epoch,lossValidation)
        drawnow
    end
end

%Preprocess the testing data to proceed with testing the model
[ATest,XTest,atomSymbolTest] = preprocessData(adjacencyDataTest,featureDataTest,atomDataTest);
XTest = (XTest - meanXTrain)./sqrt(varXTrain);
XTest = dlarray(XTest);

%Make the predictions with the trained parameters using the test dataset
%Convert the predictions (probabilities) to categorical labels using the onehotdecode function
YTest = model(parameters,XTest,ATest); %probabilities
YTest = onehotdecode(YTest,classes,2); %categorical labels

%Get the final accuracy of the model comparing the predictions with the real values using the mean function
accuracy = mean(YTest == atomSymbolTest)



%Print the confusion matrix to visualize the percentages of hit and miss
%for each class, displaying both true values and predictions (true positives, false positives)
figure
cm = confusionchart(atomSymbolTest,YTest,ColumnSummary="column-normalized",RowSummary="row-normalized");
title("GCN Confusion Chart");

%Make some predictions on unlabeled data to display graphic results
numExamples = 4;
adjacencyDataNew = adjacencyDataTest(:,:,1:numExamples);
featureDataNew = featureDataTest(1:numExamples,:);

predictions = modelPredictions(parameters,featureDataNew,adjacencyDataNew,meanXTrain,varXTrain,classes);

%Visualize the predictions with the molviewer function.
%For each molecule, we create the graph representation and label the nodes with the predictions
for i = 1:numExamples
    numNodes = find(any(adjacencyDataTest(:,:,i)),1,"last");
    A = adjacencyDataTest(1:numNodes,1:numNodes,i);

    G = graph(A);
    G.Nodes.atoms = string(predictions{i});
    G.Nodes.position1 = positionData(idxTest(i),1:numNodes,1)';
    G.Nodes.position2 = positionData(idxTest(i),1:numNodes,2)';
    G.Nodes.position3 = positionData(idxTest(i),1:numNodes,3)';
    G.Nodes.features = featureDataNew(i,1:numNodes)';

    newmol2file=strcat(mat(idxTest(i)).name(1:end-5),'_predicted.mol2');
    error = graph2mol(newmol2file,G);
    if ~error
        cd 02-ligands-coordinates\
        [error,graphl]=mol2graph(mat(idxTest(i)).name);
        cd ..
        filetestname = strcat(mat(idxTest(i)).name(1:end-5),'_test.mol2');
        Gtest = struct2G(graphl);
        errorr = graph2mol(filetestname,Gtest);
        molviewer(filetestname);
        molviewer(newmol2file);
    else
        disp('Writing Error')
    end
end