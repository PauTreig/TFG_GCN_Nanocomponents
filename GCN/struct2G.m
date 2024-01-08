%Function that converts a graph in a struct form into a graph object
function G = struct2G(graphl)
    A = graphl.Edges;
    A = double(A);
    numNodes = find(any(A),1,"last");
    G = graph(A);
    positionData1 = zeros(1,numNodes);
    positionData2 = zeros(1,numNodes);
    positionData3 = zeros(1,numNodes);
    featureData = zeros(1,numNodes);
    for i = 1:numNodes
        atomData(1,i) = atomicSymbol(graphl.Nodes(i).atom);
        positionData1(1,i) = graphl.Nodes(i).position(1);
        positionData2(1,i) = graphl.Nodes(i).position(2);
        positionData3(1,i) = graphl.Nodes(i).position(3);
        featureData(1,i) = graphl.Nodes(i).feature;
    end
    G.Nodes.atoms = atomData(1,1:numNodes)';
    G.Nodes.position1 = positionData1(1,1:numNodes)';
    G.Nodes.position2 = positionData2(1,1:numNodes)';
    G.Nodes.position3 = positionData3(1,1:numNodes)';
    G.Nodes.features = featureData(1,1:numNodes)';
end