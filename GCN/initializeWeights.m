%This function initializes the weights in a random way
%Takes as input the classes and the number of inputs and outputs, depending
%on which layer the model is
function weights = initializeWeights(sz,numOut,numIn,className)

arguments
    sz
    numOut
    numIn
    className = 'single'
end

Z = 2*rand(sz,className) - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end