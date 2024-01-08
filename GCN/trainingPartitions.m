%TrainingPartitions creates random indices according to the percentages
%inputted via the arguments
%For example: if numObservations is 100 (has 100 indices), number of partitions is 3 and percentages are 0.8,0.1,0.1
%then the first partition will contain 80 indices, the second one 10 and
%the last one the last 10 partitions.
function varargout = trainingPartitions(numObservations,splits)

arguments
    numObservations (1,1) {mustBePositive}
    splits {mustBeVector,mustBeInRange(splits,0,1,"exclusive"),mustSumToOne}
end

numPartitions = numel(splits);
varargout = cell(1,numPartitions);
idx = randperm(numObservations);
idxEnd = 0;

for i = 1:numPartitions-1
    idxStart = idxEnd + 1;
    idxEnd = idxStart + floor(splits(i)*numObservations) - 1;
    varargout{i} = idx(idxStart:idxEnd);
end

%Last partition
varargout{end} = idx(idxEnd+1:end);

end

function mustSumToOne(v)
if sum(v,"all") ~= 1
    error("Value must sum to one.")
end

end
