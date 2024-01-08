%Function that normalizes the adjacency matrices using its degree and the identity matrix
function ANorm = normalizeAdjacency(A)

%Sum identity matrix (add self-connections)
A = A + speye(size(A));

%Degree
degree = sum(A, 2);
degreeInvSqrt = sparse(sqrt(1./degree));

%Apply formula
ANorm = diag(degreeInvSqrt) * A * diag(degreeInvSqrt);

end