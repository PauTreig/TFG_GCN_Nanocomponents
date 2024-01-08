%Function that converts the atomic number into its according atomic symbol
%The function supports atoms "H", "C", "N", "O", "S", "F", "B" and "I"
%Which are the atoms that there are in our database
function [symbol,count] = atomicSymbol(atomicNum)

numSymbols = numel(atomicNum);
symbol = strings(numSymbols, 1);
count = strings(numSymbols,1);

hCount = 0;
cCount = 0;
nCount = 0;
oCount = 0;
sCount = 0;
fCount = 0;
bCount = 0;
iCount = 0;

for i = 1:numSymbols
    switch atomicNum(i)
        case 1
            symbol(i) = "H";
            hCount = hCount + 1;
            count(i) = "H" + hCount;
        case 6
            symbol(i) = "C";
            cCount = cCount + 1;
            count(i) = "C" + cCount;
        case 7
            symbol(i) = "N";
            nCount = nCount + 1;
            count(i) = "N" + nCount;
        case 8
            symbol(i) = "O";
            oCount = oCount+1;
            count(i) = "O" + oCount;
        case 9
            symbol(i) = "F";
            oCount = fCount+1;
            count(i) = "F" + fCount;
        case 16
            symbol(i) = "S";
            sCount = sCount + 1;
            count(i) = "S" + sCount;
        case 35
            symbol(i) = "B";
            oCount = bCount+1;
            count(i) = "B" + bCount;
        case 53
            symbol(i) = "I";
            oCount = iCount+1;
            count(i) = "I" + iCount;
        otherwise
            
    end
end

end
