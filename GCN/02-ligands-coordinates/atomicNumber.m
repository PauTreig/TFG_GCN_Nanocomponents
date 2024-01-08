%Function that converts the atomic symbol into its according atomic number
%The function supports atoms "H", "C", "N", "O", "S", "F", "B" and "I"
%Which are the atoms that there are in our database
function [number] = atomicNumber(symbol)

switch symbol
    case "H"
        number=1;
    case "C"
        number=6;
    case "N"
        number=7;
    case "O"
        number=8;
    case "F"
        number=9;
    case "S"
        number=16;
    case "B"
        number=35;
    case "I"
        number=53;
    otherwise
        number=99;
end