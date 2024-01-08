function [error,mol2file]=xyz2mol2(file)
   fileID = fopen(file,'r');
   if fileID >=0
    [folder, basefilename, extension] = fileparts(file);
    mol2file=strcat(basefilename,'.mol2');
    pythoncall=['py xyz2mol2.py ',file,' bond_rules.txt'];
    mol2data=evalc('system(pythoncall)');
    for i=1:strlength(mol2data)
        if mol2data(i) == newline
            mol2data(i-1) = newline;
        end
    end
    mol2datafix = regexprep(mol2data, '\n\n+', '\n');
    fileID2 = fopen(mol2file,'w');
    fprintf(fileID2,'%s',mol2datafix(1:end-12));
    fclose(fileID2);
   end
   error=fileID<0;
end