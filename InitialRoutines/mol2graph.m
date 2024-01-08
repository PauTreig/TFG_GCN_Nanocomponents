function [error,Graph]=mol2graph(filename)
   fileID = fopen(filename,'r');
   if fileID >=0
   nothingtodo=fscanf(fileID,'%s',2);
   Graph.numnodes=fscanf(fileID,'%d',1);
   Graph.numedges=fscanf(fileID,'%d',1);
   nothingtodo=fscanf(fileID,'%s',5);
   for i=1:Graph.numnodes
       stringatom=fscanf(fileID,'%s', 1);
       Graph.Nodes(i).atom=stringatom(1);
       nothingtodo=fscanf(fileID,'%s',6);
       Graph.Nodes(i).charge=fscanf(fileID,'%f', 1);
       nothingtodo=fscanf(fileID,'%s',1);
   end
   mat=zeros(Graph.numnodes,Graph.numnodes,'int8');
   for i=1:Graph.numnodes
       nothingtodo=fscanf(fileID,'%d',1);
       colTwo=fscanf(fileID,'%d',1);
       colThree=fscanf(fileID,'%d',1);
       %colFour=fscanf(fileID,'%s',1);
       %if(colFour=="ar")
           %colFour=101;
       %elseif(colFour=="am")
           %colFour=102;
       %else
           %colFour=str2num(colFour);
       %end
       %fprintf("Col two: %d\n", colTwo);
       %fprintf("Col three: %d\n", colThree);
       %fprintf("Col four: %d\n", colFour);
       mat(colTwo,colThree)=1;
       mat(colThree,colTwo)=1;
   end
   Graph.Edges=mat;
   fclose(fileID);
   end
   error=fileID<0;
end