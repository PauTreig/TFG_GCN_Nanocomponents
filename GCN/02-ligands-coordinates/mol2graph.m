%This function gets a mol2 via parameter and returns the graph of the
%observation according to that mol2 file, including features, adjacency and labels

function [error,Graph]=mol2graph(filename)
   fileID = fopen(filename,'r');
   if fileID >=0
   nothingtodo=fscanf(fileID,'%s',2);
   Graph.numnodes=fscanf(fileID,'%d',1);
   Graph.numedges=fscanf(fileID,'%d',1);
   nothingtodo=fscanf(fileID,'%s',5);
   for i=1:Graph.numnodes
       stringatom=fscanf(fileID,'%s', 1);
       natom=atomicNumber(stringatom(1));
       Graph.Nodes(i).atom=natom;
       Graph.Nodes(i).position=fscanf(fileID,'%f',3);
       nothingtodo=fscanf(fileID,'%s',3);
       Graph.Nodes(i).feature=fscanf(fileID,'%f', 1);
       nothingtodo=fscanf(fileID,'%s',1);
   end
   mat=zeros(Graph.numnodes,Graph.numnodes,'int8');
   for i=1:Graph.numedges
       nothingtodo=fscanf(fileID,'%d',1);
       colTwo=fscanf(fileID,'%d',1);
       colThree=fscanf(fileID,'%d',1);
       colFour=fscanf(fileID,'%s',1);
       if(colFour=="am")
           nothingtodo=fscanf(fileID,'%s',1);
       end
       mat(colTwo,colThree)=1;
       mat(colThree,colTwo)=1;
       %Graph.Nodes(i).edge=[colTwo,colThree];
   end
   Graph.Edges=mat;
   fclose(fileID);
   end
   error=fileID<0;
end