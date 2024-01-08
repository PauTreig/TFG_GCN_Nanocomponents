%{
Si la distància entre 2 àtoms és menor que la suma dels seus radis
covalents, es consideren enllaçats.
Radi covalent de l'oxigen: 0.74 Â
Radi covalent del titani: 1.66 Â
mesures en armstrongs (Â). 1 Â = 100 picometres
%}
function [error,Graph]=xyz2graph(filename)
   fileID = fopen(filename,'r');
   if fileID >=0
    Graph.numnodes = fscanf(fileID,'%d',1);
    nothingtodo=fgetl(fileID);
    nothingtodo=fgetl(fileID);
    covradi_O = 0.74;
    covradi_Ti = 1.66;
    numtotalenllacos = 0;
    for i=1:Graph.numnodes
       Graph.Nodes(i).atom=fscanf(fileID,'%s', 1);
       Graph.Nodes(i).position=fscanf(fileID,'%f',3);
       Graph.Nodes(i).radi=((Graph.Nodes(i).position(1)^2)+(Graph.Nodes(i).position(2)^2)+(Graph.Nodes(i).position(3)^2))^(1/2);
   end
   mat=zeros(Graph.numnodes,Graph.numnodes,'int8');
   for i=1:Graph.numnodes
      Graph.Nodes(i).numenllacos = 0;
      Graph.Nodes(i).bondedatoms = {};
      for j=i+1:Graph.numnodes
          if i ~= j
           x = (Graph.Nodes(j).position(1) - Graph.Nodes(i).position(1))^2;
           y = (Graph.Nodes(j).position(2) - Graph.Nodes(i).position(2))^2;
           z = (Graph.Nodes(j).position(3) - Graph.Nodes(i).position(3))^2;
           d = (x+y+z)^(1/2);
           if Graph.Nodes(i).atom == "Ti" && Graph.Nodes(j).atom == "Ti"
               threshold = covradi_Ti*2;
           elseif (Graph.Nodes(i).atom == "Ti" && Graph.Nodes(j).atom == "O") || (Graph.Nodes(i).atom == "O" && Graph.Nodes(j).atom == "Ti")
               threshold = covradi_Ti + covradi_O;
           elseif Graph.Nodes(i).atom == "O" && Graph.Nodes(j).atom == "O"
               threshold = covradi_O*2;
           else
               threshold = 0;
           end
           if d <= threshold
                mat(i,j)=1;
                mat(j,i)=1;
                numtotalenllacos = numtotalenllacos + 1;
           end
          end
      end
      Graph.Nodes(i).numenllacos = nnz(mat(i, :));
      indexes = find(mat(i, :));
      for k=1:length(indexes)
          Graph.Nodes(i).bondedatoms{end+1} = Graph.Nodes(indexes(k)).atom;
      end
   end
   Graph.Edges=mat;
   disp('Total enllacos: ');
   disp(numtotalenllacos);
   fclose(fileID);
   end
   error=fileID<0;
end