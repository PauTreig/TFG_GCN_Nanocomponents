%This function will receive a graph via parameter containing features, labels and adjacency and
%will return a built and valid mol2 file containing the data from that graph

function error=graph2mol(filename,G)
   fileID = fopen(filename,'w');
   if fileID >=0
     fprintf(fileID,'%s\n','@<TRIPOS>MOLECULE');
     fprintf(fileID,'%s\n','MOL2_file_created_from_graph');
     fprintf(fileID,'%d ',size(G.Nodes,1));
     fprintf(fileID,'%d\n',size(G.Edges,1));
     fprintf(fileID,'%s\n','SMALL');
     fprintf(fileID,'%s\n','USER_CHARGES');
     fprintf(fileID,'%s\n','@<TRIPOS>ATOM');
     for i=1:size(G.Nodes,1)
         fprintf(fileID,'%d   ',i);
         fprintf(fileID,'%s   ',G.Nodes.atoms(i));
         fprintf(fileID,'%.4f   ',G.Nodes.position1(i));
         fprintf(fileID,'%.4f   ',G.Nodes.position2(i));
         fprintf(fileID,'%.4f   ',G.Nodes.position3(i));
         fprintf(fileID,'%s   ',G.Nodes.atoms(i));
         fprintf(fileID,'%s   ',"1");
         fprintf(fileID,'%s   ',"LIG");
         fprintf(fileID,'%.4f\n',G.Nodes.features(i));
     end
     fprintf(fileID,'%s\n','@<TRIPOS>BOND');
     for i=1:size(G.Edges,1)
         fprintf(fileID,'%d   ',i);
         fprintf(fileID,'%d   ',G.Edges.EndNodes(i,1));
         fprintf(fileID,'%d   ',G.Edges.EndNodes(i,2));
         fprintf(fileID,'%s\n',"1");
     end
     fclose(fileID);
   end
   error=fileID<0;
end