function error=graph2mol(filename,Graph)
   fileID = fopen(filename,'w');
   if fileID >=0
     fprintf(fileID,'%s\n','@<TRIPOS>MOLECULE');
     fprintf(fileID,'%s\n','MOL2_file_created_from_graph');
     fprintf(fileID,'%d ',Graph.numnodes);
     fprintf(fileID,'%d\n',Graph.numedges);
     fprintf(fileID,'%s\n','SMALL');
     fprintf(fileID,'%s\n','USER_CHARGES');
     fprintf(fileID,'%s\n','@<TRIPOS>ATOM');
     for i=1:Graph.numnodes
         fprintf(fileID,'%d   ',i);
         fprintf(fileID,'%s   ',atomicSymbol(Graph.Nodes(i).natom));
         fprintf(fileID,'%.6f   ',Graph.Nodes(i).position(1));
         fprintf(fileID,'%.6f   ',Graph.Nodes(i).position(2));
         fprintf(fileID,'%.6f   ',Graph.Nodes(i).position(3));
         fprintf(fileID,'%s   ',atomicSymbol(Graph.Nodes(i).natom));
         fprintf(fileID,'%s   ',"1");
         fprintf(fileID,'%s   ',"LIG");
         fprintf(fileID,'%.6f\n',Graph.Nodes(i).charge);
     end
     fprintf(fileID,'%s\n','@<TRIPOS>BOND');
     for i=1:Graph.numedges
         fprintf(fileID,'%d   ',i);
         fprintf(fileID,'%d   ',Graph.Nodes(i).edge(1));
         fprintf(fileID,'%d   ',Graph.Nodes(i).edge(2));
         fprintf(fileID,'%s\n',"1");
     end
     fclose(fileID);
   end
   error=fileID<0;
end