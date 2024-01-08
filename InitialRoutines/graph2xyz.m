function error=graph2xyz(filename,Graph)
   fileID = fopen(filename,'w');
   if fileID >=0
     fprintf(fileID,'%d\n',Graph.numnodes);
     fprintf(fileID,'%s\n','XYZ_file_created_from_graph');
     for i=1:Graph.numnodes
         fprintf(fileID,'%s   ',Graph.Nodes(i).atom);
         fprintf(fileID,'%.6f   ',Graph.Nodes(i).position(1));
         fprintf(fileID,'%.6f   ',Graph.Nodes(i).position(2));
         fprintf(fileID,'%.6f   ',Graph.Nodes(i).position(3));
         fprintf(fileID,'%.2f   ',Graph.Nodes(i).radi);
         fprintf(fileID,'%d\n',Graph.Nodes(i).numenllacos);
     end
   fclose(fileID);
   end
   error=fileID<0;
end