%This script generates a matlab file with a graph given a file with mol2 format. 
% genrates the file matlab _ligand that contains a graph.

clc;clear all;close all

%% Test Ligand
target='3nm_cristal';

%Reading
xyz_file=strcat('./',target,'.xyz');
graph_file=strcat('./',target,'_graph');
[error,graph]=xyz2graph(xyz_file);
molviewer(xyz_file);
if ~error
    save(graph_file,'graph');
    disp('Reading Done')
else
    disp('Reading Error')
end

%Writing
xyz_file2=strcat(target,'_new2.xyz');
error=graph2xyz(xyz_file2,graph);
if ~error
    molviewer(xyz_file2);
    disp('Writing Done')
else
    disp('Writing Error')
end
