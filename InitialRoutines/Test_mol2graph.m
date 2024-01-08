%This script generates a matlab file with a graph given a file with mol2 format. 
% genrates the file matlab _ligand that contains a graph.

clc;clear all;close all

%% Test Ligand
target='Mpro-x0689_ligand';

mol_file=strcat('./02-ligands-coordinates/',target,'.mol2');
molviewer(mol_file);

graph_file=strcat('./','_',target);
[error,graph]=mol2graph(mol_file);
if ~error
    save(graph_file,'graph');
    disp('Ligand Done')
else
    disp('Ligand Error')
end