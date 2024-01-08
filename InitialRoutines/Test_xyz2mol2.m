%https://es.mathworks.com/matlabcentral/answers/3082-is-it-possible-to-run-python-code-in-matlab
% https://github.com/qzhang62/vasp_analysis/blob/master/xyz2mol2.py

clc;clear all;close all
[filename, pathname]=uigetfile('C:\Users\Pau\Desktop\TFG\Nanospheres\*.xyz','Compound');
file=strcat(pathname,filename);
molviewer(file);
[error,mol2file]=xyz2mol2(file);
if ~error
    molviewer(mol2file);
    disp('Conversion Done');
else
    disp('File Error');
end
