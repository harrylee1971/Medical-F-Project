addpath('Data/output_folder/')
clc; clear; close all;

img = niftiread('KKI2009-01-MPRAGE.nii');
img = double(img);
img = img / max(img(:));

% 取中間 sagittal
slice = round(size(img,1)/2);

figure;
imshow(rot90(squeeze(img(slice,:,:))), []);
colormap gray;
title(['Axial Slice = ', num2str(slice)]);

%KKI2009-42-MPRAGE