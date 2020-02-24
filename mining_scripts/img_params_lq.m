function [ret_img2048] = img_params_lq(img)
%Statistics of image

%% Stats for image as is

%Dimensions and number of pixels
stats.smallestDim = min(size(img));

%% Resize the image to 2048

%Crop the larger dimension of the image so that both dimensions are the
%same
crop = imcrop(img, [1, 1, stats.smallestDim-1, stats.smallestDim-1]);

%Resize the cropped image to 2048x2048. It says 2048 in my code: ignore
%that; it's 2048
img2048 = imresize(crop, [2048, 2048], 'method', 'box');

%Extrema
stats.min2048 = min(min(img2048));
stats.max2048 = max(max(img2048));

%Other stats

%Make rescaled copy before noise calculations
img_0to1 = (img2048-stats.min2048) / (stats.max2048-stats.min2048);
ret_img2048 = single(img_0to1);

end