function [stats, ret_img2048] = img_params(img)
%Statistics of image

%% Stats for image as is

%Dimensions and number of pixels
stats.smallestDim = min(size(img));
stats.imageDims = size(img);
stats.num_px = stats.imageDims(1) * stats.imageDims(2);

%Extrema
stats.min = min(min(img));
stats.max = max(max(img));

%Nonzeros
stats.numberNonZero = nnz(img);
stats.proportionZero = stats.numberNonZero / stats.num_px;

%Negative values
stats.numNegative = sum(img(:) < 0);
stats.proportionNegative = stats.numNegative / stats.num_px;

%% Resize the image to 2048

%Crop the larger dimension of the image so that both dimensions are the
%same
crop = imcrop(img, [1, 1, stats.smallestDim-1, stats.smallestDim-1]);

%Resize the cropped image to 2048x2048. It says 2048 in my code: ignore
%that; it's 2048
img2048 = imresize(crop, [2048, 2048], 'method', 'box');

%Estimate noise
stats.noise2048 = estimate_noise(img2048);

%Standard moments
stats.mean2048 = mean(img2048(:));
stats.stddev2048 = std2(img2048);
stats.skewness = skewness(img2048(:));
stats.kurtosis = kurtosis(img2048(:));

%Extrema
stats.min2048 = min(min(img2048));
stats.max2048 = max(max(img2048));

%Other stats
stats.median2048 = median(img2048(:));
stats.coeffVariation2048 = 100*stats.stddev2048/stats.mean2048;

img2048_sqred = img2048.^2;
stats.rms2048 = sqrt(mean(img2048_sqred(:)));

fftOriginal = fft2(double(img2048));
shiftedFFT = fftshift(fftOriginal);
shiftedFFTMagnitude = abs(shiftedFFT);
midRow = 1025;
midCol = 1025;
maxRadius = ceil(sqrt(1025^2 + 1025^2));
radialProfile = zeros(maxRadius, 1);
radialFreqs = zeros(maxRadius, 1);
for col = 1 : 2048
	for row = 1 : 2048
		radius = sqrt((row - midRow) ^ 2 + (col - midCol) ^ 2);
		thisIndex = ceil(radius) + 1;
		radialProfile(thisIndex) = radialProfile(thisIndex) + shiftedFFTMagnitude(row, col);
        
        radialFreqs(thisIndex) = radius/maxRadius;
	end
end
radialProfile = radialProfile / sum(radialProfile);
radialProfile = radialProfile .* radialFreqs;


stats.meanFreq2048 = sum(radialProfile);
stats.stddevFreq2048 = std(radialProfile);
stats.skewnessFreq2048 = skewness(radialProfile);
stats.kurtosisFreq2048 = kurtosis(radialProfile);

%Make rescaled copy before noise calculations
img_0to1 = (img2048-stats.min2048) / (stats.max2048-stats.min2048);
ret_img2048 = single(img_0to1);

%Noise standard moments
img2048(img2048 < 0) = 0;
img2048 = sqrt(img2048);

stats.mean_of_noiseFromGauss = mean(img2048(:));
stats.stddev_of_noiseFromGauss = std2(img2048);
stats.skewness_of_noiseFromGauss = skewness(img2048(:));
stats.kurtosis_of_noiseFromGauss = kurtosis(img2048(:));
stats.ratio_of_meanNoise_to_mean = stats.mean_of_noiseFromGauss / stats.mean2048;

%% Stats for images scaled between 0 and 1

%Estimate noise
stats.noise2048_for_0to1 = estimate_noise(img_0to1);

%Standard moments
stats.mean2048_for_0to1 = mean(img_0to1(:));
stats.stddev2048_for_0to1 = std2(img_0to1);

%Other stats
stats.median2048_for_0to1 = median(img_0to1(:));
stats.coeffVariation2048_for_0to1 = 100*stats.stddev2048_for_0to1/stats.mean2048_for_0to1;

img_0to1_sqred = img_0to1.^2;
stats.rms_0to1 = sqrt(mean(img_0to1_sqred(:)));

%Noise standard moments
img_0to1(img_0to1 < 0) = 0;
img_0to1 = sqrt(img_0to1);

stats.mean_of_noiseFromGauss_for_0to1 = mean(img_0to1(:));
stats.stddev_of_noiseFromGauss_for_0to1 = std2(img_0to1);
stats.skewness_of_noiseFromGauss_for_0to1 = skewness(img_0to1(:));
stats.kurtosis_of_noiseFromGauss_for_0to1 = kurtosis(img_0to1(:));
stats.ratio_of_meanNoise_to_mean_for_0to1 = stats.mean_of_noiseFromGauss_for_0to1 / ...
    stats.mean2048_for_0to1;

end