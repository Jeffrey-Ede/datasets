topDir = '//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/stats/'

files = dir( strcat(topDir, '**/compendium.mat') );

f = fopen('E:/stills_all_profile/data.txt', 'w');

fprintf(f, 'smallestDim, num_px, min, max, numberNonZero, proportionNonZero, numNegative, proportionNegative, noise2048, mean2048, stddev2048, skewness, kurtosis, min2048, max2048, median2048, coeffVariation2048, rms2048, meanFreq2048, stddevFreq2048, skewnessFreq2048, kurtosisFreq2048, mean_of_noiseFromGauss, stddev_of_noiseFromGauss, skewness_of_noiseFromGauss, kurtosis_of_noiseFromGauss, ratio_of_meanNoise_to_mean, noise2048_for_0to1, mean2048_for_0to1, stddev2048_for_0to1, median2048_for_0to1, coeffVariation2048_for_0to1, rms_0to1, mean_of_noiseFromGauss_for_0to1, stddev_of_noiseFromGauss_for_0to1, skewness_of_noiseFromGauss_for_0to1, kurtosis_of_noiseFromGauss_for_0to1, ratio_of_meanNoise_to_mean_for_0to1\r\n');

L = numel(files);
for i = 1:L
    %disp(i)
    
    load(strcat(files(i).folder, '/', files(i).name));
    
    L2 = numel(compendium)
    
    for j = 1:L2

        smallestDim = single(compendium(j).smallestDim);
        fprintf(f, '%f, ', smallestDim);

        num_px = single(compendium(j).num_px);
        fprintf(f, '%f, ', num_px);

        min = single(compendium(j).min);
        fprintf(f, '%f, ', min);

        max = single(compendium(j).max);
        fprintf(f, '%f, ', max);

        numberNonZero = single(compendium(j).numberNonZero);
        fprintf(f, '%f, ', numberNonZero);

        proportionNonZero = single(compendium(j).proportionZero);
        fprintf(f, '%f, ', proportionNonZero);

        numNegative = single(compendium(j).numNegative);
        fprintf(f, '%f, ', numNegative);

        proportionNegative = single(compendium(j).proportionNegative);
        fprintf(f, '%f, ', proportionNegative);

        noise2048 = single(compendium(j).noise2048);
        fprintf(f, '%f, ', noise2048);

        mean2048 = single(compendium(j).mean2048);
        fprintf(f, '%f, ', mean2048);

        stddev2048 = single(compendium(j).stddev2048);
        fprintf(f, '%f, ', stddev2048);

        skewness = single(compendium(j).skewness);
        fprintf(f, '%f, ', skewness);

        kurtosis = single(compendium(j).kurtosis);
        fprintf(f, '%f, ', kurtosis);

        min2048 = single(compendium(j).min2048);
        fprintf(f, '%f, ', min2048);

        max2048 = single(compendium(j).max2048);
        fprintf(f, '%f, ', max2048);

        median2048 = single(compendium(j).median2048);
        fprintf(f, '%f, ', median2048);

        coeffVariation2048 = single(compendium(j).coeffVariation2048);
        fprintf(f, '%f, ', coeffVariation2048);

        rms2048 = single(compendium(j).rms2048);
        fprintf(f, '%f, ', rms2048);

        meanFreq2048 = single(compendium(j).meanFreq2048);
        fprintf(f, '%f, ', meanFreq2048);

        stddevFreq2048 = single(compendium(j).stddevFreq2048);
        fprintf(f, '%f, ', stddevFreq2048);

        skewnessFreq2048 = single(compendium(j).skewnessFreq2048);
        fprintf(f, '%f, ', skewnessFreq2048);

        kurtosisFreq2048 = single(compendium(j).kurtosisFreq2048);
        fprintf(f, '%f, ', kurtosisFreq2048);

        mean_of_noiseFromGauss = single(compendium(j).mean_of_noiseFromGauss);
        fprintf(f, '%f, ', mean_of_noiseFromGauss);

        stddev_of_noiseFromGauss = single(compendium(j).stddev_of_noiseFromGauss);
        fprintf(f, '%f, ', stddev_of_noiseFromGauss);

        skewness_of_noiseFromGauss = single(compendium(j).skewness_of_noiseFromGauss);
        fprintf(f, '%f, ', skewness_of_noiseFromGauss);

        kurtosis_of_noiseFromGauss = single(compendium(j).kurtosis_of_noiseFromGauss);
        fprintf(f, '%f, ', kurtosis_of_noiseFromGauss);

        ratio_of_meanNoise_to_mean = single(compendium(j).ratio_of_meanNoise_to_mean);
        fprintf(f, '%f, ', ratio_of_meanNoise_to_mean);

        noise2048_for_0to1 = single(compendium(j).noise2048_for_0to1);
        fprintf(f, '%f, ', noise2048_for_0to1);

        mean2048_for_0to1 = single(compendium(j).mean2048_for_0to1);
        fprintf(f, '%f, ', mean2048_for_0to1);

        stddev2048_for_0to1 = single(compendium(j).stddev2048_for_0to1);
        fprintf(f, '%f, ', stddev2048_for_0to1);

        median2048_for_0to1 = single(compendium(j).median2048_for_0to1);
        fprintf(f, '%f, ', median2048_for_0to1);

        coeffVariation2048_for_0to1 = single(compendium(j).coeffVariation2048_for_0to1);
        fprintf(f, '%f, ', coeffVariation2048_for_0to1);

        rms_0to1 = single(compendium(j).rms_0to1);
        fprintf(f, '%f, ', rms_0to1);

        mean_of_noiseFromGauss_for_0to1 = single(compendium(j).mean_of_noiseFromGauss_for_0to1);
        fprintf(f, '%f, ', mean_of_noiseFromGauss_for_0to1);

        stddev_of_noiseFromGauss_for_0to1 = single(compendium(j).stddev_of_noiseFromGauss_for_0to1);
        fprintf(f, '%f, ', stddev_of_noiseFromGauss_for_0to1);

        skewness_of_noiseFromGauss_for_0to1 = single(compendium(j).skewness_of_noiseFromGauss_for_0to1);
        fprintf(f, '%f, ', skewness_of_noiseFromGauss_for_0to1);

        kurtosis_of_noiseFromGauss_for_0to1 = single(compendium(j).kurtosis_of_noiseFromGauss_for_0to1);
        fprintf(f, '%f, ', kurtosis_of_noiseFromGauss_for_0to1);

        ratio_of_meanNoise_to_mean_for_0to1 = single(compendium(j).ratio_of_meanNoise_to_mean_for_0to1);
        fprintf(f, '%f\r\n', ratio_of_meanNoise_to_mean_for_0to1);
    end
end
