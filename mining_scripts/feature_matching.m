dir = '\\flexo.ads.warwick.ac.uk\Shared39\EOL2100\2100\Users\Jeffrey-Ede\series72\stack1\';

fixed = imread(strcat(dir,'img13.tif'));
fixed = fixed(:,1:3800);
moving = imread(strcat(dir,'img14.tif'));
moving = moving(:,1:3800);

L = numel(stacks);
for i=1:L
    transforms = []
    
    %load series
    
    L_series = numel(series);
    
    for j=2:L_series
        [optimizer, metric] = imregconfig('multimodal');
        optimizer.MaximumIterations = 200;

        tform_mini = imregtform(imresize(series(j), 0.1, 'Method', 'box'), ...
            imresize(series(j-1), 0.1, 'Method', 'box'), ...
            'affine', optimizer, metric);
        
        [optimizer, metric] = imregconfig('multimodal');
        optimizer.MaximumIterations = 2000;

        tform = imregtform(series(j), series(j-1), 'affine', optimizer, metric, ...
            'InitialTransformation', tform_mini);

        %Save results

        %movingRegistered = imwarp(moving,tform,'OutputView',imref2d(size(fixed)));
        % figure
        % imshowpair(fixed, movingRegistered,'Scaling','joint')
        
        transforms = [transforms, tform];
    end

    save(strcat([save_loc, num2str(i), '.mat']), 'transforms');
end