%Harvest images from the microscopy data servers to train neural networks
%with

topDir = '//flexo.ads.warwick.ac.uk/Shared40/JEOL2000/JEOL 2100+';
if topDir(end) ~= '/'
    topDir = strcat(topDir, '/');
end

outDirTop ='F:/stills_all/stills';
if outDirTop(end) ~= '/' 
    outDirTop = strcat(outDirTop, '/');
end

statSavePeriod = 200; %Save stats every _ images
statsDir = 'F:/stills_all/original_filenames/100001-110000/';
if statsDir(end) ~= '/' 
    statsDir = strcat(statsDir, '/');
end
% 
% filesDir = '//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/stats/2100plus_dm4';
% if filesDir(end) ~= '/' 
%     filesDir = strcat(filesDir, '/');
% end

fprintf("Finding files...\n");

files = dir( strcat(topDir, '**/*.dm3') )

%Save file source locations
% save(strcat(filesDir, 'files.mat'), 'files');
% 
% fprintf("Files saved...\n");

%load('//flexo.ads.warwick.ac.uk/shared39/EOL2100/2100/Users/Jeffrey-Ede/datasets/compendium1')
%load('//flexo.ads.warwick.ac.uk/shared39/EOL2100/2100/Users/Jeffrey-Ede/datasets/files')
%%Harvest files
reaping = 100001;
compendium = [];
L = numel(files);
for i = 1:L
    
    disp( strcat("Image ", num2str(i), " of ", num2str(L), "...") );
    
    if files(i).bytes/1024 < 50000
    
        try
            name = strcat(files(i).folder,'/',files(i).name);
            evalc( '[tags, img] = dmread(name)' );   

            if tags.InImageMode.Value == 1 && numel(size(img)) == 2
                
                %Get image stats and a cropped then resized 2048x2048 image 
                [img2048] = img_params_lq(img);

                %Save data to TIF
                savename = strcat('reaping', num2str(reaping));

                t = Tiff(strcat(outDirTop, savename, '.tif'), 'w'); 
                tagstruct.ImageLength = size(img2048, 1); 
                tagstruct.ImageWidth = size(img2048, 2); 
                tagstruct.Compression = Tiff.Compression.None; 
                tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP; 
                tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
                tagstruct.BitsPerSample = 32; 
                tagstruct.SamplesPerPixel = 1; 
                tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; 
                t.setTag(tagstruct); 
                t.write(img2048); 
                t.close();
                
                %Save original location of image
                compendium = [compendium, [reaping, name]];
                
                reaping = reaping+1;
                
            end
        catch
            warning(num2str(i));
        end
    end
    
    %Leapfrog save the compendium every 100 images in case something goes wrong
    if mod(i, statSavePeriod) == 0
        save(strcat(statsDir, 'compendium1.mat'), 'compendium');
    end
    if mod(i, statSavePeriod)-idivide(int32(statSavePeriod), int32(2)) == 0
        save(strcat(statsDir, 'compendium2.mat'), 'compendium');
    end
end

%Save final statistics
save(strcat(statsDir, 'compendium.mat'), 'compendium');

disp('Finished!');