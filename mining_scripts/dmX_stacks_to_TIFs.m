topDir = '//flexo.ads.warwick.ac.uk/shared39/EOL2100/2100/Users/Jeffrey-Ede/series72';
if topDir(end) ~= '/' 
    topDir = strcat(topDir, '/');
end

outDir = 'E:/dump/series72';
if outDir(end) ~= '/' 
    outDir = strcat(outDir, '/');
end

files = dir( strcat(topDir, '/*.dm*') );

start = 0;
L = numel(files);
for i = 1:L
    
    %Updata the user
    fprintf(strcat("Stack ", num2str(i), " of ", num2str(L), '...\n'));
    
    name = strcat(topDir, files(i).name);
    [tags, img] = dmread(name);
    
    %Save the stack as a series of TIFs
    stack_name = strcat('stack', num2str(start+i), '/');
    mkdir(strcat(outDir, stack_name))
    dims = size(img);
    for j = 1:dims(3)
        
        %Save data to TIF
        img_name = strcat('img', num2str(j));

        t = Tiff(strcat(outDir, stack_name, img_name, '.tif'), 'w'); 
        tagstruct.ImageLength = size(img(:, :, j), 1); 
        tagstruct.ImageWidth = size(img(:, :, j), 2); 
        tagstruct.Compression = Tiff.Compression.None; 
        tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP; 
        tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
        tagstruct.BitsPerSample = 32; 
        tagstruct.SamplesPerPixel = 1;
        tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; 
        t.setTag(tagstruct); 
        t.write(img(:, :, j)); 
        t.close();
        
    end
end

%%For testing:
function img = scale0to1(input)

min_val = min(min(input));
max_val = max(max(input));

img = (input-min_val) / (max_val-min_val);

end