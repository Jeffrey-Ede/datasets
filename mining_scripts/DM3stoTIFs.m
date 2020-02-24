function [] = DM3stoTIFs(inDir, outDir)
%Converts all the files in a directory to TIFs
if inDir(end) ~= '/' 
    inDir = strcat(inDir, '/');
end
if outDir(end) ~= '/' 
    outDir = strcat(outDir, '/');
end

files = dir(strcat(inDir, '*.dm3'));
L = length(files);
for i = 1:L
    %Print progress
    disp(strcat({'Converting '}, num2str(i), {' of '}, num2str(L), '...'));
    
    %Extract image data from dm3 file
    evalc( 'dm3struct = DM3Import( strcat(inDir, files(i).name) )' );
    img = dm3struct.image_data .* dm3struct.intensity.scale;
    
    %Save data to TIF
    [~,name,~] = fileparts(files(i).name);
    
    t = Tiff(strcat(outDir, name, '.tif'), 'w'); 
    tagstruct.ImageLength = size(img, 1); 
    tagstruct.ImageWidth = size(img, 2); 
    tagstruct.Compression = Tiff.Compression.None; 
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP; 
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack; 
    tagstruct.BitsPerSample = 32; 
    tagstruct.SamplesPerPixel = 1; 
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; 
    t.setTag(tagstruct); 
    t.write(single(img)); 
    t.close();
end

end