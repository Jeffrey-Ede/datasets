PARENT_DIR = "//Desktop-sa1evjv/f/ARM_scans/";
SAVE_DIR = "//Desktop-sa1evjv/h/small_scans/";

files = dir(PARENT_DIR+"*"+".tif");

L = length(files);
for i = 1:L
    s = files(i);
    f = s.folder + "\\" + s.name;
    img = imread(f);
    shape = size(img);
    
    side = min(shape);
    img = img(1:side, 1:side);
    
    small_img = imresize(img, [96, 96], 'method', 'box');
%     imshow(small_img/max(max(small_img)))
%     pause
%     imwrite(small_img, SAVE_DIR+num2str(i)+".tif", "tiff")
    
    t = Tiff(SAVE_DIR+num2str(i)+".tif", 'w'); 
    tagstruct.ImageLength = size(small_img, 1); 
    tagstruct.ImageWidth = size(small_img, 2); 
    tagstruct.Compression = Tiff.Compression.None; 
    tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP; 
    tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample = 32; 
    tagstruct.SamplesPerPixel = 1; 
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky; 
    t.setTag(tagstruct); 
    t.write(small_img); 
    t.close();
end

