function [tags, image] = dmread(filename)
% function ret = dmread(filename)
% Read Digital Micrograph version 3 and 4 file and return the tag structure
% and the image.
%
% Andreas Korinek, korinek@mcmaster.ca 
% March 2014
%
% This code was written using the reverse engineered information from
% http://rsb.info.nih.gov/ij/plugins/DM3Format.gj.html
%
 
fid = fopen(filename,'r','l');

if fid == -1
    error('Could not open file.');
end

c = onCleanup(@()fclose(fid));

version = fread(fid,1,'*int32',0,'b');

%create root tags structure
tags = struct();

if version == 3
    fseek(fid,4,'cof');
elseif version == 4
    fseek(fid,8,'cof');
end
byteord = fread(fid,1,'*int32',0,'b');

fseek(fid,2,'cof');
if version == 3
    ntags = fread(fid,1,'*int32',0,'b');
elseif version == 4
    ntags = fread(fid,1,'*int64',0,'b');
end

%read in all tags
for i=1:ntags
    [tag, fid] = readTag(fid, byteord, version);
    
    %some tags have no name, so let's create one
    if isfield(tag,'tname') == false
        tag.tname = ['Unnamed' num2str(x)];
        x = x + 1;
    end
    
    %make valid field name from tag name and remove tag name from structure
    fname = genvarname(tag.tname);
    tag = rmfield(tag,'tname');
    
    %attach tag to root tag structure
    tags.(fname) = tag;
end

%try to return the image data in convenient form
if nargout == 2
    try
        if isfield(tags.ImageList.Unnamed1.ImageData.Dimensions,'Unnamed2') == 0
            %image
            image = reshape(tags.ImageList.Unnamed1.ImageData.Data.Value,tags.ImageList.Unnamed1.ImageData.Dimensions.Unnamed0.Value,tags.ImageList.Unnamed1.ImageData.Dimensions.Unnamed1.Value)';
        else
            %3D image, e.g. EELS map
            image = permute(reshape(tags.ImageList.Unnamed1.ImageData.Data.Value,tags.ImageList.Unnamed1.ImageData.Dimensions.Unnamed0.Value,tags.ImageList.Unnamed1.ImageData.Dimensions.Unnamed1.Value, tags.ImageList.Unnamed1.ImageData.Dimensions.Unnamed2.Value),[2 1 3]);
        end
    catch
        warning('No image found, returning raw tags only.');
        image = [];
    end
end

end



function [tagstruct,fid] = readTag(fid, byteord, version)
%this function read one tag, can be called recursively for subtags

tagstruct = struct();

%read tag type
itag = dec2hex(fread(fid,1,'*int8'));

%read tag name, if there is any
ltname = fread(fid,1,'*int16',0,'b');
if ltname > 0
    tagstruct.tname = char(fread(fid,ltname,'*char')');
end

if version == 4
    fread(fid,1,'int64',0,'b'); %this is the length of the tag, we don't use it at the moment, but might come in handy
end

switch itag
    case '14' %tag directory
        fseek(fid,2,'cof');
        if version == 3
            ntags = fread(fid,1,'int32',0,'b');
        elseif version == 4
            ntags = fread(fid,1,'int64',0,'b');
        end
        
        x = 0;
        for i=1:ntags
            [out,fid] = readTag(fid,byteord, version);
            
            %some tags have no name, so let's create one
            if isfield(out,'tname') == false
                out.tname = ['Unnamed' num2str(x)];
                x = x + 1;
            end
            fname = genvarname(out.tname);
            out = rmfield(out,'tname');
            
            %attach tag to tag structure
            tagstruct.(fname) = out;
        end
        
        return;
        
    case '15' %tag
        fseek(fid,4,'cof'); %skip string "%%%%"
        if version == 3
            ninfo = fread(fid,1,'int32',0,'b');
            info = fread(fid,ninfo,'int32',0,'b');
        elseif version == 4
            ninfo = fread(fid,1,'int64',0,'b');
            info = fread(fid,ninfo,'int64',0,'b');
        end
        
        switch info(1)
            case 2 %short
                precision{1} = '*int16';
                nelem = 1;
            case 3 %long
                precision{1} = '*int32';
                nelem = 1;
            case 4 %ushort or unicode string
                precision{1} = '*uint16';
                nelem = 1;
            case 5 %ulong
                precision{1} = '*uint32';
                nelem = 1;
            case 6 %float
                precision{1} = '*single';
                nelem = 1;
            case 7 %double
                precision{1} = '*double';
                nelem = 1;
            case 8 %boolean
                precision{1} = '*uint8';
                nelem = 1;
            case 9 %char
                precision{1} = '*char';
                nelem = 1;
            case 10 %octet
                precision{1} = '*int8'; %???
                nelem = 1;
            case 11 %long long
                precision{1} = '*int64';
                nelem = 1;
            case 12 %unsigned long long
                precision{1} = '*uint64';
                nelem = 1;
            case 15 %group
                nentries = info(3);
                precision = cell(nentries,1);
                nelem = zeros(nentries,1,'int64');
                for j = 1:nentries
                    precision{j} = get_precision(info(3+2*j));
                    nelem(j) = 1;
                end
            case 18 %string
                error('TODO'); %don't know what to do in this case, have not encountered it
            case 20 %array of numbers or groups
                if info(2) == 15
                    nentries = info(4);
                    precision = cell(nentries,1);
                    nelem = zeros(nentries,1,'int64');
                    for j = 1:nentries
                        precision{j} = get_precision(info(4+2*j));
                        nelem(j) = info(end);
                    end
                else
                    precision{1} = get_precision(info(2));
                    nelem = info(3);
                end
            otherwise
                error('Unknown tag element');
        end
        nn = numel(precision);
        for k=1:nn
            if byteord == 0 %little endian
                if nn == 1
                    tagstruct.Value = fread(fid,nelem(k),precision{k},0,'b');
                else
                    tagstruct.Value{k} = fread(fid,nelem(k),precision{k},0,'b');
                end
            else %little endian
                if nn == 1
                    tagstruct.Value = fread(fid,nelem(k),precision{k});
                else
                    tagstruct.Value{k} = fread(fid,nelem(k),precision{k});
                end
            end
        end
        
    case '0' %end of file
    otherwise
        error('Unknown Tag identifier.');
end
end

function precision = get_precision(number)
switch number
    case 2 %short
        precision = '*int16';
    case 3 %long
        precision = '*int32';
    case 4 %ushort or unicode string
        precision = '*uint16';
    case 5 %ulong
        precision = '*uint32';
    case 6 %float
        precision = '*single';
    case 7 %double
        precision = '*double';
    case 8 %boolean
        precision = '*uint8';
    case 9 %char
        precision = '*char';
    case 10 %octet
        precision = '*int8'; % don't know if this is correct
    case 11 %long long
        precision = '*int64';
    case 12 %unsigned long long
        precision{1} = '*uint64';
                
    otherwise
        error('Unknown precision');
end
end