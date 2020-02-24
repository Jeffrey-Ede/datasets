% DM3Import.m
% ----------------
% Robert A. McLeod
% 09 May 2012
% Author contact: robbmcleod@gmail.com
%
% Citation for this algorithm: Not required although thanks would be 
% appreciated if you find this script useful. If you would like to cite the
% Mathworks web-page:
%       http://www.mathworks.com/matlabcentral/fileexchange/29351
% then perhaps that would aid others in the community in fiding this tool.
%
%
% This script acts to import files from Gatan's .DM3 file format, utilized
% for electron microscopy, into a MATLAB structure.  The fields of the
% MATLAB structure can then be referenced with the dot-operator.
%
% IF YOU HAVE TROUBLE WITH .DM3 FILES FROM A PARTICULAR TEM PLEASE SEND ME
% AN EXAMPLE .DM3 FILE AND I WILL FIX THE CODE TO HANDLE IT. Different
% machines encode the file in different ways, and I don't know what they
% are until I have examples.
% 
% For example, one can load and display a file with the appropriate
% scaled axes in nanometers and electrons with the following example script:
% 
%   dm3struct = DM3Import( 'RandomBrightfieldImage.dm3' );
%   N = length( dm3struct.image_data );
%   imagesc( (1:N).*dmstruct.xaxis.scale, (1:N).*dmstruct.yaxis.scale, ...
%       dm3struct.image_data.*dm3struct.intensity.scale );
% 
% This script currently imports images, EELS spectra, and spectral images.
% Note that images generally appear transposed compared to how they are
% displayed in Digital Micrograph; this is because the [xy] axes in the
% MATLAB are reversed in comparison to DM.  You can correct this by 
% displaying the image transpose, i.e. imagesc( dm3struct.image_data.' );
%
% It can also import tags from files, to allow the state of the microscope
% to be saved.  This is part of a development project with Hitachi
% High-technologies (with Michael Bergen, michael.bergen@gmail.com).  
% 
% This script was constructed largely by parsing example DM3 files with a
% hex editor.  It uses regular expressions to find the tags that indicate
% the start points of the various fields within the files, and then strips
% the data. To the best of my knowledge there is not existing documentation
% on the actual object used to write DM3 format files so it is not practical
% to output to DM3 format. We are considering writing some sort of
% associated script for DM to import TIFF files with an associated text
% file, but it's not a high priority at the moment.


function [dm3struct] = DM3Import( dm3filename )
    global dm3file;
    global endianstyle;


    % Search for .dm3 file extension.  If not present, append it.
    location = regexp( dm3filename, '\.dm3', 'ONCE' );
    if( isempty(location) )
        dm3filename = [dm3filename, '.dm3' ];
    end
    disp( [ 'Opening ', dm3filename ] );
    % Remember that DM3 tags are 'big endian' but the data itself is
    % machine dependant (and 'little endian' for PC)
    [dm3file, message] = fopen( dm3filename, 'r', 'b' );
    if dm3file == -1
        % Error on file open
        dm3struct = [];
        warning( message );
        return
    end
    % First read version info, 
    %   (1) is version, == 3 for .DM3
    %   (2) is file size in bytes
    %   (3) is ordering, 0 for big-endian, 1 for little-endian (_ONLY_ data is
    %       in this format, tags are all big-endian).
    dm3fileinfo = fread( dm3file, 3, 'uint32');
    if dm3fileinfo(1) ~= 3
        error( 'File not in .DM3 format' )
    end
    if dm3fileinfo(3) == 0
        endianstyle = 'b'; % Data in big-endian format
    elseif dm3fileinfo(3) == 1
        endianstyle = 'l'; % Data in little-endian format
    end
    dm3struct = struct();
    
    % This puts the file into a character format that can be used with the
    % MATLAB regular expression function regexp().
    frewind(dm3file);
    metadm3file = fread(dm3file, 'uchar=>char' )';
    N = numel(metadm3file);
    
    % The tag DataType%%%% indicates whether the data object is either:
    %   x05x02 - Spectra
    %   x05x07 - 2D Image (also x01, x11, x12)
    %   x05x17 - Display screen (garbage)
    % It's about 14 bytes long.
    % The second tag appears 21 bytes after the 'D' in DataType%%%%
    datatypeloc = regexp( metadm3file, 'DataType%%%%' );
    datatype = zeros( size( datatypeloc) );
    % Count the number of spectra and images
    for index = 1:numel(datatypeloc)
        % Go through each DataType%%%% and see if we need to grab data or
        % not 
        datatype(index) = uint8(metadm3file(datatypeloc(index)+20));
    end

    spectracount = sum( datatype == 2 );
    % The tag 'Data%%%%' indicates the start of the data, but there's 
    % typically two images in all gain reference subtracted images.    
    % The first image is usually of type 'long' and is the saved RGB
    % image.  The second is the actual floating-point data.
    % Data in DM is stored row-wise.
    dataloc = regexp( metadm3file, 'Data%%%%' ); % Search whole file for Data%%%% tags
    
    spectraindex = 0;
    for index = 1:numel(datatypeloc)
        % Initialize dimensions
        ydim = 1; % may be overwritten, if greater than one
        zdim = 1; % may be overwritten, if great than one
         
        if( datatype(index) == 23 ) % Display (throw away)
            disp( 'Throwing away display' )
            % Do nothing
        elseif( datatype(index) == 2 ) % Spectra (maybe, DM3 is very inconsistent)

            spectraindex = spectraindex + 1;
            % The tag group 'Dimensions' contains two TagDatas with no names that
            % are the x and y-axis dimensions.  The lack of names mean we need to
            % do some fancy footwork to grab the correct locations.
            % Sometimes there are three %%%% tags after Dimensions, which
            % is the case for spectral images (i.e. 3x3x1024)
            % We can take advantage of the fact that 'Dimensions' seems to
            % be followed by 'PixelDepth' to limit the extent of the search
            dimensionloc = regexp( metadm3file(datatypeloc(index):N), 'Dimensions' );
            dimensionloc = dimensionloc(1) + datatypeloc(index) -1;
            
            pixeldepthloc = regexp( metadm3file(datatypeloc(index):N), 'PixelDepth' );
            pixeldepthloc = pixeldepthloc(1) + datatypeloc(index) -1;
            
            if( isempty( pixeldepthloc ) )
                warning( 'PixelDepth location not found, script may crash' )
            end

            xdimloc = regexp( metadm3file(dimensionloc:pixeldepthloc), '%%%%' );
            xdimloc = xdimloc(1) + dimensionloc - 1;
            xdim = readTagData( xdimloc );
            
            % Since this MAY be a spectra, there MAY NOT be a ydimloc
            if( spectracount == 1 )
               
                ydimloc = regexp( metadm3file(xdimloc+4:pixeldepthloc), '%%%%' );
                % disp( 'FIXME: trying to understand parsing' );
                if( ~ isempty( ydimloc ) ) % This doesn't work either
                    ydimloc = ydimloc(1) + (xdimloc+4) - 1;
                    % Check for spectral image third %%%%
                    zdimloc = regexp( metadm3file(ydimloc+4:pixeldepthloc), '%%%%' );
                    if( ~ isempty( zdimloc ) )
                        zdimloc = zdimloc(1) + (ydimloc+4) - 1;
                        disp( 'Parsing Spectral Image' );
                        ydim = readTagData( ydimloc );
                        zdim = readTagData( zdimloc );

                        dm3struct.image_data = reshape( readTagData( dataloc(index) ), ydim, xdim, zdim );
                    else
                        disp( 'Parsing Image' )
                        ydim = readTagData( ydimloc ); % Need to make failsafe
                        dm3struct.image_data = reshape( readTagData( dataloc(index) ), xdim, ydim );
                    end
                else
                    disp( strcat('Parsing Spectra #', int2str(spectraindex) ) )
                    dm3struct.spectra_data{spectraindex} = reshape( readTagData( dataloc(index) ), ydim, xdim );
                end
                
            else
                disp( strcat('Parsing Spectra #', int2str(spectraindex) ) )
                dm3struct.spectra_data{spectraindex} = reshape( readTagData( dataloc(index) ), ydim, xdim );
            end
        elseif( datatype(index) == 7 || datatype(index) == 10 || datatype(index) == 11 || datatype(index) == 1 ) % 2-D image
            disp( 'Parsing Image' )
            % The tag group 'Dimensions' contains two TagDatas with no names that
            % are the x and y-axis dimensions.  The lack of names mean we need to
            % do some fancy footwork to grab the correct locations.
            dimensionloc = regexp( metadm3file(datatypeloc(index):N), 'Dimensions' );
            dimensionloc = dimensionloc(1) + datatypeloc(index) -1;

            xdimloc = regexp( metadm3file(dimensionloc:N), '%%%%' );
            xdimloc = xdimloc(1) + dimensionloc - 1;
            xdim = readTagData( xdimloc );
            
            ydimloc = regexp( metadm3file(xdimloc+4:N), '%%%%' );
            ydimloc = ydimloc(1) + (xdimloc+4) - 1;
            ydim = readTagData( ydimloc );
            
            % The tag 'Data%%%%' indicates the start of the data, but there's two
            % of them!  The first is usually of type 'long' and is the saved RGB
            % image.  The second is the actual floating-point data.
            % Data in DM is stored row-wise.

            % Since data is in a simple vector format, we need to reshape it into a
            % matrix.
            dm3struct.image_data = reshape( readTagData( dataloc(index) ), xdim, ydim );
            % Transpose matrix to get x and y-axis same as Digital
            % Micrograph
            dm3struct.image_data = dm3struct.image_data.';
        elseif( datatype(index) == 3  ) % 2-D complex image
            disp( 'Parsing Complex Image' )
            % The tag group 'Dimensions' contains two TagDatas with no names that
            % are the x and y-axis dimensions.  The lack of names mean we need to
            % do some fancy footwork to grab the correct locations.
            dimensionloc = regexp( metadm3file(datatypeloc(index):N), 'Dimensions' );
            dimensionloc = dimensionloc(1) + datatypeloc(index) -1;

            xdimloc = regexp( metadm3file(dimensionloc:N), '%%%%' );
            xdimloc = xdimloc(1) + dimensionloc - 1;
            xdim = readTagData( xdimloc )
            
            ydimloc = regexp( metadm3file(xdimloc+4:N), '%%%%' );
            ydimloc = ydimloc(1) + (xdimloc+4) - 1;
            ydim = readTagData( ydimloc )
            
            % The tag 'Data%%%%' indicates the start of the data, but there's two
            % of them!  The first is usually of type 'long' and is the saved RGB
            % image.  The second is the actual floating-point data.
            % Data in DM is stored row-wise.

            % Since data is in a simple vector format, we need to reshape it into a
            % matrix.
            test = readTagData( dataloc(index) );
            size( test );
            
            dm3struct.image_data = reshape( readTagData( dataloc(index) ), xdim, ydim );
            % Transpose matrix to get x and y-axis same as Digital
            % Micrograph
            dm3struct.image_data = dm3struct.image_data.';
            
        else
            disp( [ 'Warning, unknown data type: ', num2str(datatype(index)) ] )
        end
        
    end
    

    % Now get the 'Intensity' and 'Dimension' calibration data
    % 'Intensity' calibration is contained in 'Brightness' TagGroup (not a
    % solitary TagData unfortunately.
    % Similar issue for 'Dimension' calibration.
    % Somewhat tricky due to the issue of repeated tags.
    % All of these appear after 2nd 'ImageData' taggroup, i.e. they are a
    % part of the 2nd 'ImageData' taggroup...
    
    calibrateloc = regexp( metadm3file, 'Calibrations' );
    if( spectracount == 0 ) % IMAGES
            calibrateloc_curr = calibrateloc(2);
            % Okay so we only want the locations of those tags that occur after the
            % 2nd 'Calibrations' TagGroup.

            % This can probably be done vector wise with the ? operator.
            temploc = regexp( metadm3file, 'Scale%%%%' );
            scaleloc = [];
            for index = 1:numel(temploc)
                if temploc(index) > calibrateloc_curr
                    scaleloc = [scaleloc, temploc(index)];
                end
            end
            temploc = regexp( metadm3file, 'Origin%%%%' );
            originloc = [];
            for index = 1:numel(temploc)
                if temploc(index) > calibrateloc_curr
                    originloc = [originloc, temploc(index)];
                end
            end
            temploc = regexp( metadm3file, 'Units%%%%' );
            unitloc = [];
            for index = 1:numel(temploc)
                if temploc(index) > calibrateloc_curr
                    unitloc = [unitloc, temploc(index)];
                end
            end

            dm3struct.intensity = struct( 'scale', readTagData( scaleloc(1) ), ...
                    'origin', readTagData( originloc(1) ), ...
                    'units', char( readTagData( unitloc(1) ).' ) );    

            dm3struct.xaxis = struct( 'scale', readTagData( scaleloc(2) ), ...
                            'origin', readTagData( originloc(2) ), ...
                            'units', char( readTagData( unitloc(2) ).' ) );

            % Images should have a yaxis
            if( ydim ~= 1 )
                dm3struct.yaxis = struct( 'scale', readTagData( scaleloc(3) ), ...
                                        'origin', readTagData( originloc(3) ), ...
                                        'units', char( readTagData( unitloc(3) ).' ) );
            end 
            % There is typically only a z-axis for spectral images
            if( zdim ~= 1 )
                dm3struct.zaxis = struct( 'scale', readTagData( scaleloc(4) ), ...
                                        'origin', readTagData( originloc(4) ), ...
                                        'units', char( readTagData( unitloc(4) ).' ) );
            end 
            
    else % SPECTRA
        for spectraindex = 1:spectracount
            calibrateloc_curr = calibrateloc(spectraindex+1);
            % Okay so we only want the locations of those tags that occur after the
            % 2nd 'Calibrations' TagGroup.

            % This can probably be done vector wise with the ? operator.
            temploc = regexp( metadm3file, 'Scale%%%%' );
            scaleloc = [];
            for index = 1:numel(temploc)
                if temploc(index) > calibrateloc_curr
                    scaleloc = [scaleloc, temploc(index)];
                end
            end
            temploc = regexp( metadm3file, 'Origin%%%%' );
            originloc = [];
            for index = 1:numel(temploc)
                if temploc(index) > calibrateloc_curr
                    originloc = [originloc, temploc(index)];
                end
            end
            temploc = regexp( metadm3file, 'Units%%%%' );
            unitloc = [];
            for index = 1:numel(temploc)
                if temploc(index) > calibrateloc_curr
                    unitloc = [unitloc, temploc(index)];
                end
            end

            dm3struct.intensity = struct( 'scale', readTagData( scaleloc(1) ), ...
                    'origin', readTagData( originloc(1) ), ...
                    'units', char( readTagData( unitloc(1) ).' ) );    
            % Create _seperate_ intensity and xaxis entries for each
            if spectracount > 1 % Probably a series of spectra, store them in a cell-array

                dm3struct.xaxis{spectraindex} = struct( 'scale', readTagData( scaleloc(2) ), ...
                                'origin', readTagData( originloc(2) ), ...
                                'units', char( readTagData( unitloc(2) ).' ) );
            else % Probably an image, only recording one x-axis and intensity
                dm3struct.xaxis = struct( 'scale', readTagData( scaleloc(2) ), ...
                                'origin', readTagData( originloc(2) ), ...
                                'units', char( readTagData( unitloc(2) ).' ) );
            end
            % There may not be a y-axis for spectra 
            if( ydim ~= 1 )
                dm3struct.yaxis = struct( 'scale', readTagData( scaleloc(3) ), ...
                                        'origin', readTagData( originloc(3) ), ...
                                        'units', char( readTagData( unitloc(3) ).' ) );
            end 
            % There is typically only a z-axis for spectral images
            if( zdim ~= 1 )
                dm3struct.zaxis = struct( 'scale', readTagData( scaleloc(4) ), ...
                                        'origin', readTagData( originloc(4) ), ...
                                        'units', char( readTagData( unitloc(4) ).' ) );
            end 
        end
    end
    
 
    

% scale6 is the intensity scale, scale7/8 are the x-y dimension
% scales.
    
    % Many of these microscope data tags may not exist, hence the need for isempty checks  
    magloc = regexp( metadm3file, 'Indicated Magnification%%%%' );
    if( ~ isempty( magloc ) )
        dm3struct.mag = readTagData( magloc(1) );
    end
    voltloc = regexp( metadm3file, 'Voltage%%%%' );
    if( ~ isempty( voltloc ) )
        dm3struct.voltage_kV = readTagData( voltloc(1) );
    end
    opmodeloc = regexp( metadm3file, 'Operation Mode%%%%' );
    if( ~ isempty( opmodeloc ) )
        % For whatever reason this is stored as an array and not a string.
        dm3struct.operation_mode = char( readTagData( opmodeloc(1) ).' );
    end 
    ecurrentloc = regexp( metadm3file, 'Emission Current \(µA\)%%%%');
    if( ~ isempty( ecurrentloc ) )
        dm3struct.emission_current_uA = readTagData( ecurrentloc(1) );
    end
    csloc = regexp( metadm3file, 'Cs\(mm\)%%%%' );
    if( ~ isempty( csloc ) )
        dm3struct.Cs_mm = readTagData( csloc(1) );
    end
    pcurrentloc = regexp( metadm3file, 'Probe Current \(nA\)%%%%' );
    if( ~ isempty( pcurrentloc ) )
        dm3struct.probe_current_nA = readTagData( pcurrentloc(1) );
    end
    psizeloc = regexp( metadm3file, 'Probe Size \(nm\)%%%%' );
    if( ~ isempty( psizeloc ) )
        dm3struct.probe_size_nm = readTagData( psizeloc(1) );
    end
    
    %There are a bunch of 'Name%%%%' tags.
    nameloc = regexp( metadm3file, [4, 'Name%%%%'] );
    if( ~ isempty( nameloc ) )
        % Since we're matching to (size)Name%%%% we need to advance the
        % position one.
        nameloc = nameloc(1) + 1;
        % For whatever reason this is stored as an array and not a string.
        dm3struct.image_name = char( readTagData( nameloc ).' );
    end


    specimenloc = regexp( metadm3file, 'Specimen%%%%' );
    if( ~ isempty( specimenloc ) )
        % For whatever reason this is stored as an array and not a string.
        dm3struct.specimen_info = char( readTagData( specimenloc(1) ).' );
    end
    opnameloc = regexp( metadm3file, 'Operator%%%%' );
    if( ~ isempty( opnameloc ) )
        % For whatever reason this is stored as an array and not a string.
        dm3struct.operator_name = char( readTagData( opnameloc(1) ).' );
    end
    micronameloc = regexp( metadm3file, 'Microscope%%%%' );
    if( ~ isempty( micronameloc ) )
        % For whatever reason this is stored as an array and not a string.
        dm3struct.microscope_name = char( readTagData( micronameloc(1) ).' );
    end
    
    TGHitachiloc = regexp( metadm3file, 'Hitachi' );
    if( ~ isempty( TGHitachiloc ) )
        % For whatever reason this is stored as an array and not a string.
        dm3struct.Hitachi = readTagGroupData( TGHitachiloc(1) );
    end
    
    imagetextlocs = regexp(metadm3file, 'Text%%%%'); %find all image_text instances
    if( ~ isempty( imagetextlocs ) )
        for k=1:length(imagetextlocs)
            image_text = char( readTagData( imagetextlocs(k) ).' ); 
            image_text = regexprep(image_text,char(8232),char(10)); %change dm3 newline to matlab newline
            dm3struct.image_text{k} = image_text;
        end
    end
    
    fclose(dm3file);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [tgd] = readTagGroupData( fileloc )
    % Read in a TagGroup and assign structure fieldnames using tag labels
    % This function was written by Michael Bergen

    global dm3file;
    
    fseek( dm3file, fileloc-3 , 'bof' );
    labelsize = fread( dm3file, 1, 'short' );

    %this label is not used
    if( labelsize ~= 0 )
        label = fread( dm3file, labelsize, 'uchar=>char')';
    end
    % Read in the 1000 crap and throw it away
    fread( dm3file, 5, 'uchar' );
    numSubTags = fread(dm3file, 1, 'uchar' );
    for k=1:numSubTags
        tagType = fread(dm3file, 1, 'uchar' );
        labelsize = fread( dm3file, 1, 'short' );

        %location to pass to readTagData or readTagGroupData
        currfileloc = ftell(dm3file); 

        % label is used to define structure fieldnames
        if( labelsize ~= 0 )
            label = fread( dm3file, labelsize, 'uchar=>char')';
        end
        
        if(tagType == 20) %a sub tag group
            
            tgd.(label) = readTagGroupData(currfileloc+1); %recursive
        elseif(tagType == 21) %just a tag
            
            tgd.(label) = char( readTagData( currfileloc+1 ).' );
        end
    end
       
    
end

function [td] = readTagData( fileloc )
    global dm3file;
    global endianstyle;

    % Set the file positon to the beginning of the matched regular
    % expression
    % File_location - 1 is the actual location of the start of the label
    % (since MATLAB arrays start counting at 1).
    % File_location - 3 is number of characters (bytes) in the label, so
    % we can read in and discard that information.
    fseek( dm3file, fileloc-3 , 'bof' );
    labelsize = fread( dm3file, 1, 'short' );
    % CURRENTLY NOT USING THE LABEL
    if( labelsize ~= 0 )
        label = fread( dm3file, labelsize, 'uchar=>char')';
    end
    % Read in the %%%% crap and throw it away
    fread( dm3file, 4, 'uchar' );
    
    % Ndef is number of data definitions, 
    %     for a simple type this will = 1,
    %     for a string this will = 2,
    %     an array of a simple type will = 3,
    %     structs have 1+2*f where f=number of fields in struct
    % I don't actuall seem to use Ndef...
    Ndef = fread(dm3file, 1, 'uint32' );
    tagdatatype = parseDataType();

    switch tagdatatype
    case 'DM3array'
        td = readDM3Array();
    case 'DM3string'
        % Strings are in _unicode_ (2 byte/char) format
        td = readDM3String();
    case 'DM3struct'
        % A 'simple' structure.
        td = readDM3Struct();
    case 'empty'
            % Do nothing;
        td = [];
    otherwise % simple type
        td = fread(dm3file, 1, tagdatatype, 0, endianstyle );
    end % case


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [typestring] = parseDataType()
    global dm3file;
    % Function parses the next 4 bytes and returns a string that represents the
    % associated data type.

    tagdatatype = fread(dm3file, 1, 'uint32' );

    % Parse tagdatatypes into MATLAB types
    % SHORT   = 2,
    % LONG    = 3,
    % USHORT  = 4,
    % ULONG   = 5,
    % FLOAT   = 6,
    % DOUBLE  = 7,
    % BOOLEAN = 8,
    % CHAR    = 9,
    % OCTET   = 10, i.e. a byte
    % STRUCT  = 15,
    % STRING  = 18,
    % ARRAY   = 20
    if( isempty( tagdatatype ) )
        disp( 'Warning, parseDataType: ignoring empty Tag' );
        typestring = 'empty';
    else
        switch tagdatatype
        case 2
            typestring = 'short';
        case 3
            typestring = 'long';
        case 4
            typestring = 'uint16';
        case 5
            typestring = 'uint32';
        case 6
            typestring = 'float32';
        case 7
            typestring = 'double';
        case 8
            typestring = 'ubit8';
        case 9
            typestring = 'uchar';
        case 10
            typestring = 'uint8';
        case 15
            typestring = 'DM3struct';
        case 18
            typestring = 'DM3string';
        case 20
            typestring = 'DM3array';
        otherwise
            error( strcat('Parse Error, unknown data TagType found: ', int2str(tagdatatype) ) );
        end % case
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [td] = readDM3Array()
    global dm3file;
    global endianstyle;
    
        %Hopefully the program doesn't do arrays of arrays or strings or 
    %there will be a bug to deal with.
    tagdatatype = parseDataType();
    arraylength = fread(dm3file, 1, 'uint32' );
    td = zeros( arraylength, 1 );
    
    if strcmp( tagdatatype, 'DM3struct' )
        % Ugh, the ever fun array of structs
        disp( 'I assume any array of struct is a complex image' )
        for index = 1:arraylength
            td(index) = readDM3Struct();
        end
    elseif strcmp( tagdatatype, 'DM3array' )
        %The slightly less enjoyable array or arrays
        disp( 'I do not read arrays of arrays' );
%         for index = 1:arraylength
%             td(index) = readDM3Array();
%         end
    elseif strcmp( tagdatatype, 'DM3string' )
        % And last the array of strings..
        disp('I do not read arrays of strings' );
%         for index = 1:arraylength
%             td(index) = readDM3String();
%         end
    elseif strcmp( tagdatatype, 'empty' )
        % Do nothing.
    else % a simple array
        td = fread(dm3file, arraylength, tagdatatype, 0, endianstyle);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [s] = readDM3Struct()
    global dm3file;
    global endianstyle;
    % This function only reads simple structs.  If the DM3 files contains
    % other structs, strings, or arrays inside the struct this function
    % will force an error.
    
    % Struct name length is usually == 0
    structnamelength = fread(dm3file, 1, 'uint32' );
    Nfields = fread(dm3file, 1, 'uint32' );
    
        % Note that 'fieldsnamelength' is often zero, hopefully fread handles
    % this gracefully or debugging may be necessary.
    fieldsnamelength = zeros( Nfields, 1 );
    fieldsdatatype = cell( Nfields, 1 );
    for index = 1:Nfields
        % Note that fieldsnamelength(i) is often zero, and when this is the
        % case it doesn't appear to exist at all...
        fieldsnamelength(index) = fread(dm3file, 1, 'uint32' );
        fieldsdatatype(index) = {parseDataType()};
    end
    
    % Now read in the struct data.  Can structs contain arrays, other
    % structs, or strings?  Most likely...
    if( structnamelength ~= 0 )
        s.name =  fread(dm3file,structnamelength,'uchar');
    else
        %s.name = '';
    end
    
    % Pre-allocation
    s.names = zeros( Nfields, 1 );
    s.data = zeros( Nfields, 1 );
    % Read in field names and field data
    for index = 1:Nfields
        if fieldsnamelength(index) ~= 0
            s.names(index) = fread(dm3file,fieldsnamelength,'uchar');
        else
            %s.fieldnames(index) = [];
        end
        
        if strcmp( fieldsdatatype(index), 'DM3struct' )
            % Ugh, the ever fun struct of structs
            error( 'I do not parse structs in a struct' );
        elseif strcmp( fieldsdatatype(index), 'DM3array' )
            %The slightly less enjoyable array in a struct
            error( 'I do not parse arrays in a struct' );
        elseif strcmp( fieldsdatatype(index), 'DM3string' )
            % And last the string in a struct
            error( 'I do not parse strings in a struct' );
        else % a simple struct field element
            s.data(index) = fread(dm3file,1,char(fieldsdatatype(index)),0,endianstyle);
        end
        
    end
    
end

function [s] = readDM3String( )
    global dm3file;
    % Strings are in _unicode_ (2 byte/char) format
    arraylength = fread(dm3file, 1, 'uint32' );
    rawstring = fread(dm3file, arraylength, '*char' )';
    s = char( rawstring.' );
    % If this doesn't work use the skip functionality of fread
end
