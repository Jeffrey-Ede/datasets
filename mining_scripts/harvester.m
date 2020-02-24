%Harvest images from the microscopy data servers to train neural networks
%with
topDir = '//flexo.ads.warwick.ac.uk/Shared39/EOL2100/2100/Users/';
if topDir(end) ~= '/' 
    topDir = strcat(topDir, '/');
end

files = dir( strcat(topDir, '**/*.dm3') );

fprintf("update...\n");

count = 0;
count2048 = 0;
count1024 = 0;

mean1000 = 0;
mean2000 = 0;
mean3000 = 0;
mean5000 = 0;

for i = 1:numel(files)
    
    if files(i).bytes/1024 < 50000
    
        try
            name = strcat(files(i).folder,'/',files(i).name);
            evalc( 'dm3struct = DM3Import(name)' );   

            if dm3struct.operation_mode == "IMAGING"

                count = count+1;
                fprintf(strcat(num2str(count), '\n'));

                [w, h] = size(dm3struct.image_data);
                if w >= 2048 && h >= 2048
                   count2048 = count2048+1;
                   fprintf(strcat(num2str(count2048), '\n'));
                end

                if w >= 1024 && h >= 1024
                   count1024 = count1024+1;
                   fprintf(strcat(num2str(count1024), '\n'));
                end
                
                total = sum(sum(dm3struct.image_data));
                mean2048 = total / (2048*2048);
                mean1024 = total / (1024*1024);
                
                if mean2048 >= 1000
                    mean1000 = mean1000+1;
                end
                if mean2048 >= 2000
                    mean2000 = mean2000+1;
                end
                if mean2048 >= 3000
                    mean3000 = mean3000+1;
                end
                if mean2048 >= 5000
                    mean5000 = mean5000+1;
                end
                
                disp(strcat({'mean2048: '}, num2str(mean2048), {' mean1024: '}, num2str(mean1024)));
                
                disp(num2str(mean1000));
                disp(num2str(mean2000));
                disp(num2str(mean3000));
                disp(num2str(mean5000));
                
                fprintf(strcat(num2str(count), '\n'));
                fprintf(strcat(num2str(count2048), '\n'));
                fprintf(strcat(num2str(count1024), '\n'));
            end
        catch
            warning(num2str(i));
        end
    end
end