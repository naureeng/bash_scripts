% Step 1: 3D Tiff with Shading Correction

% script that takes merfish TIFF files
% - splits channels
% - creates maximum intensity projections

%% variables to be set at each run
experimentName  = 'me190106Merfish_oldLib_32x'; % provide experiment name, no spaces!

% set directories
dataDir     = 'nfs//winstor//isogai//smfish_data//disk8//Mat//'; % specify directory of data, no spaces!
thisExpDir  = fullfile(dataDir,experimentName);

%% DO NOT TOUCH
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% MAIN ANALYSIS CODE %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% this should stay the same across experiments

ResultsDir  = 'nfs//winstor//isogai//stella//processed';  % specify results path, no spaces!
if ~exist(fullfile(ResultsDir,experimentName),'dir')
    mkdir(fullfile(ResultsDir,experimentName));
end
thisResultDir = fullfile(ResultsDir,experimentName);

% set parameters
nHybRounds      = 8;            % number of hybridisation rounds
startSlice      = 12;
finishSlice     = 201;
numberSlices    = (finishSlice-startSlice+1)/5;

%% loop through experiment
% get folders of rounds within the experiment
roundFolders    = dir(thisExpDir);                     % this gets a list of all files within the directory
isFolder_rounds = [roundFolders(3:end).isdir];      % in case there are other files in directory, and 1&2 are hidden folders that came up here that we don't care about

if sum(isFolder_rounds) < nHybRounds                % then the stacks are not in separate round folders
    
    for hh = 3:length(roundFolders)
        if ~roundFolders(hh).isdir
            thisFile        = roundFolders(hh).name;
            thisFileParts   = strsplit(thisFile,'.');
            if strcmp(thisFileParts{2},'tif')
                
                fileParts   = strsplit(thisFileParts{1},'_');
                hybNum      = fileParts{1};
                stackNum    = fileParts{3};
                stackName   = strcat(fileParts{2},'_',fileParts{3});
                
                switch hybNum
                    case 'hyb1' %'Round 1'
                        dapiName    = 'hyb1_dapi';
                        fidName1    = 'fid_1';
                        cy3Name     = 'bit_1';
                        cy5Name     = 'bit_5';
                        fidName2    = 'fid_5';
                        
                    case 'hyb2' %'Round 2'
                        dapiName    = 'hyb2_dapi';
                        fidName1    = 'fid_2';
                        cy3Name     = 'bit_2';
                        cy5Name     = 'bit_6';
                        fidName2    = 'fid_6';
                        
                    case 'hyb3' %'Round 3'
                        dapiName    = 'hyb3_dapi';
                        fidName1    = 'fid_3';
                        cy3Name     = 'bit_3';
                        cy5Name     = 'bit_7';
                        fidName2    = 'fid_7';
                        
                    case 'hyb4' %'Round 4'
                        dapiName    = 'hyb4_dapi';
                        fidName1    = 'fid_4';
                        cy3Name     = 'bit_4';
                        cy5Name     = 'bit_8';
                        fidName2    = 'fid_8';
                        
                    case 'hyb5' %'Round 5'
                        dapiName    = 'hyb5_dapi';
                        fidName1    = 'fid_9';
                        cy3Name     = 'bit_9';
                        cy5Name     = 'bit_13';
                        fidName2    = 'fid_13';
                        
                    case 'hyb6' %'Round 6'
                        dapiName    = 'hyb6_dapi';
                        fidName1    = 'fid_10';
                        cy3Name     = 'bit_10';
                        cy5Name     = 'bit_14';
                        fidName2    = 'fid_14';
                        
                    case 'hyb7' %'Round 7'
                        dapiName    = 'hyb7_dapi';
                        fidName1    = 'fid_11';
                        cy3Name     = 'bit_11';
                        cy5Name     = 'bit_15';
                        fidName2    = 'fid_15';
                        
                    case 'hyb8' %'Round 8'
                        dapiName    = 'hyb8_dapi';
                        fidName1    = 'fid_12';
                        cy3Name     = 'bit_12';
                        cy5Name     = 'bit_16';
                        fidName2    = 'fid_16';
                        
                    case 'hyb9'
                        dapiName = 'hyb9_dapi';
                        fidName1 = 'oligoDT';
                        cy3Name = 'toehold1';
                        cy5Name = 'toehold2';
                        
                    otherwise   %  then this folder contains something else and we want to skip going through it
                        continue
                end
                
                
                if ~exist(fullfile(ResultsDir,experimentName,stackName,strcat(cy3Name,'.tif')))
                    
                    thisFilePath = fullfile(thisExpDir,thisFile);
                    info         = imfinfo(thisFilePath);
                    
                    disp(['splitting channels & applying maximum intensity projection to zstack ' stackNum ' ' hybNum]);
                    
                    num_images  = numel(info);
                    
                    imageData   = [];
                    images      = {};
                    
                    % apply shading correction
                    for j = startSlice:finishSlice
                        thisIm    = imread(thisFilePath,j);
                        backgrIm  = imgaussfilt(thisIm,150);
                        switch hybNum
                            case 'hyb9'
                                if ismember(j,dapiInds) || ismember(j,oligoInds)       % for the dapi & oligoDT images, division correction works better
                                    backgrIm_scaled = double(backgrIm)/mean(double(backgrIm(:)));
                                    corrIm = double(thisIm)./backgrIm_scaled;   % need to do this division in double format, else there are formatting problems
                                    images{j} = uint16(corrIm);
                                else                                                   % for bit images, background subtraction works better
                                    images{j} = thisIm - uint16(backgrIm);
                                end
                            otherwise
                                if ismember(j,dapiInds)         % for the dapi images, division correction works better
                                    backgrIm_scaled = double(backgrIm)/mean(double(backgrIm(:)));
                                    corrIm = double(thisIm)./backgrIm_scaled;   % need to do this division in double format, else there are formatting problems
                                    images{j} = uint16(corrIm);
                                else                            % for bit images, background subtraction works better
                                    images{j} = thisIm - uint16(backgrIm);
                                end
                        end
                    end
                    
                    % dapi channel
                    imageData.dapi = cat(3,images{startSlice:5:finishSlice});
                    % fiducial channel - 488
                    imageData.fid = cat(3,images{startSlice+1:5:finishSlice});
                    % cy3 channel - 561
                    imageData.cy3 = cat(3,images{startSlice+2:5:finishSlice});
                    % cy5 channel - 647
                    imageData.cy5 = cat(3,images{startSlice+3:5:finishSlice});
                    
                            ExportTiff(fullfile(thisResultDir,stackName,strcat(dapiName,'_BS','.tif')), '' , imageData.dapi);
                            ExportTiff(fullfile(thisResultDir,stackName,strcat(fidName1,'_BS','.tif')), '' , imageData.fid);
                            ExportTiff(fullfile(thisResultDir,stackName,strcat(fidName2,'_BS','.tif')), '' , imageData.fid);
                            ExportTiff(fullfile(thisResultDir,stackName,strcat(cy3Name,'_BS','.tif')), '' , imageData.cy3);
                            
                           
                    switch hybNum
                        case 'hyb9'
                            continue
                        otherwise
                            ExportTiff(fullfile(thisResultDir,stackName,strcat(cy5Name,'_BS','.tif')), '' , imageData.cy5);
                    end
                    
                    
                    % to avoid problems, clear the data generated for the next iteration
                    clear thisFilePath info num_images imageData images
                    
                else
                    disp([hybNum ' ' stackName ' has already been processed, skipping to next stack']);
                end
                
            end
        end
    end
    
    %     tifList = dir(fullfile(thisExpDir,'*.tif'));     % this creates a list of all the tif files in the folder
    
else
    
    % loop to go through round folders
    for rr = 1:length(isFolder_rounds)
        
        if isFolder_rounds(rr)
            thisRoundPath   = fullfile(thisExpDir,roundFolders(2+rr).name);
            
            % define bit names here
            switch roundFolders(2+rr).name
                case 'Hyb1' %'Round 1'
                    dapiName    = 'hyb1_dapi';
                    fidName1    = 'fid_1';
                    cy3Name     = 'bit_1';
                    cy5Name     = 'bit_5';
                    fidName2    = 'fid_5';
                    
                case 'Hyb2' %'Round 2'
                    dapiName    = 'hyb2_dapi';
                    fidName1    = 'fid_2';
                    cy3Name     = 'bit_2';
                    cy5Name     = 'bit_6';
                    fidName2    = 'fid_6';
                    
                case 'Hyb3' %'Round 3'
                    dapiName    = 'hyb3_dapi';
                    fidName1    = 'fid_3';
                    cy3Name     = 'bit_3';
                    cy5Name     = 'bit_7';
                    fidName2    = 'fid_7';
                    
                case 'Hyb4' %'Round 4'
                    dapiName    = 'hyb4_dapi';
                    fidName1    = 'fid_4';
                    cy3Name     = 'bit_4';
                    cy5Name     = 'bit_8';
                    fidName2    = 'fid_8';
                    
                case 'Hyb5' %'Round 5'
                    dapiName    = 'hyb5_dapi';
                    fidName1    = 'fid_9';
                    cy3Name     = 'bit_9';
                    cy5Name     = 'bit_13';
                    fidName2    = 'fid_13';
                    
                case 'Hyb6' %'Round 6'
                    dapiName    = 'hyb6_dapi';
                    fidName1    = 'fid_10';
                    cy3Name     = 'bit_10';
                    cy5Name     = 'bit_14';
                    fidName2    = 'fid_14';
                    
                case 'Hyb7' %'Round 7'
                    dapiName    = 'hyb7_dapi';
                    fidName1    = 'fid_11';
                    cy3Name     = 'bit_11';
                    cy5Name     = 'bit_15';
                    fidName2    = 'fid_15';
                    
                case 'Hyb8' %'Round 8'
                    dapiName    = 'hyb8_dapi';
                    fidName1    = 'fid_12';
                    cy3Name     = 'bit_12';
                    cy5Name     = 'bit_16';
                    fidName2    = 'fid_16';
                    
                case 'Hyb9'
                    dapiName = 'hyb9_dapi';
                    fidName1 = 'oligoDT';
                    cy3Name = 'toehold1';
                    cy5Name = 'toehold2';
                    
                otherwise   %  then this folder contains something else and we want to skip going through it
                    continue
            end
            
            disp(['loading stacks from ' roundFolders(2+rr).name]);
            
            % get folders of tiles within the round
            tileFolders     = dir(thisRoundPath);
            isFolder_tiles  = [tileFolders(3:end).isdir];
            
            if isempty(find(isFolder_tiles==1))     % then there are no separate folders for the different tiles
                % go directly through the stacks
                dataFiles       = dir(thisRoundPath);
                dataFileNames   = {dataFiles(3:end).name}';
                
                % loop to go through stacks
                for zz = 1:length(dataFileNames)
                    %%
                    % load files, run maximum intensity projections & save resulting files
                    
                    % not all files are .tif files, only continue with the ones that are
                    if ~isempty(strfind(dataFileNames{zz},'.tif'))
                        
                        thisFilePath = fullfile(thisRoundPath,dataFileNames{zz});
                        info         = imfinfo(thisFilePath);
                        [~,stackName,~] = fileparts(info(1).Filename);
                        if strfind(stackName,'hyb')   % then the stack name also contains the hybridisation round in it, which we want to omit for later file naming purposes
                            stackName = stackName(5:end);
                            if stackName(1) == '_'
                                stackName = stackName(2:end);
                            end
                            
                        end
                        if ~exist(fullfile(ResultsDir,experimentName,stackName,strcat(cy3Name,'.tif')))
                            % in case
                            disp(['splitting channels & applying maximum intensity projection to ' stackName]);
                            num_images  = numel(info);
                            
                            imageData   = [];
                            images      = {};
                            
                            % apply shading correction
                            for j = startSlice:finishSlice
                                thisIm    = imread(thisFilePath,j);
                                backgrIm  = imgaussfilt(thisIm,150);
                                switch roundFolders(2+rr).name
                                    case 'Hyb9'
                                        if ismember(j,dapiInds) || ismember(j,oligoInds)       % for the dapi & oligoDT images, division correction works better
                                            backgrIm_scaled = double(backgrIm)/mean(double(backgrIm(:)));
                                            corrIm = double(thisIm)./backgrIm_scaled;   % need to do this division in double format, else there are formatting problems
                                            images{j} = uint16(corrIm);
                                        else                                                   % for bit images, background subtraction works better
                                            images{j} = thisIm - uint16(backgrIm);
                                        end
                                    otherwise
                                        if ismember(j,dapiInds)         % for the dapi images, division correction works better
                                            backgrIm_scaled = double(backgrIm)/mean(double(backgrIm(:)));
                                            corrIm = double(thisIm)./backgrIm_scaled;   % need to do this division in double format, else there are formatting problems
                                            images{j} = uint16(corrIm);
                                        else                            % for bit images, background subtraction works better
                                            images{j} = thisIm - uint16(backgrIm);
                                        end
                                end
                            end
                            
                            % dapi channel
                            imageData.dapi = cat(3,images{startSlice:5:finishSlice});
                            % fiducial channel - 488
                            imageData.fid = cat(3,images{startSlice+1:5:finishSlice});
                            % cy3 channel - 561
                            imageData.cy3 = cat(3,images{startSlice+2:5:finishSlice});
                            % cy5 channel - 647
                            imageData.cy5 = cat(3,images{startSlice+3:5:finishSlice});
                            
                            % save maximum intensity projections
                            if ~exist(fullfile(thisResultDir,stackName),'dir')
                                mkdir(fullfile(thisResultDir,stackName));
                            end
                
                            saveastiff(uint16(imageData.dapi), fullfile(thisResultDir,stackName,strcat(dapiName,'_BS','.tif')));
                            saveastiff(uint16(imageData.fid), fullfile(thisResultDir,stackName,strcat(fidName1,'_BS','.tif')));
                            saveastiff(uint16(imageData.fid), fullfile(thisResultDir,stackName,strcat(fidName2,'_BS','.tif')));
                            saveastiff(uint16(imageData.cy3), fullfile(thisResultDir,stackName,strcat(cy3Name,'_BS','.tif')));
                            saveastiff(uint16(imageData.cy5), fullfile(thisResultDir,stackName,strcat(cy5Name,'_BS','.tif')));
                     
                            switch roundFolders(2+rr).name
                                case 'Hyb9'
                                    continue
                                otherwise
                                    ExportTiff(fullfile(thisResultDir,stackName,strcat(fidName2,'_BS','.tif')), '' , imageData.fid);
                            end
                            %                             imwrite(imageData.MAXcy5,fullfile(thisResultDir,stackName,strcat(cy5Name,'.tif')));
                            
                            
                            % to avoid problems, clear the data generated for the next iteration
                            clear thisFilePath info num_images imageData images
                        else
                            disp([stackName ' has already been processed, skipping to next stack']);
                        end
                    end
                    
                end % end of stack loop
                
                
            else
                % loop to go through tile folders
                warning('Looping through tile folders - this part has not been needed so far, and may be buggy!');
                pause;
                for tt = 1:length(isFolder_tiles)
                    thisTilePath    = fullfile(thisRoundPath,tileFolders(2+tt).name);
                    
                    % get files within the data directory
                    dataFiles       = dir(thisTilePath);            % this gets a list of all files within the directory
                    dataFileNames   = {dataFiles(3:end).name};      % this extracts their names only into a cell
                    
                    % loop to go through stacks
                    for zz = 1:length(dataFileNames)
                        %%
                        % load files, run maximum intensity projections & save resulting files
                        
                        thisFilePath = fullfile(thisTilePath,dataFileNames{zz});
                        info        = imfinfo(thisFilePath);
                        num_images  = numel(info);
                        
                        imageData   = [];
                        images      = {};
                        
                        % apply shading correction
                        for j = startSlice:finishSlice
                            thisIm    = imread(thisFilePath,j);
                            backgrIm  = imgaussfilt(thisIm,150);
                            switch roundFolders(2+rr).name
                                case 'Hyb9'
                                    if ismember(j,dapiInds) || ismember(j,oligoInds)       % for the dapi & oligoDT images, division correction works better
                                        backgrIm_scaled = double(backgrIm)/mean(double(backgrIm(:)));
                                        corrIm = double(thisIm)./backgrIm_scaled;   % need to do this division in double format, else there are formatting problems
                                        images{j} = uint16(corrIm);
                                    else                                                   % for bit images, background subtraction works better
                                        images{j} = thisIm - uint16(backgrIm);
                                    end
                                otherwise
                                    if ismember(j,dapiInds)         % for the dapi images, division correction works better
                                        backgrIm_scaled = double(backgrIm)/mean(double(backgrIm(:)));
                                        corrIm = double(thisIm)./backgrIm_scaled;   % need to do this division in double format, else there are formatting problems
                                        images{j} = uint16(corrIm);
                                    else                            % for bit images, background subtraction works better
                                        images{j} = thisIm - uint16(backgrIm);
                                    end
                            end
                        end
                        
                        % dapi channel
                        imageData.dapi = cat(3,images{startSlice:5:finishSlice});
                        % fiducial channel - 488
                        imageData.fid = cat(3,images{startSlice+1:5:finishSlice});
                        % cy3 channel - 561
                        imageData.cy3 = cat(3,images{startSlice+2:5:finishSlice});
                        % cy5 channel - 647
                        imageData.cy5 = cat(3,images{startSlice+3:5:finishSlice});
                        
                            ExportTiff(fullfile(thisResultDir,stackName,strcat(dapiName,'_BS','.tif')), '' , imageData.dapi);
                            ExportTiff(fullfile(thisResultDir,stackName,strcat(fidName1,'_BS','.tif')), '' , imageData.fid);
                            ExportTiff(fullfile(thisResultDir,stackName,strcat(fidName2,'_BS','.tif')), '' , imageData.fid);
                            ExportTiff(fullfile(thisResultDir,stackName,strcat(cy3Name,'_BS','.tif')), '' , imageData.cy3);
                            ExportTiff(fullfile(thisResultDir,stackName,strcat(cy5Name,'_BS','.tif')), '' , imageData.cy5);
                                  
                        % to avoid problems, clear the data generated for the next iteration
                        clear thisFilePath info num_images imageData images
                        
                    end % end of stack loop
                    
                end % end of tile loop
            end
        end % end of "if folder" if loop
        clear dapiName fidName cy3Name cy5Name
    end % end of round loop
    
end

disp('done.');
% toc
