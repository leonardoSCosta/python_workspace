close all; clear; clc;
%% LOAD IMAGE DATA
load network_1.mat
%% Paths
image_path = './Images/';
label_path = './Frame_Labels/PSPI/';

aux = split(genpath(label_path), ':');
exp_path = aux(2:end);

%% Read labels
labels = [];
for i = 1:length(exp_path)
    
    if endsWith(exp_path{i}, "ff")
        Files=dir(exp_path{i});
        
        for k=3:length(Files)
           FileNames=Files(k).folder + "/" + Files(k).name;
           fileID = fopen(FileNames,'r');
           labels = [labels, fscanf(fileID, '%f')];
           fclose(fileID);
        end
    end
end

labels = labels';
labels_cat = categorical(labels);


%% Create Datastore
ds = imageDatastore(image_path, 'FileExtensions', '.png', 'IncludeSubfolders', true, 'Labels', labels);
ds.ReadFcn = @customReadDatastoreImage;

imdsNew = transform(ds,@transformFcn,'IncludeInfo',true);

% training_data = zeros(200, 200, 3, length(ds.Files), 'uint8');

% for i = 1:length(ds.Files)
%     training_data(:,:,:,i) = imresize(imread(ds.Files{i}), [200 200]);
%     fprintf("Files read %d\n", i);
%     if i == 300
%         break
%     end
% end

% [imTrain, imTest]  = splitEachLabel(ds, 1, 'randomize');
%% Train
options = trainingOptions('adam', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Verbose',true, ...
    'Plots','training-progress');

net = trainNetwork(imdsNew, lgraph_1, options);


%% Functions
function [dataOut,info] = transformFcn(data,info)
    if(length(info) > 1)
        numRows = length(info);
        dataOut = cell(numRows,2);
    else
        % if readsize = 1
        numRows = 1;
        data = {data};
    end
    for idx = 1:numRows

        % Randomized 90 degree rotation
        imgOut = rot90(data{idx,1},randi(4)-1);

        % Return the label from info struct as the 
        % second column in dataOut.
        dataOut(idx,:) = {imgOut,info.Label(idx)};

    end
end

function data = customReadDatastoreImage(filename)
    % code from default function: 
    onState = warning('off', 'backtrace'); 
    c = onCleanup(@() warning(onState)); 
    data = imread(filename); % added lines: 
    data = data(:,:,min(1:3, end)); 
    data = imresize(data,[200 200]);
end