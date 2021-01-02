% close all; clear; clc;
%% LOAD IMAGE DATA
load network_1.mat
%% Paths
image_path = './Images/042-ll042/ll042t1aaaff/';
test_path = './Images/043-jh043/jh043t1aeaff/';
label_path = './Frame_Labels/PSPI/042-ll042/ll042t1aaaff/';

aux = split(genpath(label_path), ':');
exp_path = aux(2:end);

%% Read labels
labels = [];
% % for i = 1:length(exp_path)
% %     
% %     if endsWith(exp_path{i}, "ff")
        Files=dir(aux{1});
        
        for k=3:length(Files)
           FileNames=Files(k).folder + "/" + Files(k).name;
           fileID = fopen(FileNames,'r');
           labels = [labels, fscanf(fileID, '%f')];
           fclose(fileID);
        end
%     end
% end

labels = labels';
% labels_cat = categorical(labels);

labelsTest = [];
        Files=dir('./Frame_Labels/PSPI/043-jh043/jh043t1aeaff/');
        
        for k=3:length(Files)
           FileNames=Files(k).folder + "/" + Files(k).name;
           fileID = fopen(FileNames,'r');
           labelsTest = [labelsTest, fscanf(fileID, '%f')];
           fclose(fileID);
        end

%% Create Datastore
ds = imageDatastore(image_path, 'FileExtensions', '.png', 'IncludeSubfolders', true, 'Labels', labels);
dsTest = imageDatastore(test_path, 'FileExtensions', '.png', 'IncludeSubfolders', true, 'Labels', labelsTest);

% for i = 500 : length(ds.Files)
%     ds.Files = setdiff(ds.Files,ds.Files{i});
% end
% 
% labels = labels(500:end)';
% ds.Labels = labels;

ds.ReadFcn = @customReadDatastoreImage;
dsTest.ReadFcn = @customReadDatastoreImage;

imdsNew = transform(ds,@transformFcn,'IncludeInfo',true);
imdsNewTest = transform(dsTest,@transformFcn,'IncludeInfo',true);

% training_data = zeros(200, 200, 3, length(ds.Files), 'uint8');

% for i = 1:length(ds.Files)s
%     training_data(:,:,:,i) = imresize(imread(ds.Files{i}), [200 200]);
%     fprintf("Files read %d\n", i);
%     if i == 300
%         break
%     end
% end

% [imTrain, imTest]  = splitEachLabel(ds, 1, 'randomize');
%% Train
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 5, ...
    'Verbose', true, ...
    'ExecutionEnvironment', 'cpu',...
    'MiniBatchSize', 2,...
    'Plots', 'training-progress');

net = trainNetwork(imdsNew, lgraph_1, options);
%%
res = cell2mat(predict(net, imdsNewTest));
plot(res); hold on;
plot(labelsTest);
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