% MODEL : LSTM ENCODER-DECODER MODEL on NPLab3D Dataset
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/2254a676903f6540ba45afc44873f6493c208131
%% load package to MATLAB search path
addpath('/home/hxz244'); pathLoader('umpoo');
%% environment parameters
istart  = 40000;
taskid  = 'NPLAB2048';
taskdir = fileparts(mfilename('fullpath'));
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% model parameters
framesize = [32, 32];
npixel    = prod(framesize);
nhidunit  = 2048;
nframes   = 20;
ncodefrms = 10;
%% create/load units and model
if istart == 0
    encoder = PHLSTM.randinit(nhidunit, npixel);
    decoder = PHLSTM.randinit(nhidunit, [], npixel);
    predict = PHLSTM.randinit(nhidunit, [], npixel);
else
    load(fullfile(savedir, sprintf(namept, istart)));
    encoder = Evolvable.loaddump(encoderdump);
    decoder = Evolvable.loaddump(decoderdump);
    predict = Evolvable.loaddump(predictdump);
end
encoder.stateAheadof(decoder).stateAheadof(predict);
model = Model(encoder, decoder, predict);
%% load dataset
load(fullfile(datadir, 'nplab3d.mat'));
nplab3d.enableSliceMode(nframes);
dataset = nplab3d;
%% create prevnet
encoderInput = FrameSlicer(ncodefrms, 'front', 0).appendto(dataset.data).aheadof(encoder.DI{1});
decoderRefer = FrameReorder('reverse').appendto(encoderInput);
decoderReferFix = Reshaper().appendto(decoderRefer);
predictRefer = FrameSlicer(nframes - ncodefrms, 'back', 0).appendto(dataset.data);
predictReferFix = Reshaper().appendto(predictRefer);
prevnet = Model(encoderInput, decoderRefer, decoderReferFix, predictRefer, predictReferFix);
%% create other data source
zerogen = DataGenerator('zero', nhidunit, 'tmode', 10);
zerogen.data.connect(decoder.DI{1});
zerogen.data.connect(predict.DI{1});
%% create postnet
decoderAct = Activation('logistic'); decoderAct.appendto(decoder.DO{1});
predictAct = Activation('logistic'); predictAct.appendto(predict.DO{1});
postnet = Model(decoderAct, predictAct);
%% create objectives
decoderObj = Likelihood('mse');
decoderObj.x.connect(decoderAct.O{1});
decoderObj.ref.connect(decoderReferFix.O{1});
predictObj = Likelihood('mse');
predictObj.x.connect(predictAct.O{1});
predictObj.ref.connect(predictReferFix.O{1});
%% create error generator
errgen = DataGenerator('zero', nhidunit, 'tmode', 10, '-errmode');
errgen.data.connect(encoder.DO{1});
%% create task
task = CustomTask(taskid, taskdir, model, {dataset, zerogen}, {decoderObj, predictObj}, {}, ...
    'prevnet', prevnet, 'postnet', postnet, 'errgen', errgen, ...
    'iteration', istart, '-nosave');
%% setup optmizator
opt = HyperParam.getOptimizer();
opt.gradmode('basic');
opt.stepmode('adapt', 'estimatedChange', 1e-2);            
opt.enableRcdmode(3);
%% run task
latestsave = [];
for i = 1 : 12
    task.run(10, 500, 64, 256);
    encoderdump = encoder.dump();
    decoderdump = decoder.dump();
    predictdump = predict.dump();
    save(fullfile(savedir, sprintf(namept, task.iteration)), ...
        'encoderdump', 'decoderdump', 'predictdump', '-v7.3');
    if not(isempty(latestsave))
        delete(latestsave);
    end
    latestsave = fullfile(savedir, sprintf(namept, task.iteration));
end
