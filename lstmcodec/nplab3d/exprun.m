% MODEL : LSTM Codec on NPLab3D dataset
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/d145b9582f8fa3ea1ce65b0da6721e3498236535
%% check environment
ishpc = isunix && not(ismac);
%% load package to MATLAB search path
if ishpc
    addpath('/home/hxz244'); 
    pathLoader('umpoo');
end
%% model parameters
if ishpc
    nhidunit  = 1536;
    nloop     = 10;
    nepoch    = 10;
    nbatch    = 500;
    batchsize = 32;
    validsize = 128;
    taskdir   = fileparts(mfilename('fullpath'));
else
    nhidunit  = 64;
    nloop     = 3;
    nepoch    = 3;
    nbatch    = 7;
    batchsize = 8;
    validsize = 32;
    taskdir   = pwd();
end
nwhiten = 269;
nframeEncoder = 15;
nframePredict = 15;
initEstch = 1e-3;
%% environment parameters
istart  = 0;
taskid  = ['LSTMCODEC', num2str(nhidunit), 'NPLAB3D'];
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% load dataset and parameter setup
load(fullfile(datadir, 'nplab3d.mat'));
framesize = nplab3d.stat.smpsize;
npixel    = prod(framesize);
%% create whitening module
whitening = StatisticTransform(nplab3d.stat, 'mode', 'whiten').appendto(nplab3d.data);
whitening.compressOutput(nwhiten);
stat = whitening.getKernel(framesize);
%% create/load units and model
if istart == 0
    encoder = PHLSTM.randinit(nhidunit, nwhiten);
    predict = PHLSTM.randinit(nhidunit, [], nwhiten);
else
    load(fullfile(savedir, sprintf(namept, istart)));
    encoder = BuildingBlock.loaddump(encoderdump);
    predict = BuildingBlock.loaddump(predictdump);
end
encoder.stateAheadof(predict);
% create model as collection of core units
model = Model(encoder, predict);
%% create prevnet
inputSlicer  = FrameSlicer(nframeEncoder, 'front', 0).appendto(nplab3d.data).aheadof(whitening);
outputSlicer = FrameSlicer(nframePredict, 'front', nframeEncoder).appendto(nplab3d.data);
outputShaper = Reshaper().appendto(outputSlicer);
whitening.aheadof(encoder.DI{1});
% create model for pre-processing
prevnet = Model(whitening, inputSlicer, outputSlicer, outputShaper);
%% create postnet
modelPredict = LinearTransform(stat.decode, stat.offset(:)).appendto(predict.DO{1});
predictAct = SimpleActivation('sigmoid').appendto(modelPredict);
% freeze dewhitening units
modelPredict.freeze();
% create model for post-process
postnet = Model(modelPredict, predictAct);
%% create zero generators
zerogen = DataGenerator('zero', nhidunit, 'tmode', nframePredict);
zerogen.data.connect(predict.DI{1});
errgen = DataGenerator('zero', nhidunit, 'tmode', nframeEncoder, '-errmode');
errgen.data.connect(encoder.DO{1});
%% create objectives
lossfun = Likelihood('mse');
lossfun.x.connect(predictAct.O{1});
lossfun.ref.connect(outputShaper.O{1});
%% create task
task = CustomTask(taskid, taskdir, model, {nplab3d, zerogen}, lossfun, {}, ...
    'prevnet', prevnet, 'postnet', postnet, 'errgen', errgen, ...
    'iteration', istart, '-nosave');
%% setup optmizator
opt = HyperParam.getOptimizer();
opt.gradmode('basic');
opt.stepmode('adapt', 'estimatedChange', initEstch);
opt.enableRcdmode(3);
%% run task
latestsave = [];
for i = 1 : nloop
    task.run(nepoch, nbatch, batchsize, validsize);
    encoderdump = encoder.dump();
    predictdump = predict.dump();
    save(fullfile(savedir, sprintf(namept, task.iteration)), ...
        'encoderdump', 'predictdump', '-v7.3');
    if not(isempty(latestsave))
        delete(latestsave);
    end
    latestsave = fullfile(savedir, sprintf(namept, task.iteration));
end