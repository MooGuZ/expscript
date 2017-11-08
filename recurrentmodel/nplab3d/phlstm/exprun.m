% MODEL : Separated Recurrent Model on NPLab3D dataset
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/ed6f503953393fe8a17cb3ac6ecd6d79f59944c6
%% check environment
ishpc = isunix && not(ismac);
%% load package to MATLAB search path
if ishpc
    addpath('/home/hxz244'); 
    pathLoader('umpoo');
end
%% model parameters
if ishpc
    nhidunit  = 1024;
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
nbases = 1024;
nframeEncoder = 15;
nframePredict = 15;
initEstch = 1e-3;
%% environment parameters
istart  = 0;
taskid  = ['PHLSTM', num2str(nhidunit), 'NPLAB3D'];
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% load dataset and parameter setup
load(fullfile(datadir, 'nplab3d.mat'));
framesize = nplab3d.stat.smpsize;
npixel    = prod(framesize);
%% load COModel bases
comodel = load(fullfile(datadir, 'comodel_nplab3d.mat'));
% create whitening module
whitening = StatisticTransform('whiten', nplab3d.stat).appendto(nplab3d.data);
stat      = whitening.getKernel(framesize);
%% create/load units and model
if istart == 0
    ampEncoder  = PHLSTM.randinit(nhidunit);
    ampPredict  = PHLSTM.randinit(nhidunit, [], nbases);
    angEncoder  = PHLSTM.randinit(nhidunit);
    angPredict  = PHLSTM.randinit(nhidunit, [], nbases);
    reTransform = LinearTransform.randinit(stat.sizeout, nhidunit);
    imTransform = LinearTransform.randinit(stat.sizeout, nhidunit);
else
    load(fullfile(savedir, sprintf(namept, istart)));
    ampEncoder  = Interface.loaddump(ampencoderdump);
    ampPredict  = Interface.loaddump(amppredictdump);
    angEncoder  = Interface.loaddump(angencoderdump);
    angPredict  = Interface.loaddump(angpredictdump);
    reTransform = Interface.loaddump(retransformdump);
    imTransform = Interface.loaddump(imtransformdump);
end
ampEncoder.stateAheadof(ampPredict);
angEncoder.stateAheadof(angPredict);
% create assistant units
crdTransform = Cart2Polar().appendto( ...
    reTransform, imTransform).aheadof(ampEncoder.DI{1}, angEncoder.DI{1});
model = Model(reTransform, imTransform, crdTransform, ampEncoder, ampPredict, angEncoder, angPredict);
%% create prevnet
inputSlicer  = FrameSlicer(nframeEncoder, 'front', 0).appendto( ...
    whitening).aheadof(reTransform).aheadof(imTransform);
outputSlicer = FrameSlicer(nframePredict, 'front', nframeEncoder).appendto(whitening);
prevnet = Model(whitening, inputSlicer, outputSlicer);
%% create postnet
ampact = SimpleActivation('ReLU').appendto(ampPredict.DO{1});
angact = SimpleActivation('tanh').appendto(angPredict.DO{1});
angscaler = Scaler(pi).appendto(angact);
cotransform = PolarCLT(comodel.rweight, comodel.iweight, zeros(stat.sizeout, 1)).appendto( ...
    ampact, angscaler);
cotransform.freeze();
postnet = Model(ampact, angact, angscaler, cotransform);
%% create zero generators
zerogen = DataGenerator('zero', nhidunit, 'tmode', nframeEncoder);
zerogen.data.connect(ampPredict.DI{1});
zerogen.data.connect(angPredict.DI{1});
errgen = DataGenerator('zero', nhidunit, 'tmode', nframeEncoder, '-errmode');
errgen.data.connect(ampEncoder.DO{1});
errgen.data.connect(angEncoder.DO{1});
%% create objectives
lossfun = Likelihood('mse', stat.pixelweight);
lossfun.x.connect(cotransform.O{1});
lossfun.ref.connect(outputSlicer.O{1});
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
    ampencoderdump  = ampEncoder.dump();
    amppredictdump  = ampPredict.dump();
    angencoderdump  = angEncoder.dump();
    angpredictdump  = angPredict.dump();
    retransformdump = reTransform.dump();
    imtransformdump = imTransform.dump();
    save(fullfile(savedir, sprintf(namept, task.iteration)), ...
        'ampencoderdump', 'amppredictdump', 'retransformdump', ...
        'angencoderdump', 'angpredictdump', 'imtransformdump', '-v7.3');
    if not(isempty(latestsave))
        delete(latestsave);
    end
    latestsave = fullfile(savedir, sprintf(namept, task.iteration));
end