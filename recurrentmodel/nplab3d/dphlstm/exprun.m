% MODEL : Separated Recurrent Model on NPLab3D dataset
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/a99a5b5501918960829d3e9106969c132e6a3313
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
    nhidunit  = 1024;
    nloop     = 3;
    nepoch    = 3;
    nbatch    = 3;
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
taskid  = ['DPHLSTM', num2str(nhidunit), 'NPLAB3D'];
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
% if istart == 0
%     encoder     = DPHLSTM.randinit(nhidunit, [], [], [], []);
%     predict     = DPHLSTM.randinit(nhidunit, [], nbase, [], nbase);
%     reTransform = LinearTransform.randinit(stat.sizeout, nhidunit);
%     imTransform = LinearTransform.randinit(stat.sizeout, nhidunit);
% else
    load(fullfile(savedir, sprintf(namept, istart)));
    encoder     = Interface.loaddump(encoderdump);
    predict     = Interface.loaddump(predictdump);
    reTransform = Interface.loaddump(retransformdump);
    imTransform = Interface.loaddump(imtransformdump);
% end
encoder.stateAheadof(predict);
% create assistant units
crdTransform = Cart2Polar().appendto( ...
    reTransform, imTransform).aheadof(encoder.DI{1}, encoder.DI{2});
model = Model(reTransform, imTransform, crdTransform, encoder, predict);
%% create prevnet
inputSlicer  = FrameSlicer(nframeEncoder, 'front', 0).appendto( ...
    whitening).aheadof(reTransform).aheadof(imTransform);
outputSlicer = FrameSlicer(nframePredict, 'front', nframeEncoder).appendto(whitening);
prevnet = Model(whitening, inputSlicer, outputSlicer);
%% create postnet
ampact = SimpleActivation('ReLU').appendto(predict.DO{1});
angact = SimpleActivation('tanh').appendto(predict.DO{2});
angscaler = Scaler(pi).appendto(angact);
cotransform = PolarCLT(comodel.rweight, comodel.iweight, zeros(stat.sizeout, 1)).appendto( ...
    ampact, angscaler);
cotransform.freeze();
postnet = Model(ampact, angact, angscaler, cotransform);
%% create zero generators
zerogen = DataGenerator('zero', nhidunit, 'tmode', nframeEncoder);
zerogen.data.connect(predict.DI{1});
zerogen.data.connect(predict.DI{2});
errgen = DataGenerator('zero', nhidunit, 'tmode', nframeEncoder, '-errmode');
errgen.data.connect(encoder.DO{1});
errgen.data.connect(encoder.DO{2});
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
    encoderdump     = encoder.dump();
    predictdump     = predict.dump();
    retransformdump = reTransform.dump();
    imtransformdump = imTransform.dump();
    save(fullfile(savedir, sprintf(namept, task.iteration)), ...
        'encoderdump', 'predictdump', 'retransformdump', 'imtransformdump', '-v7.3');
    if not(isempty(latestsave))
        delete(latestsave);
    end
    latestsave = fullfile(savedir, sprintf(namept, task.iteration));
end