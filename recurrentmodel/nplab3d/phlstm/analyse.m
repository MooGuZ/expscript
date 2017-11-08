% MODEL : Separated Recurrent Model on NPLab3D dataset
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/ed6f503953393fe8a17cb3ac6ecd6d79f59944c6
%% model parameters
nbases   = 1024;
nhidunit = 1024;
nframeEncoder = 15;
nframePredict = 15;
%% enviroment variables
istart  = 70e3;
taskid  = ['SRM', num2str(nhidunit), 'NPLAB3D'];
taskdir = pwd();
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
plotdir = fullfile(taskdir, 'fig');
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
%% load model
load(fullfile(savedir, sprintf(namept, istart)));
ampEncoder  = Interface.loaddump(ampencoderdump);
ampPredict  = Interface.loaddump(amppredictdump);
angEncoder  = Interface.loaddump(angencoderdump);
angPredict  = Interface.loaddump(angpredictdump);
reTransform = Interface.loaddump(retransformdump);
imTransform = Interface.loaddump(imtransformdump);
% connection LSTMs
ampEncoder.stateAheadof(ampPredict);
angEncoder.stateAheadof(angPredict);
% create assistant units
crdTransform = Cart2Polar().appendto( ...
    reTransform, imTransform).aheadof(ampEncoder.DI{1}, angEncoder.DI{1});
% build model
model = Model(reTransform, imTransform, crdTransform, ...
    ampEncoder, ampPredict, angEncoder, angPredict);
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
recompModel = LinearTransform(stat.decode, stat.offset(:)).appendto(cotransform);
recompRefer = LinearTransform(stat.decode, stat.offset(:)).appendto(outputSlicer);
postnet = Model(ampact, angact, angscaler, cotransform, recompModel, recompRefer);
%% create zero generators
zerogen = DataGenerator('zero', nhidunit, 'tmode', nframeEncoder);
zerogen.data.connect(ampPredict.DI{1});
zerogen.data.connect(angPredict.DI{1});
errgen = DataGenerator('zero', nhidunit, 'tmode', nframeEncoder, '-errmode');
errgen.data.connect(ampEncoder.DO{1});
errgen.data.connect(angEncoder.DO{1});
%% get test sample
nplab3d.next(8);
zerogen.next(8);
prevnet.forward();
model.forward();
postnet.forward();
%% show results
animorg  = nplab3d.data.packagercd;
animref  = recompRefer.O{1}.packagercd.reshape(framesize);
animpred = recompModel.O{1}.packagercd.reshape(framesize);
% animview({animorg, animref, animpred});
%% save animations
save(fullfile(plotdir, [taskid, '-ITER', num2str(istart), '-Sample.mat']), ...
    'animorg', 'animref', 'animpred', '-v7.3');

