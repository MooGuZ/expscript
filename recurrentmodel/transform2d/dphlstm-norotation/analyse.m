% MODEL : Recurrent Model on NPLab3D dataset without rotation
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/fa8fbe3bbea346bcac944338a8a1873cec8258d2
%% model parameters
nbases   = 1024;
nhidunit = 1024;
nframeEncoder = 15;
nframePredict = 15;
%% enviroment variables
istart  = 1e4;
taskid  = ['DPHLSTM', num2str(nhidunit), 'TRANSFORM2D-NOROT'];
taskdir = exproot();
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
plotdir = fullfile(taskdir, 'fig');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% load dataset and parameter setup
dataset   = Transform2D('nobjects', 1, 'nframes', nframeEncoder + nframePredict, 'rotation', 0);
framesize = dataset.framesize;
%% load COModel bases
comodel = load(fullfile(datadir, 'comodel_transform2d.mat'));
%% create stunit module
load(fullfile(datadir, 'statrans_transform2d.mat'));
stunit = BuildingBlock.loaddump(stdump);
stunit.compressOutput(size(comodel.iweight, 1));
stunit.frozen = true;
stat = stunit.getKernel(framesize);
%% load model
load(fullfile(savedir, sprintf(namept, istart)));
encoder     = BuildingBlock.loaddump(encoderdump);
predict     = BuildingBlock.loaddump(predictdump);
reTransform = BuildingBlock.loaddump(retransformdump);
imTransform = BuildingBlock.loaddump(imtransformdump);
% connection LSTMs
encoder.stateAheadof(predict);
% create assistant units
crdTransform = Cart2Polar().appendto( ...
    reTransform, imTransform).aheadof(encoder.DI{1}, encoder.DI{2});
% build model
model = Model(reTransform, imTransform, crdTransform, encoder, predict);
%% create prevnet
inputSlicer  = FrameSlicer(nframeEncoder, 'front', 0).appendto( ...
    dataset.data).aheadof(stunit);
outputSlicer = FrameSlicer(nframePredict, 'front', nframeEncoder).appendto(dataset.data);
% connect stunit
stunit.aheadof(reTransform).aheadof(imTransform);
% compose prevnet
prevnet = Model(stunit, inputSlicer, outputSlicer);
%% create postnet
ampact = SimpleActivation('ReLU').appendto(predict.DO{1});
angact = SimpleActivation('tanh').appendto(predict.DO{2});
angscaler = Scaler(pi).appendto(angact);
cotransform = PolarCLT(comodel.rweight, comodel.iweight, zeros(stat.sizeout, 1)).appendto( ...
    ampact, angscaler);
dewhiten = LinearTransform(stat.decode, stat.offset(:)).appendto(cotransform);
postnet = Model(ampact, angact, angscaler, cotransform, dewhiten);
%% create zero generators
zerogen = DataGenerator('zero', nhidunit, 'tmode', nframeEncoder);
zerogen.data.connect(predict.DI{1});
zerogen.data.connect(predict.DI{2});
errgen = DataGenerator('zero', nhidunit, 'tmode', nframeEncoder, '-errmode');
errgen.data.connect(encoder.DO{1});
errgen.data.connect(encoder.DO{2});
%% get test sample
dataset.next(32);
zerogen.next(32);
prevnet.forward();
model.forward();
postnet.forward();
%% show results
animorg  = inputSlicer.O{1}.packagercd;
animref  = outputSlicer.O{1}.packagercd.reshape(framesize);
animpred = dewhiten.O{1}.packagercd.reshape(framesize);
animview({animorg, animref, animpred});
%% save animations
save(fullfile(plotdir, [taskid, '-ITER', num2str(istart), '-Sample.mat']), ...
    'animorg', 'animref', 'animpred', '-v7.3');

