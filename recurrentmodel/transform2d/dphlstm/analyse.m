% MODEL : Recurrent Model on NPLab3D dataset
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/9d1238ded5abfa7d7eca2c2f50795fc9e0eded79
%% model parameters
nbases   = 1024;
nhidunit = 1024;
nwhiten  = 512;
nframeEncoder = 15;
nframePredict = 15;
%% enviroment variables
istart  = 5e4;
taskid  = ['DPHLSTM', num2str(nhidunit), 'TRANSFORM2D'];
taskdir = exproot();
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
plotdir = fullfile(taskdir, 'fig');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% load dataset and parameter setup
dataset   = Transform2D('nobjects', 1, 'nframes', nframeEncoder + nframePredict);
framesize = dataset.framesize;
npixel    = prod(framesize);
%% create stunit module
load(fullfile(datadir, 'statrans_transform2d.mat'));
stunit = Interface.loaddump(stdump);
stunit.compressOutput(nwhiten);
stunit.frozen = true;
stat = stunit.getKernel(framesize);
%% load model
load(fullfile(savedir, sprintf(namept, istart)));
encoder     = Interface.loaddump(encoderdump);
predict     = Interface.loaddump(predictdump);
reTransform = Interface.loaddump(retransformdump);
imTransform = Interface.loaddump(imtransformdump);
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
%% load COModel bases
comodel = load(fullfile(datadir, 'comodel_transform2d.mat'));
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
dataset.next(8);
zerogen.next(8);
prevnet.forward();
model.forward();
postnet.forward();
%% show results
animorg  = inputSlicer.O{1}.packagercd;
animref  = outputSlicer.O{1}.packagercd.reshape(framesize);
animpred = dewhiten.O{1}.packagercd.reshape(framesize);
animview({animorg, animref, animpred});
%% save animations
% save(fullfile(plotdir, [taskid, '-ITER', num2str(istart), '-Sample.mat']), ...
%     'animorg', 'animref', 'animpred', '-v7.3');

