% MODEL : 2 Layers LSTM ENCODER-DECODER MODEL on Moving MNIST
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/55a25c44f3c5bdd19ee94c63393e243845b8480d
%% load package to MATLAB search path
addpath('/home/hxz244'); pathLoader('umpoo');
%% environment parameters
istart  = 20000;
taskid  = 'MLAYER2048';
taskdir = fileparts(mfilename('fullpath'));
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% model parameters
framesize = [64, 64];
npixel    = prod(framesize);
nhidunit  = 2048;
nframes   = 20;
%% create/load units and model
if istart == 0
    encoder1L = PHLSTM.randinit(nhidunit, npixel);
    encoder2L = PHLSTM.randinit(nhidunit);
    decoder1L = PHLSTM.randinit(nhidunit);
    decoder2L = PHLSTM.randinit(nhidunit, [], npixel);
    predict1L = PHLSTM.randinit(nhidunit);
    predict2L = PHLSTM.randinit(nhidunit, [], npixel);
else
    load(fullfile(savedir, sprintf(namept, istart)));
    encoder1L = Evolvable.loaddump(encoder1ldump);
    encoder2L = Evolvable.loaddump(encoder2ldump);
    decoder1L = Evolvable.loaddump(decoder1ldump);
    decoder2L = Evolvable.loaddump(decoder2ldump);
    predict1L = Evolvable.loaddump(predict1ldump);
    predict2L = Evolvable.loaddump(predict2ldump);
end
encoder1L.DO{1}.connect(encoder2L.DI{1});
encoder2L.stateAheadof(decoder1L).stateAheadof(predict1L);
decoder1L.DO{1}.connect(decoder2L.DI{1});
predict1L.DO{1}.connect(predict2L.DI{1});
model = Model(encoder1L, encoder2L, decoder1L, decoder2L, predict1L, predict2L);
%% load dataset
load(fullfile(datadir, 'mmnist.mat'));
mmnist.nframes = nframes;
mmnist.canvasSize = framesize;
%% create prevnet
encoderInput = FrameSlicer(10, 'front', 0).appendto(mmnist.data).aheadof(encoder1L.DI{1});
decoderRefer = FrameReorder('reverse').appendto(encoderInput);
decoderReferFix = Reshaper().appendto(decoderRefer);
predictRefer = FrameSlicer(10, 'back', 0).appendto(mmnist.data);
predictReferFix = Reshaper().appendto(predictRefer);
prevnet = Model(encoderInput, decoderRefer, decoderReferFix, predictRefer, predictReferFix);
%% create other data source
zerogen = DataGenerator('zero', nhidunit, 'tmode', 10);
zerogen.data.connect(decoder1L.DI{1});
zerogen.data.connect(predict1L.DI{1});
%% create postnet
decoderAct = Activation('logistic'); decoderAct.appendto(decoder2L.DO{1});
predictAct = Activation('logistic'); predictAct.appendto(predict2L.DO{1});
postnet = Model(decoderAct, predictAct);
%% create objectives
decoderObj = Likelihood('cross-entropy');
decoderObj.x.connect(decoderAct.O{1});
decoderObj.ref.connect(decoderReferFix.O{1});
predictObj = Likelihood('cross-entropy');
predictObj.x.connect(predictAct.O{1});
predictObj.ref.connect(predictReferFix.O{1});
%% create error generator
errgen = DataGenerator('zero', nhidunit, 'tmode', 10, '-errmode');
errgen.data.connect(encoder2L.DO{1});
%% create task
task = CustomTask(taskid, taskdir, model, {mmnist, zerogen}, {decoderObj, predictObj}, {}, ...
    'prevnet', prevnet, 'postnet', postnet, 'errgen', errgen, ...
    'iteration', istart, '-nosave');
%% setup optmizator
opt = HyperParam.getOptimizer();
opt.gradmode('basic');
opt.stepmode('adapt', 'estimatedChange', 4e-3);
opt.enableRcdmode(3);
%% run task
latestsave = [];
for i = 1 : 10
    task.run(10, 500, 32, 64);
    encoder1ldump = encoder1L.dump();
    encoder2ldump = encoder2L.dump();
    decoder1ldump = decoder1L.dump();
    decoder2ldump = decoder2L.dump();
    predict1ldump = predict1L.dump();
    predict2ldump = predict2L.dump();
    save(fullfile(savedir, sprintf(namept, task.iteration)), ...
        'encoder1ldump', 'decoder1ldump', 'predict1ldump', ...
        'encoder2ldump', 'decoder2ldump', 'predict2ldump', ...
        '-v7.3');
    if not(isempty(latestsave))
        delete(latestsave);
    end
    latestsave = fullfile(savedir, sprintf(namept, task.iteration));
end
