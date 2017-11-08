% MODEL : LSTM ENCODER-DECODER MODEL on Moving MNIST
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/55a25c44f3c5bdd19ee94c63393e243845b8480d
%% load package to MATLAB search path
addpath('/home/hxz244'); pathLoader('umpoo');
%% environment parameters
istart  = 0;
taskid  = 'UNIT256';
taskdir = fileparts(mfilename('fullpath'));
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% model parameters
framesize = [64, 64];
npixel    = prod(framesize);
nhidunit  = 256;
nframes   = 20;
%% create/load units and model
if istart == 0
    encoder = PHLSTM.randinit(nhidunit, npixel);
    decoder = PHLSTM.randinit(nhidunit, npixel, npixel);
    predict = PHLSTM.randinit(nhidunit, npixel, npixel);
else
    load(fullfile(savedir, sprintf(namept, istart)));
    encoder = Evolvable.loaddump(encoderdump);
    decoder = Evolvable.loaddump(decoderdump);
    predict = Evolvable.loaddump(predictdump);
end
encoder.stateAheadof(decoder).stateAheadof(predict);
model = Model(encoder, decoder, predict);
%% load dataset
load(fullfile(datadir, 'mmnist.mat'));
mmnist.nframes = nframes;   
mmnist.canvasSize = framesize;
%% create prevnet
encoderInput = FrameSlicer(10, 'front', 0).appendto(mmnist.data).aheadof(encoder.DI{1});
decoderRefer = FrameReorder('reverse').appendto(encoderInput);
decoderReferFix = Reshaper().appendto(decoderRefer);
decoderInput = FrameSlicer(9, 'front', 1).appendto(mmnist.data);
decoderInputFixA = FrameReorder('reverse').appendto(decoderInput);
decoderInputFixB = FrameInsert(1, 'front', 0).appendto(decoderInputFixA).aheadof(decoder.DI{1});
predictRefer = FrameSlicer(10, 'back', 0).appendto(mmnist.data);
predictReferFix = Reshaper().appendto(predictRefer);
predictInput = FrameSlicer(9, 'back', 1).appendto(mmnist.data);
predictInputFix = FrameInsert(1, 'front', 0).appendto(predictInput).aheadof(predict.DI{1});
prevnet = Model(encoderInput, decoderRefer, decoderReferFix, decoderInput, decoderInputFixA, ...
    decoderInputFixB, predictRefer, predictReferFix, predictInput, predictInputFix);
%% create postnet
decoderAct = Activation('logistic'); decoderAct.appendto(decoder.DO{1});
predictAct = Activation('logistic'); predictAct.appendto(predict.DO{1});
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
errgen.data.connect(encoder.DO{1});
%% create task
task = CustomTask(taskid, taskdir, model, mmnist, {decoderObj, predictObj}, {}, ...
    'prevnet', prevnet, 'postnet', postnet, 'errgen', errgen, ...
    'iteration', istart, '-nosave');
%% setup optmizator
opt = HyperParam.getOptimizer();
opt.gradmode('basic');
opt.stepmode('adapt', 'estimatedChange', 1e-2);            
opt.enableRcdmode(3);
%% run task
% latestsave = [];
for i = 1 : 10
    task.run(10, 500, 64, 128);
    encoderdump = encoder.dump();
    decoderdump = decoder.dump();
    predictdump = predict.dump();
    save(fullfile(savedir, sprintf(namept, task.iteration)), ...
        'encoderdump', 'decoderdump', 'predictdump', '-v7.3');
    % if not(isempty(latestsave))
    %     delete(latestsave);
    % end
    % latestsave = fullfile(savedir, sprintf(namept, task.iteration));
end
