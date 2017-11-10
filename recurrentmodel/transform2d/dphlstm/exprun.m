% MODEL : Complex Recurrent Model on Transform2D dataset
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/74052446370095b0e8260d877774af0e18e04c04
%% Function Start Here
function exprun(istart, initEstch, nloop, swLinBase)
% default setttings
if not(exist('istart', 'var')),    istart = 0;        end
if not(exist('initEstch', 'var')), initEstch = 1e-3;  end
if not(exist('nloop', 'var')),     nloop = 10;        end
if not(exist('swLinBase', 'var')), swLinBase = false; end
%% check environment
ishpc = isunix && not(ismac);
%% load package to MATLAB search path
if ishpc
    addpath('/home/hxz244'); 
    pathLoader('umpoo');
end
%% model parameters
if ishpc
    nepoch    = 20;
    nbatch    = 500;
    batchsize = 32;
    validsize = 128;
    swSave    = true;
else
    nepoch    = 10;
    nbatch    = 3;
    batchsize = 8;
    validsize = 32;
    swSave    = false;
end
% structure settings
nbases   = 1024;
nhidunit = 1024;
nframeEncoder = 15;
nframePredict = 15;
%% environment parameters
taskdir = exproot();
if swLinBase
    taskid = ['DPHLSTM', num2str(nhidunit), 'TRANSFORM2D-EVBASE'];
else
    taskid  = ['DPHLSTM', num2str(nhidunit), 'TRANSFORM2D'];
end
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% load dataset and parameter setup
dataset   = Transform2D('nobjects', 1, 'nframes', nframeEncoder + nframePredict);
framesize = dataset.framesize;
%% load linear bases
comodel = load(fullfile(datadir, 'comodel_transform2d.mat'));
%% create whitening unit
load(fullfile(datadir, 'statrans_transform2d.mat'));
whitening = BuildingBlock.loaddump(stdump);
whitening.compressOutput(size(comodel.iweight, 1));
whitening.frozen = true;
stat = whitening.getKernel(framesize);
%% create/load units and model
if istart == 0
    encoder     = DPHLSTM.randinit(nhidunit, [], []);
    predict     = DPHLSTM.randinit(nhidunit, [], nbases);
    reTransform = LinearTransform.randinit(stat.sizeout, nhidunit);
    imTransform = LinearTransform.randinit(stat.sizeout, nhidunit);
else
    load(fullfile(savedir, sprintf(namept, istart)));
    encoder     = BuildingBlock.loaddump(encoderdump);
    predict     = BuildingBlock.loaddump(predictdump);
    reTransform = BuildingBlock.loaddump(retransformdump);
    imTransform = BuildingBlock.loaddump(imtransformdump);
end
encoder.stateAheadof(predict);
% create assistant units
crdTransform = Cart2Polar().appendto( ...
    reTransform, imTransform).aheadof(encoder.DI{1}, encoder.DI{2});
model = Model(reTransform, imTransform, crdTransform, encoder, predict);
%% create prevnet
inputSlicer  = FrameSlicer(nframeEncoder, 'front', 0).appendto( ...
    dataset.data).aheadof(whitening);
outputSlicer = FrameSlicer(nframePredict, 'front', nframeEncoder).appendto(dataset.data);
outputShaper = Reshaper().appendto(outputSlicer);
% connect whitening
whitening.aheadof(reTransform).aheadof(imTransform);
% compose prevnet
prevnet = Model(whitening, inputSlicer, outputSlicer, outputShaper);
%% create postnet
ampact = SimpleActivation('ReLU').appendto(predict.DO{1});
angact = SimpleActivation('tanh').appendto(predict.DO{2});
angscaler = Scaler(pi).appendto(angact);
cotransform = PolarCLT(comodel.rweight, comodel.iweight, zeros(stat.sizeout, 1)).appendto( ...
    ampact, angscaler);
if not(swLinBase)
    cotransform.freeze();
end
dewhiten = LinearTransform(stat.decode, stat.offset(:)).appendto(cotransform).freeze();
postnet = Model(ampact, angact, angscaler, cotransform, dewhiten);
%% create zero generators
zerogen = DataGenerator('zero', nhidunit, 'tmode', nframeEncoder);
zerogen.data.connect(predict.DI{1});
zerogen.data.connect(predict.DI{2});
errgen = DataGenerator('zero', nhidunit, 'tmode', nframeEncoder, '-errmode');
errgen.data.connect(encoder.DO{1});
errgen.data.connect(encoder.DO{2});
%% create objectives
lossfun = Likelihood('mse');
lossfun.x.connect(dewhiten.O{1});
lossfun.ref.connect(outputShaper.O{1});
%% create task
task = CustomTask(taskid, taskdir, model, {dataset, zerogen}, lossfun, {}, ...
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
    if swSave
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
end