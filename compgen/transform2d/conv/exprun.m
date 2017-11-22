% MODEL : Complex Generative Model on Transform2D dataset
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/d145b9582f8fa3ea1ce65b0da6721e3498236535
function model = exprun(istart, initEstch, nepoch)
%% check environment
ishpc = isunix && not(ismac);
%% load package to MATLAB search path
if ishpc
    addpath('/home/hxz244'); 
    pathLoader('umpoo');
end
%% apply default setting if necessary
if not(exist('istart', 'var'))
    istart = 0;
end
if not(exist('initEstch', 'var'))
    initEstch = 1e-3;
end
if not(exist('nepoch', 'var'))
    nepoch = 10;
end
%% model parameters
if ishpc
    nchannel  = 32;
    nbatch    = 100;
    batchsize = 8;
    validsize = 128;
    taskopt   = {};
else
    nchannel  = 8;
    nbatch    = 3;
    batchsize = 4;
    validsize = 32;
    taskopt   = {'-nosave'};
end
fltsize = [5, 5];
nfilter = 1;
%% environment parameters
taskdir = exproot();
taskid  = ['CCONVGEN', num2str(nchannel), 'TRANSFORM2D'];
savedir = fullfile(taskdir, 'records');
% datadir = fullfile(taskdir, 'data');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% load dataset and parameter setup
dataset = Transform2D();
%% create/load units and model
if istart == 0
    model = GenerativeUnit(PolarCCT.randinit(fltsize, nchannel, nfilter), @PolarCCT.normalize);
else
    load(fullfile(savedir, sprintf(namept, istart)));
    model = BuildingBlock.loaddump(modeldump);
end
model.appendto(dataset.data);
model.kernel.useCOModelNormalization = true;
model.O{1}.addPrior('cauchy', 'stdvar', sqrt(2));
model.O{1}.addPrior('slow', 'dim', 3, 'stdvar', sqrt(2));
model.noiseStdvar = 0.3;
%% create task
task = GenerativeTask(taskid, taskdir, model, dataset, ...
    'iteration', istart, taskopt{:});
%% setup inference options
model.inferOption = struct( ...
    'Method',      'bb', ...
    'Display',     'off', ...
    'MaxIter',     20, ...
    'MaxFunEvals', 30);
%% setup optmizator
opt = HyperParam.getOptimizer();
opt.gradmode('basic');
opt.stepmode('adapt', 'estimatedChange', initEstch);
opt.enableRcdmode(3);
%% run task
task.run(nepoch, nbatch, batchsize, validsize);
%% END
