% MODEL : Complex Generative Model on NPLab3D Dataset
% CODE  : 
function exprun(istart, initEstch, nepoch)
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
    nhidunit  = 1024;
    nbatch    = 1000;
    batchsize = 32;
    validsize = 128;
    taskopt   = {};
else
    nhidunit  = 1024;
    nbatch    = 5;
    batchsize = 8;
    validsize = 32;
    taskopt   = {'-nosave'};
end
whitenSizeOut = 269;
%% environment parameters
taskdir = exproot();
taskid  = ['COMPGEN', num2str(nhidunit), 'NPLAB3D'];
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% load dataset and parameter setup
load(fullfile(datadir, 'nplab3d.mat'))
dataset   = nplab3d.shuffle();
framesize = dataset.stat.smpsize;
%% load statistic information
stunit = StatisticTransform(dataset.stat, 'mode', 'whiten');
stunit.compressOutput(whitenSizeOut);
stunit.appendto(dataset.data);
stunit.frozen = true;
stat = stunit.getKernel(framesize);
%% create/load units and model
if istart == 0
    model = GenerativeUnit(PolarCLT.randinit(nhidunit, whitenSizeOut));
else
    load(fullfile(savedir, sprintf(namept, istart)));
    model = BuildingBlock.loaddump(modeldump);
end
model.appendto(stunit);
model.kernel.useCOModelNormalization = true;
model.I{1}.objweight = sqrt(stat.pixelweight);
model.O{1}.addPrior('cauchy', 'stdvar', sqrt(2));
model.O{1}.addPrior('slow', 'stdvar', sqrt(2));
model.noiseStdvar = 0.3;
%% create task
task = GenerativeTask(taskid, taskdir, model, dataset, ...
    'prevnet', stunit, 'iteration', istart, taskopt{:});
%% setup inference options
model.inferOption = struct( ...
    'Method',      'bb', ...
    'Display',     'off', ...
    'MaxIter',     40, ...
    'MaxFunEvals', 50);
%% setup optmizator
opt = HyperParam.getOptimizer();
opt.gradmode('basic');
opt.stepmode('adapt', 'estimatedChange', initEstch);
opt.enableRcdmode(3);
%% run task
task.run(nepoch, nbatch, batchsize, validsize);
%% END
