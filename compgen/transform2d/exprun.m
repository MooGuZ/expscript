% MODEL : Complex Generative Model on Transform2D dataset
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/c5ed9b6fc7cfe750ad85cea4068bfd74249ec5b8
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
    nbatch    = 500;
    batchsize = 32;
    validsize = 128;
    taskopt   = {};
else
    nhidunit  = 1024;
    nbatch    = 3;
    batchsize = 8;
    validsize = 32;
    taskopt   = {'-nosave'};
end
whitenSizeOut = 512;
%% environment parameters
taskdir = exproot();
taskid  = ['COMPGEN', num2str(nhidunit), 'TRANSFORM2D'];
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% load dataset and parameter setup
dataset = Transform2D();
framesize = dataset.framesize;
%% load statistic information
load(fullfile(datadir, 'statrans_transform2d.mat'));
stunit = Interface.loaddump(stdump);
stunit.frozen = true;
stunit.compressOutput(whitenSizeOut);
stunit.appendto(dataset.data);
stat = stunit.getKernel(framesize);
%% create/load units and model
if istart == 0
    model = GenerativeUnit(PolarCLT.randinit(nhidunit, whitenSizeOut));
else
    load(fullfile(savedir, sprintf(namept, istart)));
    model = Interface.loaddump(modeldump);
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
