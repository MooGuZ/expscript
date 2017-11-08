% MODEL : Complex Generative Model on Transform2D dataset
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/c5ed9b6fc7cfe750ad85cea4068bfd74249ec5b8
%% model parameters
nhidunit = 1024;
validsize = 8;
whitenSizeOut = 512;
%% enviroment variables
istart  = 5e4;
taskid  = ['COMPGEN', num2str(nhidunit), 'TRANSFORM2D'];
taskdir = exproot();
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
plotdir = fullfile(taskdir, 'fig');
namept  = [taskid, '-ITER%d-DUMP.mat'];
%% load dataset and parameter setup
dataset = Transform2D();
framesize = dataset.framesize;
%% load statistic information
load(fullfile(datadir, 'statrans_transform2d.mat'));
stunit = Interface.loaddump(stdump);
stunit.compressOutput(whitenSizeOut);
stunit.frozen = true;
% connect to dataset
stunit.appendto(dataset.data);
% get statistic information
stat = stunit.getKernel(framesize);
%% load units and model
load(fullfile(savedir, sprintf(namept, istart)));
model = Interface.loaddump(modeldump);
model.I{1}.objweight = stat.pixelweight;
model.O{1}.addPrior('cauchy', 'stdvar', sqrt(2));
model.O{1}.addPrior('slow',   'stdvar', sqrt(2));
model.noiseStdvar = 0.3;
% connect to whitening unit
model.appendto(stunit);
%% setup inference option
model.inferOption = struct( ...
    'Method',      'lbfgs',  ...
    'Display',     'iter', ...
    'MaxIter',     40,    ...
    'MaxFunEvals', 50);
%% reconstruction process
dpkg = dataset.next(validsize);
wpkg = stunit.forward(dpkg);
[alpha, phi] = model.forward(wpkg);
rwpkg = model.backward(alpha, phi);
rpkg  = stunit.backward(rwpkg);
%% calculate objectives
[likelihood, prior] = model.status();
fprintf('Objective Value :  %.5e <L:%.2e| P:%.2e>\n', likelihood + prior, likelihood, prior);
%% show animation
animview({dpkg, rpkg});
%% END
