% MODEL : Complex Generative Model on NPLab3D dataset
% CODE  : 
%% model parameters
nhidunit = 1024;
validsize = 8;
whitenSizeOut = 269;
%% enviroment variables
istart  = 2e4;
taskid  = ['COMPGEN', num2str(nhidunit), 'NPLAB3D'];
taskdir = exproot();
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
plotdir = fullfile(taskdir, 'fig');
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
%% setup inference option
model.inferOption = struct( ...
    'Method',      'bb',  ...
    'Display',     'iter', ...
    'MaxIter',     100,    ...
    'MaxFunEvals', 110);
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
