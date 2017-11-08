% MODEL : LSTM ENCODER-DECODER MODEL on NPLAB3D
% CODE  : https://github.com/MooGuZ/UMPrest.OO/commit/5bf1b0676235820300760dfda9d435135d3d6721
%% load package to MATLAB search path
addpath('/home/hxz244'); pathLoader('umpoo');
%% environment parameters
istart  = 20000;
taskid  = 'LSTMCODEC';
taskdir = fileparts(mfilename('fullpath'));
savedir = fullfile(taskdir, 'records');
datadir = fullfile(taskdir, 'data');
namept  = [taskid, '-ITER%d-DUMP.mat'];
estch   = 1e-3;
%% model parameters
sizein  = 1024;
sizeout = 1024;
nframes = 7;
%% create/load units and model
if istart == 0
    encoder = PHLSTM.randinit(sizein, sizeout);
    decoder = PHLSTM.randinit(sizein, sizeout);
else
    load(fullfile(savedir, sprintf(namept, istart)));
    encoder = Evolvable.loaddump(encoderdump);
    decoder = Evolvable.loaddump(decoderdump);
end
encoder.setupOutputMode('last');
decoder.enableSelfeed(nframes - 1);
encoder.aheadof(decoder);
model = Model(encoder, decoder);
%% create side path
reverser = FrameReorder('reverse');
reshaper = Reshaper().appendto(reverser);
sidepath = Model(reverser, reshaper);
%% create objectives
objective = Likelihood('mse');
%% load dataset
load(fullfile(datadir, 'NPLab3D.mat'));
nplab3d.enableSliceMode(nframes);
%% connect units to dataset and objective
nplab3d.data.connect(model.I{1});
nplab3d.data.connect(sidepath.I{1});
objective.x.connect(model.O{1});
objective.ref.connect(sidepath.O{1});
%% create task
task = CustomTask(taskid, taskdir, model, nplab3d, objective, {}, ...
    'sidepath', sidepath, 'iteration', istart, '-nosave');
%% setup optmizator
opt = HyperParam.getOptimizer();
opt.gradmode('basic');
opt.stepmode('adapt', 'estimatedChange', estch);
% opt.enableRcdmode(3);
opt.disableRcdmode();
%% run task
% latestsave = [];
for i = 1 : 5
    task.run(20, 500, 64, 128);
    encoderdump = encoder.dump();
    decoderdump = decoder.dump();
    save(fullfile(savedir, sprintf(namept, task.iteration)), ...
        'encoderdump', 'decoderdump', '-v7.3');
    % if not(isempty(latestsave))
    %     delete(latestsave);
    % end
    % latestsave = fullfile(savedir, sprintf(namept, task.iteration));
    estch = estch / 2;
    opt.stepmode('adapt', 'estimatedChange', estch);
end
