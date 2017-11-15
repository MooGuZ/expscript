function [bir, mir] = nnconv(probScale, ntry, batchsize)
useGPU  = logical(gpuDeviceCount);
verstr  = version('-release');
ver2017 = str2double(verstr(1:4)) >= 2017;
% default values
if not(exist('batchsize', 'var')), batchsize = 1; end
% calculate fundamental paramters
frmsize  = [probScale, probScale];
nchannel = ceil(log2(probScale));
fltsize  = ceil(sqrt(frmsize));
nfilter  = nchannel;
padding  = { ...
    floor(fltsize(1) / 2), floor(fltsize(2) / 2), ...
    floor((fltsize(1) - 1) / 2), floor((fltsize(2) - 1) / 2)};
% use gpu version if possible
if useGPU
    bifunc = @nnet.internal.cnngpu.convolveForward2D;
else
    bifunc = @nnet.internal.cnnhost.convolveForward2D;
end
% generate filter
f = randn([fltsize, nchannel, nfilter]);
if useGPU
    f = gpuArray(f);
end
b = zeros(1, 1, nfilter);
% data generator
datagen = DataGenerator('normal', [frmsize, nchannel]);
% experiment start
bitime = 0; % timer of build-in function 
mitime = 0; % timer of my implementation
for i = 1 : ntry
    data = datagen.next(batchsize).data;
    if useGPU
        data = gpuArray(data);
    end
    % run my implementation
    tstart = tic;
    mir = MathLib.nnconv(data, f, b, 'same');
    etime = toc(tstart);
    mitime = mitime + etime;
    % run build-in function
    if ver2017
        tstart = tic;
        bir = bifunc(data, f, padding{:}, 1, 1) + b;
        etime = toc(tstart);
    else
        tstart = tic; 
        bir = bifunc(data, f, padding{1}, padding{2}, 1, 1) + b; 
        etime = toc(tstart);
    end
    bitime = bitime + etime;
end
bitime = bitime / batchsize / ntry;
mitime = mitime / batchsize / ntry;
% print results
fprintf('Average Running Time for Build-In Function : %.4e\n', bitime);
fprintf('Average Running Time for My Implementation : %.4e\n', mitime);
