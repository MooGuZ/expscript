function nnconv(probScale, ntry)
useGPU = logical(gpuDeviceCount);
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
    data = datagen.next().data;
    if useGPU
        data = gpuArray(data);
    end
    % run my implementation
    tstart = tic;
    MathLib.nnconv(data, f, b, 'same');
    etime = toc(tstart);
    mitime = mitime + etime;
    % run build-in function
    tstart = tic; 
    bifunc(data, f, padding{:}, 1, 1) + b; 
    etime = toc(tstart);
    bitime = bitime + etime;
end
% print results
fprintf('Average Running Time for Build-In Function : %.4e\n', bitime / ntry);
fprintf('Average Running Time for My Implementation : %.4e\n', mitime / ntry);

        
