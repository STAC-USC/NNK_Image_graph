clc;
clear;
close all;
%%
image_name = 'lena';
sigma_d = 2; sigma_f = 10/255;
wsz = 5; % Window size is (2*wsz + 1). For e.g wsz=5 --> 11x11 window
%%
img = im2double(imread([image_name '.png']));
shape = size(img);
h=shape(1); w=shape(2);
N = h*w;
if length(shape) == 3
    c=shape(3);
else
    c = 1;
end
%%
[bf.W, bf.per_pixel_time] = bilateral_graph(img, wsz, 2*sigma_f^2, 2*sigma_d^2);
bf.graph_construction_t = sum(bf.per_pixel_time(:));
[nnk.W, nnk.per_pixel_time] = smart_nnk_inverse_kernel_graph(img, wsz, 2*sigma_f^2, 2*sigma_d^2);
nnk.graph_construction_t = sum(nnk.per_pixel_time(:));
%%
bf.sparsity = length(find(bf.W))/2;
nnk.sparisty = length(find(nnk.W))/2;
%%
f = reshape(img,N,1);
f_ac = f - (f'*ones(N,1))*ones(N,1)/N; % remove the dc part 
f_energy = f_ac'*f_ac; % normalized the input signal
%% SGW parameters 
params.m=6; % Polynomial degree to use for function approximation
params.Nscales=6; % No. of wavelet frequency bands
%%
tic
bf.wavelets = get_wavelets(f_ac, bf.W, params);
bf.sgwt_t = toc;
tic
nnk.wavelets = get_wavelets(f_ac, nnk.W, params);
nnk.sgwt_t = toc;
%%
plot_wavelets(bf, nnk, h, w)
print(['results/' image_name '_wavelet_images.eps'], '-deps');
%% plot energy
bf.total_energy = 0; nnk.total_energy = 0;
bf.energy=zeros(params.Nscales+1,1); nnk.energy=zeros(params.Nscales+1,1);
for k = 1:params.Nscales+1
    bf.energy(k) = sum(bf.wavelets{:,k}.^2)/f_energy;
    nnk.energy(k) = sum(nnk.wavelets{:,k}.^2)/f_energy;
end
%%
figure(); hold on; grid on;
plot(1:params.Nscales+1, cumsum(bf.energy), 'b*-', 'LineWidth', 1.5, 'DisplayName', 'BF Graph');
plot(1:params.Nscales+1, cumsum(nnk.energy), 'rx-', 'LineWidth', 1.5, 'DisplayName', 'Proposed Graph');
ylim([0,3]);
title(['Poly. degree = ' num2str(params.m)]);
legend('show'); legend('Location', 'southeast');
set(gca, 'FontSize', 14);
print(['results/' image_name '_wavelet_energy.eps'], '-deps');
%%
display(bf)
display(nnk)
%%
save(['results/' image_name '_data.mat'], 'bf', 'nnk', 'wsz', 'sigma_d', 'sigma_f', 'params', 'image_name');
%%
function plot_wavelets(bf, nnk, h, w)
    figure('units','normalized','outerposition',[0 0 1 1])
    n_wavelets = length(bf.wavelets);
    for k=1:n_wavelets
        bf_w = bf.wavelets{:,k}; nnk_w = nnk.wavelets{:,k};
        subplot(2,n_wavelets,k); imagesc(reshape(bf_w, h,w));
        colormap('gray');axis off; title(['Var: ' num2str(var(bf_w), '%e')]);
        subplot(2,n_wavelets,n_wavelets+k); imagesc(reshape(nnk_w, h, w));
        colormap('gray');axis off; title(['Var: ' num2str(var(nnk_w), '%e')]);
    end
    sgtitle('Different bands of Spectral Wavelet - BF(top) vs NNK(bottom)');
end

%%
function [wavelet_output] = get_wavelets(f, W, params)
    L = sparse(sgwt_laplacian(W));
    lmax = sgwt_rough_lmax(L);
    arange = [0, lmax];
    %% Construct 
    designtype='abspline3'; % designtype - type of filter used in the spectral domain 
    [g, t] = sgwt_filter_design(lmax,params.Nscales,'designtype',designtype);
%     [A, B] = sgwt_framebounds(g, 0, lmax);
%     fprintf('Framebounds: A = %f , B = %f', A, B);
    c = cell(numel(g),1);
    for k=1:numel(g)
        c{k} = sgwt_cheby_coeff(g{k},params.m,params.m+1,arange);
    end

    wavelet_output =  cell(1,params.Nscales+1);
    wavelet_output(:,:) = sgwt_cheby_op(f, L, c, arange);
end