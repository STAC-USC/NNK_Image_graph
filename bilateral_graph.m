function [W, per_pixel_timing] = bilateral_graph(image, wsz, sigma_f, sigma_d)
shape = size(image);
h=shape(1); w=shape(2);
N = h*w;
%% Mask position values
fname = ['mask_distances/updated_wsz_' num2str(wsz) '.mat'];
if ~exist(fname, 'file')
    generate_mask_distance(wsz);
end
load(fname, 'mask_size', 'mask_positions', 'D_d_i');
%%
g_d_i = exp(-D_d_i./(sigma_d));
%%
neighbor_indices = ones(N, mask_size - 1);
weight_values = zeros(N, mask_size - 1);
per_pixel_timing = zeros(h,w);
%% NNK Filtering 
for row=1:h
    for col=1:w
        tic;
        row_values = row+mask_positions(:,2);
        col_values = col+mask_positions(:,1);
        g_curr = g_d_i;
        if (row<wsz+1)||(row>h-wsz-1)||(col<wsz+1)||(col>w-wsz-1)  
            outside_image_index = (row_values <=0 | row_values>h | col_values<=0 | col_values>w);
            row_values(outside_image_index) = []; col_values(outside_image_index) = [];
            g_curr(outside_image_index)=[];
        end
        
        neighbor_length = length(row_values);
        
        p_values = image(sub2ind(size(image),row_values, col_values));
        p_values = p_values(:);
        p_i = image(row,col); 
        mask_intensities = p_values - p_i;
        weights = g_curr.*exp(-(mask_intensities).^2/(sigma_f)); %
        %% Output calculation
        nbr_index = (row_values -1)*w + col_values;
        node_index = (row-1)*w + col;
        neighbor_indices(node_index, 1:neighbor_length) = nbr_index;
        weight_values(node_index, 1:neighbor_length) = weights;
        %%
        per_pixel_timing(row,col) = toc;
    end
end
row_indices = repmat(1:N, mask_size - 1,1)';
W = sparse(row_indices(:), neighbor_indices(:), weight_values(:), N, N);
W = max(W, W');
end