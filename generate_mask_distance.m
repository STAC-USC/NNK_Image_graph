function generate_mask_distance(wsz)
%%%% Generate and save distances for various mask sizes
mask_size = (2*wsz+1)^2;
center = floor(0.5*mask_size)+1;
[x,y] = meshgrid(-wsz:wsz, -wsz:wsz);
x = x(:); x(center) = []; 
y = y(:); y(center) = []; 
%%
mask_positions = [x y]; % x and y are column vectors
mask_inter_positions = cat(3, x-x', y-y');
%% Additional arrays 
fname = ['mask_distances/' 'updated_wsz_' num2str(wsz) '.mat'];
D_d_i = mask_positions(:,1).^2 + mask_positions(:,2).^2;
D_d_neighbors = mask_inter_positions(:,:,1).^2 + mask_inter_positions(:,:,2).^2;
[~, index] = sort(D_d_i);
sorted_positions = mask_positions(index, :);
nodes_on_line_indexes = cell(mask_size - 1, 1);
thresh_multiplier = cell(mask_size - 1, 1);
%%
for itr = 1:length(index)-1
    Delta = (sorted_positions(itr, :) - sorted_positions)*sorted_positions(itr,:)';
    nodes_to_compare = find(Delta <= 0); % This condition makes sure we are looking only in one direction
    nodes_to_keep = nodes_to_compare > itr;
    nodes_on_line_indexes{itr,1}=nodes_to_compare(nodes_to_keep);
    thresh_multiplier{itr,1}=Delta(nodes_on_line_indexes{itr,1});
end

%% Save mat
save(fname, 'wsz', 'center', 'mask_size', 'mask_positions',...
    'sorted_positions', 'D_d_i', 'D_d_neighbors', 'nodes_on_line_indexes', 'thresh_multiplier', 'index');
end
