clear all;

wd = '';
addpath(strcat(wd, 'luppi2023'));
addpath(strcat(wd, 'luppi2023/functions'));
javaaddpath(strcat(wd, 'luppi2023/functions/infodynamics.jar'));

parpool("local", 20);

subj_list = readlines(strcat(wd, 'data/subjects_reinder326.txt'));

parc_str = 'schaefer100x7';
num_nodes = 100;
tau = 1;
MMI_CCS = 'MMI';

hcp_dir = ''

full_res = zeros(326, 4, num_nodes, num_nodes, 16);
for subj_i=1:326
    for run_i=1:4
        disp(['Doing subj=', num2str(subj_i), ', run=', num2str(run_i)]);
        curr_fname = strcat('subj-', subj_list(subj_i), '_run-', num2str(run_i), '_atlas-', parc_str, '.ptseries.nii');
        curr_data = squeeze(niftiread(strcat(hcp_dir, curr_fname)));
        % curr_data(:, [4, 39]) = []; only for aparc
        curr_data = curr_data';

        for row = 1:num_nodes
            % disp(['Doing row=', num2str(row)]);
            parfor col = 1:num_nodes
                if row == col
                    continue
                end
                curr_res = PhiIDFull([curr_data(row, :); curr_data(col, :)], tau, MMI_CCS);
                full_res(subj_i, run_i, row, col, :) = cell2mat(struct2cell(curr_res));
            end
        end
    end
end

save(strcat(wd, 'outputs/HCP_S1200_', parc_str, '_PhiIDFull_', MMI_CCS, '.mat'),"full_res","-v7");
% save("./outputs/PhiIDFull_CCS.mat","full_res","-v7");
% save("./outputs/PhiIDFullDiscrete_MMI.mat","full_res","-v7");
% save("./outputs/PhiIDFullDiscrete_CCS.mat","full_res","-v7");