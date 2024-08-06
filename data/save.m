fnames = fieldnames(ratings);
for i = 1:numel(fnames)
    tableName = ['table_' fnames{i} '.csv'];
    writetable(ratings.([fnames{i}]), tableName);
end


fnames = fieldnames(projDataGlove);
for i = 1:numel(fnames)
    tableName = ['glove_' fnames{i} '.csv'];
    writematrix(projDataGlove.([fnames{i}]), tableName);
end
