function [Rec] = SRC(tr_dat, tt_dat, trls, ttls)
lambda = 1e-3;
tolerance = 1e-3;

maxIteration = 1000;
isNonnegative = false;
for k = 1:size(tt_dat,2)
    [tmp_c, tmp_iter] = SolveHomotopy(tr_dat, tt_dat(:,k), ...   
                'maxIteration', maxIteration,...
                'isNonnegative', isNonnegative, ...
                'lambda', lambda, ...
                'tolerance', tolerance);
    coef(:,k) = tmp_c;
end

% --- In check, i.e. classification
for indTest = 1:size(tt_dat,2)
    predict_ID(indTest) = IDcheck(tr_dat, coef(:,indTest), tt_dat(:,indTest), trls);
end
cornum      =   sum(predict_ID==ttls);
% recognition rate
Rec   =   [cornum/length(ttls)];
fprintf(['The classification result of SRC is about ' num2str(Rec) '% \n']);

