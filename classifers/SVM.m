function Rec = SVM(tr_dat, tt_dat, trls, ttls)

addpath('./SVM/');

trls = reshape(trls,[],1);
ttls = reshape(ttls,[],1);
tr_dat = sparse(tr_dat);
tt_dat = sparse(tt_dat);


Model = train(trls, tr_dat,'liblinear_options','col');
[PLabel] = predict(ttls, tt_dat, Model,'liblinear_options','col');
Rec = sum(PLabel==ttls)/length(ttls);
% fprintf(['The classification result of SVM is about ' num2str(Rec) '\n']);